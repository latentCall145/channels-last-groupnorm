#include <ATen/cuda/Exceptions.h> // AT_CUDA_CHECK
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <ATen/Tensor.h> // torch tensor
#include <c10/core/ScalarType.h>
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#include "Welford.h"
#define THREADS_PER_BLOCK 128 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput

template <typename T> __global__ void fused_kernel(
        const T* X,
        const T* weight,
        const T* bias,
        const int H,
        const int W,
        const int C,
        const int G,
        const float eps,
        T* Y,
        T* means,
        T* rstds
  ) {
  /*
  dimGrid(N, G, 1)
  dimBlock(D, d)
  fused_kernel<T><<<dimGrid, dimBlock, sizeof(T) * H * W * D>>>(
      X_data, weight_data, bias_data,
      H, W, C, G, eps,
      Y_data, mean_data, rstd_data
  );
  */
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  const int D = C / G;
  extern __shared__ char X_tmp[];
  T *X_shmem = reinterpret_cast<T*>(X_tmp);
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  const int HWC = H * W * C;
  const int Hw = H * W / d;
#pragma unroll 8
  for (int i = 0; i < Hw; ++i) {
    int reduce_idx = i * d * G * D + threadIdx.y * G * D + blockIdx.y * D + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * HWC + reduce_idx];
    X_shmem[i * d * D + threadIdx.y * D + threadIdx.x] = x;
    val = welford_op.reduce(val, static_cast<T_ACC>(x));
  }

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();
  for (int stride = TPB / 2; stride >= 1; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  T_ACC mean, var;
  thrust::tie(var, mean) = welford_op.project(vals_reduced[0]);
  T_ACC rstd = rsqrt(var + static_cast<T_ACC>(eps));

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    means[blockIdx.x * G + blockIdx.y] = mean;
    rstds[blockIdx.x * G + blockIdx.y] = rstd;
  }

  const int c = blockIdx.y * D + threadIdx.x;
  T w = weight[c];
  T b = bias[c];
  T_ACC fused_scale = rstd * w;
  T_ACC fused_bias = -mean * fused_scale + b;

#pragma unroll 8
  for (int i = 0; i < Hw; ++i) {
    T_ACC x = static_cast<T_ACC>(X_shmem[i * d * D + threadIdx.y * D + threadIdx.x]);
    int reduce_idx = i * d * G * D + threadIdx.y * G * D + blockIdx.y * D + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    int X_idx = blockIdx.x * HWC + reduce_idx;
    Y[X_idx] = x * fused_scale + fused_bias;
  }
}

template <typename T>
struct alignas(4 * sizeof(T)) float_vec {
  T x, y, z, w;
};

template <typename T> __global__ void fused_kernelV4(
        const T* X,
        const T* weight,
        const T* bias,
        const int H,
        const int W,
        const int C,
        const int G,
        const float eps,
        T* Y,
        T* means,
        T* rstds
  ) {
  /*
  dimGrid(N, G, 1)
  dimBlock(D/4, d)
  fused_kernel<T><<<dimGrid, dimBlock, sizeof(T) * H * W * D>>>(
      X_data, weight_data, bias_data,
      H, W, C, G, eps,
      Y_data, mean_data, rstd_data
  );
  */
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  using V = float_vec<T>;
  using V_ACC = float_vec<T_ACC>;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  const int D = C / G;
  const int D4 = D / 4;
  extern __shared__ char X_tmp[];
  //T *X_shmem = reinterpret_cast<T*>(X_tmp);
  V *X_shmem = reinterpret_cast<V*>(X_tmp);
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);
  const V *X_vec = reinterpret_cast<const V*>(X);
  V *Y_vec = reinterpret_cast<V*>(Y);

  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  const int HWC = H * W * C;
  const int Hw = H * W / d;
#pragma unroll 8
  for (int i = 0; i < Hw; ++i) {
    int reduce_idx = i * d * G * D4 + threadIdx.y * G * D4 + blockIdx.y * D4 + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    //T x = X[blockIdx.x * HWC + reduce_idx];
    V x = X_vec[blockIdx.x * HWC/4 + reduce_idx];
    X_shmem[i * d * D4 + threadIdx.y * D4 + threadIdx.x] = x;
    val = welford_op.reduce(val, static_cast<T_ACC>(x.x));
    val = welford_op.reduce(val, static_cast<T_ACC>(x.y));
    val = welford_op.reduce(val, static_cast<T_ACC>(x.z));
    val = welford_op.reduce(val, static_cast<T_ACC>(x.w));
  }

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();
  for (int stride = TPB / 2; stride >= 1; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  T_ACC mean, var;
  thrust::tie(var, mean) = welford_op.project(vals_reduced[0]);
  T_ACC rstd = rsqrt(var + static_cast<T_ACC>(eps));

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    means[blockIdx.x * G + blockIdx.y] = mean;
    rstds[blockIdx.x * G + blockIdx.y] = rstd;
  }

  const int c = blockIdx.y * D + threadIdx.x * 4;
  V w = ((V*)&weight[c])[0];
  V b = ((V*)&bias[c])[0];
  V_ACC fused_scale = {rstd * w.x, rstd * w.y, rstd * w.z, rstd * w.w};
  V_ACC fused_bias = {-mean * fused_scale.x + b.x, -mean * fused_scale.y + b.y, -mean * fused_scale.z + b.z, -mean * fused_scale.w + b.w};

#pragma unroll 8
  for (int i = 0; i < Hw; ++i) {
    //T_ACC x = static_cast<T_ACC>(X_shmem[i * d * D + threadIdx.y * D + threadIdx.x]);
    V x = X_shmem[i * d * D4 + threadIdx.y * D4 + threadIdx.x];
    int reduce_idx = i * d * G * D4 + threadIdx.y * G * D4 + blockIdx.y * D4 + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    int X_idx = blockIdx.x * HWC/4 + reduce_idx;
    T y_x, y_y, y_z, y_w;
    y_x = static_cast<T_ACC>(x.x) * fused_scale.x + fused_bias.x;
    y_y = static_cast<T_ACC>(x.y) * fused_scale.y + fused_bias.y;
    y_z = static_cast<T_ACC>(x.z) * fused_scale.z + fused_bias.z;
    y_w = static_cast<T_ACC>(x.w) * fused_scale.w + fused_bias.w;
    Y_vec[X_idx] = {y_x, y_y, y_z, y_w};
  }
}

template <typename T>
void fused_gn_fwd(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    T eps,
    at::Tensor& Y,
    at::Tensor& means,
    at::Tensor& rstds) {
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();
  T* weight_data = weight.mutable_data_ptr<T>();
  T* bias_data = bias.mutable_data_ptr<T>();
  T* Y_data = Y.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  const int D = C / G;
  int blockDimX = D / 4;
  int blockDimY = THREADS_PER_BLOCK / blockDimX;

  //fused_kernel<T><<<dimGrid, dimBlock, sizeof(T) * H * W * D>>>(
  //    X_data, weight_data, bias_data,
  //    H, W, C, G, eps,
  //    Y_data, mean_data, rstd_data
  //);
  fused_kernelV4<T><<<dim3(N, G), dim3(blockDimX, blockDimY), sizeof(T) * H * W * D>>>(
      X_data, weight_data, bias_data,
      H, W, C, G, eps,
      Y_data, mean_data, rstd_data
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<at::Tensor> gn_nhwc_cuda_fwd_fused(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    float eps) {
  const int N = X.size(0);

  at::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  at::Tensor X_out = at::empty_like(X_nhwc);
  at::Tensor means = at::empty({N, G}, weight.options());
  at::Tensor rstds = at::empty({N, G}, weight.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_forward_fused", [&]() {
      fused_gn_fwd<scalar_t>(
          X_nhwc,
          weight, bias,
          G, eps,
          X_out, means, rstds
      );
  });
  return {X_out.permute({0, 3, 1, 2}), means, rstds};
}
