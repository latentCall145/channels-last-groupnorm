#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/Exceptions.h> // AT_CUDA_CHECK
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty_like.h>
#include <ATen/OpMathType.h> // opmath_t
#include <ATen/ops/empty.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <ATen/Tensor.h> // torch tensor
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <c10/core/ScalarType.h>
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#include "Welford.h"
#define THREADS_PER_BLOCK 128 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput

// Reduces a value across the y-threads of a threadblock
template <typename T, class ReduceOp>
__device__ void
full_reduce(
    T val,
    const ReduceOp& op,
    T* output_buffer
    ) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  output_buffer[tid] = val;
  __syncthreads();

  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= 1; stride >>= 1) {
    if (tid < stride)
      output_buffer[tid] = op.combine(output_buffer[tid], output_buffer[tid + stride]);
    __syncthreads();
    }
}
template <typename T>
__global__ void
fused_kernel(
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
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, G, blockdim = d, D

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  const int D = C / G;
  //extern __shared__ char X_tmp[];
  extern __shared__ float* X_tmp[];
  T *X_shmem = reinterpret_cast<T*>(X_tmp);
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int HWC = H * W * C;
  const int HWd = H * W * D / THREADS_PER_BLOCK;
#pragma unroll 8
  for (int i = 0; i < HWd; ++i) {
    int reduce_idx = i * THREADS_PER_BLOCK * G + threadIdx.y * D * G + blockIdx.y * D + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * HWC + reduce_idx];
    X_shmem[i * blockDim.y * D + threadIdx.y * D + threadIdx.x] = x;
    val = welford_op.reduce(val, static_cast<T_ACC>(x)); // last arg isn't used in src
  }

  full_reduce(val, welford_op, vals_reduced);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + blockIdx.y] = m1;
    rstds[blockIdx.x * G + blockIdx.y] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }

  const int ng = blockIdx.x * G + blockIdx.y;
  const int c = blockIdx.y * D + threadIdx.x;
#pragma unroll 8
  for (int i = 0; i < HWd; ++i) {
    int reduce_idx = i * THREADS_PER_BLOCK * G + threadIdx.y * D * G + blockIdx.y * D + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    int X_idx = blockIdx.x * HWC + reduce_idx;
    T_ACC x = static_cast<T_ACC>(X_shmem[i * blockDim.y * D + threadIdx.y * D + threadIdx.x]);
    T mean = means[ng];
    T rstd = rstds[ng];
    T w = weight[c];
    T b = bias[c];
    Y[X_idx] = (x - mean) * rstd * w + b;
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
  int blockDimX, blockDimY, gridDimY, gridDimZ;
  blockDimX = D;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = G;
  gridDimZ = 1;

  dim3 dimGrid(N, gridDimY, gridDimZ);
  dim3 dimBlock(blockDimX, blockDimY);

  fused_kernel<T><<<dimGrid, dimBlock, sizeof(T) * H * W * D>>>(
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
          weight,
          bias,
          G,
          eps,
          X_out,
          means,
          rstds
      );
  });
  return {X_out.permute({0, 3, 1, 2}), means, rstds};
}
