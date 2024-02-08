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
#include "Welford.h"
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

template <typename T>
__global__ void
compute_scale_biases(
        T* means,  // (N, G)
        T* rstds,  // (N, G)
        const T* weight, // (C)
        const T* bias,   // (C)
        const int G,
        const int C,
        at::acc_type<T, true>* a,            // (N, C)
        at::acc_type<T, true>* b             // (N, C)
  ) {
  // (N, f), (TPB)
  const int D = C / G;
  //const int c = threadIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int g = c / D;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  const at::acc_type<T, true> a_nc = rstds[ng] * weight[c];
  a[nc] = a_nc;
  b[nc] = -means[ng] * a_nc + bias[c];
}

template <typename T>
__global__ void
N_compute_stats(
        const T* X,
        const int C,
        const int G,
        const int HWC,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, G, blockdim = d, D

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int HWc = HWC / MAX_THREADS_PER_BLOCK;
#pragma unroll 8
  for (int i = 0; i < HWc; ++i) {
    int reduce_idx = i * MAX_THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * HWC + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x));
  }

  const int D = C / G;

  // suppose vals_reduced shape is (c, G, D), we need (G,) output
  // (c,G,D) -> (D,c,G) -> (G,)
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int c_idx = threadIdx.y;
  const int g = threadIdx.x / D;
  const int d = threadIdx.x % D;
  vals_reduced[d * blockDim.y * G + c_idx * G + g] = val;
  __syncthreads();

  for (int stride = MAX_THREADS_PER_BLOCK / 2; stride >= G; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if ((int)threadIdx.x < G && threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + threadIdx.x] = m1;
    rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
void N_gn_fwd(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    T eps,
    at::Tensor& Y,
    at::Tensor& means,
    at::Tensor& rstds) {
  using T_ACC = at::acc_type<T, true>;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  int blockDimX, blockDimY, gridDimY, gridDimZ;
  blockDimX = C;
  blockDimY = MAX_THREADS_PER_BLOCK / blockDimX;
  gridDimY = 1;
  gridDimZ = 1;

  dim3 dimGrid(N, gridDimY, gridDimZ);
  dim3 dimBlock(blockDimX, blockDimY);

  N_compute_stats<T><<<dimGrid, dimBlock>>>(
      X_data, C, G, H*W*C, eps,
      mean_data, rstd_data
  );

  //scale_shift<T>(X, weight, bias, G, Y, means, rstds);
  int TPB = MAX_THREADS_PER_BLOCK;
  if (H * W >= 1024) { // add fused scale-bias kernel to reduce num math ops on each element in the elementwise kernel if the spatial resolution is large
    const T* weight_data = weight.const_data_ptr<T>();
    const T* bias_data = bias.const_data_ptr<T>();

    const at::ScalarType kAccType =
        (X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16)
        ? at::kFloat
        : X.scalar_type();

    at::Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
    at::Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    TPB = MIN(MAX_THREADS_PER_BLOCK, C);
    if (C < MAX_THREADS_PER_BLOCK)
      TPB -= TPB % C;
    else {
      int f = 1;
      while (C % f != 0 || C / f > MAX_THREADS_PER_BLOCK || G % f != 0) {
        f++;
      }
      TPB = C / f;
    }
    compute_scale_biases<<<dim3(N, 1), TPB>>>( // note: max(D, T) threads per block
        mean_data, rstd_data,
        weight_data, bias_data,
        G, C,
        a_data, b_data);

    at::TensorIterator iter = at::TensorIteratorConfig()
      .check_all_same_dtype(std::is_same<T, T_ACC>::value) // this line relaxes requirement that all inputs/outputs are same dtype if T isn't T_ACC
      .resize_outputs(false)
      .add_owned_output(Y.view({N, H * W, C}))
      .add_owned_input(X.view({N, H * W, C}))
      .add_owned_input(a.view({N, 1, C}))
      .add_owned_input(b.view({N, 1, C}))
      .build();
   
    at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      return static_cast<T_ACC>(x) * a + b;
    });
  }
  else { // if spatial resolution small, overhead of creating the extra kernel isn't worth it
    const int D = C / G;
    at::TensorIterator iter = at::TensorIteratorConfig()
      .check_all_same_dtype(std::is_same<T, T_ACC>::value) // this line relaxes requirement that all inputs/outputs are same dtype if T isn't T_ACC
      .resize_outputs(false)
      .add_owned_output(Y.view({N, H * W, G, D}))
      .add_owned_input(X.view({N, H * W, G, D}))
      .add_owned_input(means.view({N, 1, G, 1}))
      .add_owned_input(rstds.view({N, 1, G, 1}))
      .add_owned_input(weight.view({1, 1, G, D}))
      .add_owned_input(bias.view({1, 1, G, D}))
      .build();
     
    at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T weight, T bias) -> T {
      return (static_cast<T_ACC>(x) - mean) * rstd * weight + bias;
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<at::Tensor> gn_nhwc_cuda_fwd_N_grid(
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
    "group_norm_nhwc_forward_N_grid", [&]() {
      N_gn_fwd<scalar_t>(
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
