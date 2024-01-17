#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include "scale_shift_kernel.h" // scale_shift
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

template <typename T>
__global__ void
NH_compute_stats(
        const T* X,
        const int H,
        const int W,
        const int C,
        const int G,
        at::native::WelfordData<at::acc_type<T, true>, int> *welford_data
  ) {
  /*
     C <= MAX_THREADS_PER_BLOCK (Kernel 1):
       griddim: (x=N, y=H, z=f=1); blockdim: (x=C, y=d)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Cd = TPB (threads per block)
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, C); X stride: (HWC, WC, dC, C, C, 1)
       shmem reduction: (d, C) -view-> (d, G, D) -permute-> (d, D, G) -reduce-> G
       output buffer: (N, H, 1, G)
     C > MAX_THREADS_PER_BLOCK (Kernel 2):
       griddim: (x=N, y=H, z=f); blockdim: (x=TPB, y=d=1)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, TPB); X stride: (HWC, WC, dC, C, TPB, 1)
       shmem reduction: (TPB,) -view-> (1, G/f, D) -permute-> (1, D, G/f) -reduce-> G/f
       output buffer: (N, H, f, G/f)
  */
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int Wc = W / d;
#pragma unroll 8
  for (int i = 0; i < Wc; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWC stride
    reduce_idx += blockIdx.y * W * C; // dim 1, WC stride
    reduce_idx += i * d * C; // dim 2, dC stride
    reduce_idx += threadIdx.y * C; // dim 3, C stride
    reduce_idx += blockIdx.z * TPB; // dim 4, TPB stride (in kernel 1, threadIdx.z is always 0 so this statement does nothing)
    reduce_idx += threadIdx.x; // dim 5, 1 stride
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  const int D = C / G;

  // shmem reduction
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int f = gridDim.z;
  const int gf = G / f;
  const int d_idx = threadIdx.y;
  const int gf_idx = threadIdx.x / D;
  const int D_idx = threadIdx.x % D;
  int idx = 0;
  idx += d_idx * D * gf; // dim 0, DG/f stride
  idx += D_idx * gf; // dim 1, G/f stride
  idx += gf_idx; // dim 2, 1 stride
  vals_reduced[idx] = val;
  __syncthreads();

  for (int stride = TPB / 2; stride >= gf; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if (tid < gf) {
    int out_idx = 0;
    out_idx += blockIdx.x * H * G; // dim 0, HG stride
    out_idx += blockIdx.y * G; // dim 1, G stride
    out_idx += blockIdx.z * gf; // dim 2, G/f stride
    out_idx += threadIdx.x; // dim 3, 1 stride
    welford_data[out_idx] = vals_reduced[tid];
    //welford_data[blockIdx.x * H * G + blockIdx.y * G + threadIdx.x] = vals_reduced[threadIdx.x];
  }
}

template <typename T>
__global__ void
NH_compute_stats(
        at::native::WelfordData<at::acc_type<T, true>, int> *welford_data,
        const int H,
        const int G,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  /*
     griddim: (x=N, y=f); blockdim: (x=G/f, y=d)
      d = num. spatial elements (from H dimension) each thread-block processes in parallel
      Gd/f = TPB (threads per block)
     welford_data shape: (N, H, f, G/f) -view-> (N, H/d, d, f, G/f); X stride: (HG, dG, G, G/f, 1)
     shmem reduction: (d, G/f) -reduce-> G/f
     output buffer: (N, f, G/f) -view-> (N, G)
  */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const int TPB = blockDim.y * blockDim.x;
  const int f = gridDim.y;
  const int d = blockDim.y;
  const int gf = G / f;

#pragma unroll 8
  for (int i = 0; i < H / d; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * G; // dim 0, stride HG
    reduce_idx += i * d * G; // dim 1, stride dG
    reduce_idx += threadIdx.y * G; // dim 2, stride G
    //reduce_idx += threadIdx.x; // dim 3, stride 1
    reduce_idx += blockIdx.y * gf; // dim 3, stride G/f (if f = 1, this adds nothing)
    reduce_idx += threadIdx.x; // dim 4, stride 1
    //WelfordType x = welford_data[blockIdx.x * H * G + i * THREADS_PER_BLOCK + tid];
    WelfordType x = welford_data[reduce_idx];
    val = welford_op.combine(val, x);
  }

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();

  for (int stride = TPB / 2; stride >= gf; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if (tid < gf) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[tid]);
    int out_idx = 0;
    out_idx += blockIdx.x * G; // dim 0, G stride
    //out_idx += threadIdx.x; // dim 1, 1 stride
    out_idx += blockIdx.y * gf; // dim 1, G/f stride
    out_idx += threadIdx.x; // dim 2, 1 stride
    //means[blockIdx.x * G + threadIdx.x] = m1;
    //rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    means[out_idx] = m1;
    rstds[out_idx] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
void NH_gn_fwd(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    T eps,
    torch::Tensor& Y,
    torch::Tensor& means,
    torch::Tensor& rstds) {
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);

  using WelfordType = at::native::WelfordData<at::acc_type<T, true>, int>;
  torch::Tensor welford_tensor = torch::empty({N, H, G, sizeof(WelfordType)}, X.options().dtype(torch::kByte));
  WelfordType *welford_data = reinterpret_cast<WelfordType *>(welford_tensor.mutable_data_ptr());
  
  int blockDimX, blockDimY, f, TPB;
  TPB = MIN(MAX_THREADS_PER_BLOCK, W * C);
  blockDimX = MIN(TPB, C);
  blockDimY = TPB / blockDimX;
  f = MAX(C / TPB, 1); // note: impossible for f > 1 AND blockDimY > 1
  NH_compute_stats<<<dim3(N, H, f), dim3(blockDimX, blockDimY)>>>(
      X_data, H, W, C, G, 
      welford_data
  );

  TPB = MIN(MAX_THREADS_PER_BLOCK, H * G / f);
  blockDimX = MIN(TPB, G / f);
  blockDimY = TPB / blockDimX;
  NH_compute_stats<<<dim3(N, f), dim3(blockDimX, blockDimY)>>>(
          welford_data,
          H, G, eps,
          mean_data, rstd_data
    );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_fwd_NH_grid(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    float eps) {
  const int N = X.size(0);

  torch::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  torch::Tensor X_out = torch::empty_like(X_nhwc);
  torch::Tensor means = torch::empty({N, G}, weight.options());
  torch::Tensor rstds = torch::empty({N, G}, weight.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_forward_NH_grid", [&]() {
      NH_gn_fwd<scalar_t>(
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
