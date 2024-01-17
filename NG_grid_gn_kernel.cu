#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include "scale_shift_kernel.h" // scale_shift
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define THREADS_PER_BLOCK 128 // low threads per block bad because less occupancy, high threads per block bad because of smaller reduction loops -> more instruction overhead

template <typename T>
__global__ void
NG_compute_stats(
        const T* X,
        const int H,
        const int W,
        const int C,
        const int G,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  /*
     D >= 8: (Kernel 1)
       griddim: (x=N, y=G); blockdim: (x=D, y=d)
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Dd = THREADS_PER_BLOCK
       X shape: (N, H, W, G, D) -view-> (N, HW/d, d, G, D); X stride: (HWC, dGD, GD, D, 1)
       shmem reduction: (d, D) -reduce-> 1
       output buffer: (N, G) -view-> (N, G, 1) (this is consistent with kernel 2 where g = 1)
     D < 8: (Kernel 2)
       griddim: (x=N, y=G/g); blockdim: (x=e, y=d)
        g = num. groups each block computes in parallel
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        e = num. elements loaded in one coalesced read
        Dg = e
        ed = Ddg = THREADS_PER_BLOCK
       X shape: (N, H, W, G, D) -view-> (N, HW/d, d, G/g, e); X stride: (HWC, dGD, GD, e, 1)
       shmem reduction: (d, e) -view-> (d, g, D) -permute-> (d, D, g) -reduce-> g
       output buffer: (N, G) -view-> (N, G/g, g)
   */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  //const int HWC = HWd * THREADS_PER_BLOCK * G;
  const int D = C / G;
  const int e = blockDim.x;
  const int kernel_option = (D == e) ? 1 : 2;
#pragma unroll 8
  for (int i = 0; i < H * W / (int)blockDim.y; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWGD stride
    reduce_idx += i * blockDim.y * G * D; // dim 1, dGD stride
    reduce_idx += threadIdx.y * G * D; // dim 2, GD stride
    if (kernel_option == 1)
      reduce_idx += blockIdx.y * D; // dim 3, D stride
    else if (kernel_option == 2)
      reduce_idx += blockIdx.y * e; // dim 3, e stride
    reduce_idx += threadIdx.x; // dim 4, 1 stride
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int g = e / D;
  int reduce_elems = -1;
  int idx = 0;
  if (kernel_option == 1) {
    reduce_elems = 1;
    idx = tid;
  }
  else if (kernel_option == 2) {
    reduce_elems = g;
    const int d_idx = threadIdx.y;
    const int g_idx = threadIdx.x / D;
    const int D_idx = threadIdx.x % D;
    idx += d_idx * D * g; // dim 0, Dg stride
    idx += D_idx * g; // dim 1, g stride
    idx += g_idx; // dim 2, 1 stride
  }
  vals_reduced[idx] = val;
  __syncthreads();

  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= reduce_elems; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // place value into output buffer
  if (tid < reduce_elems) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    int out_idx = 0;
    out_idx += blockIdx.x * G; // dim 0, G stride
    out_idx += blockIdx.y * g; // dim 1, g stride
    out_idx += threadIdx.x; // dim 2, 1 stride
    means[out_idx] = m1;
    rstds[out_idx] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
void NG_gn_fwd(
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
  const int D = C / G;
  int blockDimX, blockDimY, gridDimY;
  const int e = 8;
  blockDimX = D >= e ? D : e;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = D >= e ? G : G / (e / D);

  dim3 dimGrid(N, gridDimY);
  dim3 dimBlock(blockDimX, blockDimY);

  NG_compute_stats<<<dimGrid, dimBlock>>>(
      X_data, H, W, C, G, eps,
      mean_data, rstd_data
  );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_fwd_NG_grid(
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
    "group_norm_nhwc_forward_NG_grid", [&]() {
      NG_gn_fwd<scalar_t>(
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
