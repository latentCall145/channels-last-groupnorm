#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <ATen/native/cuda/Loops.cuh> // gpu kernel
#include <ATen/AccumulateType.h> // acc_type
#include <c10/core/ScalarType.h>
#include <thrust/pair.h> // thrust::pair
#include <torch/torch.h> // torch tensor
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

template <typename T>
__global__ void
NH_spatial_loop(
      const T* dy_data,
      const T* X_data,
      const int N,
      const int H,
      const int W,
      const int C,
      at::acc_type<T, true>* xdy_sum_data,
      at::acc_type<T, true>* dy_sum_data) {
  /*
     Performs a loop over the spatial dimension W, loading and summing dy and X. Spatial dimension H is processed in a separate kernel.
     C <= MAX_THREADS_PER_BLOCK (Kernel 1):
       griddim: (x=N, y=H, z=f=1); blockdim: (x=C, y=d)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Cd = TPB (threads per block)
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
       shmem reduction: (d, C) -reduce-> C
       output buffer: (N, H, C) OR
       output buffer: (N, C, H) OR
       output buffer: (H, N, C)
     C > MAX_THREADS_PER_BLOCK (Kernel 2):
       griddim: (x=N, y=H, z=f); blockdim: (x=TPB, y=d=1)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, TPB); X stride: (HWC, WC, dC, C, TPB, 1)
       shmem reduction: (d, TPB) -reduce-> TPB
       output buffer: (N, H, f, TPB) -view-> (N, H, C) OR
       output buffer: (N, f, TPB, H) -view-> (N, C, H) OR
       output buffer: (H, N, f, TPB) -view-> (H, N, C)
   */

  using T_ACC = at::acc_type<T, true>;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;
  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  const int w = W / d;

#pragma unroll 8
  for (int i = 0; i < w; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWC stride
    reduce_idx += blockIdx.y * W * C; // dim 1, WC stride
    reduce_idx += i * d * C; // dim 2, dC stride
    reduce_idx += threadIdx.y * C; // dim 3, C stride
    reduce_idx += blockIdx.z * TPB; // dim 4, TPB stride (in kernel 1, threadIdx.z is always 0 so this statement does nothing)
    reduce_idx += threadIdx.x; // dim 5, 1 stride
    xdy_sum += static_cast<T_ACC>(dy_data[reduce_idx]) * static_cast<T_ACC>(X_data[reduce_idx]);
    dy_sum += static_cast<T_ACC>(dy_data[reduce_idx]);
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[2 * tid] = xdy_sum;
  vals_reduced[2 * tid + 1] = dy_sum;
  __syncthreads();

#pragma unroll 8
  for (int stride = TPB; stride >= 2 * C; stride >>= 1) { // stopping at 2 * C (instead of just C) since we are reducing 2*C values
    if (tid < stride) {
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
    }
  }

  // put reduced outputs into return buffers
  if (tid < C) {
    int out_idx = 0;
    out_idx += blockIdx.x * H * C; // dim 0, HC stride
    out_idx += blockIdx.y * C; // dim 1, C stride
    out_idx += blockIdx.z * TPB; // dim 2, TPB stride (if f=1, this line is a no-op)
    out_idx += threadIdx.x; // dim 3, 1 stride

    //out_idx += blockIdx.x * C * H; // dim 0, CH stride
    //out_idx += blockIdx.z * TPB * H; // dim 1, TPB*H stride (if f=1, this line is a no-op)
    //out_idx += threadIdx.x * H; // dim 2, H stride
    //out_idx += blockIdx.y; // dim 3, 1 stride

    //out_idx += blockIdx.y * N * C; // dim 0, NC stride
    //out_idx += blockIdx.x * C; // dim 1, C stride
    //out_idx += blockIdx.z * TPB; // dim 2, TPB stride (if f=1, this line is a no-op)
    //out_idx += threadIdx.x; // dim 3, 1 stride

    xdy_sum_data[out_idx] = vals_reduced[2 * tid];
    dy_sum_data[out_idx] = vals_reduced[2 * tid + 1];
  }
}

template <typename T>
__global__ void
compute_bwd_scale_biases(
    const T* mean_data,
    const T* rstd_data,
    const T* weight_data,
    at::acc_type<T, true>* xdy_sum_data,
    at::acc_type<T, true>* dy_sum_data,
    const int H,
    const int W,
    const int C,
    const int G,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data) {
  /*
     griddim: (x=N); blockdim: (x=C)
      d = num. spatial elements (from HW dimension) each thread-block processes in parallel
      Cd = TPB (threads per block)
     X shape: (N, C) -view-> (N, G, D) -permute-> (N, D, G) -reduce-> (N, G)
     shmem reduction: (D, G) -reduce-> G
     output buffer: (N, G)
   */
  using T_ACC = at::acc_type<T, true>;
  const int D = C / G;
  const int n = blockIdx.x;
  const int c = threadIdx.x;
  const int g = c / D;
  const int d = c % D;
  const T_ACC gamma_v = static_cast<T_ACC>(weight_data[c]);

  extern __shared__ char vals_reduced_uncasted[]; // size 2*C, C for sum1, C for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nc = n * C + c;
  int idx = 0;
  idx += d * G;
  idx += g;
  vals_reduced[2 * idx] = xdy_sum_data[nc] * gamma_v;
  vals_reduced[2 * idx + 1] = dy_sum_data[nc] * gamma_v;
  __syncthreads();

#pragma unroll 8
  for (int stride = C; stride >= 2 * G; stride >>= 1) {
    if (tid < stride) {
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
    }
  }

  if (tid < G) {
    const int ng = n * G + tid;
    const T_ACC sum1 = vals_reduced[2 * tid];
    const T_ACC sum2 = vals_reduced[2 * tid + 1];
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * H * W);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean_data[ng]) - sum1) *
        static_cast<T_ACC>(rstd_data[ng]) * static_cast<T_ACC>(rstd_data[ng]) *
        static_cast<T_ACC>(rstd_data[ng]) * s;
    coef2_data[ng] = x;
    coef3_data[ng] = -x * static_cast<T_ACC>(mean_data[ng]) -
      sum2 * static_cast<T_ACC>(rstd_data[ng]) * s;
  }
}

template <typename T>
__global__ void
compute_dweight_dbias(
    const T* mean_data,
    const T* rstd_data,
    at::acc_type<T, true>* xdy_sum_data,
    at::acc_type<T, true>* dy_sum_data,
    const int N,
    const int C,
    const int G,
    T* dweight_data,
    T* dbias_data) {
  // gridDim: (x=1), blockDim: (x=C)
  using T_ACC = at::acc_type<T, true>;
  const int c = threadIdx.x;
  const int D = C / G;
  const int g = c / D;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;

#pragma unroll 8
  for (int n = 0; n < N; ++n) {
    const int nc = n * C + c;
    const int ng = n * G + g;
    sum1 += ((xdy_sum_data[nc] - dy_sum_data[nc] * static_cast<T_ACC>(mean_data[ng])) * static_cast<T_ACC>(rstd_data[ng]));
    sum2 += dy_sum_data[nc];
  }
  dweight_data[c] = sum1;
  dbias_data[c] = sum2;
}

template <typename T>
void gn_bwd(
      const torch::Tensor& dy_nhwc,
      const torch::Tensor& X_nhwc,
      const torch::Tensor& weight,
      const torch::Tensor& mean,
      const torch::Tensor& rstd,
      const int G,
      torch::Tensor& dX,
      torch::Tensor& dweight,
      torch::Tensor& dbias
  ) {
  using T_ACC = at::acc_type<T, true>;
  const int N = X_nhwc.size(0);
  const int H = X_nhwc.size(1);
  const int W = X_nhwc.size(2);
  const int C = X_nhwc.size(3);
  const int D = C / G;

  const T* dy_data = dy_nhwc.const_data_ptr<T>();
  const T* X_data = X_nhwc.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();
  const T* weight_data = weight.const_data_ptr<T>();

  const auto kAccType =
      (X_nhwc.scalar_type() == at::kHalf || X_nhwc.scalar_type() == at::kBFloat16)
      ? at::kFloat
      : X_nhwc.scalar_type();

  const int TPB = MIN(MAX_THREADS_PER_BLOCK, H * W * C);
  const int blockDimX = MIN(TPB, C);
  const int blockDimY = TPB / blockDimX;
  const int f = MAX(C / TPB, 1); // note: impossible for f > 1 AND blockDimY > 1
  torch::Tensor xdy_sum = at::empty({N, H, C}, X_nhwc.options().dtype(kAccType));
  torch::Tensor dy_sum = at::empty({N, H, C}, X_nhwc.options().dtype(kAccType));
  T_ACC* xdy_sum_data = xdy_sum.mutable_data_ptr<T_ACC>();
  T_ACC* dy_sum_data = dy_sum.mutable_data_ptr<T_ACC>();

  dim3 dimGrid(N, H, f);
  dim3 dimBlock(blockDimX, blockDimY);
  NH_spatial_loop<<<dimGrid, dimBlock, sizeof(T_ACC) * 2 * TPB>>>(
      dy_data, X_data, 
      N, H, W, C,
      xdy_sum_data, dy_sum_data);

  // sum over H dimension
  //TODO:
  xdy_sum = xdy_sum.sum(1);
  dy_sum = dy_sum.sum(1);

  //xdy_sum = xdy_sum.sum(2);
  //dy_sum = dy_sum.sum(2);

  //xdy_sum = xdy_sum.sum(0);
  //dy_sum = dy_sum.sum(0);

  xdy_sum_data = xdy_sum.mutable_data_ptr<T_ACC>();
  dy_sum_data = dy_sum.mutable_data_ptr<T_ACC>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  T* dweight_data = dweight.mutable_data_ptr<T>();
  T* dbias_data = dbias.mutable_data_ptr<T>();
  // For small batch size, do colwise reduce directly.
  compute_dweight_dbias<<<1, C>>>(
      mean_data, rstd_data,
      xdy_sum_data, dy_sum_data,
      N, C, G,
      dweight_data, dbias_data);

  torch::Tensor coef1 = at::empty({N, G, D}, X_nhwc.options().dtype(kAccType));
  torch::Tensor coef2 = at::empty({N, G}, X_nhwc.options().dtype(kAccType));
  torch::Tensor coef3 = at::empty({N, G}, X_nhwc.options().dtype(kAccType));
  T_ACC* coef2_data = coef2.mutable_data_ptr<T_ACC>();
  T_ACC* coef3_data = coef3.mutable_data_ptr<T_ACC>();

  at::TensorIterator iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                  .add_output(coef1)
                  .add_owned_input(rstd.view({N, G, 1}))
                  .add_owned_input(weight.view({1, G, D}))
                  .build();
  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T rstd, T weight) -> T_ACC {
    return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(weight);
  });

  compute_bwd_scale_biases<<<N, C, sizeof(T_ACC) * 2 * C>>>(
      mean_data, rstd_data, weight_data,
      xdy_sum_data, dy_sum_data,
      H, W, C, G,
      coef2_data, coef3_data);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                  .resize_outputs(false)
                  .add_owned_output(dX.view({N, H * W, G, D}))
                  .add_owned_input(dy_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(X_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(coef1.view({N, 1, G, D}))
                  .add_owned_input(coef2.view({N, 1, G, 1}))
                  .add_owned_input(coef3.view({N, 1, G, 1}))
                  .build();
  at::native::gpu_kernel(
      iter, [] GPU_LAMBDA(T dy, T x, T_ACC coef1, T_ACC coef2, T_ACC coef3) -> T {
        return coef1 * static_cast<T_ACC>(dy) + coef2 * static_cast<T_ACC>(x) + 
                coef3;
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> gn_nhwc_cuda_bwd(
    const torch::Tensor& dy,
    const torch::Tensor& X,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const torch::Tensor& weight,
    const int G
  ) {
  const int C = X.size(1);
  torch::Tensor dy_nhwc = dy.permute({0, 2, 3, 1});
  torch::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  torch::Tensor dX = torch::empty_like(X_nhwc);
  torch::Tensor dweight = torch::empty({C}, X.options());
  torch::Tensor dbias = torch::empty({C}, X.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_backward", [&]() {
      gn_bwd<scalar_t>(
          dy_nhwc, X_nhwc,
          weight, mean, rstd,
          G,
          dX, dweight, dbias
      );
  });
  return {dX.permute({0, 3, 1, 2}), dweight, dbias};
}
