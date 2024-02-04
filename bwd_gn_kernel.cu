//#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
//#include <ATen/native/cuda/Loops.cuh> // gpu kernel
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/Tensor.h> // at::tensor
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <c10/core/ScalarType.h>
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

template <typename T>
__device__ void
sum_reduce(
    T vals_reduced,
    const int start_stride,
    const int end_stride
  ) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll 8
  for (int stride = start_stride; stride >= end_stride; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
  }
}

template <typename T>
__global__ void
spatial_loop(
      const T* dy_data,
      const T* X_data,
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
       output buffer: (N, C, H)
     C > MAX_THREADS_PER_BLOCK (Kernel 2):
       griddim: (x=N, y=H, z=f); blockdim: (x=TPB, y=d=1)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, TPB); X stride: (HWC, WC, dC, C, TPB, 1)
       shmem reduction: (d, TPB) -reduce-> TPB
       output buffer: (N, f, TPB, H) -view-> (N, C, H)
   */

  using T_ACC = at::acc_type<T, true>;
  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  const int w = W / d;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;

#pragma unroll 8
  for (int i = 0; i < w; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWC stride
    reduce_idx += blockIdx.y * W * C; // dim 1, WC stride
    reduce_idx += i * d * C; // dim 2, dC stride
    reduce_idx += threadIdx.y * C; // dim 3, C stride
    reduce_idx += blockIdx.z * TPB; // dim 4, TPB stride (in kernel 1, threadIdx.z is always 0 so this statement does nothing)
    reduce_idx += threadIdx.x; // dim 5, 1 stride
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    xdy_sum += dy_elem * X_data[reduce_idx];
    dy_sum += dy_elem;
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[2 * tid] = xdy_sum;
  vals_reduced[2 * tid + 1] = dy_sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB, 2 * C);

  // put reduced outputs into return buffers
  if (tid < C) {
    int out_idx = 0;
    out_idx += blockIdx.x * C * H; // dim 0, CH stride
    out_idx += blockIdx.z * TPB * H; // dim 1, TPB*H stride (if f=1, this line is a no-op)
    out_idx += threadIdx.x * H; // dim 2, H stride
    out_idx += blockIdx.y; // dim 3, 1 stride

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
    at::acc_type<T, true>* coef1_data,
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
  const int nc = n * C + c;
  const T_ACC gamma_v = static_cast<T_ACC>(weight_data[c]);

  extern __shared__ char vals_reduced_uncasted[]; // size 2*C, C for sum1, C for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  int idx = 0;
  idx += d * G;
  idx += g;
  vals_reduced[2 * idx] = xdy_sum_data[nc] * gamma_v;
  vals_reduced[2 * idx + 1] = dy_sum_data[nc] * gamma_v;
  __syncthreads();
  sum_reduce(vals_reduced, C, 2 * G);

  const int ng = n * G + g;
  const T_ACC mean_elem = static_cast<T_ACC>(mean_data[ng]);
  const T_ACC rstd_elem = static_cast<T_ACC>(rstd_data[ng]);
  coef1_data[nc] = rstd_elem * weight_data[c];

  if (d == 0) {
    const T_ACC sum1 = vals_reduced[2 * g];
    const T_ACC sum2 = vals_reduced[2 * g + 1];
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * H * W);
    const T_ACC x = (sum2 * mean_elem - sum1) * (rstd_elem * rstd_elem * rstd_elem * s);
    coef2_data[ng] = x;
    coef3_data[ng] = (-x * mean_elem) - (sum2 * rstd_elem * s);
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

template <typename T, int size_per_float>
struct alignas(2 * size_per_float) f2 {
  T x, y;
};

template <typename T, int size_per_float>
struct alignas(4 * size_per_float) f4 {
  T x, y, z, w;
};

template <typename T, int size_per_float>
struct alignas(8 * size_per_float) f8 {
  T x, y, z, w, a, b, c, d;
};

template <typename T, int LOOP_I>
__global__ void
dx_elem_kernel(
    const T* dy_data,
    const T* X_data,
    at::acc_type<T, true>* coef1_data,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  /*
     (N, H, W, C) -> (N, H, W, G, D) = (NHWC/512/I, I, 512)
     C < 512 ->      (N, Hw, I, d, C) -> (N, Hw, I, d, G, D)
                  .add_owned_output(dX.view({N, H * W, G, D}))
                  .add_owned_input(dy_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(X_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(coef1.view({N, 1, G, D}))
                  .add_owned_input(coef2.view({N, 1, G, 1}))
                  .add_owned_input(coef3.view({N, 1, G, 1}))
     */

  using T_ACC = at::acc_type<T, true>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % C;
  //const int g = (G * c) / C;
  const int nc = n * C + c;
  const int ng = nc * G / C;
  T_ACC coef1 = coef1_data[nc];
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
#pragma unroll LOOP_I
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    // (coef1 * static_cast<T_ACC>(dy)) + (coef2 * static_cast<T_ACC>(x)) + coef3;
    dx_data[idx] = (coef1 * static_cast<T_ACC>(dy_data[idx])) + (coef2 * static_cast<T_ACC>(X_data[idx])) + coef3;
  }
}

template <typename T, int LOOP_I, typename V, typename V_ACC>
__global__ void
dx_elem_kernelV(
    const T* dy_data,
    const T* X_data,
    at::acc_type<T, true>* coef1_data,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  using T_ACC = at::acc_type<T, true>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / sizeof(V));
  const int g = (G * c) / (C / sizeof(V));
  const int nc = n * (C / sizeof(V)) + c;
  const int ng = n * G + g;
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
  const V *dy_vec = reinterpret_cast<const V*>(dy_data);
  const V *X_vec = reinterpret_cast<const V*>(X_data);
  V *dx_vec = reinterpret_cast<V*>(dx_data);
  V_ACC *coef1_vec = reinterpret_cast<V_ACC*>(coef1_data);
  V_ACC tmp_coef1 = coef1_vec[nc];
#pragma unroll LOOP_I
  for (int i  = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    //dx_data[idx] = (coef1 * static_cast<T_ACC>(dy_data[idx])) + (coef2 * static_cast<T_ACC>(X_data[idx])) + coef3;

    V tmp_dy = dy_vec[idx];
    V tmp_X = X_vec[idx];

    if (sizeof(V) == 2) {
      T dx_x, dx_y;
      dx_x = (tmp_coef1.x * tmp_dy.x) + (coef2 * tmp_X.x) + coef3;
      dx_y = (tmp_coef1.y * tmp_dy.y) + (coef2 * tmp_X.y) + coef3;
      dx_vec[idx] = {dx_x, dx_y};
    }
    else if (sizeof(V) == 4) {
      T dx_x, dx_y, dx_z, dx_w;
      dx_x = (tmp_coef1.x * tmp_dy.x) + (coef2 * tmp_X.x) + coef3;
      dx_y = (tmp_coef1.y * tmp_dy.y) + (coef2 * tmp_X.y) + coef3;
      dx_z = (tmp_coef1.z * tmp_dy.z) + (coef2 * tmp_X.z) + coef3;
      dx_w = (tmp_coef1.w * tmp_dy.w) + (coef2 * tmp_X.w) + coef3;
      dx_vec[idx] = {dx_x, dx_y, dx_z, dx_w};
    }
    else if (sizeof(V) == 8) {
      T dx_x, dx_y, dx_z, dx_w, dx_a, dx_b, dx_c, dx_d;
      dx_x = (tmp_coef1.x * tmp_dy.x) + (coef2 * tmp_X.x) + coef3;
      dx_y = (tmp_coef1.y * tmp_dy.y) + (coef2 * tmp_X.y) + coef3;
      dx_z = (tmp_coef1.z * tmp_dy.z) + (coef2 * tmp_X.z) + coef3;
      dx_w = (tmp_coef1.w * tmp_dy.w) + (coef2 * tmp_X.w) + coef3;
      dx_a = (tmp_coef1.a * tmp_dy.a) + (coef2 * tmp_X.a) + coef3;
      dx_b = (tmp_coef1.b * tmp_dy.b) + (coef2 * tmp_X.b) + coef3;
      dx_c = (tmp_coef1.c * tmp_dy.c) + (coef2 * tmp_X.c) + coef3;
      dx_d = (tmp_coef1.d * tmp_dy.d) + (coef2 * tmp_X.d) + coef3;
      dx_vec[idx] = {dx_x, dx_y, dx_z, dx_w, dx_a, dx_b, dx_c, dx_d};
    }
  }
}

template <typename T, int LOOP_I>
__global__ void
dx_elem_kernelV2(
    const T* dy_data,
    const T* X_data,
    at::acc_type<T, true>* coef1_data,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  using T_ACC = at::acc_type<T, true>;
  using T_vec_dtype = f2<T, sizeof(T)>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / 2);
  const int g = (G * c) / (C / 2);
  const int nc = n * (C / 2) + c;
  const int ng = n * G + g;
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
  const T_vec_dtype *dy_vec = reinterpret_cast<const T_vec_dtype*>(dy_data);
  const T_vec_dtype *X_vec = reinterpret_cast<const T_vec_dtype*>(X_data);
  T_vec_dtype *dx_vec = reinterpret_cast<T_vec_dtype*>(dx_data);
  f2<T_ACC, 4> *coef1_vec = reinterpret_cast<f2<T_ACC, 4>*>(coef1_data);
  f2<T_ACC, 4> tmp_coef1 = coef1_vec[nc];
#pragma unroll LOOP_I
  for (int i  = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    //dx_data[idx] = (coef1 * static_cast<T_ACC>(dy_data[idx])) + (coef2 * static_cast<T_ACC>(X_data[idx])) + coef3;

    T_vec_dtype tmp_dy = dy_vec[idx];
    T_vec_dtype tmp_X = X_vec[idx];
    T dx_x, dx_y;
    dx_x = (tmp_coef1.x * tmp_dy.x) + (coef2 * tmp_X.x) + coef3;
    dx_y = (tmp_coef1.y * tmp_dy.y) + (coef2 * tmp_X.y) + coef3;
    dx_vec[idx] = {dx_x, dx_y};
  }
}

template <typename T, int LOOP_I>
__global__ void
dx_elem_kernelV4(
    const T* dy_data,
    const T* X_data,
    at::acc_type<T, true>* coef1_data,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  /*
                  .add_owned_output(dX.view({N, H * W, G, D}))
                  .add_owned_input(dy_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(X_nhwc.view({N, H * W, G, D}))
                  .add_owned_input(coef1.view({N, 1, G, D}))
                  .add_owned_input(coef2.view({N, 1, G, 1}))
                  .add_owned_input(coef3.view({N, 1, G, 1}))
     */
  using T_ACC = at::acc_type<T, true>;
  using T_vec_dtype = f4<T, sizeof(T)>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / 4);
  const int g = (G * c) / (C / 4);
  const int nc = n * (C / 4) + c;
  const int ng = n * G + g;
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
  const T_vec_dtype *dy_vec = reinterpret_cast<const T_vec_dtype*>(dy_data);
  const T_vec_dtype *X_vec = reinterpret_cast<const T_vec_dtype*>(X_data);
  T_vec_dtype *dx_vec = reinterpret_cast<T_vec_dtype*>(dx_data);
  f4<T_ACC, 4> *coef1_vec = reinterpret_cast<f4<T_ACC, 4>*>(coef1_data);
  f4<T_ACC, 4> tmp_coef1 = coef1_vec[nc];
#pragma unroll LOOP_I
  for (int i  = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    //dx_data[idx] = (coef1 * static_cast<T_ACC>(dy_data[idx])) + (coef2 * static_cast<T_ACC>(X_data[idx])) + coef3;

    T_vec_dtype tmp_dy = dy_vec[idx];
    T_vec_dtype tmp_X = X_vec[idx];
    T dx_x, dx_y, dx_z, dx_w;
    dx_x = (tmp_coef1.x * tmp_dy.x) + (coef2 * tmp_X.x) + coef3;
    dx_y = (tmp_coef1.y * tmp_dy.y) + (coef2 * tmp_X.y) + coef3;
    dx_z = (tmp_coef1.z * tmp_dy.z) + (coef2 * tmp_X.z) + coef3;
    dx_w = (tmp_coef1.w * tmp_dy.w) + (coef2 * tmp_X.w) + coef3;
    dx_vec[idx] = {dx_x, dx_y, dx_z, dx_w};
  }
}

template <typename T, int LOOP_I>
__global__ void
dx_elem_kernelV8(
    const T* dy_data,
    const T* X_data,
    at::acc_type<T, true>* coef1_data,
    at::acc_type<T, true>* coef2_data,
    at::acc_type<T, true>* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  using T_ACC = at::acc_type<T, true>;
  using T_vec_dtype = f8<T, sizeof(T)>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / 8);
  const int g = (G * c) / (C / 8);
  const int nc = n * (C / 8) + c;
  const int ng = n * G + g;
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
  const T_vec_dtype *dy_vec = reinterpret_cast<const T_vec_dtype*>(dy_data);
  const T_vec_dtype *X_vec = reinterpret_cast<const T_vec_dtype*>(X_data);
  T_vec_dtype *dx_vec = reinterpret_cast<T_vec_dtype*>(dx_data);
  f8<T_ACC, 4> *coef1_vec = reinterpret_cast<f8<T_ACC, 4>*>(coef1_data);
#pragma unroll LOOP_I
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;

    f8 tmp_coef1 = coef1_vec[nc];
    T_vec_dtype tmp_dy = dy_vec[idx];
    T_vec_dtype tmp_X = X_vec[idx];
    T dx_x, dx_y, dx_z, dx_w, dx_a, dx_b, dx_c, dx_d;
    dx_x = (tmp_coef1.x * static_cast<T_ACC>(tmp_dy.x)) + (coef2 * static_cast<T_ACC>(tmp_X.x)) + coef3;
    dx_y = (tmp_coef1.y * static_cast<T_ACC>(tmp_dy.y)) + (coef2 * static_cast<T_ACC>(tmp_X.y)) + coef3;
    dx_z = (tmp_coef1.z * static_cast<T_ACC>(tmp_dy.z)) + (coef2 * static_cast<T_ACC>(tmp_X.z)) + coef3;
    dx_w = (tmp_coef1.w * static_cast<T_ACC>(tmp_dy.w)) + (coef2 * static_cast<T_ACC>(tmp_X.w)) + coef3;
    dx_a = (tmp_coef1.a * static_cast<T_ACC>(tmp_dy.a)) + (coef2 * static_cast<T_ACC>(tmp_X.a)) + coef3;
    dx_b = (tmp_coef1.b * static_cast<T_ACC>(tmp_dy.b)) + (coef2 * static_cast<T_ACC>(tmp_X.b)) + coef3;
    dx_c = (tmp_coef1.c * static_cast<T_ACC>(tmp_dy.c)) + (coef2 * static_cast<T_ACC>(tmp_X.c)) + coef3;
    dx_d = (tmp_coef1.d * static_cast<T_ACC>(tmp_dy.d)) + (coef2 * static_cast<T_ACC>(tmp_X.d)) + coef3;
    dx_vec[idx] = {dx_x, dx_y, dx_z, dx_w, dx_a, dx_b, dx_c, dx_d};
  }
}

template <typename T>
void run_gn_bwd_kernels(
      const at::Tensor& dy_nhwc,
      const at::Tensor& X_nhwc,
      const at::Tensor& weight,
      const at::Tensor& mean,
      const at::Tensor& rstd,
      const int G,
      at::Tensor& dX,
      at::Tensor& dweight,
      at::Tensor& dbias
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

  const c10::ScalarType kAccType =
      (X_nhwc.scalar_type() == c10::ScalarType::Half || X_nhwc.scalar_type() == c10::ScalarType::BFloat16)
      ? at::kFloat
      : X_nhwc.scalar_type();

  at::Tensor xdy_dy_sum = at::empty({2, N, C, H}, X_nhwc.options().dtype(kAccType));
  T_ACC* xdy_sum_data = xdy_dy_sum.mutable_data_ptr<T_ACC>();
  T_ACC* dy_sum_data = xdy_sum_data + N * C * H;
  int TPB = MIN(MAX_THREADS_PER_BLOCK, H * W * C);
  const int blockDimX = MIN(TPB, C);
  const int blockDimY = TPB / blockDimX;
  const int f = MAX(C / TPB, 1); // note: impossible for f > 1 AND blockDimY > 1
  spatial_loop<<<dim3(N, H, f), dim3(blockDimX, blockDimY), sizeof(T_ACC) * 2 * TPB>>>(
      dy_data, X_data, 
      H, W, C,
      xdy_sum_data, dy_sum_data);
  // sum over H dimension
  xdy_dy_sum = xdy_dy_sum.sum(3); // xdy_dy_sum shape now (2, N, C)
  xdy_sum_data = xdy_dy_sum.mutable_data_ptr<T_ACC>();
  dy_sum_data = xdy_sum_data + N * C;
  //C10_CUDA_KERNEL_LAUNCH_CHECK();

  T* dweight_data = dweight.mutable_data_ptr<T>();
  T* dbias_data = dbias.mutable_data_ptr<T>();
  compute_dweight_dbias<<<1, C>>>(
      mean_data, rstd_data,
      xdy_sum_data, dy_sum_data,
      N, C, G,
      dweight_data, dbias_data);
  //C10_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor coef1 = at::empty({N, C}, X_nhwc.options().dtype(kAccType));
  at::Tensor coef2 = at::empty({N, G}, X_nhwc.options().dtype(kAccType));
  at::Tensor coef3 = at::empty({N, G}, X_nhwc.options().dtype(kAccType));
  T_ACC* coef1_data = coef1.mutable_data_ptr<T_ACC>();
  T_ACC* coef2_data = coef2.mutable_data_ptr<T_ACC>();
  T_ACC* coef3_data = coef3.mutable_data_ptr<T_ACC>();
  compute_bwd_scale_biases<<<N, C, sizeof(T_ACC) * 2 * C>>>(
      mean_data, rstd_data, weight_data,
      xdy_sum_data, dy_sum_data,
      H, W, C, G,
      coef1_data, coef2_data, coef3_data);
  //C10_CUDA_KERNEL_LAUNCH_CHECK();

  /*
  at::TensorIterator iter = at::TensorIteratorConfig()
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
        return (coef1 * static_cast<T_ACC>(dy)) + (coef2 * static_cast<T_ACC>(x)) + coef3;
      });
      */

  T *dx_data = dX.mutable_data_ptr<T>();
  const int ELEM_KERNEL_TPB = 512;
  const int LOOP_I = 8;

  if (D == 1)
    dx_elem_kernel<T, LOOP_I><<<N * H * W * C / ELEM_KERNEL_TPB / LOOP_I, ELEM_KERNEL_TPB>>>(
        dy_data, X_data,
        coef1_data, coef2_data, coef3_data,
        N, C, G,
        dx_data
        );
  else if (D == 2)
    dx_elem_kernelV2<T, LOOP_I><<<N * H * W * C / ELEM_KERNEL_TPB / LOOP_I / 2, ELEM_KERNEL_TPB>>>(
        dy_data, X_data,
        coef1_data, coef2_data, coef3_data,
        N, C, G,
        dx_data
        );
  else if (D < 8)
    dx_elem_kernelV4<T, LOOP_I><<<N * H * W * C / ELEM_KERNEL_TPB / LOOP_I / 4, ELEM_KERNEL_TPB>>>(
        dy_data, X_data,
        coef1_data, coef2_data, coef3_data,
        N, C, G,
        dx_data
        );
  else
    dx_elem_kernelV8<T, LOOP_I><<<N * H * W * C / ELEM_KERNEL_TPB / LOOP_I / 8, ELEM_KERNEL_TPB>>>(
        dy_data, X_data,
        coef1_data, coef2_data, coef3_data,
        N, C, G,
        dx_data
        );
  //C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> gn_nhwc_cuda_bwd(
    const at::Tensor& dy,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& weight,
    const int G
  ) {
  const int C = X.size(1);
  at::Tensor dy_nhwc = dy.permute({0, 2, 3, 1});
  at::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  at::Tensor dX = at::empty_like(X_nhwc);
  at::Tensor dweight = at::empty({C}, X.options());
  at::Tensor dbias = at::empty({C}, X.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    c10::ScalarType::Half,
    c10::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_backward", [&]() {
      run_gn_bwd_kernels<scalar_t>(
          dy_nhwc, X_nhwc,
          weight, mean, rstd,
          G,
          dX, dweight, dbias
      );
  });
  return {dX.permute({0, 3, 1, 2}), dweight, dbias};
}
