#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/ScalarType.h>
#include "gn_kernel.h"
#include "vecs.h"
#define MAX_THREADS_PER_BLOCK 512
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b
#define DEBUG(format, args...) if (0) fprintf(stderr, format, args)

template <typename T>
struct acc_type { using type = float; };
template <>
struct acc_type<double> { using type = double; };

typedef struct block_params {
  int t; // threads per block
  int d; // dimensionality (number of data rows per threadblock)
  int f; // factor (number of different threadblocks needed to represent one row of data) 
} block_params_t;

template <typename T>
__device__ void
sum_reduce(
    T vals_reduced,
    const int start_stride,
    const int end_stride
  ) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int reduce_n = 2 * start_stride / end_stride;

#pragma unroll
  for (int stride = start_stride; stride >= end_stride && reduce_n % 2 == 0 && stride % end_stride == 0; stride >>= 1, reduce_n >>= 1) {
    if (tid < stride)
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
  }

  if (tid < end_stride)
#pragma unroll
    for (int i = 1; i < reduce_n; ++i)
      vals_reduced[tid] += vals_reduced[tid + i * end_stride];
  __syncthreads();
}

template <typename T>
__global__ void
width_reduce(
      const T* dy_data,
      const T* X_data,
      const int H,
      const int W,
      const int C,
      typename acc_type<T>::type *xdy_dy_sum_data) {
  /*
     Performs a loop over the spatial dimension W, loading and summing dy and X. Spatial dimension H is processed in a separate kernel.
     C <= MAX_THREADS_PER_BLOCK (Kernel 1):
       griddim: (x=N, y=H, z=f=1); blockdim: (x=C, y=d)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Cd = TPB (threads per block)
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
       shmem reduction: (d, C) -reduce-> C
       output buffer: (N, C, H, 2)
     C > MAX_THREADS_PER_BLOCK (Kernel 2):
       griddim: (x=N, y=H, z=f); blockdim: (x=TPB, y=d=1)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, TPB); X stride: (HWC, WC, dC, C, TPB, 1)
       shmem reduction: (d, TPB) -reduce-> TPB
       output buffer: (N, f, TPB, H, 2) -view-> (N, C, H, 2)
   */
  using T_ACC = typename acc_type<T>::type;

  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;

  const int w = ceil((float)W / d);
  int i;
#pragma unroll
  for (i = 0; i < w - 1; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWC stride
    reduce_idx += blockIdx.y * W * C; // dim 1, WC stride
    reduce_idx += i * d * C; // dim 2, dC stride
    reduce_idx += threadIdx.y * C; // dim 3, C stride
    reduce_idx += blockIdx.z * TPB; // dim 4, TPB stride (in kernel 1, threadIdx.z is always 0 so this statement does nothing)
    reduce_idx += threadIdx.x; // dim 5, 1 stride
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    T_ACC X_elem = static_cast<T_ACC>(X_data[reduce_idx]);
    xdy_sum += dy_elem * X_elem;
    dy_sum += dy_elem;
  }
  if ((int)(i * d + threadIdx.y) < W) { // last iteration to deal with inputs with weird width sizes
    int reduce_idx = blockIdx.x * H * W * C + blockIdx.y * W * C + i * d * C + threadIdx.y * C + blockIdx.z * TPB + threadIdx.x;
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    T_ACC X_elem = static_cast<T_ACC>(X_data[reduce_idx]);
    xdy_sum += dy_elem * X_elem;
    dy_sum += dy_elem;
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[2 * tid] = xdy_sum;
  vals_reduced[2 * tid + 1] = dy_sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB, 2 * C);

  // put reduced outputs into return buffers
  if (tid < C) {
    int out_idx = 0;
    out_idx += blockIdx.x * C * H; // dim 0, CH stride
    out_idx += (blockIdx.z * TPB + tid) * H; // dim 1, TPB*H stride (if f=1, this line is a no-op)
    out_idx += blockIdx.y; // dim 3, 1 stride

    xdy_dy_sum_data[2 * out_idx] = vals_reduced[2 * tid];
    xdy_dy_sum_data[2 * out_idx + 1] = vals_reduced[2 * tid + 1];
  }
}

template <typename T>
__global__ void
height_reduce(
    T *xdy_dy_sum_data, // no need to specify T_ACC as T is already an accumulation type
    const int H,
    const int C,
    T *xdy_sum_data,
    T *dy_sum_data
  ) {
  const int TPB = blockDim.x;
  const int tid = threadIdx.x;

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[];
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);
  T sum = 0;
  int i;
#pragma unroll
  for (i = 0; i < ceil((float)2 * H / TPB) - 1; ++i) {
    int idx = 0;
    idx += blockIdx.x * C * H * 2;
    idx += blockIdx.y * H * 2;
    idx += i * TPB;
    idx += tid;
    sum += xdy_dy_sum_data[idx];
  }
  if (i * TPB + tid < 2 * H)
    sum += xdy_dy_sum_data[blockIdx.x * C * H * 2 + blockIdx.y * H * 2 + i * TPB + tid];

  vals_reduced[tid] = sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB / 2, 2);

  // put reduced outputs into return buffers
  if (tid == 0) {
    int out_idx = blockIdx.x * C + blockIdx.y;
    xdy_sum_data[out_idx] = vals_reduced[0];
    dy_sum_data[out_idx] = vals_reduced[1];
    //if (blockIdx.x < 2 && blockIdx.y < 2 && blockIdx.z == 0)
    //  printf("N: %d, C: %d, xdy sum: %f, dy sum: %f, vals_reduced_size: %d, TPB: %d\n", blockIdx.x, blockIdx.y, vals_reduced[0], vals_reduced[1], vals_reduced_size, TPB);
  }
}

template <typename T>
__global__ void
compute_bwd_scale_biases(
    const T* mean_data,
    const T* rstd_data,
    const T* weight_data,
    typename acc_type<T>::type* xdy_sum_data,
    typename acc_type<T>::type* dy_sum_data,
    const int H,
    const int W,
    const int C,
    const int G,
    typename acc_type<T>::type* coef1_data,
    typename acc_type<T>::type* coef2_data,
    typename acc_type<T>::type* coef3_data) {
  /*
     griddim: (x=N, y=f); blockdim: (x=C/f)
      d = num. spatial elements (from HW dimension) each thread-block processes in parallel
      Cd = TPB (threads per block)
     X shape: (N, C) -view-> (N, G, D) -permute-> (N, D, G) -reduce-> (N, G)
     shmem reduction: (D, G) -reduce-> G
     output buffer: (N, G)
   */
  using T_ACC = typename acc_type<T>::type;
  const int D = C / G;
  const int f = gridDim.y;
  const int Gf = G / f;
  const int n = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int g = c / D;
  const int d = c % D;
  const int nc = n * C + c;
  const T_ACC gamma_v = static_cast<T_ACC>(weight_data[c]);

  extern __shared__ char vals_reduced_uncasted[]; // size 2*C, C for sum1, C for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  int idx = 0;
  idx += d * G / f;
  idx += g % Gf;
  vals_reduced[2 * idx] = xdy_sum_data[nc] * gamma_v;
  vals_reduced[2 * idx + 1] = dy_sum_data[nc] * gamma_v;
  __syncthreads();
  sum_reduce(vals_reduced, C / f, 2 * G / f);

  const int ng = n * G + g;
  const T_ACC mean_elem = static_cast<T_ACC>(mean_data[ng]);
  const T_ACC rstd_elem = static_cast<T_ACC>(rstd_data[ng]);
  coef1_data[nc] = rstd_elem * weight_data[c];

  if (d == 0) {
    const T_ACC sum1 = vals_reduced[2 * (g % Gf)];
    const T_ACC sum2 = vals_reduced[2 * (g % Gf) + 1];
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * H * W);
    const T_ACC x = (sum2 * mean_elem - sum1) * rstd_elem * rstd_elem * rstd_elem * s;
    coef2_data[ng] = x;
    coef3_data[ng] = (-x * mean_elem) - (sum2 * s * rstd_elem);
  }
}

template <typename T>
__global__ void
compute_dweight_dbias(
    const T* mean_data,
    const T* rstd_data,
    typename acc_type<T>::type *xdy_sum_data,
    typename acc_type<T>::type *dy_sum_data,
    const int N,
    const int C,
    const int G,
    T* dweight_data,
    T* dbias_data) {
  // gridDim: (x=f), blockDim: (x=C / f)
  using T_ACC = typename acc_type<T>::type;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int D = C / G;
  const int g = c / D;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;

#pragma unroll
  for (int n = 0; n < N; ++n) {
    const int nc = n * C + c;
    const int ng = n * G + g;
    sum1 += (xdy_sum_data[nc] - dy_sum_data[nc] * static_cast<T_ACC>(mean_data[ng])) * static_cast<T_ACC>(rstd_data[ng]);
    sum2 += static_cast<T_ACC>(dy_sum_data[nc]);
  }
  dweight_data[c] = sum1;
  dbias_data[c] = sum2;
}

template <typename T, int LOOP_I, int vec_elems>
__global__ void
dx_elem_kernel(
    const T* dy_data,
    const T* X_data,
    typename acc_type<T>::type* coef1_data,
    typename acc_type<T>::type* coef2_data,
    typename acc_type<T>::type* coef3_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  using T_ACC = typename acc_type<T>::type;
  using V = float_vec<T, vec_elems>;
  using V_ACC = float_vec<T_ACC, vec_elems>;
  const int f = gridDim.y;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) % (C / vec_elems);
  const int g = (G * c) / (C / vec_elems);
  const int nc = n * (C / vec_elems) + c;
  const int ng = n * G + g;
  T_ACC coef2 = coef2_data[ng];
  T_ACC coef3 = coef3_data[ng];
  const V *dy_vecs = reinterpret_cast<const V*>(dy_data);
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  V *dx_vecs = reinterpret_cast<V*>(dx_data);
  V_ACC coef1_vec = reinterpret_cast<V_ACC*>(coef1_data)[nc];
#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * f * blockDim.x;
    idx += i * f * blockDim.x;
    idx += blockIdx.y * blockDim.x;
    idx += threadIdx.x;

    V dy_vec = dy_vecs[idx];
    V X_vec = X_vecs[idx];

    if constexpr (vec_elems == 1)
      dx_vecs[idx] = {(coef1_vec.x * dy_vec.x) + ((coef2 * X_vec.x) + coef3)};
    else if constexpr (vec_elems == 2) {
      dx_vecs[idx] = {
        (coef1_vec.x * dy_vec.x) + ((coef2 * X_vec.x) + coef3),
        (coef1_vec.y * dy_vec.y) + ((coef2 * X_vec.y) + coef3),
      };
    }
    else if constexpr (vec_elems == 4) {
      dx_vecs[idx] = {
        (coef1_vec.x * dy_vec.x) + ((coef2 * X_vec.x) + coef3),
        (coef1_vec.y * dy_vec.y) + ((coef2 * X_vec.y) + coef3),
        (coef1_vec.z * dy_vec.z) + ((coef2 * X_vec.z) + coef3),
        (coef1_vec.w * dy_vec.w) + ((coef2 * X_vec.w) + coef3),
      };
    }
  }
}

inline block_params_t simple_calc_block_params(int ideal_num_threads, int threads_per_row, int d_preferred_divisibility = 1, int f_divides = -1) {
  /*
  d_preferred_divisibility: if possible, make d a multiple of d_preferred divisibilty (useful for kernels which require inputs to be divisible by the number of threads per block e.g. elementwise kernel)
  f_divides: parameter if user needs to explicitly specify a stricter requirement on the divisibility of the number of threads per block
    - e.g. fwd with C = 2560, G = 32, TPB = 480 wouldn't work since that means 32 groups are split over f=5 blocks (5.333 groups per block)
    - e.g. fwd with C = 2560, G = 32, TPB = 320 would work since that means 32 groups are split over f=8 blocks (4 groups per block), you could say that f divides 32 (f_divides=32)
  */
  int TPB, d = 1, f = 1;
  f_divides = f_divides == -1 ? threads_per_row : f_divides;
  TPB = MIN(MAX_THREADS_PER_BLOCK, ideal_num_threads);
  if (threads_per_row < TPB)
    if (TPB / threads_per_row / d_preferred_divisibility != 0)
      d = d_preferred_divisibility * (TPB / threads_per_row / d_preferred_divisibility); // prefer d being a multiple of d_preferred_divisibility (e.g. for C=96, prefer TPB=38d_preferred_divisibility and not d_preferred_divisibility80) since it makes it more likely that TPB will be able to use the fast elementwise kernel
    else
      d = TPB / threads_per_row;
  else
    while (f_divides % f != 0 || threads_per_row / f > MAX_THREADS_PER_BLOCK)
      ++f;
  TPB = threads_per_row * d / f;
  return {TPB, d, f};
}

#define ELEM_DEBUG 0

template <typename T>
void run_gn_bwd_kernels(
      const T *dy_data,
      const T *X_data,
      const T *weight_data,
      const T *mean_data,
      const T *rstd_data,
      const int N,
      const int H,
      const int W,
      const int C,
      const int G,
      T *dx_data,
      T *dweight_data,
      T *dbias_data
  ) {
  using T_ACC = typename acc_type<T>::type;
  const int D = C / G;

  T_ACC* xdy_dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C * H * 2);

  // sum over W dim
  {
    auto [TPB, d, f] = simple_calc_block_params(W * C, C, 4, G);
    DEBUG("starting width reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    width_reduce<<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB>>>(
        dy_data, X_data, 
        H, W, C,
        xdy_dy_sum_data);
  }

  T_ACC* xdy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC* dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  // sum over H dim
  {
    auto [TPB, d, f] = simple_calc_block_params(2 * H, 2);
    DEBUG("starting height reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    //height_reduce<<<dim3(N, C), TPB, sizeof(T_ACC) * host_next_power_of_2(TPB)>>>(
    height_reduce<<<dim3(N, C), TPB, sizeof(T_ACC) * TPB>>>(
        xdy_dy_sum_data,
        H, C,
        xdy_sum_data, dy_sum_data);
  }
  c10::cuda::CUDACachingAllocator::raw_delete(xdy_dy_sum_data);

  // compute weight/bias grads
  {
    auto [TPB, d, f] = simple_calc_block_params(C, C, 1, G);
    DEBUG("starting compute dweight dbias, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    compute_dweight_dbias<<<f, C / f>>>(
        mean_data, rstd_data,
        xdy_sum_data, dy_sum_data,
        N, C, G,
        dweight_data, dbias_data);
  }

  T_ACC *coef1_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC *coef2_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);
  T_ACC *coef3_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);
  // compute fused scales/biases for dx elementwise kernel
  {
    auto [TPB, d, f] = simple_calc_block_params(C, C, 1, G);
    DEBUG("starting bwd scale biases, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    compute_bwd_scale_biases<<<dim3(N, f), C / f, sizeof(T_ACC) * 2 * C / f>>>(
        mean_data, rstd_data, weight_data,
        xdy_sum_data, dy_sum_data,
        H, W, C, G,
        coef1_data, coef2_data, coef3_data);
  }

  {
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;
    auto [TPB, d, f] = simple_calc_block_params(H * W * C, C, 1, G);
    if (!ELEM_DEBUG && ((H * W * C) % (TPB * 8 * f * vec_elems) == 0)) {
      const int LOOP_I = 8;
      const int num_blocks = N * H * W * C / TPB / LOOP_I / f;
      //DEBUG("dx elem kernel starting, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, num blocks (before vectors): %d\n", N, H, W, C, G, D, num_blocks);
      if (D % 4 == 0)
        dx_elem_kernel<T, LOOP_I, 4><<<dim3(num_blocks / 4, f), TPB>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, N, C, G, dx_data);
      else if (D % 2 == 0)
        dx_elem_kernel<T, LOOP_I, 2><<<dim3(num_blocks / 2, f), TPB>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, N, C, G, dx_data);
      else
        dx_elem_kernel<T, LOOP_I, 1><<<dim3(num_blocks / 1, f), TPB>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, N, C, G, dx_data);
    }
    else // relatively slow fallback
      dx_elem_kernel<T, 1, 1><<<dim3(N * H * W, f), C / f>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, N, C, G, dx_data);
  }

  c10::cuda::CUDACachingAllocator::raw_delete(xdy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(dy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef1_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef2_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef3_data);
}

template void run_gn_bwd_kernels<double>(const double *dy_data, const double *X_data, const double *weight_data, const double *mean_data, const double *rstd_data, const int N, const int H, const int W, const int C, const int G, double *dx_data, double *dweight_data, double *dbias_data);
template void run_gn_bwd_kernels<float>(const float *dy_data, const float *X_data, const float *weight_data, const float *mean_data, const float *rstd_data, const int N, const int H, const int W, const int C, const int G, float *dx_data, float *dweight_data, float *dbias_data);
template void run_gn_bwd_kernels<c10::Half>(const c10::Half *dy_data, const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *mean_data, const c10::Half *rstd_data, const int N, const int H, const int W, const int C, const int G, c10::Half *dx_data, c10::Half *dweight_data, c10::Half *dbias_data);
template void run_gn_bwd_kernels<c10::BFloat16>(const c10::BFloat16 *dy_data, const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *mean_data, const c10::BFloat16 *rstd_data, const int N, const int H, const int W, const int C, const int G, c10::BFloat16 *dx_data, c10::BFloat16 *dweight_data, c10::BFloat16 *dbias_data);
