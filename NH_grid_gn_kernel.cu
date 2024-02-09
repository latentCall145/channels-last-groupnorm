#include <ATen/OpMathType.h> // opmath_t
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <thrust/pair.h> // thrust::pair
#include "gn_kernel.h"
#include "Welford.h"
#include "vecs.h"
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b
#define DEBUG(format, args...) if constexpr (0) fprintf(stderr, format, args)
#define ELEM_DEBUG 0
#define INT int

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
__global__ void
compute_scale_biases(
        T* means,  // (N, G)
        T* rstds,  // (N, G)
        const T* weight, // (C)
        const T* bias,   // (C)
        const int G,
        const int C,
        typename acc_type<T>::type* a,            // (N, C)
        typename acc_type<T>::type* b             // (N, C)
  ) {
  // (N, f), (TPB)
  const int D = C / G;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int g = c / D;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  const acc_type<T> a_nc = rstds[ng] * weight[c];
  a[nc] = a_nc;
  b[nc] = -means[ng] * a_nc + bias[c];
}

template <typename T>
__device__ T
inline relu(T x) {
  return x > 0 ? x : 0;
}

template <typename T>
__device__ T
inline silu(T x) {
  using opmath_t = at::opmath_type<T>;
  return x / (opmath_t(1) + exp(-x));
}

template <typename T>
__device__ T
inline gelu(T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kAlpha = M_SQRT1_2;
  return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
}

template <typename T>
__device__ T
inline gelu_tanh(T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  return opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + c10::cuda::compat::tanh(inner));
}

inline block_params_t calc_block_params(int ideal_num_threads, int threads_per_row, int d_preferred_divisibility = 1, int f_divides = -1) {
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

#define ACT // silu

template <typename T, int LOOP_I, int vec_elems>
__global__ void
scale_shift(
    const T* X_data,
    const T* mean_data,
    const T* rstd_data,
    const T* weight_data,
    const T* bias_data,
    const int N,
    const int C,
    const int G,
    T* y
    ) {
  using T_ACC = typename acc_type<T>::type;
  using V = float_vec<T, vec_elems>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) % (C / vec_elems);
  const int g = (G * c) / (C / vec_elems);
  const int ng = n * G + g;
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  const V *weight_vecs = reinterpret_cast<const V*>(weight_data);
  const V *bias_vecs = reinterpret_cast<const V*>(bias_data);
  V *y_vecs = reinterpret_cast<V*>(y);
  T mean = mean_data[ng];
  T rstd = rstd_data[ng];
  V weight_vec = weight_vecs[c];
  V bias_vec = bias_vecs[c];

  V fused_weight, fused_bias;
  if constexpr (vec_elems == 1) {
    fused_weight = {rstd * weight_vec.x};
    fused_bias = {-mean * fused_weight.x + bias_vec.x};
  }
  else if constexpr (vec_elems == 2) {
    fused_weight = {rstd * weight_vec.x, rstd * weight_vec.y};
    fused_bias = {-mean * fused_weight.x + bias_vec.x, -mean * fused_weight.y + bias_vec.y};
  }
  else if constexpr (vec_elems == 4) {
    fused_weight = {rstd * weight_vec.x, rstd * weight_vec.y, rstd * weight_vec.z, rstd * weight_vec.w};
    fused_bias = {-mean * fused_weight.x + bias_vec.x, -mean * fused_weight.y + bias_vec.y, -mean * fused_weight.z + bias_vec.z, -mean * fused_weight.w + bias_vec.w};
  }

#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * gridDim.y * blockDim.x;
    idx += i * gridDim.y * blockDim.x;
    idx += blockIdx.y * blockDim.x;
    idx += threadIdx.x;
    V X_vec = X_vecs[idx];

    if constexpr (vec_elems == 1)
      y_vecs[idx] = {ACT(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x)};
    else if constexpr (vec_elems == 2) {
      y_vecs[idx] = {
        ACT(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x),
        ACT(static_cast<T_ACC>(X_vec.y) * fused_weight.y + fused_bias.y),
      };
    }
    else if constexpr (vec_elems == 4) {
      y_vecs[idx] = {
        ACT(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x),
        ACT(static_cast<T_ACC>(X_vec.y) * fused_weight.y + fused_bias.y),
        ACT(static_cast<T_ACC>(X_vec.z) * fused_weight.z + fused_bias.z),
        ACT(static_cast<T_ACC>(X_vec.w) * fused_weight.w + fused_bias.w),
      };
    }
  }
}

template <typename T>
__global__ void
compute_stats_pt1(
    const T* X,
    const int H,
    const int W,
    const int C,
    const int G,
    WelfordData<typename acc_type<T>::type, INT> *welford_data
  ) {
  /*
     C <= MAX_THREADS_PER_BLOCK (Kernel 1):
       griddim: (x=N, y=H, z=f=1); blockdim: (x=C, y=d)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Cd = TPB (threads per block)
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, C); X stride: (HWC, WC, dC, C, C, 1)
       shmem reduction: (d, C) -view-> (d, G, D) -permute-> (d, D, G) -reduce-> G
       output buffer: (N, 1, G, H)
     C > MAX_THREADS_PER_BLOCK (Kernel 2):
       griddim: (x=N, y=H, z=f); blockdim: (x=TPB, y=d=1)
        f = factor of channels that each thread have to process separately
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        f * TPB = C
       X shape: (N, H, W, C) -view-> (N, H, W/d, d, f, TPB); X stride: (HWC, WC, dC, C, TPB, 1)
       shmem reduction: (TPB,) -view-> (1, G/f, D) -permute-> (1, D, G/f) -reduce-> G/f
       output buffer: (N, f, G/f, H)
  */
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

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
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x)); // last arg isn't used in src
  }
  if ((int)(i * d + threadIdx.y) < W) // last iteration to deal with inputs with weird width sizes
    val = welford_op.reduce(val, static_cast<T_ACC>(X[blockIdx.x * H * W * C + blockIdx.y * W * C + i * d * C + threadIdx.y * C + blockIdx.z * TPB + threadIdx.x]));

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

  int reduce_n = TPB / gf;
#pragma unroll
  for (int stride = TPB / 2; stride >= gf && reduce_n % 2 == 0 && stride % gf == 0; stride >>= 1, reduce_n >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  if (tid < gf) {
#pragma unroll
    for (int i = 1; i < reduce_n; ++i)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + i * gf]);

    int out_idx = 0;
    out_idx += blockIdx.x * G * H; // dim 0, GH stride
    out_idx += blockIdx.z * gf * H; // dim 1, G/f * H stride
    out_idx += tid * H; // dim 2, H stride
    out_idx += blockIdx.y; // dim 3, 1 stride
    welford_data[out_idx] = vals_reduced[tid];
  }
}

template <typename T>
__global__ void
compute_stats_pt2(
    WelfordData<typename acc_type<T>::type, INT> *welford_data,
    const int H,
    const int G,
    const T eps,
    T* means,
    T* rstds
  ) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  /*
     griddim: (x=N, y=G); blockdim: (x=H)
      d = num. spatial elements (from H dimension) each thread-block processes in parallel
      Gd/f = TPB (threads per block)
     welford_data shape: (N, G, H); X stride: (GH, H, 1)
     shmem reduction: (H) -reduce-> 1
     output buffer: (N, G)
  */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  const int TPB = blockDim.y * blockDim.x;

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);
  WelfordType val(0, 0, 0, 0);

  const int f = H / TPB;
  for (int i = 0 ; i < f; ++i) {
    int idx = 0;
    idx += blockIdx.x * G * H;
    idx += blockIdx.y * H;
    idx += i * H / f;
    idx += threadIdx.x;
    val = welford_op.combine(val, welford_data[idx]);
  }

  const int tid = threadIdx.x;
  //vals_reduced[tid] = welford_data[blockIdx.x * G * H + blockIdx.y * H + threadIdx.x];
  vals_reduced[tid] = val;
  __syncthreads();

  // next lowest power of 2 (AKA half of TPB's next highest power of 2 (or TPB if TPB is a power of 2)) - https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
  int start_stride = TPB - 1;
  start_stride |= start_stride >> 1;
  start_stride |= start_stride >> 2;
  start_stride |= start_stride >> 4;
  start_stride |= start_stride >> 8;
  start_stride |= start_stride >> 16;
  start_stride = (start_stride + 1) >> 1;

  // doing the first iteration outside the loop because of the extra condition regarding inputs with non-power-of-2 heights
  if (tid < start_stride && tid + start_stride < H / f)
    vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + start_stride]);
  __syncthreads();
#pragma unroll
  for (int stride = start_stride >> 1; stride >= 1; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  if (tid == 0) {
    T_ACC mean, var;
    thrust::tie(var, mean) = welford_op.project(vals_reduced[tid]);
    int out_idx = 0;
    out_idx += blockIdx.x * G; // dim 0, G stride
    out_idx += blockIdx.y; // dim 1, G/f stride
    means[out_idx] = mean;
    rstds[out_idx] = rsqrt(var + static_cast<T_ACC>(eps));
    //if (blockIdx.x < 2 && blockIdx.y < 2)
    //  printf("N: %d, G: %d, mean: %f, var: %f\n", blockIdx.x, blockIdx.y, mean, var);
  }
}

template <typename T>
void run_gn_fwd_kernels(
    const T *X_data,
    const T *weight_data,
    const T *bias_data,
    const int N,
    const int H,
    const int W,
    const int C,
    const int G,
    T eps,
    T *Y_data,
    T *mean_data,
    T *rstd_data) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  WelfordType *welford_data = (WelfordType*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(WelfordType) * N * G * H);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  //cudaError_t err = cudaGetLastError();
  //if (err != cudaSuccess) {fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err)); exit(-1);}
  
  {
    auto [TPB, d, f] = calc_block_params(W * C, C, 1, G);
    DEBUG("starting compute_stats 1, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", (int)N, (int)H, (int)W, (int)C, (int)G, (int)(C / G), (int)TPB, (int)d, (int)f, (int)(G / f));
    compute_stats_pt1<<<dim3(N, H, f), dim3(TPB / d, d), 0, cuda_stream>>>(
        X_data,
        H, W, C, G, 
        welford_data
    );
  }

  {
    auto [TPB, d, f] = calc_block_params(H, H);
    DEBUG("starting compute_stats 2, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", (int)N, (int)H, (int)W, (int)C, (int)G, (int)(C / G), (int)TPB, (int)d, (int)f, (int)(G / f));
    compute_stats_pt2<<<dim3(N, G), H / f, 0, cuda_stream>>>(
        welford_data,
        H, G, eps,
        mean_data, rstd_data
    );
  }

  {
    const int D = C / G;
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;
    auto [TPB, d, f] = calc_block_params(H * W * C / 8 / vec_elems, C);
    if (!ELEM_DEBUG && ((H * W * C) % (TPB * 8 * f * vec_elems) == 0)) {
      DEBUG("starting scale_shift, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", (int)N, (int)H, (int)W, (int)C, (int)G, (int)(C / G), (int)TPB, (int)d, (int)f, (int)(G / f));
      const int LOOP_I = 8;
      const int num_blocks = N * H * W * C / TPB / LOOP_I / f;
      if (vec_elems == 4)
        scale_shift<T, LOOP_I, 4><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2)
        scale_shift<T, LOOP_I, 2><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else
        scale_shift<T, LOOP_I, 1><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    }
    else {// relatively slow fallback
      DEBUG("SLOW FALLBACK, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", (int)N, (int)H, (int)W, (int)C, (int)G, (int)(C / G), (int)TPB, (int)d, (int)f, (int)(G / f));
      scale_shift<T, 1, 1><<<dim3(N * H * W, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(welford_data);
}

template void run_gn_fwd_kernels<float>(const float *X_data, const float *weight_data, const float *bias_data, const int N, const int h, const int W, const int C, const int G, float eps, float *Y_data, float *mean_data, float *rstd_data);
template void run_gn_fwd_kernels<double>(const double *X_data, const double *weight_data, const double *bias_data, const int N, const int h, const int W, const int C, const int G, double eps, double *Y_data, double *mean_data, double *rstd_data);
template void run_gn_fwd_kernels<c10::Half>(const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *bias_data, const int N, const int h, const int W, const int C, const int G, c10::Half eps, c10::Half *Y_data, c10::Half *mean_data, c10::Half *rstd_data);
template void run_gn_fwd_kernels<c10::BFloat16>(const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *bias_data, const int N, const int h, const int W, const int C, const int G, c10::BFloat16 eps, c10::BFloat16 *Y_data, c10::BFloat16 *mean_data, c10::BFloat16 *rstd_data);
