#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <thrust/pair.h>
#include "gn_kernel.h"
#include "Welford.h"
#define MAX_THREADS_PER_BLOCK 128 // seems to work slightly better than 256/512/1024
#define LOOP_I 8 // number of iterations per thread for scale_shift/dx_elem_kernel
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define CDIV(a, b) (((a) + (b) - 1) / (b))

#define DEBUG_ENABLED 0
#if DEBUG_ENABLED
#include <iostream>
#define DEBUG(format, args...) fprintf(stderr, format, args); std::cout << std::flush;
#else
#define DEBUG(format, args...) ((void)0)
#endif
#define ELEM_DEBUG 0
#define INT int // torch uses int64_t but this came at a 10% performance drop and the input sizes that I frequently use (resolutions no bigger than 1024x1024) have a number of pixels smaller than the int max value

template <typename T> struct acc_type { using type = float; };
template <> struct acc_type<double> { using type = double; };

template <typename T, int num_elems> struct float_vec;
template <typename T> struct alignas(1 * sizeof(T)) float_vec<T, 1> { T x; };
template <typename T> struct alignas(2 * sizeof(T)) float_vec<T, 2> { T x, y; };
template <typename T> struct alignas(4 * sizeof(T)) float_vec<T, 4> { T x, y, z, w; };

typedef struct block_params {
  int t; // threads per block
  int rows_per_block; // dimensionality (number of rows of data that each threadblock proceesses in parallel)
  int blocks_per_row; // factor (number of different threadblocks needed to represent one row of data) 
} block_params_t;

inline block_params_t calc_block_params(const int ideal_num_threads, const int threads_per_row, const int snap = -1) {
  /*
  ideal_num_threads: absolute upper limit of threads that a block should have (e.g. a kernel that operates on only 30 elements should have a max TPB of 30 (ideal_num_threads=30))
  threads_per_row: determines the user-specified upper limit on the size of blockDim.x
    - meant to be set to the size of the last dimension, e.g. a kernel operating on tensor sized (N, R, C) would have threads_per_row=C
  snap: an optional constraint for threads per block. If set, the returned TPB % snap = 0 or snap % TPB = 0.
    - ex: D=1280, C=2560 -> threads_per_row=2560 -> blocks_per_row=5, TPB=512 (each group consists of 1280/512=2.5 blocks - will have to deal with nasty padding)
      - ex: D=1280, C=2560 -> threads_per_row=2560, snap=1280 -> blocks_per_row=8, TPB=320 (each group consists of exactly four blocks)
  */
  int TPB, rows_per_block = 1, blocks_per_row = 1;
  TPB = MIN(MAX_THREADS_PER_BLOCK, ideal_num_threads);
  if (threads_per_row < TPB)
    rows_per_block = TPB / threads_per_row;
  else {
    blocks_per_row = CDIV(threads_per_row, TPB); // lower bound for blocks_per_row
    TPB = CDIV(threads_per_row * rows_per_block, blocks_per_row);
    while (TPB % snap != 0 && snap % TPB != 0) {
      ++blocks_per_row; // in a separate line because CDIV is a macro and would call ++blocks_per_row twice
      TPB = CDIV(threads_per_row, blocks_per_row);
    }
  }
  TPB = CDIV(threads_per_row * rows_per_block, blocks_per_row);
  return {TPB, rows_per_block, blocks_per_row};
}

template <typename T> __device__ T inline identity(T x) { return x; }
template <typename T> __device__ T inline identity_d(T /*x*/) { return 1; }
template <typename T> __device__ T inline relu(T x) { return x > 0 ? x : static_cast<T>(0); }
template <typename T> __device__ T inline relu_d(T x) { return x > 0 ? 1. : 0.; }
template <typename T> __device__ T inline silu(T x) { return x / (1 + exp(-x)); }
template <typename T> __device__ T inline silu_d(T x) {
  const T s = 1 / (1 + exp(-x));
  return s * (1 + x * (1 - s));
}
template <typename T> __device__ T inline gelu(T x) {
  constexpr float kAlpha = M_SQRT1_2;
  return x * T(0.5) * (T(1) + erf(x * kAlpha));
}
template <typename T> __device__ T inline gelu_d(T x) {
  constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  constexpr float kAlpha = M_SQRT1_2;
  const T cdf = T(0.5) * (T(1) + erf(x * kAlpha));
  const T pdf = exp(T(-0.5) * x * x) * kBeta;
  return cdf + x * pdf;
}
template <typename T> __device__ T inline gelu_tanh(T x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715;
  auto inner = kBeta * (x + kKappa * x * x * x);
  return T(0.5) * x * (T(1) + tanh(inner));
}
template <typename T> __device__ T inline gelu_tanh_d(T x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715;
  auto x_sq = x * x;
  auto x_cube = x_sq * x;
  auto inner = kBeta * (x + kKappa * x_cube);
  auto tanh_inner = tanh(inner);

  auto left = T(0.5) * x;
  auto right = T(1) + tanh_inner;

  auto left_derivative = T(0.5) * right;

  auto tanh_derivative = T(1) - tanh_inner * tanh_inner;
  auto inner_derivative = kBeta * (T(1) + T(3) * kKappa * x_sq);
  auto right_derivative = left * tanh_derivative * inner_derivative;

  return left_derivative + right_derivative;
}

__device__ int inline
next_pow2(unsigned int x) {
  // Return the closest power of 2 greater than x (if x is 0 or a power of 2, return x).
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

int closest_factor(int n) {
    // finds the largest factor of n that is less than or equal to the square root of n
    int factor = 1;
    for (int i = 1; i * i <= n; i++)
        if (n % i == 0)
            factor = i;
    return factor;
}

/////////////////// forward kernels ///////////////////

template <typename T>
__global__ void
compute_stats_pt1(
    const T *X,
    const int H,
    const int W,
    const int C,
    const int G,
    const int w_loops,
    const int D,
    const int groups_per_block,
    const int reduce_n,
    WelfordData<typename acc_type<T>::type, INT> *welford_data) {
  /*
    Computes means and rstds of X on the W (width) dimension.
    grid: (x=N, y=H, z=blocks_per_row); block: (x=TPB/rows_per_block, y=rows_per_block)
    - w_loops = W / rows_per_block
    - reduce_n = TPB / groups_per_block; // number of inputs that gets reduced to a single output
    - TPB = Cd/blocks_per_row
    if TPB < C (blocks_per_row > 1, rows_per_block=1)
      TPB = ceil(C/blocks_per_row) (aka blocks_per_row*TPB >= C)
      X shape: (N, R, C) -view-> (N, H, W, C) -view-> (N, H, W, 1, blocks_per_row, TPB); X stride: (HWC, WC, C, C, TPB, 1)
      dram reduction (per block): (W, 1, TPB) -reduce-> (1, TPB)
    else (block.x=C, block.y=rows_per_block)
      TPB = Cd
      X shape: (N, H, W, C) -view-> (N, H, W/rows_per_block, rows_per_block, 1, C); X stride: (HWC, WC, dC, C, C, 1)
      dram reduction (per block): (W/rows_per_block, rows_per_block, C) -reduce-> (rows_per_block, C)
    shmem reduction (per block):
      if G/blocks_per_row >= 1
        (TPB,) -view-> (rows_per_block, G/blocks_per_row, D) -permute-> (rows_per_block, D, G/blocks_per_row) -reduce-> G/blocks_per_row
        output buffer: (N, G, H)
      else (e.g. blocks_per_row/G > 1 aka more than one thread-block reduces one group)
        snap constraints require that D % TPB = 0 in this case so blocks_per_row/G = CDIV(blocks_per_row, G)
        (TPB,) -view-> (1, 1, D) -permute-> (1, D, 1) -reduce-> 1
        output buffer: (N, blocks_per_row*CDIV(G/blocks_per_row), H) = (N, blocks_per_row, H)
  */
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  const int TPB = blockDim.y * blockDim.x;
  const int c = blockIdx.z * blockDim.x + threadIdx.x;
  const int rows_per_block = blockDim.y;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  if (c >= C) return;
  for (int i = 0; i < w_loops; ++i) {
    int w = i * rows_per_block + threadIdx.y;
    if (w >= W) continue; // handle indices which overflow width
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C;
    reduce_idx += blockIdx.y * W * C;
    reduce_idx += w * C;
    reduce_idx += blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x));
  }

  // shmem reduction
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int blocks_per_row = gridDim.z;
  const int d_idx = threadIdx.y;
  const int block_group_idx = threadIdx.x / D;
  const int D_idx = threadIdx.x % D;

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  int idx = 0;
  idx += d_idx * D * groups_per_block;
  idx += D_idx * groups_per_block;
  idx += block_group_idx;
  vals_reduced[idx] = val;
  __syncthreads();

  for (int stride = groups_per_block * next_pow2(reduce_n) / 2; stride >= groups_per_block; stride >>= 1) {
    if (tid < stride && (tid + stride) < TPB)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  const int partial_groups = (groups_per_block > 1) ? G : blocks_per_row;
  const int partial_group_idx = blockIdx.z * groups_per_block + block_group_idx;
  if (partial_group_idx >= partial_groups) return;
  int out_idx = 0;
  out_idx += blockIdx.x * partial_groups * H;
  out_idx += partial_group_idx * H;
  out_idx += blockIdx.y;
  welford_data[out_idx] = vals_reduced[block_group_idx];
}

template <typename T>
__global__ void
compute_stats_pt2(
    WelfordData<typename acc_type<T>::type, INT> *welford_data,
    const int G,
    const int R,
    const int l,
    const T eps,
    T *means,
    T *rstds) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  /*
    Computes means and rstds of X on the H (height) dimension.
    grid: (x=N, y=G); block: (x=TPB)
    - partials_per_group = number of thread-blocks in compute_stats1 needed to reduce a single group; equal to the number of elements per group per row that will be reduced into a single element
    - R = partials_per_group * H (R = number of reduced elements per block)
    - l = partials_per_group * H / TPB (l = number of times to loop a block to reduce the H dimension as well as any partially reduced group sections)
    if blocks_per_group = 1 (i.e. partial_groups = G)
      welford_data shape: (N, G, H) -view-> (N, G, H/TPB, TPB); X stride: (GH, H, TPB, 1)
    else (i.e. blocks_per_group > 1 aka partial_groups > G)
      welford_data shape: (N, partial_groups, H) -view-> (N, G, partials_per_group*H/TPB, TPB); X stride: (G*partials_per_group*H, partials_per_group*H, TPB, 1)
    dram reduction (per block): (CDIV(partials_per_group*H, TPB), TPB) -reduce-> (TPB,)
    shmem reduction (per block): (TPB,) -reduce-> (1,)
    output buffer: (N, G)
  */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const int TPB = blockDim.x;

  for (int i = 0; i < l; ++i) {
    int r = i * TPB + threadIdx.x;
    if (r >= R) continue;
    int idx = 0;
    idx += blockIdx.x * G * R;
    idx += blockIdx.y * R;
    idx += r;
    val = welford_op.combine(val, welford_data[idx]);
  }

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int tid = threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();

  for (int stride = next_pow2(TPB) / 2; stride >= 1; stride >>= 1) {
    if (tid < stride && tid + stride < TPB)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  if (tid != 0) return;
  thrust::pair<T_ACC, T_ACC> var_mean = welford_op.project(vals_reduced[tid]);
  T_ACC var = var_mean.first;
  T_ACC mean = var_mean.second;
  int ng = 0;
  ng += blockIdx.x * G;
  ng += blockIdx.y;
  means[ng] = mean;
  rstds[ng] = rsqrt(var + static_cast<T_ACC>(eps));
}

// intrinsics used in scale_shift because pytorch upcasts fp16/bf16 in these ops 
template <typename T> __device__ T mul(T a, T b);
inline __device__ double mul(double a, double b) { return a * b; }
inline __device__ float mul(float a, float b) { return a * b; }
inline __device__ c10::Half mul(c10::Half a, c10::Half b) { return __hmul(a, b); }
inline __device__ c10::BFloat16 mul(c10::BFloat16 a, c10::BFloat16 b) { return __hmul(static_cast<__nv_bfloat16>(a), static_cast<__nv_bfloat16>(b)); }

template <typename T> __device__ T fma(T a, T b, T c);
inline __device__ c10::Half fma(c10::Half a, c10::Half b, c10::Half c) { return __hfma(static_cast<half>(a), static_cast<half>(b), static_cast<half>(c)); }
inline __device__ c10::BFloat16 fma(c10::BFloat16 a, c10::BFloat16 b, c10::BFloat16 c) { return __hfma(static_cast<__nv_bfloat16>(a), static_cast<__nv_bfloat16>(b), static_cast<__nv_bfloat16>(c)); }

template <typename T, int vec_elems, T (*act_fn)(T)>
__global__ void
scale_shift_kernel(
    const T *X_data,
    const T *mean_data,
    const T *rstd_data,
    const T *weight_data,
    const T *bias_data,
    const int R,
    const int C,
    const int G,
    const int blocks_per_elem,
    T *y) {
  /*
    Performs elementwise op (X - mean) * rstd * weight + bias. Vectorized for speed.
    LOOP_I: number of elements that each thread processes.
    vec_elems: number of elements stored for each vector.
    grid: (x=N, y=HW/LOOP_I/rows_per_block, z=blocks_per_row), block: (x=TPB/rows_per_block, y=rows_per_block)
    - C' = C / vec_elems
    - blocks_per_row = cdiv(C', TPB) (e.g. blocks_per_row*TPB ~ C')
    - note: the grid is actually (x=NR/LOOP_I/rows_per_block, y=1, z=blocks_per_row) since y/z max block size is 65K which causes issues for N=1, H=W=1024, C=256
    if rows_per_block > 1:
      X shape: (N, R, C') -view-> (N, R/LOOP_I/rows_per_block, LOOP_I, rows_per_block, 1, C');   X.stride: (RC', LOOP_I*rows_per_block*C', TPB, C', C', 1)
    if blocks_per_row > 1:
      X shape: (N, R, C') -view-> (N, R/LOOP_I/1, LOOP_I, 1, blocks_per_row, TPB); X.stride: (RC', LOOP_I*C', C', C', TPB, 1)
   */
  using V = float_vec<T, vec_elems>;
  const int Cp = C / vec_elems;
  const int rows_per_block = blockDim.y;
  const int n = blockIdx.x / blocks_per_elem;
  const int by = blockIdx.x % blocks_per_elem; // hacky way to simulate blockIdx.y since blockDim.y is limited to 65K
  const int c = blockIdx.z * blockDim.x + threadIdx.x;
  if (c >= Cp) return;

  const int g = (G * c) / Cp;
  const int ng = n * G + g;
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  const V *weight_vecs = reinterpret_cast<const V*>(weight_data);
  const V *bias_vecs = reinterpret_cast<const V*>(bias_data);
  V *y_vecs = reinterpret_cast<V*>(y);
  T mean = mean_data[ng];
  T rstd = rstd_data[ng];
  V weight_vec = weight_vecs[c];
  V bias_vec = bias_vecs[c];

  // compute fused weight/bias a,b such that (x - mean) * rstd * weight + bias = x * a + b
  V fused_weight, fused_bias;
  if constexpr (vec_elems == 1) {
    fused_weight = {mul(rstd, weight_vec.x)};
    fused_bias = {fma(-mean, fused_weight.x, bias_vec.x)};
  }
  else if constexpr (vec_elems == 2) {
    fused_weight = {
      mul(rstd, weight_vec.x),
      mul(rstd, weight_vec.y)
    };
    fused_bias = {
      fma(-mean, fused_weight.x, bias_vec.x),
      fma(-mean, fused_weight.y, bias_vec.y)
    };
  }
  else if constexpr (vec_elems == 4) {
    fused_weight = {
      mul(rstd, weight_vec.x),
      mul(rstd, weight_vec.y),
      mul(rstd, weight_vec.z),
      mul(rstd, weight_vec.w)
    };
    fused_bias = {
      fma(-mean, fused_weight.x, bias_vec.x),
      fma(-mean, fused_weight.y, bias_vec.y),
      fma(-mean, fused_weight.z, bias_vec.z),
      fma(-mean, fused_weight.w, bias_vec.w)
    };
  }

  for (int i = 0; i < LOOP_I; ++i) {
    int row = 0;
    row += by * LOOP_I * rows_per_block;
    row += i * rows_per_block;
    row += threadIdx.y;
    if (row >= R) continue;

    int idx = 0;
    idx += n * R * Cp;
    idx += row * Cp;
    idx += c;

    V X_vec = X_vecs[idx];
    
    if constexpr (vec_elems == 1)
      y_vecs[idx] = {act_fn(fma(X_vec.x, fused_weight.x, fused_bias.x))};
    else if constexpr (vec_elems == 2) {
      y_vecs[idx] = {
        act_fn(fma(X_vec.x, fused_weight.x, fused_bias.x)),
        act_fn(fma(X_vec.y, fused_weight.y, fused_bias.y)),
      };
    }
    else if constexpr (vec_elems == 4) {
      y_vecs[idx] = {
        act_fn(fma(X_vec.x, fused_weight.x, fused_bias.x)),
        act_fn(fma(X_vec.y, fused_weight.y, fused_bias.y)),
        act_fn(fma(X_vec.z, fused_weight.z, fused_bias.z)),
        act_fn(fma(X_vec.w, fused_weight.w, fused_bias.w)),
      };
    }
  }
}

// scale_shift2 and scale_shift are helper fns that define the template args of scale_shift_kernel
template <typename T, int vec_elems> void scale_shift2(dim3 grid_dim, dim3 block_dim, int act_fn_option, const T *X_data, const T *mean_data, const T *rstd_data, const T *weight_data, const T *bias_data, const int N, const int R, const int C, const int G, T *y) {
  const int blocks_per_elem = grid_dim.x / N;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (act_fn_option == 0)
    scale_shift_kernel<T, vec_elems, identity><<<grid_dim, block_dim, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, R, C, G, blocks_per_elem, y);
  else if (act_fn_option == 1)
    scale_shift_kernel<T, vec_elems, relu><<<grid_dim, block_dim, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, R, C, G, blocks_per_elem, y);
  else if (act_fn_option == 2)
    scale_shift_kernel<T, vec_elems, silu><<<grid_dim, block_dim, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, R, C, G, blocks_per_elem, y);
  else if (act_fn_option == 3)
    scale_shift_kernel<T, vec_elems, gelu><<<grid_dim, block_dim, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, R, C, G, blocks_per_elem, y);
  else if (act_fn_option == 4)
    scale_shift_kernel<T, vec_elems, gelu_tanh><<<grid_dim, block_dim, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, R, C, G, blocks_per_elem, y);
}

template <typename T> void scale_shift(dim3 grid_dim, dim3 block_dim, int vec_elems, int act_fn_option, const T *X_data, const T *mean_data, const T *rstd_data, const T *weight_data, const T *bias_data, const int N, const int R, const int C, const int G, T *y) {
  if (vec_elems == 1)
    scale_shift2<T, 1>(grid_dim, block_dim, act_fn_option, X_data, mean_data, rstd_data, weight_data, bias_data, N, R, C, G, y);
  else if (vec_elems == 2)
    scale_shift2<T, 2>(grid_dim, block_dim, act_fn_option, X_data, mean_data, rstd_data, weight_data, bias_data, N, R, C, G, y);
  else if (vec_elems == 4)
    scale_shift2<T, 4>(grid_dim, block_dim, act_fn_option, X_data, mean_data, rstd_data, weight_data, bias_data, N, R, C, G, y);
}

template <typename T>
void run_gn_fwd_kernels(
    const T *X_data,
    const T *weight_data,
    const T *bias_data,
    const int N,
    const int R,
    const int C,
    const int G,
    T eps,
    const int64_t act_fn_option,
    T *Y_data,
    T *mean_data,
    T *rstd_data) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;

  const int H = closest_factor(R);
  const int W = R / H;
  auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(W * C, C, C / G);
  const int groups_per_block = CDIV(G, blocks_per_row); // number of groups processed per block in compute_stats_pt1, needed here because it determines size of welford_data
  const int partial_groups = (groups_per_block == 1) ? blocks_per_row : G; // number of group intermediates produced per element after compute_stats_pt1, must be a multiple of G
  WelfordType *welford_data = (WelfordType*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(WelfordType) * N * partial_groups * H);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  
  // compute means/rstds over width dimension
  {
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(W * C, C, C / G); // same fn + args as the one a couple lines up but repeated for clarity
    DEBUG("starting compute_stats 1, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d, groups_per_block: %d\n", N, H, W, C, G, (C / G), TPB, rows_per_block, blocks_per_row, groups_per_block);

    const int w_loops = CDIV(W, rows_per_block);
    const int D = C / G;
    const int groups_per_block = CDIV(G, blocks_per_row); // cdiv in case G < blocks_per_row -> ceil(G/blocks_per_row) = 1
    const int reduce_n = TPB / groups_per_block; // number of inputs that gets reduced to a single output

    compute_stats_pt1<<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), 0, cuda_stream>>>(
        X_data,
        H, W, C, G, 
        w_loops, D, groups_per_block, reduce_n,
        welford_data
    );
  }

  // compute means/rstds over height dimension
  {
    const int partials_per_group = partial_groups / G; // number of blocks to process one group
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(partials_per_group * H, partials_per_group * H);
    const int R = partials_per_group * H; // R for num "R"educe elements per block
    const int l = CDIV(R, TPB);
    DEBUG("starting compute_stats 2, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d, groups_per_block: %d\n", N, H, W, C, G, (C / G), TPB, rows_per_block, blocks_per_row, groups_per_block);
    compute_stats_pt2<<<dim3(N, G), TPB, 0, cuda_stream>>>(
        welford_data,
        G, R, l, eps,
        mean_data, rstd_data
    );
  }

  // scale/shift X
  {
    const int D = C / G;
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;

    if (!ELEM_DEBUG && R % LOOP_I == 0) {
      auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(R / LOOP_I * C / vec_elems, C / vec_elems);
      DEBUG("scale shift starting (LOOP_I = 8), N: %d, R: %d, C: %d, G: %d, D: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d, vec_elems: %d, (virtual) by: %d\n", N, R, C, G, D, TPB, rows_per_block, blocks_per_row, vec_elems, CDIV(H*W, LOOP_I*rows_per_block));
      scale_shift<T>(
          dim3(N * CDIV(R, LOOP_I*rows_per_block), 1, blocks_per_row), dim3(TPB/rows_per_block, rows_per_block),
          vec_elems, act_fn_option,
          X_data, mean_data, rstd_data,
          weight_data, bias_data,
          N, R, C, G,
          Y_data
      );
    }
    else {// relatively slow fallback
      auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(C, C); // each block operates only on one spatial element (on all channels) and with vec_elems = 1
      DEBUG("SLOW FALLBACK, scale shift kernel starting, N: %d, R: %d, C: %d, G: %d, D: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d, \n", N, R, C, G, D, TPB, rows_per_block, blocks_per_row);
      scale_shift<T>(
          dim3(N * CDIV(R, LOOP_I*rows_per_block), 1, blocks_per_row), dim3(TPB/rows_per_block, rows_per_block),
          1, act_fn_option,
          X_data, mean_data, rstd_data,
          weight_data, bias_data,
          N, R, C, G,
          Y_data
      );
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(welford_data);
}

template void run_gn_fwd_kernels<float>(const float *X_data, const float *weight_data, const float *bias_data, const int N, const int R, const int C, const int G, float eps, const int64_t act_fn_option, float *Y_data, float *mean_data, float *rstd_data);
template void run_gn_fwd_kernels<double>(const double *X_data, const double *weight_data, const double *bias_data, const int N, const int R, const int C, const int G, double eps, const int64_t act_fn_option, double *Y_data, double *mean_data, double *rstd_data);
template void run_gn_fwd_kernels<c10::Half>(const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *bias_data, const int N, const int R, const int C, const int G, c10::Half eps, const int64_t act_fn_option, c10::Half *Y_data, c10::Half *mean_data, c10::Half *rstd_data);
template void run_gn_fwd_kernels<c10::BFloat16>(const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *bias_data, const int N, const int R, const int C, const int G, c10::BFloat16 eps, const int64_t act_fn_option, c10::BFloat16 *Y_data, c10::BFloat16 *mean_data, c10::BFloat16 *rstd_data);

/////////////////// backward kernels ///////////////////

template <typename T>
__device__ void
sum_reduce(
    T vals_reduced,
    const int start_size,
    const int end_size) {
  // Sums a shared buffer (vals_reduced) containing start_size values (shape (reduce_n, end_size)) into (end_size,)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int reduce_n = start_size / end_size;

  for (int stride = end_size * next_pow2(reduce_n) / 2; stride >= end_size; stride >>= 1) {
    if (tid < stride && tid + stride < start_size)
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
  }
}

template <typename T, T (*act_d_fn)(T)>
__global__ void
width_reduce(
    const T *dy_data,
    const T *X_data,
    const T *mean_data,
    const T *rstd_data,
    const T *weight_data,
    const T *bias_data,
    const int H,
    const int W,
    const int C,
    const int G,
    const int w_loops,
    typename acc_type<T>::type *xdy_dy_sum_data) {
  /*
    Loops over W (width) dimension, loading and summing dy, X, and the activation derivative of Y. Outputs stored in xdy_dy_sum_data. Spatial dimension H is processed in a separate kernel.
    grid: (x=N, y=H, z=blocks_per_row); blockdim: (x=TPB/rows_per_block, y=rows_per_block)
      TPB = Cd/blocks_per_row
    if TPB < C (blocks_per_row > 1, rows_per_block=1)
      C = blocks_per_row*TPB
      X shape: (N, H, W, C) -view-> (N, H, W, 1, blocks_per_row, TPB); X stride: (HWC, WC, C, C, TPB, 1)
      dram reduction (per block): (W, 1, TPB) -reduce-> (TPB,)
    else (block.x=C, block.y=rows_per_block)
      TPB = Cd
      X shape: (N, H, W, C) -view-> (N, H, W/rows_per_block, rows_per_block, 1, C); X stride: (HWC, WC, dC, C, C, 1)
      dram reduction (per block): (W/rows_per_block, rows_per_block, C) -reduce-> (rows_per_block, C)
    shmem reduction (per block): (TPB, 2) -> (rows_per_block, C/blocks_per_row, 2) -reduce-> (C/blocks_per_row, 2) (the 2 comes from storing both xdy_sum and dy_sum in the same buffer)
    output buffer: (N, blocks_per_row, C/blocks_per_row, H, 2) -view-> (N, C, H, 2)
      xdy_dy_sum_data[:, :, :, 0] = x * dy * activation_derivative((x-mean)*rstd*weight+bias)
      xdy_dy_sum_data[:, :, :, 1] = dy * activation_derivative((x-mean)*rstd*weight+bias)
   */
  using T_ACC = typename acc_type<T>::type;

  const int TPB = blockDim.y * blockDim.x;
  const int rows_per_block = blockDim.y;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;

  const int n = blockIdx.x;
  int c = blockIdx.z * blockDim.x + threadIdx.x;
  int g = G * c / C;
  const int ng = n * G + g;
  if (c >= C) return;
  T_ACC fused_scale = rstd_data[ng] * weight_data[c];
  T_ACC fused_bias = -mean_data[ng] * fused_scale + bias_data[c];

  for (int i = 0; i < w_loops; ++i) {
    int w = i * rows_per_block + threadIdx.y;
    if (w >= W) continue; // handle overflowing indices
    int reduce_idx = 0;
    reduce_idx += n * H * W * C;
    reduce_idx += blockIdx.y * W * C;
    reduce_idx += w * C;
    reduce_idx += blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T_ACC X_elem = static_cast<T_ACC>(X_data[reduce_idx]);
    T_ACC X_norm = X_elem * fused_scale + fused_bias;
    T_ACC d_act = act_d_fn(X_norm);
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    xdy_sum += dy_elem * d_act * X_elem;
    dy_sum += dy_elem * d_act;
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if (TPB > C) {
    vals_reduced[2 * tid] = xdy_sum;
    vals_reduced[2 * tid + 1] = dy_sum;
    __syncthreads();
    sum_reduce(vals_reduced, 2 * TPB, 2 * C); // does nothing if rows_per_block=1
    xdy_sum = vals_reduced[2 * tid];
    dy_sum = vals_reduced[2 * tid + 1];
  }

  // put reduced outputs into return buffers
  if (threadIdx.y != 0) return; 
  // at this point ty = 0 and c < C -> maximum of C threads/block reaching past this point (fewer threads if blocks_per_row > 1)
  int out_idx = 0;
  out_idx += n * C * H;
  out_idx += c * H;
  out_idx += blockIdx.y;

  xdy_dy_sum_data[2 * out_idx] = xdy_sum;
  xdy_dy_sum_data[2 * out_idx + 1] = dy_sum;
}

template <typename T>
__global__ void
height_reduce(
    T *xdy_dy_sum_data, // no need to specify T_ACC as T is already an accumulation type
    const int H,
    const int C,
    const int l,
    T *xdy_sum_data,
    T *dy_sum_data) {
  /*
    Same thing as width_reduce but over the H (height) instead of the width dimension.
    - l = CDIV(2H, TPB) (computed on CPU for speed); number of loops for DRAM reduction
    grid: (x=N, y=C); block: (x=2H/blocks_per_row)
    X shape: (N, C, H, 2) -view-> (N, C, blocks_per_row, H/blocks_per_row, 2); X stride: (2CH, 2H, 2H/blocks_per_row, H/blocks_per_row, 1)
    dram reduction (per block): (blocks_per_row, H/blocks_per_row, 2) -reduce-> (H/blocks_per_row, 2)
    shmem reduction (per block): (H/blocks_per_row, 2) -reduce-> (2,)
    output buffer: (N, C, 2)
   */
  const int TPB = blockDim.x;
  const int tid = threadIdx.x;

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[];
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);

  T sum = 0;
  for (int i = 0; i < l; ++i) {
    const int h = i * TPB + tid;
    if (h >= 2 * H) continue;
    int idx = 0;
    idx += blockIdx.x * C * H * 2;
    idx += blockIdx.y * H * 2;
    idx += h;
    sum += xdy_dy_sum_data[idx];
  }

  vals_reduced[tid] = sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB, 2);

  // put reduced outputs into return buffers
  if (tid != 0) return;
  int out_idx = blockIdx.x * C + blockIdx.y;
  xdy_sum_data[out_idx] = vals_reduced[0];
  dy_sum_data[out_idx] = vals_reduced[1];
}

template <typename T>
__global__ void
compute_dweight_dbias(
    const T *mean_data,
    const T *rstd_data,
    typename acc_type<T>::type *xdy_sum_data,
    typename acc_type<T>::type *dy_sum_data,
    const int N,
    const int C,
    const int G,
    const int D,
    T *dweight_data,
    T *dbias_data) {
  /*
    Computes derivatives wrt the weight and bias. 
    grid: (x=blocks_per_row), block: (x=C/blocks_per_row)
   */
  using T_ACC = typename acc_type<T>::type;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c >= C) return;
  const int g = c / D;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;

  for (int n = 0; n < N; ++n) {
    const int nc = n * C + c;
    const int ng = n * G + g;
    sum1 += (xdy_sum_data[nc] - mean_data[ng] * dy_sum_data[nc]) * rstd_data[ng];
    sum2 += dy_sum_data[nc];
  }
  dweight_data[c] = sum1;
  dbias_data[c] = sum2;
}

template <typename T>
__global__ void
compute_bwd_scale_biases(
    const T *mean_data,
    const T *rstd_data,
    const T *weight_data,
    const T *bias_data,
    typename acc_type<T>::type *xdy_sum_data,
    typename acc_type<T>::type *dy_sum_data,
    const int R,
    const int C,
    const int G,
    const int D,
    typename acc_type<T>::type *fused_scale_data,
    typename acc_type<T>::type *fused_bias_data,
    typename acc_type<T>::type *coef1_data,
    typename acc_type<T>::type *coef2_data) {
  /*
    Calculates coefficients to reduce computation on the elementwise kernel.
    - fused_scale: rstd * weight
    - fused_bias: -mean * rstd * weight + bias
      - fused_scale * x + fused_bias = (x - mean) / std * weight + bias
    - coef1/2: some derivative terms
    griddim: (x=N, y=G); blockdim: (x=D/blocks_per_row)
    - blocks_per_row = num. elements to loop over to traverse D
    - C/blocks_per_row = TPB (threads per block)
    X shape: (N, C) -view-> (N, G, D) -permute-> (N, D, G) -reduce-> (N, G)
    shmem reduction: (D, G) -reduce-> G
    output buffer:
      fused_scale/bias: (N, C)
      coef1/2: (N, G)
   */
  using T_ACC = typename acc_type<T>::type;
  const int TPB = blockDim.x;
  const int blocks_per_row = CDIV(D, TPB);
  const int n = blockIdx.x;
  const int g = blockIdx.y;

  const int ng = n * G + g;
  const T_ACC mean_elem = static_cast<T_ACC>(mean_data[ng]);
  const T_ACC rstd_elem = static_cast<T_ACC>(rstd_data[ng]);

  extern __shared__ char vals_reduced_uncasted[]; // contains next_pow2(2 * TPB) values (instead of next_pow2(TPB)) because we're storing 2 values per thread (xdy_sum, dy_sum)
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  T_ACC xdy_gamma = 0;
  T_ACC dy_gamma = 0;
  for (int i = 0; i < blocks_per_row; ++i) {
    const int rows_per_block = i * TPB + threadIdx.x;
    if (rows_per_block >= D) continue;
    const int c = g * D + rows_per_block;
    const int nc = n * C + c;
    const T_ACC gamma = static_cast<T_ACC>(weight_data[c]);
    fused_scale_data[nc] = rstd_elem * gamma;
    fused_bias_data[nc] = -mean_elem * rstd_elem * weight_data[c] + bias_data[c];

    xdy_gamma += gamma * xdy_sum_data[nc];
    dy_gamma += gamma * dy_sum_data[nc];
  }
  vals_reduced[2 * threadIdx.x] = xdy_gamma;
  vals_reduced[2 * threadIdx.x + 1] = dy_gamma;
  __syncthreads();
  sum_reduce(vals_reduced, 2 * TPB, 2);

  if (threadIdx.x != 0) return;
  const T_ACC xdy_gamma_sum = vals_reduced[0];
  const T_ACC dy_gamma_sum = vals_reduced[1];
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * R);
  const T_ACC x = (mean_elem * dy_gamma_sum - xdy_gamma_sum) * s * rstd_elem * rstd_elem * rstd_elem;
  coef1_data[ng] = x;
  coef2_data[ng] = -mean_elem * x - (dy_gamma_sum * s * rstd_elem);
}

template <typename T, int vec_elems, T (*act_d_fn)(T)>
__global__ void
dx_elem_kernel(
    const T *dy_data,
    const T *X_data,
    typename acc_type<T>::type *fused_scale_data,
    typename acc_type<T>::type *fused_bias_data,
    typename acc_type<T>::type *coef1_data,
    typename acc_type<T>::type *coef2_data,
    const int R,
    const int C,
    const int G,
    const int blocks_per_elem,
    T *dx_data) {
  /*
    Performs elementwise kernel to calculate gradients wrt X. Vectorized for speed.
    LOOP_I: number of elements that each thread processes.
    vec_elems: number of elements stored for each vector.
    grid: (x=N, y=HW/LOOP_I, y=blocks_per_row), block: (x=TPB)
    - C' = C / vec_elems
    - blocks_per_row = cdiv(C', TPB) (e.g. blocks_per_row*TPB ~ C')
    - blocks_per_elem = gridDim.x / N (computed on CPU for speed)
    - note: the grid is actually (x=NR/LOOP_I/rows_per_block, y=1, z=blocks_per_row) since y/z max block size is 65K which causes issues for N=1, R=1024*1024, C=256
    if rows_per_block > 1:
      X shape: (N, R, C') -view-> (N, R/LOOP_I/rows_per_block, LOOP_I, rows_per_block, 1, C');   X.stride: (RC', LOOP_I*rows_per_block*C', TPB, C', C', 1)
    if blocks_per_row > 1:
      X shape: (N, R, C') -view-> (N, R/LOOP_I/1, LOOP_I, 1, blocks_per_row, TPB); X.stride: (RC', LOOP_I*C', C', C', TPB, 1)
   */
  using T_ACC = typename acc_type<T>::type;
  using V = float_vec<T, vec_elems>;
  using V_ACC = float_vec<T_ACC, vec_elems>;
  const int Cp = C / vec_elems;
  const int rows_per_block = blockDim.y;
  const int n = blockIdx.x / blocks_per_elem;
  const int by = blockIdx.x % blocks_per_elem; // hacky way to simulate blockIdx.y since blockDim.y is limited to 65K
  const int c = blockIdx.z * blockDim.x + threadIdx.x;
  if (c >= Cp) return;

  const int g = (G * c) / Cp;
  const int nc = n * Cp + c;
  const int ng = n * G + g;
  T_ACC coef1 = coef1_data[ng];
  T_ACC coef2 = coef2_data[ng];
  const V *dy_vecs = reinterpret_cast<const V*>(dy_data);
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  V *dx_vecs = reinterpret_cast<V*>(dx_data);
  V_ACC fused_scale_vec = reinterpret_cast<V_ACC*>(fused_scale_data)[nc];
  V_ACC fused_bias_vec = reinterpret_cast<V_ACC*>(fused_bias_data)[nc];

  for (int i = 0; i < LOOP_I; ++i) {
    int row = 0;
    row += by * LOOP_I * rows_per_block;
    row += i * rows_per_block;
    row += threadIdx.y;
    if (row >= R) continue;

    int idx = 0;
    idx += n * R * Cp;
    idx += row * Cp;
    idx += c;

    V dy_vec = dy_vecs[idx];
    V X_vec = X_vecs[idx];

    if constexpr (vec_elems == 1) {
      V X_norm = {X_vec.x * fused_scale_vec.x + fused_bias_vec.x};
      dx_vecs[idx] = {
        (fused_scale_vec.x * dy_vec.x * act_d_fn(X_norm.x))
          + (coef1 * X_vec.x + coef2)
      };
    }
    else if constexpr (vec_elems == 2) {
      V X_norm = {
        X_vec.x * fused_scale_vec.x + fused_bias_vec.x,
        X_vec.y * fused_scale_vec.y + fused_bias_vec.y,
      };
      dx_vecs[idx] = {
        (fused_scale_vec.x * dy_vec.x * act_d_fn(X_norm.x))
          + (coef1 * X_vec.x + coef2),
        (fused_scale_vec.y * dy_vec.y * act_d_fn(X_norm.y))
          + (coef1 * X_vec.y + coef2),
      };
    }
    else if constexpr (vec_elems == 4) {
      V X_norm = {
        X_vec.x * fused_scale_vec.x + fused_bias_vec.x,
        X_vec.y * fused_scale_vec.y + fused_bias_vec.y,
        X_vec.z * fused_scale_vec.z + fused_bias_vec.z,
        X_vec.w * fused_scale_vec.w + fused_bias_vec.w,
      };
      dx_vecs[idx] = {
        (fused_scale_vec.x * dy_vec.x * act_d_fn(X_norm.x))
          + (coef1 * X_vec.x + coef2),
        (fused_scale_vec.y * dy_vec.y * act_d_fn(X_norm.y))
          + (coef1 * X_vec.y + coef2),
        (fused_scale_vec.z * dy_vec.z * act_d_fn(X_norm.z))
          + (coef1 * X_vec.z + coef2),
        (fused_scale_vec.w * dy_vec.w * act_d_fn(X_norm.w))
          + (coef1 * X_vec.w + coef2),
      };
    }
  }
}

// dx_elem2 and dx_elem are helper fns that define the template args of dx_elem_kernel
template <typename T, int vec_elems> void dx_elem2(dim3 grid_dim, dim3 block_dim, int act_fn_option, const T *dy_data, const T *X_data, typename acc_type<T>::type *fused_scale_data, typename acc_type<T>::type *fused_bias_data, typename acc_type<T>::type *coef1_data, typename acc_type<T>::type *coef2_data, const int N, const int R, const int C, const int G, T *dx_data) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int blocks_per_elem = grid_dim.x / N;
  if (act_fn_option == 0)
    dx_elem_kernel<T, vec_elems, identity_d><<<grid_dim, block_dim, 0, cuda_stream>>>(dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, R, C, G, blocks_per_elem, dx_data);
  else if (act_fn_option == 1)
    dx_elem_kernel<T, vec_elems, relu_d><<<grid_dim, block_dim, 0, cuda_stream>>>(dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, R, C, G, blocks_per_elem, dx_data);
  else if (act_fn_option == 2)
    dx_elem_kernel<T, vec_elems, silu_d><<<grid_dim, block_dim, 0, cuda_stream>>>(dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, R, C, G, blocks_per_elem, dx_data);
  if (act_fn_option == 3)
    dx_elem_kernel<T, vec_elems, gelu_d><<<grid_dim, block_dim, 0, cuda_stream>>>(dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, R, C, G, blocks_per_elem, dx_data);
  if (act_fn_option == 4)
    dx_elem_kernel<T, vec_elems, gelu_tanh_d><<<grid_dim, block_dim, 0, cuda_stream>>>(dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, R, C, G, blocks_per_elem, dx_data);
}

template <typename T> void dx_elem(dim3 grid_dim, dim3 block_dim, int vec_elems, int act_fn_option, const T *dy_data, const T *X_data, typename acc_type<T>::type *fused_scale_data, typename acc_type<T>::type *fused_bias_data, typename acc_type<T>::type *coef1_data, typename acc_type<T>::type *coef2_data, const int N, const int R, const int C, const int G, T *dx_data) {
  if (vec_elems == 1)
    dx_elem2<T, 1>(grid_dim, block_dim, act_fn_option, dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, N, R, C, G, dx_data);
  else if (vec_elems == 2)
    dx_elem2<T, 2>(grid_dim, block_dim, act_fn_option, dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, N, R, C, G, dx_data);
  else if (vec_elems == 4)
    dx_elem2<T, 4>(grid_dim, block_dim, act_fn_option, dy_data, X_data, fused_scale_data, fused_bias_data, coef1_data, coef2_data, N, R, C, G, dx_data);
}

template <typename T>
void run_gn_bwd_kernels(
    const T *dy_data,
    const T *X_data,
    const T *weight_data,
    const T *bias_data,
    const T *mean_data,
    const T *rstd_data,
    const int N,
    const int R,
    const int C,
    const int G,
    const int64_t act_fn_option,
    T *dx_data,
    T *dweight_data,
    T *dbias_data) {
  using T_ACC = typename acc_type<T>::type;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int D = C / G;

  const int H = closest_factor(R);
  const int W = R / H;
  T_ACC* xdy_dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C * H * 2);

  // sum over W dim
  {
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(W * C, C);
    DEBUG("starting width reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d\n", N, H, W, C, G, TPB, rows_per_block, blocks_per_row);
    const int w_loops = CDIV(W, rows_per_block);
    if (act_fn_option == 0)
      width_reduce<T, identity_d><<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, w_loops, xdy_dy_sum_data);
    else if (act_fn_option == 1)
      width_reduce<T, relu_d><<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, w_loops, xdy_dy_sum_data);
    else if (act_fn_option == 2)
      width_reduce<T, silu_d><<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, w_loops, xdy_dy_sum_data);
    else if (act_fn_option == 3)
      width_reduce<T, gelu_d><<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, w_loops, xdy_dy_sum_data);
    else if (act_fn_option == 4)
      width_reduce<T, gelu_tanh_d><<<dim3(N, H, blocks_per_row), dim3(TPB / rows_per_block, rows_per_block), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, w_loops, xdy_dy_sum_data);
  }

  T_ACC* xdy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC* dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  // sum over H dim
  {
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(2 * H, 2);
    DEBUG("starting height reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d\n", N, H, W, C, G, TPB, rows_per_block, blocks_per_row);
    const int l = CDIV(2 * H, TPB);
    height_reduce<<<dim3(N, C), TPB, sizeof(T_ACC) * TPB, cuda_stream>>>(
        xdy_dy_sum_data,
        H, C, l,
        xdy_sum_data, dy_sum_data);
  }
  c10::cuda::CUDACachingAllocator::raw_delete(xdy_dy_sum_data);

  // compute weight/bias grads
  {
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(C, C);
    DEBUG("starting compute dweight dbias, N: %d, R: %d, C: %d, G: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d\n", N, R, C, G, TPB, rows_per_block, blocks_per_row);
    compute_dweight_dbias<<<blocks_per_row, TPB, 0, cuda_stream>>>(
        mean_data, rstd_data,
        xdy_sum_data, dy_sum_data,
        N, C, G, D,
        dweight_data, dbias_data);
  }

  T_ACC *fused_scale_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC *fused_bias_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC *coef1_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);
  T_ACC *coef2_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);

  // compute fused scales/biases for X grads
  {
    auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(D, D);
    DEBUG("starting bwd scale biases, N: %d, R: %d, C: %d, G: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d\n", N, R, C, G, TPB, rows_per_block, blocks_per_row);
    compute_bwd_scale_biases<<<dim3(N, G), TPB, sizeof(T_ACC) * 2*TPB, cuda_stream>>>(
        mean_data, rstd_data, weight_data, bias_data,
        xdy_sum_data, dy_sum_data,
        R, C, G, D,
        fused_scale_data, fused_bias_data, coef1_data, coef2_data);
  }

  // compute X grads
  {
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;

    if (!ELEM_DEBUG && R % LOOP_I == 0) {
      auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(R / LOOP_I * C / vec_elems, C / vec_elems);
      DEBUG("dx elem kernel starting, N: %d, R: %d, C: %d, G: %d, D: %d, TPB: %d, rows_per_block: %d, blocks_per_row: %d, vec_elems: %d\n", N, R, C, G, D, TPB, rows_per_block, blocks_per_row, vec_elems);
      dx_elem<T>(
          dim3(N * CDIV(R, LOOP_I*rows_per_block), 1, blocks_per_row), dim3(TPB/rows_per_block, rows_per_block),
          vec_elems, act_fn_option,
          dy_data, X_data,
          fused_scale_data, fused_bias_data,
          coef1_data, coef2_data,
          N, R, C, G,
          dx_data
      );
    }
    else { // relatively slow fallback
      auto [TPB, rows_per_block, blocks_per_row] = calc_block_params(C, C); // each block operates only on one spatial element (on all channels) and with vec_elems = 1
      DEBUG("SLOW FALLBACK, dx elem kernel starting, N: %d, R: %d, C: %d, G: %d, D: %d, TPB: %d, blocks_per_row: %d\n", N, R, C, G, D, TPB, blocks_per_row);
      dx_elem<T>(
          dim3(N * R/rows_per_block, 1, blocks_per_row), dim3(TPB/rows_per_block, rows_per_block),
          1, act_fn_option,
          dy_data, X_data,
          fused_scale_data, fused_bias_data,
          coef1_data, coef2_data,
          N, R, C, G,
          dx_data
      );
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(xdy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(dy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(fused_scale_data);
  c10::cuda::CUDACachingAllocator::raw_delete(fused_bias_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef1_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef2_data);
}

template void run_gn_bwd_kernels<double>(const double *dy_data, const double *X_data, const double *weight_data, const double *bias_data, const double *mean_data, const double *rstd_data, const int N, const int R, const int C, const int G, const int64_t act_fn_option, double *dx_data, double *dweight_data, double *dbias_data);
template void run_gn_bwd_kernels<float>(const float *dy_data, const float *X_data, const float *weight_data, const float *bias_data, const float *mean_data, const float *rstd_data, const int N, const int R, const int C, const int G, const int64_t act_fn_option, float *dx_data, float *dweight_data, float *dbias_data);
template void run_gn_bwd_kernels<c10::Half>(const c10::Half *dy_data, const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *bias_data, const c10::Half *mean_data, const c10::Half *rstd_data, const int N, const int R, const int C, const int G, const int64_t act_fn_option, c10::Half *dx_data, c10::Half *dweight_data, c10::Half *dbias_data);
template void run_gn_bwd_kernels<c10::BFloat16>(const c10::BFloat16 *dy_data, const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *bias_data, const c10::BFloat16 *mean_data, const c10::BFloat16 *rstd_data, const int N, const int R, const int C, const int G, const int64_t act_fn_option, c10::BFloat16 *dx_data, c10::BFloat16 *dweight_data, c10::BFloat16 *dbias_data);
