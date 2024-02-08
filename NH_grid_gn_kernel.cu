//#include <ATen/cuda/Exceptions.h> // AT_CUDA_CHECK
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty_like.h>
#include <ATen/OpMathType.h> // opmath_t
#include <ATen/ops/empty.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <ATen/Tensor.h> // torch tensor
#include <c10/core/ScalarType.h>
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#include "Welford.h"
#include "vecs.h"
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
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int g = c / D;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  const at::acc_type<T, true> a_nc = rstds[ng] * weight[c];
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

#define ACT 

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
  using T_ACC = at::acc_type<T, true>;
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
NH_compute_stats_pt1(
    const T* X,
    const int H,
    const int W,
    const int C,
    const int G,
    WelfordData<at::acc_type<T, true>, int> *welford_data
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
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
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
NH_compute_stats_pt2(
    WelfordData<at::acc_type<T, true>, int> *welford_data,
    const int H,
    const int G,
    const float eps,
    T* means,
    T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
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

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = welford_data[blockIdx.x * G * H + blockIdx.y * H + threadIdx.x];
  __syncthreads();

  // next lowest power of 2 (AKA half of the next highest power of 2) - https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
  int start_stride = TPB - 1;
  start_stride |= start_stride >> 1;
  start_stride |= start_stride >> 2;
  start_stride |= start_stride >> 4;
  start_stride |= start_stride >> 8;
  start_stride |= start_stride >> 16;
  start_stride = (start_stride + 1) >> 1;

  // doing the first iteration outside the loop because of the extra condition regarding inputs with non-power-of-2 heights
  if (tid < start_stride && tid + start_stride < H)
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
  }
}

#define TENSORIT_DEBUG 0
#include <ATen/native/cuda/Loops.cuh>

template <typename T>
void NH_gn_fwd(
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

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);

  using T_ACC = at::acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int>;
  at::Tensor welford_tensor = at::empty({N, G, H, sizeof(WelfordType)}, X.options().dtype(at::kByte));
  WelfordType *welford_data = reinterpret_cast<WelfordType*>(welford_tensor.mutable_data_ptr());
  
  int TPB, d = 1, f = 1;
  TPB = MIN(MAX_THREADS_PER_BLOCK, W * C);
  if (C < TPB)
    if (TPB / C / 4 != 0)
      d = 4 * (TPB / C / 4); // prefer d being a multiple of 4 (e.g. for C=96, prefer TPB=384 and not 480) since it makes it more likely that TPB will be able to use the fast elementwise kernel
    else
      d = TPB / C;
  else
    while (C % f != 0 || C / f > MAX_THREADS_PER_BLOCK || G % f != 0)
      ++f;
  TPB = C * d / f;
  //printf("starting compute_stats, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", N, H, W, C, G, C / G, TPB, d, f, G / f);
  NH_compute_stats_pt1<<<dim3(N, H, f), dim3(TPB / d, d)>>>(
      X_data,
      H, W, C, G, 
      welford_data
  );

  NH_compute_stats_pt2<<<dim3(N, G), H>>>(
      welford_data,
      H, G, eps,
      mean_data, rstd_data
  );

  const int LOOP_I = 8;
  const int D = C / G;
  int vec_elems;
  if (D % 4 == 0) vec_elems = 4;
  else if (D % 2 == 0) vec_elems = 2;
  else vec_elems = 1;
  if (!TENSORIT_DEBUG && ((H * W * C) % (TPB * LOOP_I * f * vec_elems) == 0)) {
    const T* weight_data = weight.const_data_ptr<T>();
    const T* bias_data = bias.const_data_ptr<T>();
    T* Y_data = Y.mutable_data_ptr<T>();

    //printf("starting scale_shift N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, f: %d, G/f: %d,\n", N, H, W, C, G, C / G, TPB, f, G / f);
    const int num_blocks = N * H * W * C / TPB / LOOP_I / f;
    if (vec_elems == 4)
      scale_shift<T, LOOP_I, 4><<<dim3(num_blocks / vec_elems, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    else if (vec_elems == 2)
      scale_shift<T, LOOP_I, 2><<<dim3(num_blocks / vec_elems, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    else
      scale_shift<T, LOOP_I, 1><<<dim3(num_blocks / vec_elems, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
  }
  else {
    //printf("using TensorIterator, N: %d H %d W %d C %d G %d TPB %d f %d\n", N, H, W, C, G, TPB, f);
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

std::vector<at::Tensor> gn_nhwc_cuda_fwd_NH_grid(
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
    "group_norm_nhwc_forward_NH_grid", [&]() {
      NH_gn_fwd<scalar_t>(
          X_nhwc,
          weight, bias,
          G, eps,
          X_out, means, rstds
      );
  });
  return {X_out.permute({0, 3, 1, 2}), means, rstds};
}
