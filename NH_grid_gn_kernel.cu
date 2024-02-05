#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty_like.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <c10/core/ScalarType.h>
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h> // torch tensor
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/cuda/Exceptions.h> // AT_CUDA_CHECK
# include <ATen/OpMathType.h> // opmath_t
#include <ATen/native/cuda/Loops.cuh>

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

template <typename T, int num_elems>
struct float_vec;

template <typename T>
struct alignas(1 * sizeof(T)) float_vec<T, 1> {
  T x;
};
template <typename T>
struct alignas(2 * sizeof(T)) float_vec<T, 2> {
  T x, y;
};
template <typename T>
struct alignas(4 * sizeof(T)) float_vec<T, 4> {
  T x, y, z, w;
};
template <typename T>
struct alignas(8 * sizeof(T)) float_vec<T, 8> {
  T x, y, z, w, a, b, c, d;
};

#define ACT // silu

template <typename T, int LOOP_I, int vec_elems>
__global__ void
scale_shift_elem_kernelV(
    const T* X_data,
    at::acc_type<T, true>* a_data,
    at::acc_type<T, true>* b_data,
    const int N,
    const int C,
    T* y
    ) {
  using T_ACC = at::acc_type<T, true>;
  using V = float_vec<T, vec_elems>;
  using V_ACC = float_vec<T_ACC, vec_elems>;
  const int n = (N * blockIdx.x) / gridDim.x;
  //const int c = threadIdx.x % (C / vec_elems);
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) % (C / vec_elems);
  const int nc = n * (C / vec_elems) + c;
  const V *X_vec = reinterpret_cast<const V*>(X_data);
  V *y_vec = reinterpret_cast<V*>(y);
  V_ACC *a_vec = reinterpret_cast<V_ACC*>(a_data);
  V_ACC *b_vec = reinterpret_cast<V_ACC*>(b_data);
#pragma unroll LOOP_I
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * gridDim.y * blockDim.x;
    //idx += i * blockDim.x;
    idx += i * gridDim.y * blockDim.x;
    idx += blockIdx.y * blockDim.x;
    idx += threadIdx.x;
    V tmp_X = X_vec[idx];
    V_ACC tmp_a = a_vec[nc];
    V_ACC tmp_b = b_vec[nc];
    if constexpr (vec_elems == 1)
      y_vec[idx] = {ACT(tmp_a.x * tmp_X.x + tmp_b.x)};
    else if constexpr (vec_elems == 2) {
      T y_x, y_y;
      y_x = ACT(tmp_a.x * tmp_X.x + tmp_b.x);
      y_y = ACT(tmp_a.y * tmp_X.y + tmp_b.y);
      y_vec[idx] = {y_x, y_y};
    }
    else if constexpr (vec_elems == 4) {
      T y_x, y_y, y_z, y_w;
      y_x = ACT(tmp_a.x * tmp_X.x + tmp_b.x);
      y_y = ACT(tmp_a.y * tmp_X.y + tmp_b.y);
      y_z = ACT(tmp_a.z * tmp_X.z + tmp_b.z);
      y_w = ACT(tmp_a.w * tmp_X.w + tmp_b.w);
      y_vec[idx] = {y_x, y_y, y_z, y_w};
    }
    else if constexpr (vec_elems == 8) {
      T y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d;
      y_x = ACT(tmp_a.x * tmp_X.x + tmp_b.x);
      y_y = ACT(tmp_a.y * tmp_X.y + tmp_b.y);
      y_z = ACT(tmp_a.z * tmp_X.z + tmp_b.z);
      y_w = ACT(tmp_a.w * tmp_X.w + tmp_b.w);
      y_a = ACT(tmp_a.a * tmp_X.a + tmp_b.a);
      y_b = ACT(tmp_a.b * tmp_X.b + tmp_b.b);
      y_c = ACT(tmp_a.c * tmp_X.c + tmp_b.c);
      y_d = ACT(tmp_a.d * tmp_X.d + tmp_b.d);
      y_vec[idx] = {y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d};
    }
  }
}

template <typename T, int LOOP_I, int vec_elems>
__global__ void
small_scale_shift_elem_kernelV(
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
  const V *x_vec = reinterpret_cast<const V*>(X_data);
  const V *weight_vec = reinterpret_cast<const V*>(weight_data);
  const V *bias_vec = reinterpret_cast<const V*>(bias_data);
  V *y_vec = reinterpret_cast<V*>(y);
  T mean = mean_data[ng];
  T rstd = rstd_data[ng];
  V weight_tmp = weight_vec[c];
  V bias_tmp = bias_vec[c];
#pragma unroll LOOP_I
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * gridDim.y * blockDim.x;
    idx += i * gridDim.y * blockDim.x;
    idx += blockIdx.y * blockDim.x;
    idx += threadIdx.x;
    V tmp_X = x_vec[idx];

    if constexpr (vec_elems == 1)
      y_vec[idx] = {ACT((static_cast<T_ACC>(tmp_X.x) - mean) * rstd * weight_tmp.x + bias_tmp.x)};
    else if constexpr (vec_elems == 2) {
      T y_x, y_y;
      y_x = ACT((static_cast<T_ACC>(tmp_X.x) - mean) * rstd * weight_tmp.x + bias_tmp.x);
      y_y = ACT((static_cast<T_ACC>(tmp_X.y) - mean) * rstd * weight_tmp.y + bias_tmp.y);
      y_vec[idx] = {y_x, y_y};
    }
    else if constexpr (vec_elems == 4) {
      T y_x, y_y, y_z, y_w;
      y_x = ACT((static_cast<T_ACC>(tmp_X.x) - mean) * rstd * weight_tmp.x + bias_tmp.x);
      y_y = ACT((static_cast<T_ACC>(tmp_X.y) - mean) * rstd * weight_tmp.y + bias_tmp.y);
      y_z = ACT((static_cast<T_ACC>(tmp_X.z) - mean) * rstd * weight_tmp.z + bias_tmp.z);
      y_w = ACT((static_cast<T_ACC>(tmp_X.w) - mean) * rstd * weight_tmp.w + bias_tmp.w);
      y_vec[idx] = {y_x, y_y, y_z, y_w};
    }
    else if constexpr (vec_elems == 8) {
      T y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d;
      y_x = ACT((static_cast<T_ACC>(tmp_X.x) - mean) * rstd * weight_tmp.x + bias_tmp.x);
      y_y = ACT((static_cast<T_ACC>(tmp_X.y) - mean) * rstd * weight_tmp.y + bias_tmp.y);
      y_z = ACT((static_cast<T_ACC>(tmp_X.z) - mean) * rstd * weight_tmp.z + bias_tmp.z);
      y_w = ACT((static_cast<T_ACC>(tmp_X.w) - mean) * rstd * weight_tmp.w + bias_tmp.w);
      y_a = ACT((static_cast<T_ACC>(tmp_X.a) - mean) * rstd * weight_tmp.a + bias_tmp.a);
      y_b = ACT((static_cast<T_ACC>(tmp_X.b) - mean) * rstd * weight_tmp.b + bias_tmp.b);
      y_c = ACT((static_cast<T_ACC>(tmp_X.c) - mean) * rstd * weight_tmp.c + bias_tmp.c);
      y_d = ACT((static_cast<T_ACC>(tmp_X.d) - mean) * rstd * weight_tmp.d + bias_tmp.d);
      y_vec[idx] = {y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d};
    }
  }
}

// note: fn body inside header file as template fn don't work in non-header files
// note: when fn body is modified, must recompile all modules that call this fn so they can adopt the change (since headers aren't compiled)
template <typename T>
void scale_shift(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    at::Tensor& Y,
    at::Tensor& means,
    at::Tensor& rstds) {
  using T_ACC = at::acc_type<T, true>;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();
  T* Y_data = Y.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);

  int TPB = 512;
  int f = 1;
  if (C <= 512)
    TPB = C;
  else
    while (C % f != 0 || C / f > 512)
      ++f;
  TPB = C / f;

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

    //compute_scale_biases<<<N, C>>>( // note: max(D, T) threads per block
    //printf("starting compute_scale_biases N: %d H %d W %d C %d G %d TPB %d f %d\n", N, H, W, C, G, TPB, f);
    compute_scale_biases<<<dim3(N, f), TPB>>>( // note: max(D, T) threads per block
        mean_data, rstd_data,
        weight_data, bias_data,
        G, C,
        a_data, b_data);

    /*
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
      */
     
    //const int TPB = 512;
    const int LOOP_I = 8;
    //printf("starting elem kernel N: %d H %d W %d C %d G %d TPB %d f %d\n", N, H, W, C, G, TPB, f);
    if (C % 8 == 0)
      scale_shift_elem_kernelV<T, LOOP_I, 8><<<dim3(N * H * W * C / TPB / LOOP_I / f / 8, f), TPB>>>(
          X_data,
          a_data, b_data,
          N, C,
          Y_data
          );
    else if (C % 4 == 0)
      scale_shift_elem_kernelV<T, LOOP_I, 4><<<dim3(N * H * W * C / TPB / LOOP_I / f / 4, f), TPB>>>(X_data, a_data, b_data, N, C, Y_data);
    else if (C % 2 == 0)
      scale_shift_elem_kernelV<T, LOOP_I, 2><<<dim3(N * H * W * C / TPB / LOOP_I / f / 2, f), TPB>>>(X_data, a_data, b_data, N, C, Y_data);
    else
      scale_shift_elem_kernelV<T, LOOP_I, 1><<<dim3(N * H * W * C / TPB / LOOP_I / f / 1, f), TPB>>>(X_data, a_data, b_data, N, C, Y_data);
  }
  else { // if spatial resolution small, overhead of creating the extra kernel isn't worth it
    //printf("starting elem kernel N: %d H %d W %d C %d G %d\n", N, H, W, C, G);
    /*
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
      */
                                                                  
    const T* weight_data = weight.const_data_ptr<T>();
    const T* bias_data = bias.const_data_ptr<T>();

    //const int TPB = 128;
    const int LOOP_I = 4;
    const int D = C / G;
    if (D % 8 == 0)
      small_scale_shift_elem_kernelV<T, LOOP_I, 8><<<dim3(N * H * W * C / TPB / LOOP_I / 8 / f, f), TPB>>>(
          X_data,
          mean_data, rstd_data,
          weight_data, bias_data,
          N, C, G,
          Y_data
          );
    else if (D % 4 == 0)
      small_scale_shift_elem_kernelV<T, LOOP_I, 4><<<dim3(N * H * W * C / TPB / LOOP_I / f / 4, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    else if (D % 2 == 0)
      small_scale_shift_elem_kernelV<T, LOOP_I, 2><<<dim3(N * H * W * C / TPB / LOOP_I / f / 2, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    else
      small_scale_shift_elem_kernelV<T, LOOP_I, 1><<<dim3(N * H * W * C / TPB / LOOP_I / f / 1, f), TPB>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}


// copied from https://github.com/pytorch/pytorch/blob/b8307513e57f8beaf99daff342a23d705a417e11/aten/src/ATen/native/SharedReduceOps.h
template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

// copied from https://github.com/pytorch/pytorch/blob/b8307513e57f8beaf99daff342a23d705a417e11/aten/src/ATen/native/SharedReduceOps.h
template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOps {
  acc_scalar_t correction;
  bool take_sqrt;
  public:
    using acc_t = WelfordData<acc_scalar_t, index_t>;
    inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
      // We accumulate n in index_t to avoid cumulative rounding error, but still
      // need nf for use in combine where int32 may overflow.
      index_t new_n = acc.n + 1;
      acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);

      acc_scalar_t delta = data - acc.mean;
      
      acc_scalar_t new_mean = acc.mean + delta / new_nf;
      acc_scalar_t new_delta = data - new_mean;
      return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
      };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? std::sqrt(var) : var, mean);
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  C10_HOST_DEVICE WelfordOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

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

  //const int w = W / d;
  const int w = ceil((float)W / d);
#pragma unroll 8
  for (int i = 0; i < w; ++i) {
    if ((int)(i * d + threadIdx.y) >= W)
      continue;

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
  __syncthreads();

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

  //for (int stride = TPB / 2; stride >= gf; stride >>= 1) {
  //  if (tid < stride)
  //    vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
  //  __syncthreads();
  //  }

  if (tid < gf)
    for (int di = 1; di < d*D; ++di)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + di*gf]);
  __syncthreads();

  // put reduced outputs into return buffers
  if (tid < gf) {
    int out_idx = 0;
    out_idx += blockIdx.x * G * H; // dim 0, HG stride
    out_idx += blockIdx.z * gf * H; // dim 2, G/f stride
    out_idx += threadIdx.x * H; // dim 3, 1 stride
    out_idx += blockIdx.y; // dim 1, G stride
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
  WelfordType val(0, 0, 0, 0);
  const int TPB = blockDim.y * blockDim.x;

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = welford_data[blockIdx.x * G * H + blockIdx.y * H + threadIdx.x];
  __syncthreads();

  for (int stride = TPB / 2; stride >= 1; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if (tid < 1) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[tid]);
    int out_idx = 0;
    out_idx += blockIdx.x * G; // dim 0, G stride
    out_idx += blockIdx.y; // dim 1, G/f stride
    means[out_idx] = m1;
    rstds[out_idx] = rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

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

  using WelfordType = WelfordData<at::acc_type<T, true>, int>;
  at::Tensor welford_tensor = at::empty({N, G, H, sizeof(WelfordType)}, X.options().dtype(at::kByte));
  WelfordType *welford_data = reinterpret_cast<WelfordType *>(welford_tensor.mutable_data_ptr());
  
  int blockDimX, blockDimY, f, TPB;
  TPB = MIN(MAX_THREADS_PER_BLOCK, W * C);
  if (C < MAX_THREADS_PER_BLOCK)
    TPB -= TPB % C;
  else {
    int d = 1;
    while (C % d != 0 || C / d > MAX_THREADS_PER_BLOCK)
      d++;
    TPB = C / d;
  }


  blockDimX = MIN(TPB, C);
  blockDimY = TPB / blockDimX;
  f = MAX(C / TPB, 1); // note: impossible for f > 1 AND blockDimY > 1
  //printf("starting compute_stats_pt1 N: %d H %d W %d C %d G %d TPB %d bdx %d bdy %d f %d\n", N, H, W, C, G, TPB, blockDimX, blockDimY, f);
  NH_compute_stats_pt1<<<dim3(N, H, f), dim3(blockDimX, blockDimY)>>>(
      X_data, H, W, C, G, 
      welford_data
  );

  //printf("starting compute_stats_pt2 N: %d H %d W %d C %d G %d\n", N, H, W, C, G);
  //TPB = MIN(MAX_THREADS_PER_BLOCK, H * G / f);
  //blockDimX = MIN(TPB, G / f);
  //blockDimY = TPB / blockDimX;
  NH_compute_stats_pt2<<<dim3(N, G), H>>>(
          welford_data,
          H, G, eps,
          mean_data, rstd_data
    );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
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
