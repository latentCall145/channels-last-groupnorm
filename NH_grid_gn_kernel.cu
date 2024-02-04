//#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty_like.h>
#include <ATen/Dispatch.h> // at_dispatch macro
#include <c10/core/ScalarType.h>
#include "scale_shift_kernel.h" // scale_shift
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

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
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x)); // last arg isn't used in src
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

  //using WelfordType = at::native::WelfordData<at::acc_type<T, true>, int>;
  using WelfordType = WelfordData<at::acc_type<T, true>, int>;
  at::Tensor welford_tensor = at::empty({N, G, H, sizeof(WelfordType)}, X.options().dtype(at::kByte));
  WelfordType *welford_data = reinterpret_cast<WelfordType *>(welford_tensor.mutable_data_ptr());
  
  int blockDimX, blockDimY, f, TPB;
  TPB = MIN(MAX_THREADS_PER_BLOCK, W * C);

  blockDimX = MIN(TPB, C);
  blockDimY = TPB / blockDimX;
  f = MAX(C / TPB, 1); // note: impossible for f > 1 AND blockDimY > 1
  NH_compute_stats_pt1<<<dim3(N, H, f), dim3(blockDimX, blockDimY)>>>(
      X_data, H, W, C, G, 
      welford_data
  );

  TPB = MIN(MAX_THREADS_PER_BLOCK, H * G / f);
  blockDimX = MIN(TPB, G / f);
  blockDimY = TPB / blockDimX;
  NH_compute_stats_pt2<<<dim3(N, G), H>>>(
          welford_data,
          H, G, eps,
          mean_data, rstd_data
    );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
  //AT_CUDA_CHECK(cudaGetLastError());
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
