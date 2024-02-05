#pragma once 
#ifndef SCALE_SHIFT_KERNEL
#define SCALE_SHIFT_KERNEL
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

#define ACT 

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
     
    //const int TPB = 512;
    /*
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
      */
  }
  else { // if spatial resolution small, overhead of creating the extra kernel isn't worth it
    //printf("starting elem kernel N: %d H %d W %d C %d G %d\n", N, H, W, C, G);
    /*const int D = C / G;
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
          });*/
                                                                  
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

#endif
