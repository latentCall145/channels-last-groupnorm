#pragma once 
#include <c10/util/BFloat16.h>
#include <type_traits>
#ifndef SCALE_SHIFT_KERNEL
#define SCALE_SHIFT_KERNEL
#include <ATen/native/cuda/Loops.cuh> // gpu kernel
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h> // torch tensor
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#define SCALE_SHIFT_KERNEL_MAX_TPB 512
#define LOOP_I 8

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
  const int D = C / G;
  const int c = threadIdx.x;
  const int g = c / D;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  const at::acc_type<T, true> a_nc = rstds[ng] * weight[c];
  a[nc] = a_nc;
  b[nc] = -means[ng] * a_nc + bias[c];
}

template <typename T>
T inline relu(T x) {
  return x > 0 ? x : 0;
}

template <typename T>
T inline silu(T x) {
  using opmath_t = at::opmath_type<T>;
  return x / (opmath_t(1) + exp(-x));
}

template <typename T>
T inline gelu(T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kAlpha = M_SQRT1_2;
  return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
}

template <typename T>
T inline gelu_tanh(T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  return opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + c10::cuda::compat::tanh(inner));
}

template <typename T>
struct alignas(16) f4 {
  T x;
  T y;
  T z;
  T w;
};

template <typename T>
struct alignas(8) f4_half {
  T x;
  T y;
  T z;
  T w;
};

template <typename T>
struct alignas(32) f8 {
  T x;
  T y;
  T z;
  T w;
  T a;
  T b;
  T c;
  T d;
};

template <typename T>
struct alignas(16) f8_half {
  T x;
  T y;
  T z;
  T w;
  T a;
  T b;
  T c;
  T d;
};

template <typename T>
__global__ void
scale_shift_elem_kernel1(
    const T* x,
    at::acc_type<T, true>* a,
    at::acc_type<T, true>* b,
    const int N,
    const int C,
    T* y
    ) {
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % C;
  const int nc = n * C + c;
  const at::acc_type<T, true> a_elem = a[nc]; 
  const at::acc_type<T, true> b_elem = b[nc]; 
#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    y[idx] = a_elem * x[idx] + b_elem;
  }
}

template <typename T, typename T_vec_dtype>
__global__ void
scale_shift_elem_kernel1V4(
    const T* x,
    at::acc_type<T, true>* a,
    at::acc_type<T, true>* b,
    const int N,
    const int C,
    T* y
    ) {
  using T_ACC = at::acc_type<T, true>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / 4);
  const int nc = n * (C/4) + c;
  const T_vec_dtype *x_vec = reinterpret_cast<const T_vec_dtype*>(x);
  T_vec_dtype *y_vec = reinterpret_cast<T_vec_dtype*>(y);
  f4<T_ACC> *a_vec = reinterpret_cast<f4<T_ACC>*>(a);
  f4<T_ACC> *b_vec = reinterpret_cast<f4<T_ACC>*>(b);
#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    T_vec_dtype tmp_x = x_vec[idx];
    f4 tmp_a = a_vec[nc];
    f4 tmp_b = b_vec[nc];
    //y[4 * idx + 0] = tmp_a.x * tmp_x.x + tmp_b.x;
    //y[4 * idx + 1] = tmp_a.y * tmp_x.y + tmp_b.y;
    //y[4 * idx + 2] = tmp_a.z * tmp_x.z + tmp_b.z;
    //y[4 * idx + 3] = tmp_a.w * tmp_x.w + tmp_b.w;
    T y_x, y_y, y_z, y_w;
    y_x = tmp_a.x * tmp_x.x + tmp_b.x;
    y_y = tmp_a.y * tmp_x.y + tmp_b.y;
    y_z = tmp_a.z * tmp_x.z + tmp_b.z;
    y_w = tmp_a.w * tmp_x.w + tmp_b.w;
    y_vec[idx] = {y_x, y_y, y_z, y_w};
  }
}

template <typename T, typename T_vec_dtype>
__global__ void
scale_shift_elem_kernel1V8(
    const T* x,
    at::acc_type<T, true>* a,
    at::acc_type<T, true>* b,
    const int N,
    const int C,
    T* y
    ) {
  using T_ACC = at::acc_type<T, true>;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = threadIdx.x % (C / 8);
  const int nc = n * (C/8) + c;
  const T_vec_dtype *x_vec = reinterpret_cast<const T_vec_dtype*>(x);
  T_vec_dtype *y_vec = reinterpret_cast<T_vec_dtype*>(y);
  f8<T_ACC> *a_vec = reinterpret_cast<f8<T_ACC>*>(a);
  f8<T_ACC> *b_vec = reinterpret_cast<f8<T_ACC>*>(b);
#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * blockDim.x;
    idx += i * blockDim.x;
    idx += threadIdx.x;
    T_vec_dtype tmp_x = x_vec[idx];
    f8 tmp_a = a_vec[nc];
    f8 tmp_b = b_vec[nc];
    T y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d;
    y_x = tmp_a.x * tmp_x.x + tmp_b.x;
    y_y = tmp_a.y * tmp_x.y + tmp_b.y;
    y_z = tmp_a.z * tmp_x.z + tmp_b.z;
    y_w = tmp_a.w * tmp_x.w + tmp_b.w;
    y_a = tmp_a.a * tmp_x.a + tmp_b.a;
    y_b = tmp_a.b * tmp_x.b + tmp_b.b;
    y_c = tmp_a.c * tmp_x.c + tmp_b.c;
    y_d = tmp_a.d * tmp_x.d + tmp_b.d;
    y_vec[idx] = {y_x, y_y, y_z, y_w, y_a, y_b, y_c, y_d};
    //y[8 * idx + 0] = tmp_a.x * tmp_x.x + tmp_b.x;
    //y[8 * idx + 1] = tmp_a.y * tmp_x.y + tmp_b.y;
    //y[8 * idx + 2] = tmp_a.z * tmp_x.z + tmp_b.z;
    //y[8 * idx + 3] = tmp_a.w * tmp_x.w + tmp_b.w;
    //y[8 * idx + 4] = tmp_a.a * tmp_x.a + tmp_b.a;
    //y[8 * idx + 5] = tmp_a.b * tmp_x.b + tmp_b.b;
    //y[8 * idx + 6] = tmp_a.c * tmp_x.c + tmp_b.c;
    //y[8 * idx + 7] = tmp_a.d * tmp_x.d + tmp_b.d;
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

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  const int D = C / G;

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

    compute_scale_biases<<<N, C>>>( // note: max(D, T) threads per block
        mean_data, rstd_data,
        weight_data, bias_data,
        G, C,
        a_data, b_data);

    /*at::TensorIterator iter = at::TensorIteratorConfig()
      .check_all_same_dtype(std::is_same<T, T_ACC>::value) // this line relaxes requirement that all inputs/outputs are same dtype if T isn't T_ACC 
      .resize_outputs(false)
      .add_owned_output(Y.view({N, H * W, C}))
      .add_owned_input(X.view({N, H * W, C}))
      .add_owned_input(a.view({N, 1, C}))
      .add_owned_input(b.view({N, 1, C}))
      .build();

    at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
        return a * x + b;
        });
        */

    T* Y_data = Y.mutable_data_ptr<T>();
    //scale_shift_elem_kernel1<<<N * H * W * C / SCALE_SHIFT_KERNEL_MAX_TPB / LOOP_I, SCALE_SHIFT_KERNEL_MAX_TPB>>>(
    //    X_data,
    //    a_data, b_data,
    //    N, C,
    //    Y_data
    //    );
    if (sizeof(T) == 4)
      scale_shift_elem_kernel1V4<T, f4<T>><<<N * H * W * C / SCALE_SHIFT_KERNEL_MAX_TPB / LOOP_I / 4, SCALE_SHIFT_KERNEL_MAX_TPB>>>(
          X_data,
          a_data, b_data,
          N, C,
          Y_data
          );
      //scale_shift_elem_kernel1V8<T, f8<T>><<<N * H * W * C / SCALE_SHIFT_KERNEL_MAX_TPB / LOOP_I / 8, SCALE_SHIFT_KERNEL_MAX_TPB>>>(
      //    X_data,
      //    a_data, b_data,
      //    N, C,
      //    Y_data
      //    );
    else
      scale_shift_elem_kernel1V4<T,f4_half<T>><<<N * H * W * C / SCALE_SHIFT_KERNEL_MAX_TPB / LOOP_I / 4, SCALE_SHIFT_KERNEL_MAX_TPB>>>(
          X_data,
          a_data, b_data,
          N, C,
          Y_data
          );
      //scale_shift_elem_kernel1V8<T,f8_half<T>><<<N * H * W * C / SCALE_SHIFT_KERNEL_MAX_TPB / LOOP_I / 8, SCALE_SHIFT_KERNEL_MAX_TPB>>>(
      //    X_data,
      //    a_data, b_data,
      //    N, C,
      //    Y_data
      //    );
  }
  else { // if spatial resolution small, overhead of creating the extra kernel isn't worth it
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

#endif
