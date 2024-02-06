#include <torch/extension.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h>

void GroupNormKernelImpl(
    const at::Tensor& X,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    at::Tensor& Y,
    at::Tensor& mean,
    at::Tensor& rstd);

void GroupNormBackwardKernelImpl(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    at::Tensor& dX,
    at::Tensor& dgamma,
    at::Tensor& dbeta);

std::vector<at::Tensor> gn_nhwc_cuda_fwd_NH_grid(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    float eps);

/*
std::vector<at::Tensor> gn_nhwc_cuda_fwd_N_grid(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    float eps);

std::vector<at::Tensor> gn_nhwc_cuda_fwd_NG_grid(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    float eps);
*/

std::vector<at::Tensor> gn_nhwc_cuda_fwd_fused(
    const at::Tensor& X,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int G,
    float eps);

std::vector<at::Tensor> gn_nhwc_cuda_bwd(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    const int G);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

std::vector<at::Tensor> gn_nhwc_fwd_NH_grid(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_fwd_NH_grid(X, weight, bias, G, eps);
}

/*
std::vector<at::Tensor> gn_nhwc_fwd_N_grid(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_fwd_N_grid(X, weight, bias, G, eps);
}

std::vector<at::Tensor> gn_nhwc_fwd_NG_grid(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_fwd_NG_grid(X, weight, bias, G, eps);
}
*/

std::vector<at::Tensor> gn_nhwc_fwd_fused(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_fwd_fused(X, weight, bias, G, eps);
}

std::vector<at::Tensor> gn_nhwc_bwd(
    const at::Tensor dy,
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor means,
    const at::Tensor rstds,
    const int G) {
  CHECK_CUDA(dy);
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(means);
  CHECK_CUDA(rstds);
  at::Tensor dX = at::empty_like(X);
  at::Tensor dgamma = at::empty_like(weight);
  at::Tensor dbeta = at::empty_like(weight);
  return gn_nhwc_cuda_bwd(dy, X, means, rstds, weight, G);
}

/*
std::vector<at::Tensor> gn_nchw_forward(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);
  at::Tensor Y = at::empty_like(X);
  at::Tensor mean = at::empty({N, G}, X.options());
  at::Tensor rstd = at::empty({N, G}, X.options());
  GroupNormKernelImpl(
    X, weight, bias,
    N, C, H * W,
    G,
    eps,
    Y,
    mean,
    rstd);
  return {Y, mean, rstd};
}

std::vector<at::Tensor> gn_nchw_backward(
    const at::Tensor dy,
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor means,
    const at::Tensor rstds,
    const int G) {
  CHECK_CUDA(dy);
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(means);
  CHECK_CUDA(rstds);
  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);
  at::Tensor dX = at::empty_like(X);
  at::Tensor dgamma = at::empty_like(weight);
  at::Tensor dbeta = at::empty_like(weight);
  GroupNormBackwardKernelImpl(
      dy, X,
      means, rstds, weight,
      N, C, H * W, G,
      dX, dgamma, dbeta);

  return {dX, dgamma, dbeta};
}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //m.def("fwd_N_grid", &gn_nhwc_fwd_N_grid, "GN NHWC forward (N grid)");
  m.def("fwd_NH_grid", &gn_nhwc_fwd_NH_grid, "GN NHWC forward (NH grid)");
  //m.def("fwd_NG_grid", &gn_nhwc_fwd_NG_grid, "GN NHWC forward (NG grid)");
  m.def("fwd_fused", &gn_nhwc_fwd_fused, "GN NHWC forward_fused");
  m.def("bwd", &gn_nhwc_bwd, "GN NHWC backward");
  //m.def("nchwforward", &gn_nchw_forward, "GN NCHW forward");
  //m.def("nchwbackward", &gn_nchw_backward, "GN NCHW backward");
}
