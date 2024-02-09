#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h>
#include <torch/extension.h>
#include "gn_kernel.h"

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

std::vector<at::Tensor> gn_nhwc_cuda_bwd(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    const int G);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

std::vector<at::Tensor> gn_nhwc_fwd(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    double eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);

  at::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  at::Tensor X_out = at::empty_like(X_nhwc);
  at::Tensor means = at::empty({N, G}, weight.options());
  at::Tensor rstds = at::empty({N, G}, weight.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_forward", [&]() {
    run_gn_fwd_kernels<scalar_t>(
        X_nhwc.const_data_ptr<scalar_t>(),
        weight.const_data_ptr<scalar_t>(), bias.const_data_ptr<scalar_t>(),
        N, H, W, C, G, static_cast<scalar_t>(eps),
        X_out.mutable_data_ptr<scalar_t>(), means.mutable_data_ptr<scalar_t>(), rstds.mutable_data_ptr<scalar_t>()
    );
  });
  return {X_out.permute({0, 3, 1, 2}), means, rstds};
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

  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);
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
      dy_nhwc.const_data_ptr<scalar_t>(), X_nhwc.const_data_ptr<scalar_t>(),
      weight.const_data_ptr<scalar_t>(), means.const_data_ptr<scalar_t>(), rstds.const_data_ptr<scalar_t>(),
      N, H, W, C, G,
      dX.mutable_data_ptr<scalar_t>(), dweight.mutable_data_ptr<scalar_t>(), dbias.mutable_data_ptr<scalar_t>()
      );
  });
  return {dX.permute({0, 3, 1, 2}), dweight, dbias};
}

std::vector<at::Tensor> gn_nchw_forward(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int G,
    double eps) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &gn_nhwc_fwd, "GN NHWC forward (NH grid)");
  m.def("bwd", &gn_nhwc_bwd, "GN NHWC backward");
  m.def("nchwforward", &gn_nchw_forward, "GN NCHW forward");
  m.def("nchwbackward", &gn_nchw_backward, "GN NCHW backward");
}
