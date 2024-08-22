#pragma once
#ifndef FWD_GN_KERNEL_H
#define FWD_GN_KERNEL_H

#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

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
    const int64_t act_fn_option,
    T *Y_data,
    T *mean_data,
    T *rstd_data);

template <typename T>
void run_gn_bwd_kernels(
      const T *dy_data,
      const T *X_data,
      const T *weight_data,
      const T *bias_data,
      const T *mean_data,
      const T *rstd_data,
      const int N,
      const int H,
      const int W,
      const int C,
      const int G,
      const int64_t act_fn_option,
      T *dx_data,
      T *dweight_data,
      T *dbias_data);

//void GroupNormKernelImpl(
//    const at::Tensor& X,
//    const at::Tensor& gamma,
//    const at::Tensor& beta,
//    int64_t N,
//    int64_t C,
//    int64_t HxW,
//    int64_t group,
//    double eps,
//    at::Tensor& Y,
//    at::Tensor& mean,
//    at::Tensor& rstd);
//
//void GroupNormBackwardKernelImpl(
//    const at::Tensor& dY,
//    const at::Tensor& X,
//    const at::Tensor& mean,
//    const at::Tensor& rstd,
//    const at::Tensor& gamma,
//    int64_t N,
//    int64_t C,
//    int64_t HxW,
//    int64_t group,
//    at::Tensor& dX,
//    at::Tensor& dgamma,
//    at::Tensor& dbeta);

#endif
