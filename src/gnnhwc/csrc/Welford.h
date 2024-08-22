#pragma once
#ifndef WELFORD_H 
#define WELFORD_H 

#include <c10/cuda/CUDAMathCompat.h> // for C10_HOST_DEVICE and __ubsan_ignore_float_divide_by_zero__

// copied from https://github.com/pytorch/pytorch/blob/b8307513e57f8beaf99daff342a23d705a417e11/aten/src/ATen/native/SharedReduceOps.h
template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

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

  C10_HOST_DEVICE WelfordOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

#endif
