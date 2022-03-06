/*
 * minstar_decoder.cpp
 * Implementation of the Min* decoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "minstar_lut.hpp"
#include "yaldpc.hpp"

#include <cmath>
#include <cstdio>
#include <utility>

// messages
#define __E(j, i) this->E[j * this->_n + i]
#define __M(j, i) this->M[j * this->_n + i]

namespace yaldpc {

// constructor
template <typename LLR, typename Out>
MinStarDecoder<LLR, Out>::MinStarDecoder(LDPCCode const &code, uint64_t max_its,
                                         bool output_value,
                                         bool do_parity_check)
    : Decoder<LLR, Out>(), detail::FloodingDecoder<LLR, Out>(
                               code.H, code.m, code.n, max_its, output_value,
                               do_parity_check) {}

template <typename LLR, typename Out>
typename yaldpc::MinStarDecoder<LLR, Out>::sptr
MinStarDecoder<LLR, Out>::make(LDPCCode const &code, uint64_t max_its,
                               bool output_value, bool do_parity_check) {
  return std::make_shared<MinStarDecoder<LLR, Out>>(code, max_its, output_value,
                                                    do_parity_check);
}

// helper function for below
template <typename LLR> inline LLR plus(LLR x, LLR y) {
  // if true -> negative
  LLR sign = detail::Sign(x) != detail::Sign(y) ? -1.0 : 1.0;
  LLR a1 = detail::Abs(x);
  LLR a2 = detail::Abs(y);
  LLR t1 = a1 < a2 ? a1 : a2;
  //
  // note: log(1 + exp(-abs(x))) can be approximated nicely. check [1] for
  // details.
  // LLR t2 = std::log(1+std::exp(-(a1+a2)));
  // LLR t3 = std::log(1+std::exp(-std::abs(a1-a2)));
  //
  // use a lookup table to compute the function. the table and the lookup
  // function are precomputed in matlab...
  LLR t2 = lut_lookup(a1 + a2);
  LLR t3 = lut_lookup(a1 - a2);
  return sign * (t1 + t2 - t3);
}

// compute the check messages
template <typename LLR, typename Out>
void MinStarDecoder<LLR, Out>::check_messages() {
  for (size_t j = 0; j < this->_m; ++j) {
    size_t dc = this->B[j].size();
    if (dc == 2) {
      __E(j, this->B[j][0]) = __M(j, this->B[j][1]);
      __E(j, this->B[j][1]) = __M(j, this->B[j][0]);
    } else {
      std::vector<LLR> y(dc - 1);
      y.front() = __M(j, this->B[j][0]);
      for (size_t i = 1; i < dc - 1; ++i)
        y[i] = plus(y[i - 1], __M(j, this->B[j][i]));
      std::vector<LLR> yp(dc);
      yp.back() = __M(j, this->B[j].back());
      yp.front() = 0.0;
      for (size_t i = dc - 2; i > 0; --i)
        yp[i] = plus(yp[i + 1], __M(j, this->B[j][i]));
      __E(j, this->B[j][0]) = yp[1];
      __E(j, this->B[j].back()) = y.back();
      for (size_t i = 1; i < dc - 1; ++i)
        __E(j, this->B[j][i]) = plus(y[i - 1], yp[i + 1]);
    }
  }
}

// housekeeping

// decode using flooding algorithm
template <typename LLR, typename Out>
void MinStarDecoder<LLR, Out>::decode(LLR const *in, Out *out) {
  this->flood_decode(in, out);
}

template <typename LLR, typename Out>
std::vector<Out> MinStarDecoder<LLR, Out>::decode(std::vector<LLR> const &in) {
  assert(in.size() == this->_n);
  std::vector<Out> o(this->_output_value ? this->_k : this->_n);
  this->flood_decode(in.data(), o.data());
  return o;
}

// get decoder parameters
template <typename LLR, typename Out>
uint64_t MinStarDecoder<LLR, Out>::max_its() {
  return this->_max_its;
}
template <typename LLR, typename Out>
bool MinStarDecoder<LLR, Out>::output_value() {
  return this->_output_value;
}
template <typename LLR, typename Out>
bool MinStarDecoder<LLR, Out>::do_parity_check() {
  return this->_do_parity_check;
}
template <typename LLR, typename Out>
std::size_t MinStarDecoder<LLR, Out>::m() {
  return this->_m;
}
template <typename LLR, typename Out>
std::size_t MinStarDecoder<LLR, Out>::n() {
  return this->_n;
}
template <typename LLR, typename Out>
std::size_t MinStarDecoder<LLR, Out>::k() {
  return this->_k;
}
// template instantiation
template class MinStarDecoder<float, uint8_t>;
template class MinStarDecoder<float, float>;
template class MinStarDecoder<double, uint8_t>;
template class MinStarDecoder<double, double>;
// N.B: Can't do fixed-point Min* rn because we don't have a fixed-point
// log function.
} /* namespace  yaldpc */

#undef __M
#undef __E
