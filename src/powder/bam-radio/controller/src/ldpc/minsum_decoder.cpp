/*
 * minsum_decoder.cpp
 * Implementation of the Sum-product decoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "yaldpc.hpp"
#include <cmath>
#include <limits>
#include <utility>

// message storage
#define __E(j, i) this->E[j * this->_n + i]
#define __M(j, i) this->M[j * this->_n + i]

using std::size_t;
namespace yaldpc {

// constructor
template <typename LLR, typename Out>
MinSumDecoder<LLR, Out>::MinSumDecoder(LDPCCode const &code, uint64_t max_its,
                                       bool output_value, bool do_parity_check,
                                       LLR alpha)
    : Decoder<LLR, Out>(), detail::FloodingDecoder<LLR, Out>(
                               code.H, code.m, code.n, max_its, output_value,
                               do_parity_check),
      _alpha(alpha) {}

template <typename LLR, typename Out>
typename yaldpc::MinSumDecoder<LLR, Out>::sptr
MinSumDecoder<LLR, Out>::make(LDPCCode const &code, uint64_t max_its,
                              bool output_value, bool do_parity_check,
                              LLR alpha) {
  return std::make_shared<MinSumDecoder<LLR, Out>>(code, max_its, output_value,
                                                   do_parity_check, alpha);
}

// compute check messages
template <typename LLR, typename Out>
void MinSumDecoder<LLR, Out>::check_messages() {
  for (size_t j = 0; j < this->_m; ++j) {
    LLR f = detail::Inf<LLR>(); // minimum
    size_t f_idx = this->_n;    // index of minimum
    LLR s = detail::Inf<LLR>(); // second minimum
    bool si = false;            // keep track of sign
    size_t k = 0;               // helper iterator
    for (auto const &i : this->B[j]) {
      auto MM = __M(j, i);
      auto v = detail::Abs(MM);
      this->_si2[k] = detail::Sign(MM);
      si = (si != this->_si2[k]);
      ++k;
      if (v <= f) {
        s = f;
        f = v;
        f_idx = i;
      } else if (v <= s) {
        s = v;
      }
    }
    assert(f_idx != this->_n);
    assert(f != detail::Inf<LLR>());
    assert(s != detail::Inf<LLR>());

    k = 0;
    for (auto const &i : this->B[j]) {
      auto m = f_idx == i ? s : f;
      // subtract the outgoing message, flip the sign back if needed
      __E(j, i) = si != this->_si2[k] ? _minusalpha * m : _alpha * m;
      ++k;
    }
  }
}

// decode using flooding algorithm
template <typename LLR, typename Out>
void MinSumDecoder<LLR, Out>::decode(LLR const *in, Out *out) {
  this->flood_decode(in, out);
}

template <typename LLR, typename Out>
std::vector<Out> MinSumDecoder<LLR, Out>::decode(std::vector<LLR> const &in) {
  assert(in.size() == this->_n);
  std::vector<Out> o(this->_output_value ? this->_k : this->_n);
  this->flood_decode(in.data(), o.data());
  return o;
}

// get decoder parameters
template <typename LLR, typename Out>
uint64_t MinSumDecoder<LLR, Out>::max_its() {
  return this->_max_its;
}
template <typename LLR, typename Out>
bool MinSumDecoder<LLR, Out>::output_value() {
  return this->_output_value;
}
template <typename LLR, typename Out>
bool MinSumDecoder<LLR, Out>::do_parity_check() {
  return this->_do_parity_check;
}
template <typename LLR, typename Out> std::size_t MinSumDecoder<LLR, Out>::m() {
  return this->_m;
}
template <typename LLR, typename Out> std::size_t MinSumDecoder<LLR, Out>::n() {
  return this->_n;
}
template <typename LLR, typename Out> std::size_t MinSumDecoder<LLR, Out>::k() {
  return this->_k;
}
// template instantiation
template class MinSumDecoder<double, uint8_t>;
template class MinSumDecoder<double, double>;
template class MinSumDecoder<float, uint8_t>;
template class MinSumDecoder<float, float>;
template class MinSumDecoder<fixed, uint8_t>;
template class MinSumDecoder<fixed, fixed>;
template class MinSumDecoder<fixed64, uint8_t>;
template class MinSumDecoder<fixed64, fixed64>;
} /* namespace  yaldpc */

#undef __M
#undef __E
