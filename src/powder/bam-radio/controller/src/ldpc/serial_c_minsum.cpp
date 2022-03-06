/*
 * serial_c_minsum.cpp
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

using std::size_t;
namespace yaldpc {

// constructors
template <typename LLR, typename Out>
SerialCMinSumDecoder<LLR, Out>::SerialCMinSumDecoder(LDPCCode const &code,
                                                     uint64_t max_its,
                                                     bool output_value,
                                                     bool do_parity_check,
                                                     float alpha)
    : Decoder<LLR, Out>(), detail::SerialCDecoder<LLR, Out>(
                               code.H, code.m, code.n, max_its, output_value,
                               do_parity_check),
      _alpha(alpha), _minusalpha(-alpha) {}

template <typename LLR, typename Out>
typename yaldpc::SerialCMinSumDecoder<LLR, Out>::sptr
SerialCMinSumDecoder<LLR, Out>::make(LDPCCode const &code, uint64_t max_its,
                                     bool output_value, bool do_parity_check,
                                     float alpha) {
  return std::make_shared<SerialCMinSumDecoder<LLR, Out>>(
      code, max_its, output_value, do_parity_check, alpha);
}

// message passing algorithm
template <typename LLR, typename Out>
void SerialCMinSumDecoder<LLR, Out>::_message_passing() {
  // global pass
  for (size_t j = 0; j < this->_m; ++j) {
    auto f = detail::Inf<LLR>();
    size_t f_idx = this->_n;
    auto s = detail::Inf<LLR>();
    bool si = false;
    for (size_t k = 0; k < this->_B[j].size(); ++k) {
      auto const &i = this->_B[j][k];
      auto MM = this->_P[i] - this->_R[j][k];
      auto v = detail::Abs(MM);
      this->_si2[k] = detail::Sign(MM);
      si = (si != this->_si2[k]);
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
    // local pass
    for (size_t k = 0; k < this->_B[j].size(); ++k) {
      auto const &i = this->_B[j][k];
      auto m = f_idx == i ? s : f;
      auto msg = si != this->_si2[k] ? _minusalpha * m : _alpha * m;
      this->_R[j][k] = msg;
      this->_Q[i] += msg;
    }
  }
}

// decode
template <typename LLR, typename Out>
void SerialCMinSumDecoder<LLR, Out>::decode(LLR const *in, Out *out) {
  this->_serial_c_decode(in, out);
}

template <typename LLR, typename Out>
std::vector<Out>
SerialCMinSumDecoder<LLR, Out>::decode(std::vector<LLR> const &in) {
  assert(in.size() == this->_n);
  std::vector<Out> o(this->_output_value ? this->_k : this->_n);
  this->_serial_c_decode(in.data(), o.data());
  return o;
}

// decoder params
template <typename LLR, typename Out>
uint64_t SerialCMinSumDecoder<LLR, Out>::max_its() {
  return this->_max_its;
}
template <typename LLR, typename Out>
bool SerialCMinSumDecoder<LLR, Out>::output_value() {
  return this->_output_value;
}
template <typename LLR, typename Out>
bool SerialCMinSumDecoder<LLR, Out>::do_parity_check() {
  return this->_do_parity_check;
}
template <typename LLR, typename Out>
std::size_t SerialCMinSumDecoder<LLR, Out>::m() {
  return this->_m;
}
template <typename LLR, typename Out>
std::size_t SerialCMinSumDecoder<LLR, Out>::n() {
  return this->_n;
}
template <typename LLR, typename Out>
std::size_t SerialCMinSumDecoder<LLR, Out>::k() {
  return this->_k;
}
// template housekeeping
template class SerialCMinSumDecoder<float, uint8_t>;
template class SerialCMinSumDecoder<float, float>;
template class SerialCMinSumDecoder<double, uint8_t>;
template class SerialCMinSumDecoder<double, double>;
template class SerialCMinSumDecoder<fixed, uint8_t>;
template class SerialCMinSumDecoder<fixed, fixed>;
template class SerialCMinSumDecoder<fixed64, uint8_t>;
template class SerialCMinSumDecoder<fixed64, fixed64>;
} /* namespace  yaldpc */
