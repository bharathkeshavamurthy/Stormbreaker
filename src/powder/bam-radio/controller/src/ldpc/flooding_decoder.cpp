/*
 * flooding_decoder.cpp
 * Implementation of the flooding decoder base class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "detail.hpp"

#include <cstdio>
#include <vector>

// messages
#define __E(j, i) this->E[j * this->_n + i]
#define __M(j, i) this->M[j * this->_n + i]

namespace yaldpc {
namespace detail {

// construct a Decoder objec
template <typename LLR, typename Out>
FloodingDecoder<LLR, Out>::FloodingDecoder(GF2Matrix H, unsigned int m,
                                           unsigned int n, uint64_t max_its,
                                           bool output_value,
                                           bool do_parity_check)
    : _max_its(max_its), _output_value(output_value),
      _do_parity_check(do_parity_check), H(H), _m(m), _n(n), _k(n - m),
      A(n, std::vector<size_t>()), B(m, std::vector<size_t>()), E(m * n, 0.0),
      M(m * n, 0.0), _L(n), _z(n) {

  // search the PCM to get the connections between bit and check nodes
  for (size_t j = 0; j < m; ++j) {
    auto row = H.slice(j, j + 1, 0, n);
    for (size_t i = 0; i < n; ++i) {
      if (__ONE__ == row.data[i]) {
        B[j].push_back(i);
        A[i].push_back(j);
      }
    }
  }
  // find the highest degree of a check node to resize the signbit storage
  int sz = 0;
  for (auto const &b : B)
    if (b.size() >= sz)
      sz = b.size();
  _si2.resize(sz + 1);
}

// compute the bit messages by summing and then 'backing out'
template <typename LLR, typename Out>
void FloodingDecoder<LLR, Out>::bit_messages(LLR const *in_llr) {
  for (size_t i = 0; i < _n; ++i) {
    LLR su = 0.0;
    for (auto j : A[i])
      su += __E(j, i);
    for (auto j : A[i])
      __M(j, i) = su - __E(j, i) + in_llr[i];
  }
}

// decode a block of data
template <typename LLR, typename Out>
void FloodingDecoder<LLR, Out>::flood_decode(LLR const *in_llr, Out *out) {
  // initialization
  for (size_t i = 0; i < _n; ++i)
    for (auto j : A[i])
      __M(j, i) = in_llr[i];

  // iteration
  size_t II = 0;
  while (true) {
    // check messages (check nodes -> bit nodes)
    check_messages();

    // test and exit prematurely
    if (_do_parity_check || II == _max_its) {
      std::fill(begin(_L), end(_L), detail::Zero<LLR>());
      // compute the hard and soft decisions
      for (size_t i = 0; i < _n; ++i) {
        for (auto j : A[i])
          _L[i] += __E(j, i);
        _L[i] += in_llr[i];
        _z[i] = detail::HardDecision(_L[i]);
      }
      bool check = false;
      if (_do_parity_check) {
        // compute the parity equations
        auto syndrome = H.mult_vec(_z);
        check = (end(syndrome) ==
                 std::find_if(begin(syndrome), end(syndrome),
                              [](auto s) { return s != detail::__ZERO__; }));
      }
      if ((II == _max_its) || check) {
        // have decoded a valid codeword
        _write_results(out);
        break;
      }
    }

    // bit messages (bit nodes -> check nodes)
    bit_messages(in_llr);

    // continue
    ++II;
  }
}

// write results to the output buffers
template <>
void FloodingDecoder<double, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void FloodingDecoder<double, double>::_write_results(double *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _L[i];
}
template <> void FloodingDecoder<float, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void FloodingDecoder<float, float>::_write_results(float *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _L[i];
}
template <> void FloodingDecoder<fixed, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void FloodingDecoder<fixed, fixed>::_write_results(fixed *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _L[i];
}
template <>
void FloodingDecoder<fixed64, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <>
void FloodingDecoder<fixed64, fixed64>::_write_results(fixed64 *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _L[i];
}
// instantiate templates
template class FloodingDecoder<double, uint8_t>;
template class FloodingDecoder<double, double>;
template class FloodingDecoder<float, uint8_t>;
template class FloodingDecoder<float, float>;
template class FloodingDecoder<fixed, uint8_t>;
template class FloodingDecoder<fixed, fixed>;
template class FloodingDecoder<fixed64, uint8_t>;
template class FloodingDecoder<fixed64, fixed64>;
} /* namespace detail */
} /* namespace yaldpc */

#undef __M
#undef __E
