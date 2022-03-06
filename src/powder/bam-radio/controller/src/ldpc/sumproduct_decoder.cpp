/*
 * sumproduct_decoder.cpp
 * Implementation of the Sum-product decoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "yaldpc.hpp"
#include <cmath>
#include <utility>

// message storage
#define __E(j, i) this->E[j * this->_n + i]
#define __M(j, i) this->M[j * this->_n + i]

using std::size_t;
namespace yaldpc {

// constructor
template <typename LLR, typename Out>
SumProductDecoder<LLR, Out>::SumProductDecoder(LDPCCode const &code,
                                               uint64_t max_its,
                                               bool output_value,
                                               bool do_parity_check)
    : Decoder<LLR, Out>(), detail::FloodingDecoder<LLR, Out>(
                               code.H, code.m, code.n, max_its, output_value,
                               do_parity_check) {}

template <typename LLR, typename Out>
typename yaldpc::SumProductDecoder<LLR, Out>::sptr
SumProductDecoder<LLR, Out>::make(LDPCCode const &code, uint64_t max_its,
                                  bool output_value, bool do_parity_check) {
  return std::make_shared<SumProductDecoder<LLR, Out>>(
      code, max_its, output_value, do_parity_check);
}

// helper function for below
template <typename LLR> inline LLR phi(LLR x) {
  LLR e_x = std::exp(x);
  return std::log(e_x + 1.0) - std::log(e_x - 1.0);
}

// compute check messages
template <typename LLR, typename Out>
void SumProductDecoder<LLR, Out>::check_messages() {
  for (size_t j = 0; j < this->_m; ++j) {
    bool si = false;                          // keep track of sign
    LLR su(0.0);                              // keep track of sum
    std::vector<LLR> phis(this->B[j].size()); // save one computation of phi
    size_t k = 0;                             // helper iterator
    for (auto const &i : this->B[j]) {
      auto MM = __M(j, i);
      this->_si2[k] = detail::Sign(MM);
      si = (si != this->_si2[k]);
      phis[k] = phi(detail::Abs(MM));
      su += phis[k++]; // sum
    }
    k = 0;
    for (auto const &i : this->B[j]) {
      // subtract the outgoing message, flip the sign back if needed
      __E(j, i) =
          si != this->_si2[k] ? -phi(su - phis[k++]) : phi(su - phis[k++]);
    }
  }
}

// decode using flooding algorithm
template <typename LLR, typename Out>
void SumProductDecoder<LLR, Out>::decode(LLR const *in, Out *out) {
  this->flood_decode(in, out);
}

template <typename LLR, typename Out>
std::vector<Out>
SumProductDecoder<LLR, Out>::decode(std::vector<LLR> const &in) {
  assert(in.size() == this->_n);
  std::vector<Out> o(this->_output_value ? this->_k : this->_n);
  this->flood_decode(in.data(), o.data());
  return o;
}

// get decoder parameters
template <typename LLR, typename Out>
uint64_t SumProductDecoder<LLR, Out>::max_its() {
  return this->_max_its;
}
template <typename LLR, typename Out>
bool SumProductDecoder<LLR, Out>::output_value() {
  return this->_output_value;
}
template <typename LLR, typename Out>
bool SumProductDecoder<LLR, Out>::do_parity_check() {
  return this->_do_parity_check;
}
template <typename LLR, typename Out>
std::size_t SumProductDecoder<LLR, Out>::m() {
  return this->_m;
}
template <typename LLR, typename Out>
std::size_t SumProductDecoder<LLR, Out>::n() {
  return this->_n;
}
template <typename LLR, typename Out>
std::size_t SumProductDecoder<LLR, Out>::k() {
  return this->_k;
}
// template instantiation
template class SumProductDecoder<float, uint8_t>;
template class SumProductDecoder<float, float>;
template class SumProductDecoder<double, uint8_t>;
template class SumProductDecoder<double, double>;
// N.B: Can't do fixed-point sum-product rn because we don't have a fixed-point
// log function.
} /* namespace  yaldpc */

#undef __M
#undef __E
