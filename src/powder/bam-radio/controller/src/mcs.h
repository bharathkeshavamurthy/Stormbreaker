// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

// TODO: document this file

#ifndef aa7620e0641f926531
#define aa7620e0641f926531

#include <array>
#include <cstdint>
#include <utility>
#include "ldpc/yaldpc.hpp"

namespace bamradio {

enum class Modulation { BPSK, QPSK, QAM16, QAM32, QAM64, QAM128, QAM256 };

size_t BitsPerSymbol(Modulation m);

struct CodeRate {
  uint16_t k;
  uint16_t n;

  bool operator==(CodeRate const &R) const { return k == R.k && n == R.n; }
};

template <typename T> decltype(CodeRate::k) operator*(CodeRate R, T N) {
  return N * R.k / R.n;
}
template <typename T> decltype(CodeRate::k) operator*(T N, CodeRate R) {
  return R * N;
}
template <typename T> decltype(CodeRate::k) operator/(T N, CodeRate R) {
  return N * R.n / R.k;
}

namespace ofdm {

struct MCS {
  Modulation modulation;
  CodeRate codeRate;
  uint32_t blockLength;

  inline size_t bitsPerSymbol() const { return BitsPerSymbol(modulation); }

  static std::array<MCS, 52> const table;

  yaldpc::IEEE802Code toIEEE802Code() const;

  typedef std::pair<uint64_t, uint64_t> CodePair;
  CodePair codePair() const;

  enum Name : size_t {
    BPSK_R12_N648 = 0,
    BPSK_R23_N648,
    BPSK_R34_N648,
    BPSK_R56_N648,
    QPSK_R12_N648,
    QPSK_R23_N648,
    QPSK_R34_N648,
    QPSK_R56_N648,
    QAM16_R12_N648,
    QAM16_R23_N648,
    QAM16_R34_N648,
    QAM16_R56_N648,
    BPSK_R12_N1296,
    BPSK_R23_N1296,
    BPSK_R34_N1296,
    BPSK_R56_N1296,
    QPSK_R12_N1296,
    QPSK_R23_N1296,
    QPSK_R34_N1296,
    QPSK_R56_N1296,
    QAM16_R12_N1296,
    QAM16_R23_N1296,
    QAM16_R34_N1296,
    QAM16_R56_N1296,
    BPSK_R12_N1944,
    BPSK_R23_N1944,
    BPSK_R34_N1944,
    BPSK_R56_N1944,
    QPSK_R12_N1944,
    QPSK_R23_N1944,
    QPSK_R34_N1944,
    QPSK_R56_N1944,
    QAM16_R12_N1944,
    QAM16_R23_N1944,
    QAM16_R34_N1944,
    QAM16_R56_N1944,
    QAM32_R12_N1944,
    QAM32_R23_N1944,
    QAM32_R34_N1944,
    QAM32_R56_N1944,
    QAM64_R12_N1944,
    QAM64_R23_N1944,
    QAM64_R34_N1944,
    QAM64_R56_N1944,
    QAM128_R12_N1944,
    QAM128_R23_N1944,
    QAM128_R34_N1944,
    QAM128_R56_N1944,
    QAM256_R12_N1944,
    QAM256_R23_N1944,
    QAM256_R34_N1944,
    QAM256_R56_N1944,
    NUM_MCS
  };

  static Name stringNameToIndex(std::string const &n);
};

static_assert(MCS::NUM_MCS == MCS::table.size(),
              "All MCS table entries must be named.");
}
}

#endif
