// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#include "mcs.h"
#include <unordered_map>

namespace bamradio {

size_t BitsPerSymbol(Modulation m) {
  switch (m) {
  case Modulation::BPSK:
    return 1;
  case Modulation::QPSK:
    return 2;
  case Modulation::QAM16:
    return 4;
  case Modulation::QAM32:
    return 5;
  case Modulation::QAM64:
    return 6;
  case Modulation::QAM128:
    return 7;
  case Modulation::QAM256:
    return 8;
  }
}

namespace ofdm {

MCS::CodePair MCS::codePair() const {
  return MCS::CodePair(codeRate == CodeRate{1, 2}
                           ? 0
                           : codeRate == CodeRate{2, 3}
                                 ? 1
                                 : codeRate == CodeRate{3, 4}
                                       ? 2
                                       : codeRate == CodeRate{5, 6} ? 3 : -1,
                       blockLength);
}

yaldpc::IEEE802Code MCS::toIEEE802Code() const {
  auto const cp = codePair();
  return yaldpc::ieee80211::get(cp.first, cp.second);
}

std::array<MCS, 52> const MCS::table{
    {// uncoded
     // MCS{Modulation::BPSK, {1, 1}, 1}, MCS{Modulation::QPSK, {1, 1}, 1},
     // MCS{Modulation::QAM16, {1, 1}, 1},
     // blocklength 648
     MCS{Modulation::BPSK, {1, 2}, 648}, MCS{Modulation::BPSK, {2, 3}, 648},
     MCS{Modulation::BPSK, {3, 4}, 648}, MCS{Modulation::BPSK, {5, 6}, 648},
     MCS{Modulation::QPSK, {1, 2}, 648}, MCS{Modulation::QPSK, {2, 3}, 648},
     MCS{Modulation::QPSK, {3, 4}, 648}, MCS{Modulation::QPSK, {5, 6}, 648},
     MCS{Modulation::QAM16, {1, 2}, 648}, MCS{Modulation::QAM16, {2, 3}, 648},
     MCS{Modulation::QAM16, {3, 4}, 648}, MCS{Modulation::QAM16, {5, 6}, 648},
     // blocklength 1296
     MCS{Modulation::BPSK, {1, 2}, 1296}, MCS{Modulation::BPSK, {2, 3}, 1296},
     MCS{Modulation::BPSK, {3, 4}, 1296}, MCS{Modulation::BPSK, {5, 6}, 1296},
     MCS{Modulation::QPSK, {1, 2}, 1296}, MCS{Modulation::QPSK, {2, 3}, 1296},
     MCS{Modulation::QPSK, {3, 4}, 1296}, MCS{Modulation::QPSK, {5, 6}, 1296},
     MCS{Modulation::QAM16, {1, 2}, 1296}, MCS{Modulation::QAM16, {2, 3}, 1296},
     MCS{Modulation::QAM16, {3, 4}, 1296}, MCS{Modulation::QAM16, {5, 6}, 1296},
     // blocklength 1944
     MCS{Modulation::BPSK, {1, 2}, 1944}, MCS{Modulation::BPSK, {2, 3}, 1944},
     MCS{Modulation::BPSK, {3, 4}, 1944}, MCS{Modulation::BPSK, {5, 6}, 1944},
     MCS{Modulation::QPSK, {1, 2}, 1944}, MCS{Modulation::QPSK, {2, 3}, 1944},
     MCS{Modulation::QPSK, {3, 4}, 1944}, MCS{Modulation::QPSK, {5, 6}, 1944},
     MCS{Modulation::QAM16, {1, 2}, 1944}, MCS{Modulation::QAM16, {2, 3}, 1944},
     MCS{Modulation::QAM16, {3, 4}, 1944}, MCS{Modulation::QAM16, {5, 6}, 1944},
     MCS{Modulation::QAM32, {1, 2}, 1944}, MCS{Modulation::QAM32, {2, 3}, 1944},
     MCS{Modulation::QAM32, {3, 4}, 1944}, MCS{Modulation::QAM32, {5, 6}, 1944},
     MCS{Modulation::QAM64, {1, 2}, 1944}, MCS{Modulation::QAM64, {2, 3}, 1944},
     MCS{Modulation::QAM64, {3, 4}, 1944}, MCS{Modulation::QAM64, {5, 6}, 1944},
     MCS{Modulation::QAM128, {1, 2}, 1944},
     MCS{Modulation::QAM128, {2, 3}, 1944},
     MCS{Modulation::QAM128, {3, 4}, 1944},
     MCS{Modulation::QAM128, {5, 6}, 1944},
     MCS{Modulation::QAM256, {1, 2}, 1944},
     MCS{Modulation::QAM256, {2, 3}, 1944},
     MCS{Modulation::QAM256, {3, 4}, 1944},
     MCS{Modulation::QAM256, {5, 6}, 1944}}};

MCS::Name MCS::stringNameToIndex(std::string const &n) {
  static std::unordered_map<std::string, MCS::Name> const m{
      {"BPSK_R12_N648", MCS::BPSK_R12_N648},
      {"BPSK_R23_N648", MCS::BPSK_R23_N648},
      {"BPSK_R34_N648", MCS::BPSK_R34_N648},
      {"BPSK_R56_N648", MCS::BPSK_R56_N648},
      {"QPSK_R12_N648", MCS::QPSK_R12_N648},
      {"QPSK_R23_N648", MCS::QPSK_R23_N648},
      {"QPSK_R34_N648", MCS::QPSK_R34_N648},
      {"QPSK_R56_N648", MCS::QPSK_R56_N648},
      {"QAM16_R12_N648", MCS::QAM16_R12_N648},
      {"QAM16_R23_N648", MCS::QAM16_R23_N648},
      {"QAM16_R34_N648", MCS::QAM16_R34_N648},
      {"QAM16_R56_N648", MCS::QAM16_R56_N648},
      {"BPSK_R12_N1296", MCS::BPSK_R12_N1296},
      {"BPSK_R23_N1296", MCS::BPSK_R23_N1296},
      {"BPSK_R34_N1296", MCS::BPSK_R34_N1296},
      {"BPSK_R56_N1296", MCS::BPSK_R56_N1296},
      {"QPSK_R12_N1296", MCS::QPSK_R12_N1296},
      {"QPSK_R23_N1296", MCS::QPSK_R23_N1296},
      {"QPSK_R34_N1296", MCS::QPSK_R34_N1296},
      {"QPSK_R56_N1296", MCS::QPSK_R56_N1296},
      {"QAM16_R12_N1296", MCS::QAM16_R12_N1296},
      {"QAM16_R23_N1296", MCS::QAM16_R23_N1296},
      {"QAM16_R34_N1296", MCS::QAM16_R34_N1296},
      {"QAM16_R56_N1296", MCS::QAM16_R56_N1296},
      {"BPSK_R12_N1944", MCS::BPSK_R12_N1944},
      {"BPSK_R23_N1944", MCS::BPSK_R23_N1944},
      {"BPSK_R34_N1944", MCS::BPSK_R34_N1944},
      {"BPSK_R56_N1944", MCS::BPSK_R56_N1944},
      {"QPSK_R12_N1944", MCS::QPSK_R12_N1944},
      {"QPSK_R23_N1944", MCS::QPSK_R23_N1944},
      {"QPSK_R34_N1944", MCS::QPSK_R34_N1944},
      {"QPSK_R56_N1944", MCS::QPSK_R56_N1944},
      {"QAM16_R12_N1944", MCS::QAM16_R12_N1944},
      {"QAM16_R23_N1944", MCS::QAM16_R23_N1944},
      {"QAM16_R34_N1944", MCS::QAM16_R34_N1944},
      {"QAM16_R56_N1944", MCS::QAM16_R56_N1944},
      {"QAM32_R12_N1944", MCS::QAM32_R12_N1944},
      {"QAM32_R23_N1944", MCS::QAM32_R23_N1944},
      {"QAM32_R34_N1944", MCS::QAM32_R34_N1944},
      {"QAM32_R56_N1944", MCS::QAM32_R56_N1944},
      {"QAM64_R12_N1944", MCS::QAM64_R12_N1944},
      {"QAM64_R23_N1944", MCS::QAM64_R23_N1944},
      {"QAM64_R34_N1944", MCS::QAM64_R34_N1944},
      {"QAM64_R56_N1944", MCS::QAM64_R56_N1944},
      {"QAM128_R12_N1944", MCS::QAM128_R12_N1944},
      {"QAM128_R23_N1944", MCS::QAM128_R23_N1944},
      {"QAM128_R34_N1944", MCS::QAM128_R34_N1944},
      {"QAM128_R56_N1944", MCS::QAM128_R56_N1944},
      {"QAM256_R12_N1944", MCS::QAM256_R12_N1944},
      {"QAM256_R23_N1944", MCS::QAM256_R23_N1944},
      {"QAM256_R34_N1944", MCS::QAM256_R34_N1944},
      {"QAM256_R56_N1944", MCS::QAM256_R56_N1944}};
  auto const it = m.find(n);
  if (it == m.end()) {
    throw std::runtime_error("Bad MCS name.");
  }
  return it->second;
}
}
}
