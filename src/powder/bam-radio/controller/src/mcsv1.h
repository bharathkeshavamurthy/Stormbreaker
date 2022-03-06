// Copyright (c) 2018 Diyu Yang

#include "mcs.h"
#include "ofdm.h"
#include "options.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
auto const inf = std::numeric_limits<double>::infinity();
namespace bamradio {
namespace ofdm {
// struct for return value of findMultMatMins
struct MultMatMins {
  uint8_t rArg;        // index row vector
  uint8_t cArg;        // index col vector
  double minVal = inf; // vector for all min vals
};

class MCSAdaptAlg {
private:
  // mcs table
  std::pair<MCS::Name, SeqID::ID> const _MCS_table[4][4]{
      {{MCS::QPSK_R12_N1944, SeqID::ID::ZIG_128_12_108_12_QPSK},
       {MCS::QPSK_R23_N1944, SeqID::ID::ZIG_128_12_108_12_QPSK},
       {MCS::QPSK_R34_N1944, SeqID::ID::ZIG_128_12_108_12_QPSK},
       {MCS::QPSK_R56_N1944, SeqID::ID::ZIG_128_12_108_12_QPSK}},

      {{MCS::QAM16_R12_N1944, SeqID::ID::ZIG_128_12_108_12_QAM16},
       {MCS::QAM16_R23_N1944, SeqID::ID::ZIG_128_12_108_12_QAM16},
       {MCS::QAM16_R34_N1944, SeqID::ID::ZIG_128_12_108_12_QAM16},
       {MCS::QAM16_R56_N1944, SeqID::ID::ZIG_128_12_108_12_QAM16}},

      {{MCS::QAM32_R12_N1944, SeqID::ID::ZIG_128_12_108_12_QAM32},
       {MCS::QAM32_R23_N1944, SeqID::ID::ZIG_128_12_108_12_QAM32},
       {MCS::QAM32_R34_N1944, SeqID::ID::ZIG_128_12_108_12_QAM32},
       {MCS::QAM32_R56_N1944, SeqID::ID::ZIG_128_12_108_12_QAM32}},

      {{MCS::QAM64_R12_N1944, SeqID::ID::ZIG_128_12_108_12_QAM64},
       {MCS::QAM64_R23_N1944, SeqID::ID::ZIG_128_12_108_12_QAM64},
       {MCS::QAM64_R34_N1944, SeqID::ID::ZIG_128_12_108_12_QAM64},
       {MCS::QAM64_R56_N1944, SeqID::ID::ZIG_128_12_108_12_QAM64}},
  };

  template <std::size_t size1, std::size_t size2>
  MultMatMins findMultMatMins(std::array<std::array<double, size1>, size2> A);
  // Q function for standard normal distribution
  double qfunc(double value) { return 1 - 0.5 * erfc(-value * M_SQRT1_2); };

  double compTheoBERForMD(double dMin, double sd, uint16_t M);
  std::array<uint16_t, 4> M = {4, 16, 32, 64};
  std::array<double, 4> R = {1. / 2., 2. / 3., 3. / 4., 5. / 6.};
  // coding rate table
  float alpha = options::phy::mcs_alpha;
  uint8_t rArg_prev =
      3; // previous MCS indices. Used when no solution is found in getNextMCS
  uint8_t mArg_prev = 0; //
  std::array<double, 4> _dMin = {
      alpha * 1.1892, alpha * 0.5318, alpha * 0.3761,
      alpha * 0.2595}; // The minimum Euclidean distance between points in a
                       // MQAM constellation

  // double Gc[4][4]; // 3*4 matrix in dB
  double Gc[4] = {3.5, 1.8, 1.0, 0.0};    // in dB
  double Gd[4] = {6.64, 3.09, 1.65, 0.0}; // in dB

  double _bandwidth =
      500000; // bandwidth (Hz). From set {500E3,750E3,1E6,2E6,4E6,5E6}.
  double Rb[4][4];
  double _throughputLB =
      0; // bps. Temporarily hardcoded. Later should be provided by scheduler
  double _berUB = pow(10, -6); // probability of bit error
  size_t _mcs_idx = 0;

  // hysteresis
  bool _hyst_state;
  int const _hyst_max_pause;
  int _hyst_pause_cnt;
  float const _hyst_err_ub;
  float const _hyst_err_lb;

public:
  // constructor
  MCSAdaptAlg();

  std::pair<MCS::Name, SeqID::ID> MCSbackoff() {
    if ((rArg_prev == 0) && (mArg_prev == 0))
      return _MCS_table[0][0];
    else if (rArg_prev == 0)
      mArg_prev--;
    else
      rArg_prev--;
    return _MCS_table[mArg_prev][rArg_prev];
  };

  std::pair<MCS::Name, SeqID::ID> getNextMCS(double noise_var,
                                             float error_rate);
}; // end of class
} // end namespace ofdm
} // end namespace bamradio
