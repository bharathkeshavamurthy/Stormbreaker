// Copyright (c) 2018 Diyu Yang
// Copyright (c) 2018 Andrew C. Marcum
// Copyright (c) 2019 Tomohiro Arakawa

#include "mcsv1.h"
#include "events.h"
#include "options.h"
namespace bamradio {
namespace ofdm {

template <std::size_t size1, std::size_t size2>
MultMatMins
MCSAdaptAlg::findMultMatMins(std::array<std::array<double, size1>, size2> A) {
  MultMatMins retval;
  for (int r = 0; r < size2; r++) {
    for (int c = 0; c < size1; c++) {
      if (A[r][c] < retval.minVal) {
        retval.minVal = A[r][c];
        retval.rArg = r;
        retval.cArg = c;
      } // endif
    }   // end c loop
  }     // end r loop

  return retval;
};

double MCSAdaptAlg::compTheoBERForMD(double dMin, double sd, uint16_t M) {
  if (M == 2 || M == 4)
    return (double)qfunc(dMin / (sqrt(2) * sd));
  else
    return (double)(4 / log2(M)) * qfunc(dMin / (sqrt(2) * sd));
};

// constructor
MCSAdaptAlg::MCSAdaptAlg()
    : _hyst_state(false), _hyst_err_lb(options::phy::data::mcs_hyst_err_lb),
      _hyst_err_ub(options::phy::data::mcs_hyst_err_ub), _hyst_pause_cnt(0),
      _hyst_max_pause(10) {
  // compute Gc and Rb
  for (int m = 0; m < M.size(); ++m) {
    for (int r = 0; r < R.size(); ++r) {
      // Gc[m][r] = 10.*log10(1./(R[r]*_dMin[m]));
      Rb[m][r] = _bandwidth * log2(M[m]) * R[r];
    }
  }
};

/*
MCSAdaptation algorithm
sd: variance (not st-deviation. Sorry for the name confusion)
_dMin: const min distance. length 3 arrayy
Gc: asymptotic code gains. size 3*4 array
Rb: block error rate
_throughputLB: lower bound of throughput. Provided by scheduling algo
_berUB: upper bound of block error rate. set this to be small
*/
std::pair<MCS::Name, SeqID::ID> MCSAdaptAlg::getNextMCS(double noise_var,
                                                        float error_rate) {

  // modulation and coding scheme indices
  size_t m_idx = mArg_prev;
  size_t r_idx = rArg_prev;

  // FER hysteresis: update state
  if (!_hyst_state && error_rate >= _hyst_err_ub) {
    _hyst_state = true;
    _hyst_pause_cnt = 0;
  } else if (_hyst_state && error_rate <= _hyst_err_ub) {
    _hyst_state = false;
  }

  if (_hyst_state) { // hysteresis state
    if (_hyst_pause_cnt > _hyst_max_pause) {
      _hyst_pause_cnt = 0;
    }

    if (_hyst_pause_cnt == 0) { // find the next highest mcs
      double se_diff_min = inf;
      for (size_t m = 0; m < M.size(); ++m) {
        for (size_t r = 0; r < R.size(); ++r) {
          double const se_diff =
              log2(M[mArg_prev]) * R[rArg_prev] - log2(M[m]) * R[r];
          if (se_diff > 0 && se_diff < se_diff_min) {
            m_idx = m;
            r_idx = r;
            se_diff_min = se_diff;
          }
        }
      }
    }

    // increment pause count
    _hyst_pause_cnt++;

  } else { // hysteresis == false ... run the normal mcs adaotation
    // first convert variance to st-deviation
    double sd = sqrt(noise_var);
    size_t msize = M.size();
    size_t rsize = R.size();
    std::array<std::array<double, 4>, 4> B;
    // initialize B to 0
    int cnt = 0;
    for (int m = 0; m < msize; ++m) {
      for (int r = 0; r < rsize; ++r) {
        B[m][r] = 0;
        double sdin = sd / sqrt(pow(10., (Gc[r] + Gd[m]) / 10.));
        double ber = compTheoBERForMD(_dMin[m], sdin, M[m]);
        if ((Rb[m][r] >= _throughputLB) && (ber <= _berUB)) {
          B[m][r] = _dMin[m] - 2 * sd / sqrt(pow(10, Gc[r] / 10));
          if (B[m][r] <= 0) {
            B[m][r] = inf;
            cnt = cnt + 1;
          }
        } else {
          B[m][r] = inf;
          cnt = cnt + 1;
        }
      }
    }
    if (cnt == msize * rsize) { // hardcoded total size of M*R
      // std::cout<<"no solution found"<<std::endl;
    } else {
      // found solution
      auto Bmins = findMultMatMins(B);
      m_idx = Bmins.rArg;
      r_idx = Bmins.cArg;
    }
  }

  // save results
  mArg_prev = m_idx;
  rArg_prev = r_idx;

  return _MCS_table[m_idx][r_idx];

}; // end new MCS algo

} // namespace ofdm
} // namespace bamradio
