//  Copyright © 2017 Stephen Larew

#include "common.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace gr {
namespace bamofdm {

static const float PI = 3.14159265358979323846264338327950f;

std::vector<std::complex<float>> generate_cazac_seq(const size_t N,
                                                    const size_t M) {
  // D. Chu, "Polyphase codes with good periodic correlation properties
  // (Corresp.)," Information Theory, IEEE Transactions on, vol.18, no.4,
  // pp.531-532, Jul 1972
  // doi: 10.1109/TIT.1972.1054840
  //
  // Zadoff-Chu Sequence (a CAZAC sequence):
  // code { a_k } (k=1...N-1) of length N
  //
  // N even:
  // a_k = exp[ i (M pi k^2) / N ]
  // where M is integer relatively prime to N
  //
  // N odd:
  // a_k = exp[ i (M pi k(k+1)) / N ]
  // wher eM is integer relatively prime to N
  //
  // Interesting:
  // B.M. Popovic, "Generalized chirp-like polyphase sequences with
  // optimum correlation properties," Information Theory, IEEE
  // Transactions on, vol.38, no.4, pp.1406-1409, Jul 1992
  // doi: 10.1109/18.144727

  std::vector<std::complex<float>> cazac(N);

  unsigned int k = 0;
  std::generate(cazac.begin(), cazac.end(), [&k, N, M] {
    float t;
    if (N % 2 == 0) {
      t = (M * k * k) * PI / N;
    } else {
      t = (M * k * (k + 1)) * PI / N;
    }
    k++;
    return std::complex<float>(std::cos(t), std::sin(t));
  });

  return cazac;
}

// unit test helpers
// random symbols from some constellation.
// can generate seed with
// auto seed =
//     std::chrono::high_resolution_clock::now().time_since_epoch().count();
std::vector<gr_complex>
random_symbols(size_t n, bamradio::constellation::sptr constellation,
               size_t seed) {
  std::mt19937 mt_rand(seed);
  std::vector<gr_complex> v(n);
  // auto const points = constellation->points();
  std::vector<gr_complex> points = constellation->points();
  auto rand = std::uniform_int_distribution<>(0, points.size() - 1);
  std::generate(begin(v), end(v), [&]() {
    return constellation->get_scale() * points[rand(mt_rand)];
  });
  return v;
}

// Complex Gaussian AWGN; SNR is an absolute ratio
std::vector<gr_complex> awgn(double snr, std::vector<gr_complex> syms,
                             size_t seed) {
  std::mt19937 mt_rand(seed);
  auto randn = std::bind(std::normal_distribution<float>(0.0, 1.0), mt_rand);
  std::vector<gr_complex> v(syms.size());
  size_t k = 0;
  std::generate(begin(v), end(v), [&]() {
    auto re = 1.0 / std::sqrt(2.0 * snr) * randn();
    auto im = 1.0 / std::sqrt(2.0 * snr) * randn();
    return syms[k++] + gr_complex(re, im);
  });
  return v;
}

// vector assertions
bool assert_float_almost_equal(float const &ref, float const &x, double abs_eps,
                               double rel_eps) {
  if (std::abs(ref - x) < abs_eps) {
    return true;
  }
  if (std::abs(ref) > abs_eps) {
    if (std::abs(ref - x) / std::abs(ref) > rel_eps) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

bool assert_float_vectors_almost_equal(std::vector<float> const &ref,
                                       std::vector<float> const &x,
                                       double abs_eps, double rel_eps) {
  if (ref.size() != x.size()) {
    return false;
  }
  for (int i = 0; i < ref.size(); ++i) {
    if (!assert_float_almost_equal(ref[i], x[i], abs_eps, rel_eps)) {
      return false;
    }
  }
  return true;
}

bool assert_complex_almost_equal(gr_complex const &ref, gr_complex const &x,
                                 double abs_eps, double rel_eps) {
  if (std::abs(ref - x) < abs_eps) {
    return true;
  }
  if (std::abs(ref) > abs_eps) {
    if (std::abs(ref - x) / std::abs(ref) > rel_eps) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

bool assert_complex_vectors_almost_equal(std::vector<gr_complex> const &ref,
                                         std::vector<gr_complex> const &x,
                                         double abs_eps, double rel_eps) {
  if (ref.size() != x.size()) {
    return false;
  }
  for (int i = 0; i < ref.size(); ++i) {
    if (!assert_complex_almost_equal(ref[i], x[i], abs_eps, rel_eps)) {
      return false;
    }
  }
  return true;
}

} // namespace bamofdm
} // namespace gr
