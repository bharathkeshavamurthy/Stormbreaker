// -*- c++ -*-
//
// fftw wrapper

#ifndef fc84281ab17fc64e59a3
#define fc84281ab17fc64e59a3

#include "fft.h"
#include <fftw3.h>

#include <unordered_map>

namespace bamradio {
namespace fft {
//
// CPU FFT (FFTW)
//
class CPUFFT : public FFT {
public:
  typedef std::shared_ptr<CPUFFT> sptr;
  static sptr make(std::vector<size_t> const &sizes, bool forward, int nthreads,
                   std::string const &wisdom_filename) {
    return std::shared_ptr<CPUFFT>(
        new CPUFFT(sizes, forward, nthreads, wisdom_filename));
  }

  void execute(size_t size, fcomplex *in, fcomplex *out) const;
  std::vector<size_t> sizes() const;
  bool forward() const;

  ~CPUFFT();

private:
  CPUFFT(std::vector<size_t> const &sizes, bool forward, int nthreads,
         std::string const &wisdom_filename);

  std::unordered_map<size_t, fftwf_plan> _plans;
  bool const _forward;
};

} // namespace fft
} // namespace bamradio

#endif // fc84281ab17fc64e59a3
