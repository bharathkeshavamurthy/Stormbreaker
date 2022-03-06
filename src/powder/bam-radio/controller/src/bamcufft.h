// -*- c++ -*-
//
// cuFFT wrapper

#ifndef e653c4df35a37c536366
#define e653c4df35a37c536366

#include "fft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <unordered_map>

namespace bamradio {
namespace fft {
//
// GPU FFT (cuFFT)
//
class GPUFFT : public FFT {
public:
  typedef std::shared_ptr<GPUFFT> sptr;
  static sptr make(std::vector<size_t> const &sizes, bool forward,
                   cudaStream_t stream) {
    return std::shared_ptr<GPUFFT>(new GPUFFT(sizes, forward, stream));
  }

  void execute(size_t size, fcomplex *in, fcomplex *out) const;
  std::vector<size_t> sizes() const;
  bool forward() const;

  ~GPUFFT();

private:
  GPUFFT(std::vector<size_t> const &sizes, bool forward, cudaStream_t stream);

  std::unordered_map<size_t, cufftHandle> _plans;
  int const _type;
};
} // namespace fft
} // namespace bamradio

#endif // e653c4df35a37c536366
