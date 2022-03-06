// GPU Channelizer
// (c) Tomohiro Arakawa (tarakawa@purdue.edu)

#ifndef CHANNELIZER_H_INCLUDED
#define CHANNELIZER_H_INCLUDED

#include "fcomplex.h"

#include <complex>
#include <vector>

namespace bam {
namespace dsp {

using fcomplex = bamradio::fcomplex;

class Channelizer {
private:
  size_t const _max_in_samps;
  size_t const _num_chan;
  void *_device_in_buf;
  void *_device_freqshifted_buf; // N_stream * N_samps elements
  void *_device_out_buf;         // N_stream * N_samps elements
  void **_device_inptr_list;
  size_t *_managed_decimation_factor_list;
  float **_managed_filt_ptr_list;
  size_t *_managed_filt_size_list;
  float *_managed_freq_list;
  float *_managed_phase_list;
  size_t *_managed_out_nitems_list;
  std::vector<size_t> _offset_list;

public:
  Channelizer(size_t num_chan, size_t max_in_samps);
  ~Channelizer();
  /// Set filter taps
  void setTaps(size_t channel, std::vector<float> const &taps);
  /// Set decimation factor
  void setDecimationFactor(size_t channel, size_t factor);
  /// Set offset frequency (normalized to [-0.5, 0.5])
  void setOffsetFreq(size_t channel, float freq_normalized);
  /// Run channelizer and return number of consumed items
  size_t execute(size_t nsamps, fcomplex const *in_buf, bool copy_buf = true);
  /// Get channelizer output and return number of produced items (legacy)
  size_t getOutput(size_t channel, fcomplex *out_buf);

  /// Get the number of items produced in the last call to execute(...)
  void getNumOutput(std::vector<size_t> &vn);
  /// Get the device pointer to the output buffer
  fcomplex *getOutPtr() const { return (fcomplex *)_device_out_buf; };
};

} // namespace dsp
} // namespace bam

#endif
