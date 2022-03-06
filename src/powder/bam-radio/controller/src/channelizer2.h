// Channelizer v2. Overlap-save method with cuFFT.
//
// Copyright (c) 2018 Dennis Ogbe
//
// Reference: Proakis, John G., and Dimitris G Manolakis. Digital Signal
// Processing. 4th ed. Upper Saddle River, N.J. Pearson Prentice Hall, 2007, pp
// 486

#ifndef c9902164c66f9a2a736972a25d77
#define c9902164c66f9a2a736972a25d77

#include "bandwidth.h"
#include "buffers.h"

#include <memory>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace bam {
namespace dsp {

// n.b. some conventions: unless clear from context, we agree that the
// prefix for device pointers is d_* and the prefix for host pointers is
// h_*. managed device pointers is dm_*. as usual in this codebase, we try
// to prefix private members with _*.

class Channelizer2 {
public:
  // shared_ptr convention
  typedef std::shared_ptr<Channelizer2> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<Channelizer2>(std::forward<Args>(args)...);
  }

  ///
  /// num_chan: number of channels
  /// N: FFT size
  /// nbatch: number of batches for overlap/save
  /// subchannel_table: table containing possible impulse response and
  ///                   oversampling factor of subchannels.
  ///
  /// there are some things to think about here. one constraint we have is for
  /// all the filter impulse responses to have the same number of
  /// taps. Furthermore, all of the decimation factors in subchannel_table need
  /// to cleanly divide the number of usable samples L. One way of achieving
  /// this is to let the filter length be the least common multiple of all
  /// decimation rates and let the FFT size be an integer multiple of the number
  /// of taps.
  ///
  Channelizer2(size_t num_chan, size_t N, size_t nbatch,
               std::vector<SubChannel> const &subchannel_table);
  Channelizer2(Channelizer2 const &other) = delete;
  ~Channelizer2();

  /// load samples into device buffer. return true if buf had enough items
  /// ready, false otherwise. calls consume() on the buffer if successful.
  bool load(bamradio::PinnedComplexRingBuffer *buf);
  /// process signal.
  void execute();
  /// Get the number of items produced in the last call to execute()
  void getNumOutput(std::vector<size_t> &vn) const;
  /// Get the device pointer to the output buffer
  cuComplex *getOutPtr() const;
  /// Get the stride of the output buffer
  size_t getOutStride() const;
  /// Get the cuda streams used
  std::vector<cudaStream_t> const &getStreams() const;

  /// set the "bandwidth" (decimation + filter taps) of a specific channel
  /// index. this index must be in the range of the size of the subchannel_table
  /// given to the constructor.
  void setBandwidth(size_t channel, size_t bw_idx);
  /// Set offset frequency (normalized to [-0.5, 0.5])
  void setOffsetFreq(size_t channel, float freq_normalized);

protected:
  /// number of channels
  size_t const _num_chan;
  /// FFT size
  size_t const _N;
  /// batch size
  size_t const _nbatch;
  /// number of filter taps (same for each channel)
  size_t const _ntaps;
  /// number of usable samples per batch
  size_t const _L;
  /// number of different oversampling/decimation rates/filter taps
  size_t const _nos;

  /// number of samples to load during one call to load()
  size_t const _nload;
  /// number of samples at FFT input for one channel
  size_t const _fft_in_size;
  /// number of samples at FFT output for one channel
  size_t const _fft_out_size;
  // number of samples at the decimator input for one channel
  size_t const _dec_in_size;
  // maximum possible number of samples at the output for one channel
  size_t const _max_out_size;

  /// we try to utilize _num_chan CUDA streams
  std::vector<cudaStream_t> _streams;
  /// each stream has its own plan for forward and backward FFT
  std::vector<cufftHandle> _fft_plans;
  std::vector<cufftHandle> _ifft_plans;
  /// to figure out which filter to use for channel i, we look in this
  /// vector.
  std::vector<size_t> _bw_idx;

  /// this is where the filters and the decimation factors live
  std::vector<cuComplex *> _filters_d;
  std::vector<size_t> _os;
  /// frequency and phase per channel
  std::vector<float> _phase;
  std::vector<float> _freq;

  /// the input samples (_nload complex samples)
  cuComplex *_d_in;

  /// the intermediate buffers holding samples for each channel follow the
  /// convention established in v1 channelizer. each channel holds buf_size *
  /// num_chan samples (buf_size varying between the different buffers) and the
  /// i-th sample of the n-th channel lies in _d_buf[n * buf_size + i]

  /// the input buffer for the FFT (output buffer of mixing). _fft_in_size *
  /// _num_chan samples. (first _ntaps - 1 samples are history, otherwise
  /// _nbatch * _L samples)
  cuComplex *_d_fft_in;
  /// the output buffer of the FFT. This buffer is also
  /// used by the filtering and the IFFT stage.
  cuComplex *_d_fft_out;
  /// the input buffer to the decimator.
  cuComplex *_d_dec_in;
  /// the output buffer. we keep this on the device so that the synchronizer can
  /// take over.
  cuComplex *_d_out;

  /// initialize the on-device storage of oversample factors and filter
  /// frequency responses.
  void _initTaps(std::vector<SubChannel> const &subchannel_table);
  /// initialize on-device storage of mixing frequency and phase
  void _initPhase();
  /// initialize FFT plans and cuda streams
  void _initPlans();
  /// initialize device memory
  void _initMemory();

  /// signal processing
  void _mix();
  void _fft();
  void _filter();
  void _ifft();
  void _discard();
  void _decimate();
  void _update_history();
};

} // namespace dsp
} // namespace bam

#endif // c9902164c66f9a2a736972a25d77
