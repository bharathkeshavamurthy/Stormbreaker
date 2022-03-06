// GPU Schmidl & Cox sync metrics
//
// Copyright (c) 2018 Dennis Ogbe
//
// Compute P[d] and M[d] from raw complex samples

#ifndef b126c30d6439f1e2a89c
#define b126c30d6439f1e2a89c

#include "buffers.h"
#include "fcomplex.h"

#include <cuComplex.h>

namespace bam {
namespace dsp {

// n.b. some conventions: unless clear from context, we agree that the prefix
// for device pointers is d_* and the prefix for host pointers is h_*. as usual
// in this codebase, we try to prefix private members with _*.
//
// in both the execute and write_output function, we use the convention that
// nsamps is the input vector of available samples per channel and nreturn is an
// inout return value of the number of samples produced per channel.

using fcomplex = bamradio::fcomplex;
using outbuf = bamradio::ofdm::ChannelOutputBuffer::sptr;

class SCSync {
public:
  typedef std::shared_ptr<SCSync> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<SCSync>(std::forward<Args>(args)...);
  }

  SCSync(size_t L, size_t num_chan, size_t max_in_samps,
         std::vector<cudaStream_t> const &streams);
  ~SCSync();
  SCSync(SCSync const &other) = delete;

  /// run the computation. d_in follows the convention set in channelizer.cu
  /// (the 0-th sample of the i-th channel is at d_in[_max_in_samps * i])
  void execute(std::vector<size_t> const &nsamps, cuComplex const *d_in);

  /// write samples to the output buffers. outbuf buffers are assumed to be in
  /// host memory. it is assumed that this function also calls produce() on the
  /// output pointers.
  void write_output(std::vector<size_t> const &nsamps, cuComplex const *d_in,
                    std::vector<outbuf> &h_out);

protected:
  /// L length of sync sequence (=> 2L = length of twice repeated sync
  /// sequence)
  size_t const _L;
  /// number of channels
  size_t const _num_chan;
  /// maximum number of samples to be processed. This number MUST be the same
  /// size as the corresponding parameter of the channelizer object
  size_t const _max_in_samps;

  /// number of execution grids
  int const _ngrid = 100;
  /// number of threads per block
  int const _tpb = 512;

  /// we try to utilize _num_chan CUDA streams (these are created outside of
  /// this block)
  std::vector<cudaStream_t> const &_streams;
  /// keep a history of samples in GPU memory
  cuComplex *_d_samp_hist;
  /// the signal multiplied with its complex conjugate delayed by L samples
  cuComplex *_d_mul_conj_delay;
  /// the sample energy
  float *_d_samp_energy;

  /// history for the Pd computation
  cuComplex *_d_Pd_hist;
  /// The P[d] metric
  cuComplex *_d_Pd;
  /// history for the Rd^2 computation
  float *_d_Rd2_hist;
  /// The R[d]^2 metric
  float *_d_Rd2;
  /// The M[d] metric
  float *_d_Md;

  /// Signal Processing
  void _mul_conj_delay(std::vector<size_t> const &nsamps, cuComplex *d_in);
  void _samp_energy(std::vector<size_t> const &nsamps, cuComplex *d_in);
  void _compute_Pd(std::vector<size_t> const &nsamps);
  void _compute_Rd2(std::vector<size_t> const &nsamps);
  void _compute_Md(std::vector<size_t> const &nsamps);
  void _update_history(std::vector<size_t> const &nsamps, cuComplex *d_in);
};

} // namespace dsp
} // namespace bam

#endif // b126c30d6439f1e2a89c
