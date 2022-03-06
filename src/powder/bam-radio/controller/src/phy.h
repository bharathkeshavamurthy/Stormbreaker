// -*- c++ -*-
//
// Physical Layer.
//
// Copyright (c) 2017-2018 Dennis Ogbe
// Copyright (c) 2017-2018 Stephen Larew

#ifndef e1b89b720965927782c6
#define e1b89b720965927782c6

#include "bam_constellation.h"
#include "bbqueue.h"
#include "cc_data.h"
#include "cuda_allocator.h"
#include "fft.h"
#include "frame.h"
#include "fcomplex.h"
#include "buffers.h"
#include "statistics.h"
#include "util.h"
#include "radiocontroller_types.h"
#include "median.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>

#include <boost/asio.hpp>
#include <boost/crc.hpp>
#include <boost/optional.hpp>

// see the yaldpc/ext/FpXX headers for a reason why we are doing this
#define __4140ec62e7fc__YALDPC_FIXED_NEED_BOOST_SERIALZE__
#include "llr_format.h"
#include "ldpc/yaldpc.hpp"

#include <uhd/stream.hpp>

// debug
#define BAMRADIO_PHY_DEBUG 0

namespace bamradio {

namespace ofdm {

// given a pointer to a frame object, give me the number of OFDM Symbols, the
// number of samples per OFDM symbol, and the number of cyclic prefix samples.
typedef std::function<void(DFTSOFDMFrame::sptr frame, size_t &num_syms,
                           size_t &symbol_length, size_t &cyclic_prefix_length)>
    interpret_frame;

// Schmidl & Cox information
struct SCInfo {
  float snr;
  float domega;
};

//
// Synchronization (preambles)
//

const std::vector<fcomplex> &get_preamble_td_oversampled(size_t L, size_t O);
const std::vector<fcomplex> &get_preamble_pilot_symbols(size_t L, size_t O);

//
// Frame detection
//

class frame_detect_demux {
public:
  // function signature for the demod(...) function call. pointer to the frame
  // object (if nullptr, assume we are demodulating a header), rvalue reference
  // to a vector of complex samples, the snr detected during sync, and a pointer
  // back to the FDD that called this function in the first place
  typedef std::function<void(DFTSOFDMFrame::sptr, std::vector<fcomplex> &&,
                             SCInfo sc_info, frame_detect_demux *)>
      demodfn;
  typedef std::shared_ptr<frame_detect_demux> sptr;

  static sptr make(size_t Nc, size_t Ns, size_t num_pre_header_ofdm_syms,
                   float preamble_threshold, demodfn demod,
                   interpret_frame interpret_frame) {
    return std::make_shared<frame_detect_demux>(
        Nc, Ns, num_pre_header_ofdm_syms, preamble_threshold, demod,
        interpret_frame);
  }
  frame_detect_demux(size_t Nc, size_t Ns, size_t num_pre_header_ofdm_syms,
                     float preamble_threshold, demodfn demod,
                     interpret_frame interpret_frame);
  frame_detect_demux(frame_detect_demux const &other) = delete;
  ~frame_detect_demux();

  /// connect the FDD to a channel output and start searching
  void start(ChannelOutputBuffer::sptr inbuf, std::string const &name);
  /// stop the FDD
  void stop();
  /// notify the FDD about the result of a header demodulation
  void notify(DFTSOFDMFrame::sptr frame);

protected:
  interpret_frame _interpret_frame;

  enum class State { SEARCHING, PRE_HEADER, PRE_HEADER_WAIT, PAYLOAD } _state;

  /// input buffers
  ChannelOutputBuffer::sptr _inbuf;

  /// work thread
  std::thread _work_thread;
  std::atomic_bool _running;
  std::condition_variable _cv;
  std::mutex _mtx;
  bool _waiting;

  /// Cyclic prefix length (pre-header)
  size_t _ph_Nc;
  /// OFDM symbol length (pre-header)
  size_t _ph_Ns;
  /// Number of OFDM symbols in pre-header
  size_t const _num_pre_header_ofdm_syms;
  /// Threshold for sync metric detection
  float const _threshold;

  /// Cyclic prefix length (current)
  size_t _Nc;
  /// OFDM symbol length (current)
  size_t _Ns;

  /// Current estimate of fine frequency offset
  float _domega;
  /// Phase used for rotation
  fcomplex _phi;
  /// Capture snr and freq offset w/o modification
  SCInfo _sc_info;

  /// Number of items to consume on failed pre_header decode
  size_t _searching_skip;
  /// number of symbols remaining to output for a pre-header
  size_t _pre_header_num_syms_remaining;
  /// OFDM frame currently being processed
  DFTSOFDMFrame::sptr _current_frame;
  /// Number of OFDM symbols in payload
  size_t _num_payload_ofdm_syms;
  /// number of symbols remaining to output for a payload
  size_t _payload_num_syms_remaining;

  std::deque<float> _Md_history;
  float _current_Md;
  int _avg_len;

  demodfn _demod;
  std::vector<fcomplex> _hdr_buf;
  std::vector<fcomplex> _payl_buf;

  size_t output_ofdm_symbols(fcomplex *out, size_t nout, fcomplex const *r,
                             size_t nin, size_t max_num_ofdm_syms);

  size_t detect_preamble(float &domega, float &Md_t, ssize_t &t,
                         float const *Md, gr_complex const *Pd, size_t const N);

  size_t work(int navail, fcomplex const *r, float const *Md,
              fcomplex const *Pd);
};

// sliding window median filtering
template <typename T> class SlidingWindowFilter {
public:
  SlidingWindowFilter(float window_sec, T min, T max, size_t nbins)
      : _f(window_sec, min, max, nbins) {} // FIXME std::chrono?
  T filter(T const &value) {
    std::lock_guard<decltype(_m)> l(_m);
    _f.push(value);
    if (_f.size() == 1) {
      // FIXME: median filter should take care of this case
      return value;
    } else {
      return _f.median();
    }
  }
  void flush() {
    std::lock_guard<decltype(_m)> l(_m);
    _f.flush();
  }
  size_t size() {
    std::lock_guard<decltype(_m)> l(_m);
    _f.size();
  }

private:
  stats::Median<T> _f;
  std::mutex _m;
};

typedef SlidingWindowFilter<float> VarianceFilter;

// unified deframer function signature
typedef std::function<bool(DFTSOFDMFrame::sptr frame,
                           boost::asio::const_buffer bits, float snr,
                           float noiseVar, size_t rx_chain_idx,
                           size_t resource_idx, frame_detect_demux *src_fdd)>
    deframer;

// Interleaving/Deinterleaving (shared by Tx and Rx)
class Interleaver {
public:
  // playing fun and games with names
  static Interleaver get;

  Interleaver();
  std::vector<size_t> const &operator()(size_t n);

private:
  std::map<size_t, std::vector<size_t>> _map;
  std::mutex _mtx;
};

//
// Demodulation
//

/// Header demodulator
class Demodulator {
private:
  // resources: how many?
  int _nthreads;
  // resources: memory and objects
  struct Resource {
    // demod memory resources
    std::vector<fcomplex> fft_inbuf;   // pre-FFT samples
    std::vector<fcomplex> fft_outbuf;  // pre-FFT samples
    std::vector<fcomplex> ifft_inbuf;  // pre-IFFT samples
    std::vector<fcomplex> ifft_outbuf; // post-IFFT samples
    std::vector<fcomplex> inter_outbuf;// post-deinterleaved samples
    std::vector<fcomplex> chanest;     // channel estimate
    std::vector<float> llrs;           // demodulated LLRs
    std::vector<uint8_t> decoded_bits; // decoded bits
    // demod objects
    fft::FFT::sptr fft;
    fft::FFT::sptr ifft;
    std::map<MCS::CodePair, yaldpc::Decoder<bamradio::llr_type, uint8_t>::sptr>
        decoders;

    // construct resources from the header description
    Resource(DFTSOFDMFrameParams const &params, MCS::Name const &mcs);
    ~Resource();
    // initialize resources to zero and make sure they are sufficient
    void init(DFTSOFDMFrameParams const &params, MCS::Name const &mcs);
  };
  std::vector<Resource *> _resources;
  std::mutex _mtx;
  // resources: work threads
  boost::asio::io_context _ioctx;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _ioctx_work;
  std::vector<std::thread> _work_threads;

  // header parameters
  std::map<int, DFTSOFDMFrameParams> const _header_params;
  MCS::Name const _header_mcs;

  // misc
  deframer _deframe;

  // carrier demap helper struct
  struct carrier_demap_plan_t {
    fcomplex *in;
    fcomplex *out;
    int32_t *alloc;
    size_t ncarriers;
    size_t symbol_len;
  };

  // signal processing
  int _fft(std::vector<fcomplex> const &in, Demodulator::Resource *r,
           std::vector<OFDMSymbolParams *> const &symbols);
  int _estimate_channel(Demodulator::Resource *r,
                        std::vector<OFDMSymbolParams *> const &symbols,
                        bool interp_freq, bool interp_time, float pilot_gain);
  int _estimate_snr(Demodulator::Resource *r,
                    std::vector<OFDMSymbolParams *> const &symbols,
                    float &snr_db, float &normalized_var);
  int _equalize(Demodulator::Resource *r,
                std::vector<OFDMSymbolParams *> const &symbols);
  int _demap_subcarriers(Demodulator::Resource *r,
                         std::vector<OFDMSymbolParams *> &symbols,
                         size_t dft_spread_length);
  int _ifft(Demodulator::Resource *r,
            std::vector<OFDMSymbolParams *> const &symbols,
            size_t dft_spread_length);
  int _deinterleave(Demodulator::Resource *r,
                    size_t nsymb);
  int _llr_map(Demodulator::Resource *r,
               std::vector<OFDMSymbolParams *> const &symbols, float snr,
               size_t dft_spread_length);
  int _decode(Demodulator::Resource *r, DFTSOFDMFrame *frame, MCS::Name mcs,
              int nllr);

public:
  // attempt to demodulate. non-blocking, pushes demod work on the work queue.
  void demod(DFTSOFDMFrame::sptr frame, std::vector<fcomplex> &&raw_samples,
             SCInfo sc_info, int tx_os, int rx_chain_idx,
             std::shared_ptr<VarianceFilter> snr_filt,
             frame_detect_demux *fdd_src);

  MCS::Name header_mcs() const { return _header_mcs; }
  std::map<int, DFTSOFDMFrameParams> const &header_params() const {
    return _header_params;
  }

  Demodulator(std::string name_prefix, MCS::Name headerMcs, int nthreads,
              deframer deframe);
  ~Demodulator();
  Demodulator(Demodulator const &other) = delete;
};

/// OFDM receiver phy
class phy_rx {
public:
  // read from a ring buffer holding complex samples
  typedef ringbuffer::Ringbuffer<
      fcomplex, ringbuffer::rb_detail::memfd_nocuda_circbuf<fcomplex>>::sptr
      inbuf_ptr;

  // usual jazz
  typedef std::shared_ptr<phy_rx> sptr;
  static sptr make(MCS::Name header_mcs, int header_nthreads,
                   int payload_nthreads, deframer header_deframer,
                   deframer payload_deframer) {
    return std::shared_ptr<phy_rx>(new phy_rx(header_mcs, header_nthreads,
                                              payload_nthreads, header_deframer,
                                              payload_deframer));
  }

  // connect the RX to the input buffers FIXME make this use the radiocontroller
  // Channel type to get oversample rate...
  void connect(std::vector<ChannelOutputBuffer::sptr> const &inbufs,
               std::vector<int> const &tx_os);

  bool connected() const;

  void setOversampleRate(int which, int os);

  void flushFilters();

private:
  phy_rx(MCS::Name header_mcs, int header_nthreads, int payload_nthreads,
         deframer header_deframer, deframer payload_deframer);

  // number of channels to receive on
  size_t _nchan;

  // transmit oversample rates per rx channels
  std::vector<int> _tx_os;

  // frame demuxing
  std::vector<frame_detect_demux::sptr> _fdds;

  // header and payload demodulators
  Demodulator _hdr_demod;
  Demodulator _payl_demod;

  // SNR filters
  std::vector<std::shared_ptr<VarianceFilter>> _hdr_snr_filt;
  std::vector<std::shared_ptr<VarianceFilter>> _payl_snr_filt;
};

class phy_tx {
public:
  typedef std::shared_ptr<phy_tx> sptr;

  static sptr make(size_t nthreads) {
    return std::shared_ptr<phy_tx>(new phy_tx(nthreads));
  }

  phy_tx(size_t nthreads);
  ~phy_tx();
  phy_tx(phy_tx const &other) = delete;

  /// connect a tx streamer to the PHY
  void connect(uhd::tx_streamer::sptr stream);

  struct prepared_frame {
    DFTSOFDMFrame::sptr frame;
    Channel channel;
  };

  typedef std::function<prepared_frame()> frame_maker;
  void start(frame_maker fm);
  void continue_frame_maker();
  void stop();

protected:
  /// compute the sin^2 window
  static std::vector<float> &get_window(bool head, int N, int os);

  /// UHD streamer and mutex to protect
  uhd::tx_streamer::sptr _stream;
  bool _connected;
  std::mutex _smtx;
  std::condition_variable _stream_cv; // MUTEX _smtx
  bool _stream_avail;                 // MUTEX _smtx

  /// Resources
  // how many?
  size_t const _nthreads;
  // memory and other objects
  struct Resource {
    // memory
    std::vector<uint8_t> raw_bits;        // raw bits
    std::vector<uint8_t> coded_bits;      // coded bits
    std::vector<fcomplex> symbols;        // constellation symbols
    std::vector<fcomplex> spread_symbols; // spread constellation
    std::vector<fcomplex> inter_symbols;  // interleaved spread symbols
    std::vector<fcomplex> ifft_in;        // spread symbols + pilots
    std::vector<fcomplex> cp_buf;         // scratch buffer for CP computation
    std::vector<fcomplex> out;            // scratch buffer for output samples
    // demod objects
    fft::FFT::sptr const fft;
    fft::FFT::sptr const ifft;
    std::vector<yaldpc::Encoder::sptr> const encoders;

    // ctor and initialization
    Resource();
    void init(DFTSOFDMFrameParams const &params);
  };
  std::vector<Resource *> const _resources;

  std::vector<std::thread> _work_threads; // MUTEX _thrwmtx
  bool _work_threads_running;             // MUTEX _wmtx
  size_t _work_threads_avail;             // MUTEX _wmtx
  std::mutex _wmtx, _thrwmtx;
  std::condition_variable _work_cv; // MUTEX _wmtx

  frame_maker _fm; // MUTEX _wmtx

  // padding bits
  std::vector<uint8_t> const _random_bits;

  // this is a constant for now. The number of constellation symbols in the header.
  size_t const _header_num_data_sym;

  // signal processing
  int encode_bits(DFTSOFDMFrame::sptr frame, DFTSOFDMFrameParams const &params,
                  int ridx);
  int modulate_bits(std::vector<OFDMSymbolParams *> const &symbols, int ridx);
  int interleave(size_t nsymb, int ridx);
  int spread_symbols(std::vector<OFDMSymbolParams *> const &symbols,
                     DFTSOFDMFrameParams const &params, int ridx);
  int map_subcarriers(std::vector<OFDMSymbolParams *> const &symbols, int ridx);
  int shift_upsample(std::vector<OFDMSymbolParams *> const &symbols,
                     std::vector<fcomplex> &dest, int ridx);
  int window_cp(std::vector<OFDMSymbolParams *> const &symbols,
                std::vector<fcomplex> &samples, int ridx);
  int freq_shift(std::vector<fcomplex> &samples, double const rot_phase,
                 size_t n);
  int multiply_const(std::vector<fcomplex> &samples, float pc_factor, size_t n);
  // put samples in air
  uint64_t stream(std::vector<fcomplex> const &samples, size_t n);

  // send function implementation
  void _send(prepared_frame const &pf);
};

} // namespace ofdm

//
// Control channel PHY
//
namespace controlchannel {

// Usage:
//
// phy_ctrl_mod::Resource r;
// std::vector<fcomplex> s;
// phy_ctrl_mod m( [see below] );
// m.mod(cc_data, &r, out);

class phy_ctrl_mod {
public:
  /// shared_ptr convention
  typedef std::shared_ptr<phy_ctrl_mod> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<phy_ctrl_mod>(std::forward<Args>(args)...);
  }

  /// memory and other stuff needed to make a control channel frame. instantiate
  /// one of these outside and pass as argument to the mod(...) function
  /// below. keep around for re-use.
  struct Resource {
    std::vector<uint8_t> raw_data;            // serialized bytes
    std::vector<uint8_t> bytes;               // serialized bytes + CRC32
    std::vector<uint8_t> rs_buf;              // rs scratch space
    std::vector<uint8_t> coded_with_preamble; // encoded bytes + preamble
    uint8_t *rs_start;                        // ptr into cwp array for encoder
    std::vector<uint8_t> chunks;              // re-packed bits
    std::vector<float> symbols;               // pulse-shaped symbols
    boost::crc_optimal<32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, true, true> _crc;
  };

  /// eternally hardcoded things
  static const std::vector<uint8_t> preamble;
  static const std::vector<fcomplex> pulse_shape;

  /// construct one of these
  phy_ctrl_mod(float sample_rate, float bandwidth, size_t rs_k, size_t npoints,
               float scale);

  /// turn CCData into signal
  size_t mod(CCData::sptr cc_data, float const freq_offset_hz, Resource *r,
             std::vector<fcomplex> &out) const;

protected:
  float const _sample_rate;
  float const _bandwidth;
  float const _scale;
  size_t const _rs_k;
  size_t const _npoints;
  size_t const _bps;

  std::vector<float> const _freq_table_normalized;

  // signal processing
  size_t _crc32(Resource *r) const;
  size_t _encode(size_t ncrc, Resource *r) const;
  size_t _unpack(size_t nbytes, Resource *r) const;
  size_t _mfsk_mod(size_t nsymb, float const freq_offset_hz, Resource *r,
                   std::vector<fcomplex> &out) const;
};

} // namespace controlchannel

} // namespace bamradio

#endif // e1b89b720965927782c6
