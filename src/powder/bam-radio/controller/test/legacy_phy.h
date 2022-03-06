// Legacy PHY stuff for testing
//
// This is where old code goes to die
//
// Copyright (c) 2018 Dennis Ogbe, Stephen Larew, Tomohiro Arakawa

#ifndef b99037518cfd6533a34d
#define b99037518cfd6533a34d

#include <gnuradio/fft/fft.h>
#include <pmt/pmt.h>

#include <numeric>
#if BAMOFDM_SYNC_DEBUG
#include <gnuradio/blocks/file_sink.h>
#endif

#include <gnuradio/gr_complex.h>
#include <gnuradio/hier_block2.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/top_block.h>

namespace bamradio {
namespace ofdm {

//
// GNU Radio Frame detection and Synchronization
//

namespace legacy {
//!
//! \brief Time and fine frequency offset synchronization.
//! \ingroup ofdm
//!
class sc_sync : public gr::hier_block2 {
public:
  typedef boost::shared_ptr<sc_sync> sptr;

  static const std::vector<gr_complex> &get_preamble_td_oversampled(size_t L,
                                                                    size_t O);
  static const std::vector<gr_complex> &get_preamble_pilot_symbols(size_t L,
                                                                   size_t O);

  //!
  //! \brief Return a shared_ptr to a new instance of bamofdm::sc_sync.
  //!
  //! \param L length of sync sequence (=> 2L = length of twice repeated sync
  //! sequence)
  //! \param Nc length of cyclic prefix
  //!
  static sptr make(size_t const L, size_t const Nc);

private:
  size_t const _L;
  size_t const _Nc;

  sc_sync(size_t const L, size_t const Nc);
};

/*!
 * \brief Detect a frame and then demux frequency corrected frame sections.
 * \ingroup ofdm
 *
 * Inputs:
 * 1. Received samples
 * 2. Synchronization metric Md[n]
 * 3. Frequency offset function Pd[n]
 *
 * Outputs:
 * Each output stream is a sequence of frequency corrected OFDM symbols.
 * 1. Preamble and header (pre-header)
 * 2. Payload
 */
class frame_detect_demux : public gr::sync_block {
  // n.b. <2018-03-20 Tue> this block does not produce samples any more. my
  // knee-jerk reaction is that it should thus become a sync block. I am keeping
  // it a basic block because I assume that a sync block with multiple inputs
  // always puts the same number of inputs in the input buffers, which is not
  // necessarily what we want. This question is FFS.
  //
public:
  typedef boost::shared_ptr<frame_detect_demux> sptr;

  // function signature for the demod(...) function call. pointer to the frame
  // object (if nullptr, assume we are demodulating a header), rvalue reference
  // to a vector of complex samples, the snr detected during sync, and a pointer
  // back to the FDD that called this function in the first place
  typedef std::function<void(DFTSOFDMFrame::sptr, std::vector<fcomplex> &&,
                             float snr, frame_detect_demux *)>
      demodfn;

  /// Port ID for receiving pre-header messages.
  static pmt::pmt_t const pre_header_port_id;

  /// Port ID for publishing channel use info
  static pmt::pmt_t const channel_use_port_id;

  size_t get_pre_header_cyclic_prefix_length() const;
  size_t get_pre_header_ofdm_symbol_length() const;
  size_t get_pre_header_num_ofdm_symbols() const;

  /*!
   * \brief Return a shared_ptr to a new instance of
   * bamofdm::frame_detect_demux.
   *
   * \param Nc Number of samples in a pre-header cyclic prefix
   * \param Ns Number of samples in a pre-header OFDM symbol
   * \param num_pre_header_ofdm_syms Number of OFDM symbols in a pre-header
   * \param preamble_threshold Threshold in (0,1) for detecting a preamble
   * \param snr_tag_key Tag key for SNR estimate derived from preamble
   * \param interpret_frame Function to interpret the payload message sent to
   * the pre_header message port.
   */
  static sptr make(size_t Nc, size_t Ns, size_t num_pre_header_ofdm_syms,
                   float preamble_threshold, demodfn demod,
                   interpret_frame interpret_frame);

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

private:
  frame_detect_demux(size_t Nc, size_t Ns, size_t num_pre_header_ofdm_syms,
                     float preamble_threshold, demodfn demod,
                     interpret_frame interpret_frame);

  interpret_frame _interpret_frame;

  enum class State { SEARCHING, PRE_HEADER, PRE_HEADER_WAIT, PAYLOAD } _state;

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
  gr_complex _phi;
  /// Current SNR estimate
  float _SNR;

  /// Number of items to consume on failed pre_header decode
  size_t _searching_skip;

  /// number of symbols remaining to output for a pre-header
  size_t _pre_header_num_syms_remaining;

  /// feedback message indication successful header demodulation
  pmt::pmt_t _pre_header_wait_msg;

  /// OFDM frame currently being processed
  DFTSOFDMFrame::sptr _current_frame;

  /// Number of OFDM symbols in payload
  size_t _num_payload_ofdm_syms;

  /// number of symbols remaining to output for a payload
  size_t _payload_num_syms_remaining;

  void handle_pre_header_msg(pmt::pmt_t msg);

  size_t output_ofdm_symbols(gr_complex *out, size_t nout, gr_complex const *r,
                             size_t nin, size_t max_num_ofdm_syms);

  size_t detect_preamble(float &domega, float &SNR, ssize_t &t, float const *Md,
                         gr_complex const *Pd, size_t N);
#if 0 // FIXME rx time is not part of FDD any more...
  std::pair<uint64_t, ::uhd::time_spec_t> _last_rx_time_tag;
  // ((uint64:secs,double:frac-secs), uint64:sample-offset)
  pmt::pmt_t rx_time(std::vector<gr::tag_t> const &tags, uint64_t o) const;
#endif

  std::deque<float> _Md_history;
  float _current_Md;
  int _avg_len;

  demodfn _demod;
  std::vector<gr_complex> _hdr_buf;
  std::vector<gr_complex> _payl_buf;
};
} // namespace legacy
} // namespace ofdm
} // namespace bamradio

#endif // b99037518cfd6533a34d
