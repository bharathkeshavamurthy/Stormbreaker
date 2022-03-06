#include "common.h"
#include "phy.h"

#include "legacy_phy.h"

namespace bamradio {
namespace ofdm {
namespace legacy {

//
// Schmidl & Cox OFDM Synchronization
//

sc_sync::sptr sc_sync::make(size_t const L, size_t const Nc) {
  return gnuradio::get_initial_sptr(new sc_sync(L, Nc));
}

sc_sync::sc_sync(size_t const L, size_t const Nc)
    : gr::hier_block2(
          "sc_sync", gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::makev(2, 2, {sizeof(float), sizeof(gr_complex)})),
      _L(L), _Nc(Nc) {

  if (_L <= 1) {
    // throw std::logic_error("sc_sync: L <= 1");
    abort();
  }

  // Compute P(d)
  // P(d) = sum_{i=0}^{L-1} r[d+L+i] * conj(r[d+i])

  // o[d] = r[d+L] * conj(r[d])
  auto const mul_conj_delay =
      gr::bamofdm::lambda_sync_block_11<gr_complex, gr_complex>::make(
          [L](auto b) {
            b->set_tag_propagation_policy(gr::block::TPP_DONT);
            b->set_history(L + 1);
            b->declare_sample_delay(L);
          },
          [L](auto, auto in, auto out, auto N) {
            volk_32fc_x2_multiply_conjugate_32fc(out, in + L, in, N);
            return N;
          });

  // o[n] = sum_{i=0}^{L-1} r[n+i]
  // TODO: determine if max_noutput_items needs to be set for numerical
  // stability
  auto const Pd =
      gr::bamofdm::lambda_sync_block_11<gr_complex, gr_complex>::make(
          [L](auto b) {
            b->set_tag_propagation_policy(gr::block::TPP_DONT);
            b->set_history(L);
            b->declare_sample_delay(L - 1);
            // b->set_max_noutput_items(256);
          },
          [L](auto, auto in, auto Pd, auto N) {
            gr_complex s = std::accumulate(in, in + L, gr_complex(0.0f, 0.0f));
            Pd[0] = s;
            bool reset = false;
            for (size_t i = 1; i < N; ++i) {
              bool nreset = std::norm(in[0] / in[L]) > (500.0f * 500.0f);
              if (reset && !nreset) {
                return i;
              } else {
                reset = nreset;
              }
              s = s - in[0] + in[L];
              ++in;
              Pd[i] = s;
            }
            return N;
          });

  connect(self(), 0, mul_conj_delay, 0);
  connect(mul_conj_delay, 0, Pd, 0);
#if BAMOFDM_SYNC_DEBUG
  connect(mul_conj_delay, 0,
          gr::blocks::file_sink::make(sizeof(gr_complex),
                                      (alias() + "_mul_conj_delay.out").c_str(),
                                      false),
          0);
  connect(Pd, 0,
          gr::blocks::file_sink::make(sizeof(gr_complex),
                                      (alias() + "_Pd.out").c_str(), false),
          0);
#endif

  // Compute (R(d))^2
  // R(d) = sum_{i=0}^{L-1} |r2[d+L+i]|^2

  // o[d] = |r[d+L]|^2 = r[d+L] * conj(r[d+L])
  auto const samp_energy_delay =
      gr::bamofdm::lambda_sync_block_11<gr_complex, float>::make(
          [L](auto b) {
            b->set_tag_propagation_policy(gr::block::TPP_DONT);
            b->set_history(L + 1);
            b->declare_sample_delay(L);
          },
          [L](auto, auto in, auto out, auto N) {
            volk_32fc_magnitude_squared_32f(out, in + L, N);
            return N;
          });

  // o[n] = | sum_{i=0}^{L-1} r[n+i] |^2
  // TODO: determine if max_noutput_items needs to be set for numerical
  // stability
  auto const Rd2 = gr::bamofdm::lambda_sync_block_11<float, float>::make(
      [L](auto b) {
        b->set_tag_propagation_policy(gr::block::TPP_DONT);
        b->set_history(L);
        b->declare_sample_delay(L - 1);
        // b->set_max_noutput_items(256);
      },
      [L](auto, auto in, auto Rd2, auto N) {
        float s = std::accumulate(in, in + L, 0.0f);
        Rd2[0] = s * s;
        bool reset = false;
        for (size_t i = 1; i < N; ++i) {
          bool nreset = in[0] / in[L] > 500.0f;
          if (reset && !nreset) {
            return i;
          } else {
            reset = nreset;
          }
          s = s - in[0] + in[L];
          ++in;
          Rd2[i] = s * s;
        }
        return N;
      });

  connect(self(), 0, samp_energy_delay, 0);
  connect(samp_energy_delay, 0, Rd2, 0);
#if BAMOFDM_SYNC_DEBUG
  connect(
      samp_energy_delay, 0,
      gr::blocks::file_sink::make(
          sizeof(float), (alias() + "_samp_energy_delay.out").c_str(), false),
      0);
  connect(Rd2, 0,
          gr::blocks::file_sink::make(sizeof(float),
                                      (alias() + "_Rd2.out").c_str(), false),
          0);
#endif

  // Compute M(d)
  // M(d) = |P[d]|^2 / (R(d))^2

  // o[n] = |r0[n]|^2 / r1[n]
  auto const Md =
      gr::bamofdm::lambda_sync_block_21<gr_complex, float, float>::make(
          [](auto b) { b->set_tag_propagation_policy(gr::block::TPP_DONT); },
          [L](auto, auto Pd, auto Rd2, auto Md, auto N) {
            // Use Md (output buffer) as temp store for |Pd|^2
            volk_32fc_magnitude_squared_32f(Md, Pd, N);
            volk_32f_x2_divide_32f(Md, Md, Rd2, N);
            return N;
          });

  connect(Pd, 0, Md, 0);
  connect(Rd2, 0, Md, 1);
#if BAMOFDM_SYNC_DEBUG
  connect(Md, 0,
          gr::blocks::file_sink::make(
              sizeof(float), (symbol_name() + "_Md.out").c_str(), false),
          0);
#endif

  connect(Md, 0, self(), 0);
  connect(Pd, 0, self(), 1);
}

#define is_even(v) (((v) & static_cast<decltype(v)>(1)) == 0)
#define power_of_2(v) ((v) && !((v) & ((v)-1)))

const std::vector<gr_complex> &
sc_sync::get_preamble_pilot_symbols(const size_t L, size_t const O) {
  // Compute the peamble only once for a given L;
  static std::map<std::pair<size_t, size_t>, std::vector<gr_complex>> p;
  auto &pi = p[{L, O}];
  if (pi.size() > 0) {
    return pi;
  }

  // sync this with get_preamble_td
  const unsigned int M = 3; // gcd(M,N) = 1
  pi = gr::bamofdm::generate_cazac_seq(L, M);

  auto const ntds = O * 2 * L;

  // Insert zeros in CAZAC
  std::vector<gr_complex> cazac0(ntds, gr_complex(0.0, 0.0));
  for (size_t i = 0; i < L / 2; ++i) {
    cazac0[i * 2] = pi[i];
    cazac0[ntds - L + i * 2] = pi[L / 2 + i];
  }

  // Take IFFT of cazac.
  gr::fft::fft_complex ifft(ntds, false);
  memset(ifft.get_inbuf(), 0x00, sizeof(gr_complex) * ifft.inbuf_length());
  std::copy(cazac0.begin(), cazac0.end(), ifft.get_inbuf());
  ifft.execute();

  // Copy out IFFT result
  std::vector<gr_complex> preamble(ntds);
  std::copy_n(ifft.get_outbuf(), ntds, preamble.begin());

  // get max value after ifft as scalar
  auto const scalar = [&] {
    // i'm so sorry, this is terrible
    auto r = std::numeric_limits<double>::lowest();
    for (auto const &a : preamble) {
      auto const b = std::abs(a);
      if (b > r)
        r = b;
    }
    return r;
  }();

  // Ensure proper scaling of the channel estimate at the receiver. Since we are
  // using FFTW to do a reverse FFT (at the transmitter) and a forward FFT (at
  // the receiver), the received sequence is effectively scaled by 2 * L (the
  // length of the two FFTs). This would scale our channel estimate, if we would
  // not correct for that here.
  // auto const scalar = std::abs(preamble[0]) / (2 * L);

  for (auto &x : pi) {
    x /= scalar;
  }

  return pi;
}

const std::vector<gr_complex> &
sc_sync::get_preamble_td_oversampled(const size_t L, size_t const O) {
  // Compute the peamble only once for a given L;
  static std::map<std::pair<size_t, size_t>, std::vector<gr_complex>> preambles;

  auto &preamble = preambles[{L, O}];

  if (preamble.size() > 0) {
    return preamble;
  }

  // TODO generalize to arbitrary fft_len and set M correctly.

  if (!power_of_2(L)) {
    // throw std::logic_error("get_preamble_td: L not power of 2.");
    abort();
  }
  const unsigned int M = 3; // gcd(M,N) = 1

  // Generate CAZAC sequence.
  auto const cazac = gr::bamofdm::generate_cazac_seq(L, M);

  auto const ntds = O * 2 * L;

  // Insert zeros in CAZAC
  std::vector<gr_complex> cazac0(ntds, gr_complex(0.0, 0.0));
  for (size_t i = 0; i < L / 2; ++i) {
    cazac0[i * 2] = cazac[i];
    cazac0[ntds - L + i * 2] = cazac[L / 2 + i];
  }

  // Take IFFT of cazac.
  gr::fft::fft_complex ifft(ntds, false);
  memset(ifft.get_inbuf(), 0x00, sizeof(gr_complex) * ifft.inbuf_length());
  std::copy(cazac0.begin(), cazac0.end(), ifft.get_inbuf());
  ifft.execute();

  // Copy out IFFT result
  preamble.resize(ntds);
  std::copy_n(ifft.get_outbuf(), ntds, preamble.begin());

  // get max value after ifft as scalar
  auto const scalar = [&] {
    // i'm so sorry, this is terrible
    auto r = std::numeric_limits<double>::lowest();
    for (auto const &a : preamble) {
      auto const b = std::abs(a);
      if (b > r)
        r = b;
    }
    return r;
  }();

  for (auto &x : preamble) {
    x /= scalar;
  }

  return preamble;
}

//
// Frame Detection
//

frame_detect_demux::sptr
frame_detect_demux::make(size_t Nc, size_t Ns, size_t num_pre_header_ofdm_syms,
                         float preamble_threshold, demodfn demod,
                         interpret_frame interpret_frame) {
  return gnuradio::get_initial_sptr(
      new frame_detect_demux(Nc, Ns, num_pre_header_ofdm_syms,
                             preamble_threshold, demod, interpret_frame));
}

pmt::pmt_t const frame_detect_demux::pre_header_port_id =
    pmt::intern("pre_header");
pmt::pmt_t const frame_detect_demux::channel_use_port_id =
    pmt::intern("channel_use");

frame_detect_demux::frame_detect_demux(size_t Nc, size_t Ns,
                                       size_t num_pre_header_ofdm_syms,
                                       float preamble_threshold, demodfn demod,
                                       interpret_frame interpret_frame)
    : gr::sync_block(
          "frame_detect_demux",
          gr::io_signature::makev(
              3, 3, {sizeof(gr_complex), sizeof(float), sizeof(gr_complex)}),
          gr::io_signature::make(0, 0, 0)),
      _interpret_frame(interpret_frame), _state(State::SEARCHING), _ph_Nc(Nc),
      _ph_Ns(Ns), _num_pre_header_ofdm_syms(num_pre_header_ofdm_syms),
      _threshold(preamble_threshold), _Nc(Nc), _Ns(Ns), _domega(0.0f),
      _phi(1.0f, 0.0f), _SNR(0.0f), _searching_skip(0),
      _pre_header_num_syms_remaining(0), _pre_header_wait_msg(nullptr),
      _current_frame(nullptr), _num_payload_ofdm_syms(0),
      _payload_num_syms_remaining(0), _current_Md(0.0), _avg_len(5),
      _demod(demod) {

  // this block terminates
  set_tag_propagation_policy(gr::block::TPP_DONT);

  message_port_register_in(pre_header_port_id);
  set_msg_handler(
      pre_header_port_id,
      boost::bind(&legacy::frame_detect_demux::handle_pre_header_msg, this,
                  _1));
  message_port_register_out(channel_use_port_id);
}

inline ::uhd::time_spec_t pmt_to_time_spec(pmt::pmt_t const &p) {
  return uhd::time_spec_t(pmt::to_uint64(pmt::tuple_ref(p, 0)),
                          pmt::to_double(pmt::tuple_ref(p, 1)));
}
inline pmt::pmt_t time_spec_to_pmt(::uhd::time_spec_t const &t) {
  return pmt::make_tuple(pmt::from_uint64(t.get_full_secs()),
                         pmt::from_double(t.get_frac_secs()));
}

#if 0 // see header file
pmt::pmt_t frame_detect_demux::rx_time(std::vector<gr::tag_t> const &tags,
                                       uint64_t o) const {
  auto last_rx_time_tag = _last_rx_time_tag;

  for (auto const &t : tags) {
    if (o > t.offset) {
      last_rx_time_tag = {t.offset, pmt_to_time_spec(t.value)};
    } else {
      break;
    }
  }

  if (o < last_rx_time_tag.first) {
    return pmt::PMT_NIL;
  }

  return pmt::make_tuple(time_spec_to_pmt(last_rx_time_tag.second),
                         pmt::from_uint64(o - last_rx_time_tag.first));
}
#endif

int frame_detect_demux::work(int noutput_items,
                             gr_vector_const_void_star &input_items,
                             gr_vector_void_star &output_items) {
  auto r = (gr_complex const *)input_items[0];
  auto Md = (float const *)input_items[1];
  auto Pd = (gr_complex const *)input_items[2];

  auto const Nin = static_cast<size_t>(std::max<int>(
      0, noutput_items - (_state == State::SEARCHING ? (int)_Ns - 1 : 0)));
  if (Nin == 0) {
    return 0;
  }

  size_t nin = Nin;

#define FDDCONSUME(n)                                                          \
  do {                                                                         \
    size_t const nn = (n);                                                     \
    r += nn;                                                                   \
    Md += nn;                                                                  \
    Pd += nn;                                                                  \
    nin -= nn;                                                                 \
  } while (false)

#if 0 // see header file
  static pmt::pmt_t uhd_rx_time_tag_key = pmt::intern("rx_time");
  std::vector<gr::tag_t> rx_time_tags;
  get_tags_in_window(rx_time_tags, 0, 0, Nin, uhd_rx_time_tag_key);
  std::sort(rx_time_tags.begin(), rx_time_tags.end(),
            gr::tag_t::offset_compare)
#endif

  switch (_state) {
  // Look for preamble on the input.
  // If preamble *not* found:
  // - Consume all of input.
  // - Produce nothing.
  // - Next state: SEARCHING
  // If preamble found:
  // - Consume up to the trigger point.
  // - Produce nothing.
  // - Next state: PRE_HEADER
  case State::SEARCHING: {
    ssize_t t;
    float Md_t = 0.0;
    auto const nskip = detect_preamble(_domega, Md_t, t, Md, Pd, nin);

    if (t < 0) {
      // no preamble was detected
      // assert(nskip == nin);
      // consume
      FDDCONSUME(nskip);
      break;
    }

    if (_Md_history.empty()) {
      for (int i = 0; i < _avg_len; ++i)
        _Md_history.push_back(Md_t);
      _current_Md = Md_t;
    } else {
      _Md_history.push_back(Md_t);
      _current_Md += (_Md_history.back() - _Md_history.front()) / _avg_len;
      _Md_history.pop_front();
    }
    _SNR =
        10 * std::log10(std::sqrt(_current_Md) / (1 - std::sqrt(_current_Md)));

    // preamble detected to start at t

    // when we consume, probably need to consume at a minimum slightly more
    // than t, so remember that minimium amount to consume
    assert(nskip >= (size_t)t);
    _searching_skip = nskip - t;

    // consume up to the trigger
    FDDCONSUME(t);

#if 0 // see header file
    // msg: (rx_time:start_time, duration) tuple of channel occupation
    auto cumsg = pmt::make_tuple(
        rx_time(rx_time_tags, nitems_read(0) + Nin - nin),
        pmt::from_uint64(_num_pre_header_ofdm_syms * (_Nc + _Ns)));
    message_port_pub(channel_use_port_id, cumsg);
#endif
    // produce nothing

    // Transition to next state
    _state = State::PRE_HEADER;
    _pre_header_num_syms_remaining = _num_pre_header_ofdm_syms;
    _hdr_buf.clear();
    _hdr_buf.resize(_num_pre_header_ofdm_syms * _ph_Ns);
  } break;
  // Output freq. corrected, guard dropped OFDM symbols of pre-header.
  // If symbols left to output:
  // - Consume nothing
  // - Produce expected # pre-header symbols
  // - Write tags on first OFDM symbol
  // - Next state: PRE_HEADER
  // If all symbols output:
  // - Consume & produce nothing
  // - Next state: PRE_HEADER_WAIT
  case State::PRE_HEADER: {
    // NB: PRE_HEADER state never consumes so r points to start of pre-header
    // at every iteration within this state.
    auto const nin_offset = (_Ns + _Nc) * (_num_pre_header_ofdm_syms -
                                           _pre_header_num_syms_remaining);
    auto const nout_offset =
        _Ns * (_num_pre_header_ofdm_syms - _pre_header_num_syms_remaining);

    auto const num_syms_produced = output_ofdm_symbols(
        _hdr_buf.data() + nout_offset, _hdr_buf.size() - nout_offset,
        r + nin_offset, nin - nin_offset, _pre_header_num_syms_remaining);

    _pre_header_num_syms_remaining -= num_syms_produced;

    if (_pre_header_num_syms_remaining == 0) {
      _demod(nullptr, std::move(_hdr_buf), _SNR, this);
      _state = State::PRE_HEADER_WAIT;
      _pre_header_wait_msg = nullptr;
    }
  } break;
  // - Wait for header decode message.
  // - If decode bad:
  //   - consume searching_skip inputs
  //   - produce nothing
  //   - Next state: SEARCHING
  // - If decode good:
  //   - consume pre-header
  //   - produce nothing
  //   - save pre-header dict
  //   - Next state: PAYLOAD
  case State::PRE_HEADER_WAIT: {
    if (_pre_header_wait_msg == nullptr) {
      break;
    } else if (pmt::is_any(_pre_header_wait_msg)) {
      auto const n = _num_pre_header_ofdm_syms * (_Nc + _Ns);
      if (nin < n) {
        break;
      }
      FDDCONSUME(n);

      _current_frame = boost::any_cast<DFTSOFDMFrame::sptr>(
          pmt::any_ref(_pre_header_wait_msg));
      _interpret_frame(_current_frame, _num_payload_ofdm_syms, _Ns, _Nc);

#if 0 // see header file
      // notify channel use
      // msg: (rx_time:start_time, duration) tuple of channel occupation
      auto cumsg = pmt::make_tuple(
          rx_time(rx_time_tags, nitems_read(0) + Nin - nin),
          pmt::from_uint64(_num_payload_ofdm_syms * (_Nc + _Ns)));
      message_port_pub(channel_use_port_id, cumsg);
#endif

      _state = State::PAYLOAD;
      _searching_skip = 0;
      _payload_num_syms_remaining = _num_payload_ofdm_syms;
      _payl_buf.clear();
      _payl_buf.resize(_num_payload_ofdm_syms * _Ns);
    } else {
      if (nin < _searching_skip) {
        break;
      }
      FDDCONSUME(_searching_skip);

      _state = State::SEARCHING;
      _searching_skip = 0;
      _pre_header_wait_msg = nullptr;
      _current_frame = nullptr;
    }
  } break;
  // Output freq. corrected, guard dropped OFDM symbols of payload.
  // If symbols left to output:
  // - Consume cyc prefixed symbols
  // - Produce expected # payload symbols.
  // - Next state: PAYLOAD
  // If all symbols output:
  // - Consume & produce nothing
  // - Next state: SEARCHING
  case State::PAYLOAD: {
    auto const nout_offset =
        _Ns * (_num_payload_ofdm_syms - _payload_num_syms_remaining);
    auto const num_syms_produced = output_ofdm_symbols(
        _payl_buf.data() + nout_offset, _payl_buf.size() - nout_offset, r, nin,
        _payload_num_syms_remaining);
    auto const n = num_syms_produced * (_Nc + _Ns);
    if (_payload_num_syms_remaining == num_syms_produced) {
      // Do not consume trailing cyclic prefix at end of payload
      FDDCONSUME(n - _Nc);
    } else {
      FDDCONSUME(n);
    }
    _payload_num_syms_remaining -= num_syms_produced;
    if (_payload_num_syms_remaining == 0) {
      _demod(_current_frame, std::move(_payl_buf), _SNR, this);
      _state = State::SEARCHING;
      _pre_header_wait_msg = nullptr;
      _current_frame = nullptr;
      _Nc = _ph_Nc;
      _Ns = _ph_Ns;
    }
  } break;
  }
  return Nin - nin;
}

void frame_detect_demux::handle_pre_header_msg(pmt::pmt_t msg) {
  if (pmt::is_any(msg)) {
    assert(_pre_header_wait_msg == nullptr);
    _pre_header_wait_msg = msg;
  } else if (msg == pmt::PMT_F) {
    _pre_header_wait_msg = msg;
  } else {
    assert(false);
  }
}

size_t frame_detect_demux::output_ofdm_symbols(gr_complex *out, size_t nout,
                                               gr_complex const *r, size_t nin,
                                               size_t max_num_ofdm_syms) {
  size_t k = 0;

  while (nin >= (_Ns + _Nc) && nout >= _Ns && k < max_num_ofdm_syms) {
    auto const domega = gr_complex(cos(-_domega), sin(-_domega));
    volk_32fc_s32fc_x2_rotator_32fc(out, r, domega, &_phi, _Ns);
    ++k;
    out += _Ns;
    nout -= _Ns;
    r += (_Ns + _Nc);
    nin -= (_Ns + _Nc);
    // Advance phi past the cyclic prefix.
    _phi *= gr_complex(cos(-_domega * _Nc), sin(-_domega * _Nc));
  }

  return k;
}

size_t frame_detect_demux::detect_preamble(float &domega, float &Md_t,
                                           ssize_t &t, float const *Md,
                                           gr_complex const *Pd,
                                           size_t const N) {
  // Set t = -1 to indicate no preamble found
  t = -1;

  // n is number of items to consume
  size_t n = 0;

  // Advance the metrics (Md & Pd) past history.
  Md += (_Ns - 1);
  Pd += (_Ns - 1);

  // Detect plateau in Md metric and choose middle point
  while (n < N) {
    // Is Md greater than the threshold?
    if (Md[n] > _threshold) {
      // Ensure enough samples available to detect the plateau and advance.
      if (N - n < (_Nc + _Ns)) {
        break;
      }
      auto a = Md[n];
      // Remember the rising edge of the plateau and advance to the falling
      // edge.
      auto const s = n++;
      while (n < N && Md[n] > _threshold) {
        a += Md[n++];
      }
      // Ensure the plateau is at least 2 samples wide
      // and average value <= ~1
      auto const w = n - s;
      if (w > 1 && w < ((5 * _Nc) / 2) && (a / (float)w <= 1.3f)) {
        // Choose the middle of the detected plateau as the trigger
        t = s + w / (w > 4 ? 4 : 2); // - (2 * _L - 1);
        // t = s + w / 2; // - (2 * _L - 1);

        // Compute the fine frequency offset
        // FIXME Ns/2 should be floating point division
        domega = std::arg(Pd[t]) / (_Ns / 2);
        // Estimate SNR
        // SNR = 10 * std::log10(std::sqrt(Md[t]) / (1 - std::sqrt(Md[t])));
        Md_t = Md[t];

        // n = std::min(n + _Nc, N);
        break;
      }
    } else {
      ++n;
    }
  }

  return n;
}

size_t frame_detect_demux::get_pre_header_cyclic_prefix_length() const {
  return _ph_Nc;
}
size_t frame_detect_demux::get_pre_header_ofdm_symbol_length() const {
  return _ph_Ns;
}
size_t frame_detect_demux::get_pre_header_num_ofdm_symbols() const {
  return _num_pre_header_ofdm_syms;
}
} // namespace legacy
} // namespace ofdm
} // namespace bamradio
