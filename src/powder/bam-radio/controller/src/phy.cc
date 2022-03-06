// -*- c++ -*-
//
// Copyright (c) 2017-2018 Dennis Ogbe
// Copyright (c) 2017-2018 Stephen Larew
// Copyright (c) 2017-2018 Tomohiro Arakawa

#include "phy.h"
#include "bamfftw.h"
#include "bandwidth.h"
#include "common.h"
#include "debug.h"
#include "events.h"
#include "options.h"

#include "rs_ccsds.h"

#include <algorithm>
#include <cmath>
#include <math.h>
#include <random>
#include <type_traits>

#include <boost/format.hpp>
#include <boost/optional.hpp>

#include <gnuradio/fft/fft.h>
#include <volk/volk.h>

namespace bamradio {
namespace ofdm {

phy_rx::phy_rx(MCS::Name header_mcs, int header_nthreads, int payload_nthreads,
               deframer header_deframer, deframer payload_deframer)
    : _nchan(0),
      _hdr_demod("header_demod", header_mcs, header_nthreads, header_deframer),
      _payl_demod("payload_demod", header_mcs, payload_nthreads,
                  payload_deframer) {}

void phy_rx::connect(std::vector<ChannelOutputBuffer::sptr> const &inbufs,
                     std::vector<int> const &tx_os) {
  assert(tx_os.size() == inbufs.size());
  _tx_os = tx_os;
  _nchan = inbufs.size();
  for (size_t i = 0; i < _nchan; ++i) {
    _hdr_snr_filt.push_back(std::make_shared<VarianceFilter>(
        options::phy::data::variance_filter_window,
        options::phy::data::variance_hist_min,
        options::phy::data::variance_hist_max,
        options::phy::data::variance_hist_nbins));
    _payl_snr_filt.push_back(std::make_shared<VarianceFilter>(
        options::phy::data::variance_filter_window,
        options::phy::data::variance_hist_min,
        options::phy::data::variance_hist_max,
        options::phy::data::variance_hist_nbins));
  }
  auto headerParams = begin(_hdr_demod.header_params())->second;
  for (size_t i = 0; i < _nchan; ++i) {
    auto fdd = frame_detect_demux::make(
        // header and detection constants
        headerParams.symbols[0].second->cyclic_prefix_length,
        headerParams.symbols[0].second->symbol_length,
        headerParams.numSymbols(), options::phy::data::sync_detection_threshold,
        // demodulate something
        [this, i](auto frame, auto &&samples, auto sc_info, auto src) {
          if (frame == nullptr) {
            _hdr_demod.demod(frame, std::move(samples), sc_info, _tx_os[i], i, _hdr_snr_filt[i],
                             src);
          } else {
            _payl_demod.demod(frame, std::move(samples), sc_info, _tx_os[i], i, _payl_snr_filt[i],
                              src);
          }
        },
        // interpret_frame
        [this, i](auto frame, auto &num_symbols, auto &Ns, auto &Nc) {
          auto p = frame->payloadParams(false, _tx_os[i], 0);
          num_symbols = p.numSymbols();
          Ns = p.symbols.size() > 0 ? p.symbols[0].second->symbol_length : 0;
          Nc = p.symbols.size() > 0 ? p.symbols[0].second->cyclic_prefix_length : 0;
        });
    fdd->start(inbufs[i], (boost::format("fdd_%1%") % i).str());
    _fdds.push_back(fdd);
  }
}

bool phy_rx::connected() const { return not(_nchan == 0); }

void phy_rx::setOversampleRate(int which, int os) {
  if (!connected()) {
    throw std::runtime_error(
        "phy_rx: Cannot set oversample rate (not connected).");
  }
  assert(which < _tx_os.size());
  // FIXME make _tx_os thread safe
  _tx_os[which] = os;
}

void phy_rx::flushFilters() {
  for (auto &f : _hdr_snr_filt) {
    f->flush();
  }
  for (auto &f : _payl_snr_filt) {
    f->flush();
  }
}

// initialize resources for demodulation
// FIXME: 30 iterations hardcoded in decoder
Demodulator::Resource::Resource(DFTSOFDMFrameParams const &params,
                                MCS::Name const &mcs)
    : fft([&] {
        // FIXME potentially support more FFT sizes
        std::vector<size_t> sz;
        sz.push_back(params.symbols[0].second->symbol_length);
        return fft::CPUFFT::make(sz, true, 1, options::phy::fftw_wisdom);
      }()),
      ifft([&] {
        // FIXME potentially support more FFT sizes
        std::vector<size_t> sz;
        sz.push_back(params.dft_spread_length);
        return fft::CPUFFT::make(sz, false, 1, options::phy::fftw_wisdom);
      }()),
      decoders([&] {
        decltype(decoders) o;
        int max_its = 30; // FIXME
        for (auto const &rate_idx : yaldpc::ieee80211::valid_rates) {
          for (auto const &block_size : yaldpc::ieee80211::valid_sizes) {
            auto dd_code = yaldpc::ieee80211::get(rate_idx, block_size);
            auto full_code = yaldpc::expand(dd_code);
            o.emplace(
                MCS::CodePair(rate_idx, block_size),
                yaldpc::SerialCMinSumDecoder<bamradio::llr_type, uint8_t>::make(
                    full_code, max_its, true, true));
          }
        }
        return o;
      }()) {
  // allocate memory. n.b this is a "little" wasteful, since we are allocating
  // space for a channel estimate for every OFDM symbol. Cons: we might not need
  // that much, there might be some extra copying. Pros: equalizing is easy
  fft_inbuf.resize(params.numSymbols() *
                   params.symbols[0].second->symbol_length);
  fft_outbuf.resize(fft_inbuf.size());
  ifft_inbuf.resize(params.numSymbols() * params.dft_spread_length);
  ifft_outbuf.resize(ifft_inbuf.size());
  chanest.resize(params.numSymbols() * params.symbols[0].second->symbol_length);
  llrs.resize(params.numBits());
  decoded_bits.resize(params.numBits());
}

Demodulator::Resource::~Resource() {}

// grow (but never shrink) a demod resource
void Demodulator::Resource::init(DFTSOFDMFrameParams const &params,
                                 MCS::Name const &mcs) {
  // these sizes are needed
  auto nfft = params.numSymbols() * params.symbols[0].second->symbol_length;
  auto nifft = params.numSymbols() * params.dft_spread_length;
  auto nbits = params.numBits();
  // resize if necessary
  if (fft_inbuf.size() < nfft) {
    fft_inbuf.resize(nfft);
  }
  if (fft_outbuf.size() < nfft) {
    fft_outbuf.resize(nfft);
  }
  if (chanest.size() < nfft) {
    chanest.resize(nfft);
  }
  if (inter_outbuf.size() < nifft) {
    inter_outbuf.resize(nifft);
  }
  if (ifft_inbuf.size() < nifft) {
    ifft_inbuf.resize(nifft);
  }
  if (ifft_outbuf.size() < nifft) {
    ifft_outbuf.resize(nifft);
  }
  if (llrs.size() < nbits) {
    llrs.resize(nbits);
  }
  if (decoded_bits.size() < nbits) {
    decoded_bits.resize(nbits);
  }
  // initialize memory
  auto zerofill = [](auto &vec) { std::fill(begin(vec), end(vec), 0); };
  zerofill(fft_inbuf);
  zerofill(fft_outbuf);
  zerofill(chanest);
  zerofill(inter_outbuf);
  zerofill(ifft_inbuf);
  zerofill(ifft_outbuf);
  zerofill(llrs);
  zerofill(decoded_bits);
}

// channel estimation. <2018-03-15 Thu> This needs to go away and be replaced by
// some smart CUDA Kernels. This is copied verbatim from ls_chanest_cc_impl.cc
inline void interpolate_channel_freq(fcomplex *ofdm_symbol,
                                     fcomplex *channel_estimate,
                                     const fcomplex *pilot_symbols,
                                     const int32_t *pilot_carriers,
                                     size_t num_pilots, size_t num_symbols,
                                     float pilot_gain) {
  auto numPilots = num_pilots;
  auto ps = pilot_symbols;
  auto pc = pilot_carriers;
  auto d_sym_len = num_symbols;
  auto d_chanest = channel_estimate;

  // Copy pilot carriers and sort for linear interp
  std::vector<int32_t> pc_c(pilot_carriers, pilot_carriers + num_pilots);
  std::sort(begin(pc_c), end(pc_c));

  pc = &pc_c[0];

  // linear interpolation

  auto complex_interpolation = [](fcomplex const &x, fcomplex const &y,
                                  int const a, int const b,
                                  int const j) -> fcomplex {
    float t = float(j - a) / (b - a);
    if (x != 0.0f && y != 0.0f) {
      float argx = std::arg(x), argy = std::arg(y);
      float absx = std::abs(x), absy = std::abs(y);
      float sign = 1;
      if (std::abs(argx - argy) > M_PI)
        sign = -1;
      fcomplex v =
          std::polar((1 - t) * absx + t * absy, (1 - t) * argx + t * argy);
      return sign * v;
    } else
      return (1 - t) * x + t * y;
  };

  {
    size_t b = *pc++;

    for (size_t i = 1; i < numPilots; ++i) {
      auto const a = b;
      b = *pc++;
      if (a < d_sym_len / 2 && b >= d_sym_len / 2) {
        // we're at the boundary between highest and lowest frequency pilot
        // carriers
        // so don't interp and instead just extend out
        for (size_t i = a + 1; i < d_sym_len / 2; ++i) {
          d_chanest[i] = d_chanest[a];
        }
        for (size_t i = d_sym_len / 2; i < b; ++i) {
          d_chanest[i] = d_chanest[b];
        }
      } else {
        // do interp
        // auto const s = (d_chanest[b] - d_chanest[a]) / (float)(b - a);
        for (size_t j = a + 1; j < b; ++j) {
          // d_chanest[j] = d_chanest[a] + (float)(j - a) * s;
          d_chanest[j] =
              complex_interpolation(d_chanest[a], d_chanest[b], a, b, j);
        }
      }
    }
  }

  auto const b = pc_c.front();
  assert(b >= 0 && b < (int)d_sym_len);
  auto const ap = pc_c.back();
  assert(ap >= 0 && ap < (int)d_sym_len);
  auto const an = pc_c.back() - (int)d_sym_len;
  assert(an < 0);
  if (b - an > 1) {
    // do special cased wrapped interpolation
    // next two asserts mean there must be min two pilots, one on pos. freq
    // side and other on neg. freq side
    assert(an < 0);
    // auto const s = (d_chanest[b] - d_chanest[ap]) / (float)(b - an);
    for (int j = 0; j < b; ++j) {
      assert(j - an > 0);
      d_chanest[j] =
          complex_interpolation(d_chanest[ap], d_chanest[b], an, b, j);
    }
    for (size_t j = ap + 1; j < d_sym_len; ++j) {
      assert(j - ap > 0);
      d_chanest[j] = complex_interpolation(d_chanest[ap], d_chanest[b], an, b,
                                           j - d_sym_len);
    }
  }
}

int Demodulator::_estimate_snr(Demodulator::Resource *r,
                               std::vector<OFDMSymbolParams *> const &symbols,
                               float &snr_db, float &normalized_var) {

  // Exit early if symbols vector is empty
  if (symbols.empty()) {
    return 0;
  }

  // post-FFT data
  auto const post_fft = r->fft_outbuf.data();

  // Length of one OFDM symbol
  // n.b. assumes length of OFDM does not change within a frame
  size_t const symbol_length = symbols[0]->symbol_length;

  auto map2to1 = [symbol_length](size_t const f, size_t const t) {
    return symbol_length * t + f;
  };

  float noise_var_sum = 0;
  int noise_var_cnt = 0;
  float signal_var_sum = 0;
  int signal_var_cnt = 0;

  // iterate over all OFDM symbols
  for (size_t symbol_idx = 0; symbol_idx < symbols.size(); symbol_idx++) {

    // Get current symbol
    auto const symbol = symbols[symbol_idx];
    assert(symbol_length == symbol->symbol_length);

    // Get pilot carrier mapping
    auto np = pmt::length(symbol->pilot_carrier_mapping);
    assert(np <= symbol_length);
    auto pmap = pmt::s32vector_elements(symbol->pilot_carrier_mapping, np);

    // Get data carrier mapping
    auto nd = pmt::length(symbol->data_carrier_mapping);
    assert(nd <= symbol_length);
    auto dmap = pmt::s32vector_elements(symbol->data_carrier_mapping, nd);

    // Find null subcarriers
    std::vector<uint8_t> subcarriers(
        symbol_length); // probably faster and safer than std::vector<bool>?
    for (size_t i = 0; i < np; ++i) {
      assert(*(pmap + i) < symbol_length);
      subcarriers[*(pmap + i)] = 1;
    }
    for (size_t i = 0; i < nd; ++i) {
      assert(*(dmap + i) < symbol_length);
      subcarriers[*(dmap + i)] = 1;
    }

    // noise
    for (size_t subcarrier_idx = 0; subcarrier_idx < symbol_length;
         ++subcarrier_idx) {
      if (!subcarriers[subcarrier_idx]) {
        noise_var_sum +=
            std::norm(post_fft[map2to1(subcarrier_idx, symbol_idx)]);
        ++noise_var_cnt;
      }
    }

    // signal
    for (size_t i = 0; i < np; ++i) {
      size_t subcarrier_idx = *(pmap + i);
      signal_var_sum +=
          std::norm(post_fft[map2to1(subcarrier_idx, symbol_idx)]);
      ++signal_var_cnt;
    }
  }

  // Compute SNR
  float const signal_var = signal_var_sum / signal_var_cnt;
  float const noise_var = noise_var_sum / noise_var_cnt;
  snr_db = 10 * log10(signal_var / noise_var);

  // chanest
  auto const chanest = r->chanest.data();

  // Compute variance
  // n.b. constellation_energy is a function of constellations. Make sure that
  // this value is consistent when modifying the constellations.
  float constellation_energy = 1 / sqrt(2);
  normalized_var = constellation_energy / (signal_var / noise_var);

  return 0;
}

int Demodulator::_estimate_channel(
    Demodulator::Resource *r, std::vector<OFDMSymbolParams *> const &symbols,
    bool interp_freq, bool interp_time, float pilot_gain) {
  auto const ip = r->fft_outbuf.data();
  auto const op = r->chanest.data();
  // FIXME again, these buffer sizes need to be worked out
  assert(r->chanest.size() == r->fft_outbuf.size());

  // Exit early if symbols vector is empty
  if (symbols.empty()) {
    return 0;
  }

  // Length of one OFDM symbol
  // n.b. assumes length of OFDM does not change within a frame
  size_t const symbol_length = symbols[0]->symbol_length;

  // variables for interpolation (time axis)
  std::vector<fcomplex> chanest_prev(symbol_length, 0.0);
  std::vector<size_t> chanest_cnt(symbol_length, 0);
  std::vector<fcomplex> chanest_delta(symbol_length, 0.0);

  auto map2to1 = [symbol_length](size_t const f, size_t const t) {
    return symbol_length * t + f;
  };

  // iterate over all OFDM symbols
  for (size_t symbol_idx = 0; symbol_idx < symbols.size(); symbol_idx++) {
    // Get current symbol
    auto const symbol = symbols[symbol_idx];

    // Check the length of the symbol
    if (symbol_length != symbol->symbol_length) {
      log::doomsday(
          "OFDM frame cannot contain OFDM symbols with different length.",
          __FILE__, __LINE__);
    }

    // Copy the previous estimate except first OFDM symbol
    if (symbol_idx != 0) {
      size_t offset = symbol_length * symbol_idx;
      memcpy(op + offset, op + offset - symbol_length,
             symbol_length * sizeof(*op));
    }

    // Perform channel estimation if there are pilot symbols
    if (pmt::length(symbol->pilot_symbols) != 0) {
      {
        auto np = pmt::length(symbol->pilot_carrier_mapping);
        assert(np == pmt::length(symbol->pilot_symbols));
        auto ps = pmt::c32vector_elements(symbol->pilot_symbols, np);
        auto pc = pmt::s32vector_elements(symbol->pilot_carrier_mapping, np);

        // zero forcing
        for (size_t i = 0; i < np; ++i) {
          // scale by d_sym_len because the IFFT at TX followed by
          // FFT at RX adds a d_sym_len scale factor. yeah, i know...
          //
          // holy moly we're in some trouble here.  here's teh situation.
          // right now the TX chain does hardcoded 2x oversampling of the OFDM
          // signal, meaning the IFFT is 2x length of ofdm symbol carriers.  but
          // the RX is still critically sampled.  The constellation points are
          // polite and respect this asymmetry, but our pilot symbols are
          // barefoot children who bring toads inside the house and mess it all
          // up.  So, we need to do the scaling here and now we need to respect
          // the asymmetric FFT sizes from TX to RX.
          auto const pilot_carrier = *(pc + i);
          auto const pilot_symbol = *(ps + i);
          assert(pilot_carrier >= 0);
          assert(pilot_carrier < (int)symbol_length);
          size_t const pos = symbol_length * symbol_idx + pilot_carrier;
          op[pos] =
              ip[pos] / (pilot_symbol * (float)symbol_length * pilot_gain);

          if (symbol_idx != 0 && interp_time) {
            // Compute delta
            auto p_orig = op[map2to1(pilot_carrier,
                                     symbol_idx - chanest_cnt[pilot_carrier])];
            auto p_now = op[map2to1(pilot_carrier, symbol_idx)];
            chanest_delta[pilot_carrier] =
                (p_now - p_orig) / fcomplex(chanest_cnt[pilot_carrier], 0);
            // Interpolation
            for (size_t j = 0; j < (chanest_cnt[pilot_carrier] - 1); ++j) {
              op[map2to1(pilot_carrier,
                         symbol_idx - chanest_cnt[pilot_carrier] + 1 + j)] =
                  op[map2to1(pilot_carrier,
                             symbol_idx - chanest_cnt[pilot_carrier] + j)] +
                  chanest_delta[pilot_carrier];
            }
            chanest_cnt[pilot_carrier] = 0;
          }
        }
      }

      // Increment counter
      for (auto &v : chanest_cnt)
        v += 1;

      // Interpolation for header
      if (interp_freq) {
        auto np = pmt::length(symbol->pilot_carrier_mapping);
        auto ps = pmt::c32vector_elements(symbol->pilot_symbols, np);
        auto pc = pmt::s32vector_elements(symbol->pilot_carrier_mapping, np);
        size_t offset = symbol_length * symbol_idx;
        interpolate_channel_freq(ip + offset, op + offset, ps, pc, np,
                                 symbol_length, pilot_gain);
      }
    }
  }

  // Interpolate remainint samples
  if (interp_time) {
    for (size_t carrier_idx = 0; carrier_idx < symbol_length; ++carrier_idx) {
      // Interpolation
      for (size_t j = 0; j < (chanest_cnt[carrier_idx] - 1); ++j) {
        op[map2to1(carrier_idx,
                   symbols.size() - chanest_cnt[carrier_idx] + 1 + j)] =
            op[map2to1(carrier_idx,
                       symbols.size() - chanest_cnt[carrier_idx] + j)] +
            chanest_delta[carrier_idx];
      }
    }
  }

  return 0;
}

int Demodulator::_fft(std::vector<fcomplex> const &in, Demodulator::Resource *r,
                      std::vector<OFDMSymbolParams *> const &symbols) {
  int nfft = 0;
  assert(r->fft_inbuf.size() >= in.size());
  auto ip = const_cast<fcomplex *>(in.data());
  auto op = r->fft_outbuf.data();
  for (auto const &symbol : symbols) {
    r->fft->execute(symbol->symbol_length, ip, op);
    ip += symbol->symbol_length;
    op += symbol->symbol_length;
    nfft += symbol->symbol_length;
  }
  return nfft;
}

int Demodulator::_equalize(Demodulator::Resource *r,
                           std::vector<OFDMSymbolParams *> const &symbols) {
  auto ip = r->fft_outbuf.data();
  auto cep = r->chanest.data();
  // n.b. assumes all symbols have the same length
  auto symlen = symbols[0]->symbol_length;
  volk_32fc_x2_divide_32fc(ip, ip, cep, symbols.size() * symlen);
  return symbols.size() * symlen;
}

int Demodulator::_demap_subcarriers(Demodulator::Resource *r,
                                    std::vector<OFDMSymbolParams *> &symbols,
                                    size_t dft_spread_length) {
  // compute a "plan" for this serialization step. input
  // pointer, output pointer, pointer to carrier allocation, number of allocated
  // carriers.
  std::vector<carrier_demap_plan_t> plan;
  std::vector<OFDMSymbolParams *> new_symbols;
  plan.reserve(symbols.size());
  new_symbols.reserve(symbols.size());
  [&](auto &p, auto &ns) {
    auto *ip = r->fft_outbuf.data();
    auto *op = r->ifft_inbuf.data();
    for (auto const &symbol : symbols) {
      auto nc = pmt::length(symbol->data_carrier_mapping);
      if (nc > 0) {
        int32_t *a;
        a = const_cast<decltype(a)>(
            pmt::s32vector_elements(symbol->data_carrier_mapping, nc));
        p.emplace_back(
            carrier_demap_plan_t{ip, op, a, nc, symbol->symbol_length});
        new_symbols.push_back(symbol);
        op += dft_spread_length;
      }
      ip += symbol->symbol_length;
    }
  }(plan, new_symbols);
  // execute the serializer kernel with this new plan
  int nmapped = 0;
  assert(plan.size() > 0);
  assert(plan.size() == new_symbols.size());
  for (auto const &p : plan) {
    for (size_t i = 0; i < p.ncarriers; ++i) {
      p.out[i] = p.in[p.alloc[i]];
    }
    nmapped += p.ncarriers;
  }
  // update enclosing scope
  symbols = new_symbols;
  return nmapped;
}

int Demodulator::_ifft(Demodulator::Resource *r,
                       std::vector<OFDMSymbolParams *> const &symbols,
                       size_t dft_spread_length) {
  int nifft = 0;
  auto ip = r->ifft_inbuf.data();
  auto op = r->ifft_outbuf.data();
  for (auto const &symbol : symbols) {
    r->ifft->execute(dft_spread_length, ip, op);
    ip += dft_spread_length;
    op += dft_spread_length;
    nifft += dft_spread_length;
  }
  return nifft;
}

int Demodulator::_deinterleave(Demodulator::Resource *r,
                               size_t nsymb) {
  auto const &interleaver = Interleaver::get(nsymb);
  for (size_t i = 0; i < nsymb; ++i) {
    r->inter_outbuf[i] = r->ifft_outbuf[interleaver[i]];
  }
  return nsymb;
}

int Demodulator::_llr_map(Demodulator::Resource *r,
                          std::vector<OFDMSymbolParams *> const &symbols,
                          float snr, size_t dft_spread_length) {
  auto op = r->llrs.data();
  auto n = 0;
  for (size_t s = 0; s < symbols.size(); ++s) {
    auto const constellation = symbols[s]->constellation;
    auto const sidx = constellation->get_snr_idx(snr);
    for (size_t i = 0; i < dft_spread_length; ++i) {
      constellation->make_soft_decision(
          r->inter_outbuf[s * dft_spread_length + i], op, sidx);
      op += constellation->bits_per_symbol();
      n += constellation->bits_per_symbol();
    }
  }
  return n;
}

int Demodulator::_decode(Demodulator::Resource *r, DFTSOFDMFrame *frame,
                         MCS::Name mcs, int nllr) {
  // FIXME? hardcoded number of blocks for header
  auto nblocks = frame == nullptr ? 1 : frame->numBlocks(false);
  auto decoder = r->decoders.at(MCS::table[mcs].codePair());
  assert(r->decoded_bits.size() >= nblocks * decoder->k());
  assert(nllr >= nblocks * decoder->n());
  auto ip = r->llrs.data();
  auto op = r->decoded_bits.data();
  for (size_t i = 0; i < nblocks; ++i) {
    decoder->decode(ip, op);
    ip += decoder->n();
    op += decoder->k();
  }
  return nblocks * decoder->k();
}

// each worker thread needs to save its index into the array of resources
thread_local size_t cpu_thread_idx;

//
// OFDM Demodulation (complex samples, crtitically sampled -> bits)
//

void Demodulator::demod(DFTSOFDMFrame::sptr frame,
                        std::vector<fcomplex> &&raw_samples, SCInfo sc_info,
                        int tx_os, int rx_chain_idx,
                        std::shared_ptr<VarianceFilter> snr_filt,
                        frame_detect_demux *fdd_src) {
  boost::asio::dispatch(_ioctx, [this, frame, in = std::move(raw_samples),
                                 sc_info, tx_os, rx_chain_idx, snr_filt,
                                 fdd_src] {
#if BAMRADIO_PHY_DEBUG
    using gr::bamofdm::dump_vec;
    static unsigned long hncall = 0;
    static unsigned long pncall = 0;
    frame == nullptr ? hncall++ : pncall++;
#endif

    // figure out whether we are decoding a header or a payload
    bool const header = frame == nullptr ? true : false;

    // acquire resources and make sure we have enough memory
    auto r = _resources[cpu_thread_idx];
    auto params = header ? _header_params.at(tx_os)
                         : frame->payloadParams(false, tx_os, 0);
    auto mcs = header ? _header_mcs : frame->payloadMcs();
    r->init(params, mcs);

    // in order to more easily iterate over the OFDM symbols in this "frame",
    // we use this vector. it has numSymbols() elements and each element is a
    // raw pointer to the corresponding OFDMFrameParams struct. n.b. We should
    // add this to the OFDMSymbolParams interface (do something similar to the
    // next() call of the Sequence ID, which produces the next SymbolName)
    auto symbols = [&params] {
      std::vector<OFDMSymbolParams *> o;
      for (auto const &symbol : params.symbols)
        for (size_t i = 0; i < symbol.first; ++i)
          o.push_back(symbol.second.get());
      return o;
    }();

    // check that the number of samples given to us matches the number of
    // samples based on the DFTSOFDMFrameParams governing this demod attempt
    // (FIXME: debug only or make proper check)
    assert(in.size() ==
           std::accumulate(begin(symbols), end(symbols), 0,
                           [](auto const &accum, auto const &elem) -> int {
                             return accum + elem->symbol_length;
                           }));

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_fft_in_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             in);

    debug::DumpFrameParams((boost::format("%1%_frameparams_call_%2%.bin") %
                            (header ? "hdr" : "payl") %
                            (header ? hncall : pncall))
                               .str(),
                           params);
#endif

    // compute the FFT
    _fft(in, r, symbols);

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_fft_out_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             r->fft_outbuf.data(), r->fft_outbuf.size());
#endif

    // channel estimation
    _estimate_channel(r, symbols, header, !header, sqrt((float)tx_os));

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_chanest_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             r->chanest.data(), r->chanest.size());
#endif

    // estimate SNR and variance
    float snr = 0;
    float normalized_var = 0;
    _estimate_snr(r, symbols, snr, normalized_var);

    // equalize the samples
    _equalize(r, symbols);

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_equalized_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             r->fft_outbuf.data(), r->fft_outbuf.size());
#endif

    // subcarrier de-mapping
    // we might have to drop some full OFDM symbols. that changes
    // our symbol array
    _demap_subcarriers(r, symbols, params.dft_spread_length);

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_ifft_in_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             r->ifft_inbuf.data(), r->ifft_inbuf.size());
#endif

    // compute IFFT
    auto const nsymb = _ifft(r, symbols, params.dft_spread_length);

#if BAMRADIO_PHY_DEBUG
    dump_vec((boost::format("%1%_ifft_out_call_%2%.32fc") %
              (header ? "hdr" : "payl") % (header ? hncall : pncall))
                 .str(),
             r->ifft_outbuf.data(), r->ifft_outbuf.size());
#endif

    // deinterleave
    _deinterleave(r, nsymb);

    // demodulate symbols
    auto const normalized_var_db = -10.0 * log10(normalized_var);
    auto nllr =
        _llr_map(r, symbols, normalized_var_db, params.dft_spread_length);

    // decode bits
    auto nbits = _decode(r, frame.get(), mcs, nllr);

    // deframe
    _deframe(frame, boost::asio::buffer(r->decoded_bits.data(), nbits), snr,
             normalized_var, rx_chain_idx, cpu_thread_idx, fdd_src);
  });
} // end demod function

Demodulator::Demodulator(std::string name_prefix, MCS::Name headerMcs,
                         int nthreads, deframer deframer)
    : _nthreads(nthreads), _ioctx_work(boost::asio::make_work_guard(_ioctx)),
      _header_params([] {
        std::map<int, DFTSOFDMFrameParams> hp;
        for (auto const &sc : bam::dsp::SubChannel::table()) {
          auto params =
              DFTSOFDMFrame(
                  0, 0, {},
                  MCS::stringNameToIndex(options::phy::data::header_mcs_name),
                  MCS::stringNameToIndex(
                      options::phy::data::initial_payload_mcs_name),
                  SeqID::stringNameToIndex(
                      options::phy::data::initial_payload_symbol_seq_name),
                  0, 0)
                  .headerParams(false, sc.os);
          hp.emplace(sc.os, params.symbols);
        }
        return hp;
      }()),
      _header_mcs(headerMcs), _deframe(deframer) {
  for (size_t i = 0; i < _nthreads; ++i) {
    // add worker threads to the io_context
    _work_threads.emplace_back([this, i, name_prefix] {
      cpu_thread_idx = i;
      // set the thread name for easier debugging
      bamradio::set_thread_name(
          (boost::format("%1%_%2%") % name_prefix % i).str());
      _ioctx.run();
    });
    // initialize the resources
    _resources.push_back(
        new Resource(begin(_header_params)->second, _header_mcs));
  }
}

Demodulator::~Demodulator() {
  // stop worker threads
  _ioctx_work.reset();
  _ioctx.stop();
  for (auto &thread : _work_threads) {
    thread.join();
  }
  // delete resources
  for (auto &r : _resources) {
    delete r;
  }
}

//
// Transmitter PHY
//

std::vector<size_t> const fftsizes = [] {
  std::vector<size_t> s;
  s.resize(bam::dsp::SubChannel::table().size());
  int k = 0;
  std::generate(begin(s), end(s),
                [&k] { return bam::dsp::SubChannel::table()[k++].os * 128; });
  s.push_back(108); // for IFFT;
  s.push_back(4);
  s.push_back(8);
  s.push_back(16);
  s.push_back(32);
  s.push_back(64);
  s.push_back(128);
  s.push_back(256);
  s.push_back(2560);
  std::sort(begin(s), end(s));
  auto const l = std::unique(begin(s), end(s));
  s.erase(l, end(s));
  return s;
}();

phy_tx::Resource::Resource()
    : fft(fft::CPUFFT::make(fftsizes, true, 1, options::phy::fftw_wisdom)),
      ifft(fft::CPUFFT::make(fftsizes, false, 1, options::phy::fftw_wisdom)),
      encoders([&] {
        std::vector<yaldpc::Encoder::sptr> o;
        // create all possible encoders
        std::map<MCS::CodePair, yaldpc::Encoder::sptr> m;
        for (auto const &rate_idx : yaldpc::ieee80211::valid_rates) {
          for (auto const &block_size : yaldpc::ieee80211::valid_sizes) {
            auto dd_code = yaldpc::ieee80211::get(rate_idx, block_size);
            m.emplace(MCS::CodePair(rate_idx, block_size),
                      yaldpc::DDEncoder::make(dd_code));
          }
        }
        // keep array of pointers to them ordered by MCS index
        for (size_t mcs = 0; mcs < MCS::Name::NUM_MCS; ++mcs) {
          o.push_back(m.at(MCS::table[mcs].codePair()));
        }
        return o;
      }()) {}

void phy_tx::Resource::init(DFTSOFDMFrameParams const &params) {
  auto nbits = params.numBits();
  auto nspread = params.numSymbols() * params.dft_spread_length;
  auto nsamp = params.numTXSamples();
  // enough scratch space to fit a cyclic prefix
  auto ncp = params.symbols[0].second->cyclic_prefix_length *
             params.symbols[0].second->oversample_rate * 2;

  // grow memory reserves if necessary
  if (raw_bits.size() < nbits) {
    raw_bits.resize(nbits);
  }
  if (coded_bits.size() < nbits) {
    coded_bits.resize(nbits);
  }
  if (symbols.size() < nspread) {
    symbols.resize(nspread);
  }
  if (spread_symbols.size() < nspread) {
    spread_symbols.resize(nspread);
  }
  if (inter_symbols.size() < nspread) {
    inter_symbols.resize(nspread);
  }
  if (ifft_in.size() < nsamp) {
    ifft_in.resize(nsamp);
  }
  if (cp_buf.size() < ncp) {
    cp_buf.resize(ncp);
  }
  if (out.size() < nsamp) {
    out.resize(nsamp);
  }
  // initialize memory
  auto zerofill = [](auto &vec) { std::fill(begin(vec), end(vec), 0); };
  zerofill(raw_bits);
  zerofill(coded_bits);
  zerofill(symbols);
  zerofill(spread_symbols);
  zerofill(inter_symbols);
  zerofill(ifft_in);
  zerofill(cp_buf);
  zerofill(out);
}

phy_tx::phy_tx(size_t nthreads)
    : _stream(nullptr), _connected(false), _stream_avail(true),
      _nthreads(nthreads), _resources([nthreads] {
        std::vector<Resource *> o(nthreads);
        std::generate(begin(o), end(o), [] { return new Resource(); });
        return o;
      }()),
      _work_threads_running(false), _work_threads_avail(0), _fm(nullptr),
      _random_bits([] {
        // 8 bits/symbol, 128 symbols in an OFDM symbol + doomsday prep
        auto const max_bits = 8 * 128 + 200;
        std::mt19937_64 rng(33);
        std::uniform_int_distribution<uint8_t> dist(0, 1);
        std::vector<uint8_t> buf;
        buf.resize(max_bits);
        std::generate(begin(buf), end(buf), [&] { return dist(rng); });
        return buf;
      }()),
      _header_num_data_sym([] {
        size_t nsym = 0;
        auto params =
            DFTSOFDMFrame(
                0, 0, {},
                MCS::stringNameToIndex(options::phy::data::header_mcs_name),
                MCS::stringNameToIndex(
                    options::phy::data::initial_payload_mcs_name),
                SeqID::stringNameToIndex(
                    options::phy::data::initial_payload_symbol_seq_name),
                0, 0)
            .headerParams(true, 1); // n.b. doesn't really matter what these are.
        for (auto const &symbol : params.symbols) {
          for (size_t i = 0; i < symbol.first; ++i)
            nsym += symbol.second->numDataCarriers();
        }
        return nsym;
      }()) {
  if (_nthreads < 1) {
    log::doomsday("Need at least one transmitter thread.", __FILE__, __LINE__);
  } else if (_nthreads > 2) {
    panic("nthreads > 2 untested; out-of-order transmission possible. override "
          "at your own peril");
  }
}

phy_tx::~phy_tx() {
  stop();
  // delete resources
  for (auto &r : _resources) {
    delete r;
  }
}

std::vector<float> &phy_tx::get_window(bool head, int N, int os) {
  // cache the pre-computed windows in this function
  static std::map<int, std::vector<float>> heads;
  static std::map<int, std::vector<float>> tails;
  // I am allowed to do this.
  // https://stackoverflow.com/questions/14106653
  static std::mutex mtx;
  try {
    std::unique_lock<decltype(mtx)> l(mtx);
    if (head) {
      return heads.at(N * os);
    } else {
      return tails.at(N * os);
    }
  } catch (std::out_of_range) {
    std::vector<float> win(N * os);
    int n = 0;
    std::generate(begin(win), end(win), [&] {
      auto s = std::sin((M_PI * (0.5 + n++)) / (N * os * 2));
      return std::pow(s, 2);
    });
    std::unique_lock<decltype(mtx)> l(mtx);
    heads[N * os] = win;
    std::reverse(begin(win), end(win));
    tails[N * os] = win;
    l.unlock();
    return get_window(head, N, os);
  }
}

int phy_tx::encode_bits(DFTSOFDMFrame::sptr frame,
                        DFTSOFDMFrameParams const &params, int ridx) {
  using boost::asio::mutable_buffer;
  auto const r = _resources[ridx];
  auto const hdr_nblocks = frame->numBlocks(true);
  auto const payl_nblocks = frame->numBlocks(false);
  auto const hdr_enc = r->encoders[frame->headerMcs()];
  auto const payl_enc = r->encoders[frame->payloadMcs()];
  std::vector<size_t> offset;     // stores index of raw block beginning
  std::vector<size_t> out_offset; // index of codeword beginning
  offset.reserve(hdr_nblocks + payl_nblocks);
  out_offset.reserve(hdr_nblocks + payl_nblocks);
  int nbits = 0;

  auto write_blocks = [&](auto header) {
    auto const enc = header ? hdr_enc : payl_enc;
    auto const n = header ? hdr_nblocks : payl_nblocks;
    auto const ibase = header ? 0 : hdr_nblocks * hdr_enc->k();
    auto const obase = header ? 0 : hdr_nblocks * hdr_enc->n();
    for (int i = 0; i < n; ++i) {
      auto ofst = ibase + i * enc->k();
      auto out_ofst = obase + i * enc->n();
      offset.push_back(ofst);
      out_offset.push_back(out_ofst);
      auto b = mutable_buffer(r->raw_bits.data() + ofst, enc->k());
      frame->writeBlock(header, i, b);
    }
  };
  auto encode_blocks = [&](auto header) {
    auto const enc = header ? hdr_enc : payl_enc;
    auto const n = header ? hdr_nblocks : payl_nblocks;
    auto const base = header ? 0 : hdr_nblocks;
    for (int i = 0; i < n; ++i) {
      auto in = r->raw_bits.data() + offset[base + i];
      auto out = r->coded_bits.data() + out_offset[base + i];
      enc->encode(in, out);
      nbits += enc->n();
    }
  };

  // write raw bits to buffer
  write_blocks(true);
  write_blocks(false);

  // encode bits
  encode_blocks(true);
  encode_blocks(false);

  // pad with random bits for the last OFDM symbol
  auto const zp = params.numBits() -
                  frame->blockLength(false) * frame->numBlocks(false) -
                  frame->blockLength(true) * frame->numBlocks(true);
  memcpy(r->coded_bits.data() + out_offset.back() + payl_enc->n(),
         _random_bits.data(), zp);
  nbits += zp;
  return nbits;
}

int phy_tx::modulate_bits(std::vector<OFDMSymbolParams *> const &symbols,
                          int ridx) {
  auto const r = _resources[ridx];
  int nw = 0;
  auto bit = r->coded_bits.data();
  auto op = r->symbols.data();
  for (auto const &symbol : symbols) {
    auto constellation = symbol->constellation;
    assert(constellation != nullptr);
    auto bps = constellation->bits_per_symbol();
    for (int i = 0; i < symbol->numDataCarriers(); ++i) {
      unsigned int value = 0;
      for (int j = 0; j < bps; ++j) {
        value |= (*bit++ << (bps - 1 - j));
      }
      constellation->map_to_points_and_scale(value, op);
      op += constellation->dimensionality();
      nw += constellation->dimensionality();
    }
  }
  return nw;
}

int phy_tx::interleave(size_t nsymb, int ridx) {
  // n.b. need to interleave header and payload independently
  auto const r = _resources[ridx];
  auto const &hint = Interleaver::get(_header_num_data_sym);
  auto const &pint = Interleaver::get(nsymb - _header_num_data_sym);
  auto const hsrc = r->symbols.data();
  auto const psrc = hsrc + _header_num_data_sym;
  auto const hdst = r->inter_symbols.data();
  auto const pdst = hdst + _header_num_data_sym;
  for (size_t i = 0; i < hint.size(); ++i) {
    hdst[hint[i]] = hsrc[i];
  }
  for (size_t i = 0; i < pint.size(); ++i) {
    pdst[pint[i]] = psrc[i];
  }
  return nsymb;
}

int phy_tx::spread_symbols(std::vector<OFDMSymbolParams *> const &symbols,
                           DFTSOFDMFrameParams const &params, int ridx) {
  auto const r = _resources[ridx];
  int ns = 0;
  auto ip = r->inter_symbols.data();
  auto op = r->spread_symbols.data();
  for (auto const &symbol : symbols) {
    if (symbol->numDataCarriers() > 0) {
      assert(symbol->numDataCarriers() == params.dft_spread_length);
      r->fft->execute(params.dft_spread_length, ip, op);
      ip += params.dft_spread_length;
      op += params.dft_spread_length;
      ns += params.dft_spread_length;
    }
  }
  return ns;
}

int phy_tx::map_subcarriers(std::vector<OFDMSymbolParams *> const &symbols,
                            int ridx) {
  // n.b. all preparations for oversampling are covered in src/ofdm.cc when
  // the symbols in the symbol table are defined. This is not the prettiest
  // way of handling things, but I doubt that we will ever clean this
  // structure up. Ideally, the symbol table would be agnostic to the
  // oversampling factor and all relevant adjustments are made here, in the
  // modulator function. This is a project for when there is some downtime,
  // I suppose. (Or as part of the debt repayment package for Phase 3.)
  auto const r = _resources[ridx];
  int nmapped = 0;
  auto ip = r->spread_symbols.data();
  auto op = r->ifft_in.data();
  auto write_symbols = [](auto ofdm_symbol, auto mapping, auto n, auto in,
                          auto out) {
    auto const nfft = ofdm_symbol->symbol_length * ofdm_symbol->oversample_rate;
    for (size_t i = 0; i < n; ++i) {
      auto out_idx = *mapping++;
      if (out_idx >= ofdm_symbol->symbol_length / 2) {
        out_idx += nfft - ofdm_symbol->symbol_length;
      } else if (out_idx < 0) {
        out_idx += nfft;
      }
      assert(out_idx >= 0 && out_idx < nfft);
      out[out_idx] = *in++;
    }
  };
  for (auto const &symbol : symbols) {
    // data symbols
    auto ndata = symbol->numDataCarriers();
    auto data_map =
        pmt::s32vector_elements(symbol->data_carrier_mapping, ndata);
    write_symbols(symbol, data_map, ndata, ip, op);
    // pilot symbols
    auto npilots = symbol->numPilotCarriers();
    auto pilot_map =
        pmt::s32vector_elements(symbol->pilot_carrier_mapping, npilots);
    auto pilot_symbols =
        pmt::c32vector_elements(symbol->pilot_symbols, npilots);
    write_symbols(symbol, pilot_map, npilots, pilot_symbols, op);
    // pointer advance
    if (ndata > 0) {
      ip += ndata;
    }
    if (ndata + npilots > 0) {
      op += symbol->symbol_length * symbol->oversample_rate;
    }
    nmapped += ndata + npilots;
  }
  return nmapped;
}

int phy_tx::shift_upsample(std::vector<OFDMSymbolParams *> const &symbols,
                           std::vector<fcomplex> &dest, int ridx) {
  // n.b.  In order to minimize copies, we want to directly write into the
  // 'dest' buffer we created earlier. This means that we have to write out
  // OFDM symbols with enough spacing between them to fit their cyclic
  // prefixes and postfixes.  The cylic prefixer operation below will then
  // compute the windowed cylic prefix and postfix symbols for each OFDM
  // symbol and add them accordingly.
  auto const r = _resources[ridx];
  int nsamp = 0;
  auto ip = r->ifft_in.data();
  auto op = dest.data();
  for (auto const &symbol : symbols) {
    if (symbol->prefix) {
      // n.b. copy/paste from cyclic_prefixer_impl.cc, why not just use
      // pmt::c32vector_elements(...) ?
      size_t prefix_vec_size = 0;
      auto prefix_vec =
          pmt::uniform_vector_elements(symbol->prefix, prefix_vec_size);
      auto prefix_vec_len = prefix_vec_size / sizeof(fcomplex);
      assert(prefix_vec_len = symbol->symbol_length * symbol->oversample_rate);
      // write the prefix symbols without their cylic prefix
      op += symbol->cyclic_prefix_length * symbol->oversample_rate;
      memcpy(op, prefix_vec, prefix_vec_size);
      op += prefix_vec_len;
      nsamp += prefix_vec_len;
    }
    if (symbol->numDataCarriers() + symbol->numPilotCarriers() > 0) {
      // shift the next OFDM symbol into the time domain
      auto nfft = symbol->symbol_length * symbol->oversample_rate;
      op += symbol->cyclic_prefix_length * symbol->oversample_rate;
      r->ifft->execute(nfft, ip, op);
      op += nfft;
      ip += nfft;
      nsamp += nfft;
    }
  }
  return nsamp;
}

int phy_tx::window_cp(std::vector<OFDMSymbolParams *> const &symbols,
                      std::vector<fcomplex> &samples, int ridx) {
  // if the previous step was correct, we should be perfectly set-up to
  // compute and write out the cyclic prefixes and postfixes for each OFDM
  // symbol.
  auto const r = _resources[ridx];
  // argument bp points to zeroed-out scratch space for intermediate CP
  // calculation.
  auto do_prepostfix = [](auto sp, auto symbol, auto bp) {
    auto const os = symbol->oversample_rate;
    auto const Ncp = symbol->cyclic_prefix_length;
    auto const Nw = symbol->cyclic_postfix_length;

    auto const first = sp;
    auto const post_start = sp + symbol->symbol_length * os;
    auto const pre_start = first - (Ncp * os);

    auto const head = get_window(true, Nw, os);
    auto const tail = get_window(false, Nw, os);
    assert(head.size() == Nw * os);
    assert(tail.size() == Nw * os);

    // copy the windowed part of the cyclic prefix into the scratch space
    // and apply the window
    auto const cp_begin = post_start - Ncp * os;
    auto const cp_half = cp_begin + Nw * os;
    assert(cp_half - cp_begin == head.size());
    volk_32fc_32f_multiply_32fc(bp, cp_begin, head.data(), Nw * os);

    // add the windowed part of of the cyclic prefix to the buffer
    volk_32f_x2_add_32f((float *)pre_start, (float *)pre_start, (float *)bp,
                        Nw * os * 2);

    // copy the remaining part of the cyclic prefix into
    memcpy(pre_start + Nw * os, cp_half, sizeof(fcomplex) * (Ncp - Nw) * os);

    // memcopy the cyclic postfix to the right position and apply the window
    volk_32fc_32f_multiply_32fc(post_start, first, tail.data(), Nw * os);
  };
  auto sp = samples.data();
  auto bp = r->cp_buf.data();
  for (auto const &symbol : symbols) {
    auto NcpOs = symbol->cyclic_prefix_length * symbol->oversample_rate;
    if (symbol->prefix) {
      sp += NcpOs;
      do_prepostfix(sp, symbol, bp);
      sp += symbol->symbol_length * symbol->oversample_rate;
    }
    if (symbol->numDataCarriers() + symbol->numPilotCarriers() > 0) {
      sp += NcpOs;
      do_prepostfix(sp, symbol, bp);
      sp += symbol->symbol_length * symbol->oversample_rate;
    }
  }

  return (sp - samples.data()) + symbols.back()->cyclic_postfix_length *
                                     symbols.back()->oversample_rate;
}

int phy_tx::freq_shift(std::vector<fcomplex> &samples, double const rot_phase,
                       size_t n) {
  auto phase_incr = std::exp(fcomplex(0, rot_phase));
  fcomplex phase = 1;
  volk_32fc_s32fc_x2_rotator_32fc(samples.data(), samples.data(), phase_incr,
                                  &phase, n);
  return n;
}

int phy_tx::multiply_const(std::vector<fcomplex> &samples,
                           float const pc_factor, size_t n) {
  volk_32f_s32f_multiply_32f((float *)samples.data(), (float *)samples.data(),
                             pc_factor, n * 2);
  return n;
}

uint64_t phy_tx::stream(std::vector<fcomplex> const &samples, size_t n) {
  {
    std::unique_lock<decltype(_smtx)> l(_smtx);
    _stream_cv.wait(l, [this] { return _stream_avail; });
    _stream_avail = false;
  }
  uhd::tx_metadata_t md;
  md.start_of_burst = true;
  md.end_of_burst = false;
  md.has_time_spec = false;
  uint64_t nsent = 0;
  auto const mspp = _stream->get_max_num_samps();
  while (nsent < n) {
    nsent += _stream->send(samples.data() + nsent, std::min(mspp, n - nsent),
                           md, 1.0);
    md.start_of_burst = false;
  }
  md.end_of_burst = true;
  _stream->send("", 0, md, 1.0);
  {
    std::lock_guard<decltype(_smtx)> l(_smtx);
    _stream_avail = true;
  }
  _stream_cv.notify_all();
  NotificationCenter::shared.post(
      BurstSendEvent,
      BurstSendEventInfo{
          std::chrono::system_clock::now().time_since_epoch().count()});
  return nsent;
}

void phy_tx::start(frame_maker fm) {
  std::unique_lock<decltype(_wmtx)> wl(_wmtx);

  if (!_connected) {
    panic("Must connect phy_tx before starting.");
  }

  // Not stopped until _fm is null.
  while (_fm) {
    wl.unlock();
    stop();
    wl.lock();
  }

  _fm = fm;
  _work_threads_running = true;
  _work_threads_avail = 0;

  wl.unlock();

  std::unique_lock<decltype(_wmtx)> thrwl(_thrwmtx);

  for (size_t i = 0; i < _nthreads; ++i) {
    _work_threads.emplace_back([this, i] {
      cpu_thread_idx = i;
      bamradio::set_thread_name((boost::format("phy_tx_%1%") % i).str());

      std::unique_lock<decltype(_wmtx)> l(_wmtx);

      while (_work_threads_running) {
        auto const fmc = _fm;
        l.unlock();
        auto const pf = fmc();
        if (pf.frame) {
          _send(pf);
          l.lock();
        } else {
          l.lock();
          ++_work_threads_avail;
          _work_cv.wait(l);
          --_work_threads_avail;
        }
      }
    });
  }
}

void phy_tx::continue_frame_maker() {
  std::unique_lock<decltype(_wmtx)> l(_wmtx);
  l.unlock();
  _work_cv.notify_one();
}

void phy_tx::stop() {
  std::unique_lock<decltype(_wmtx)> wl(_wmtx);
  if (!_work_threads_running) {
    // Stopped or stopping
    return;
  }
  _work_threads_running = false;
  _work_cv.notify_all();
  wl.unlock(); // unlock after notify to preserve ordering
  std::unique_lock<decltype(_wmtx)> thrwl(_thrwmtx);
  for (auto &t : _work_threads) {
    t.join();
  }
  _work_threads.clear();
  thrwl.unlock();
  wl.lock();
  _fm = nullptr;
}

void phy_tx::_send(prepared_frame const &pf) {
  using namespace std::chrono;
  // get parameters for this frame
  auto const params = pf.frame->allParams(true, pf.channel.interp_factor, 0);
  auto const nsamp = params.numTXSamples();
  // initialize resources
  _resources[cpu_thread_idx]->init(params);
  // get symbols as vector
  auto const symbols = [&params] {
    std::vector<OFDMSymbolParams *> o;
    for (auto const &symbol : params.symbols)
      for (size_t i = 0; i < symbol.first; ++i)
        o.push_back(symbol.second.get());
    return o;
  }();
  // timing
  int64_t t_code_ns, t_mod_ns, t_spread_ns, t_inter_ns, t_map_ns, t_shift_ns,
      t_cp_ns, t_mix_ns, t_scale_ns, t_stream_ns;
#define PHYTX_TIMED_EXEC(expr, val)                                            \
  do {                                                                         \
    auto const tick = steady_clock::now();                                     \
    (expr);                                                                    \
    (val) = duration_cast<nanoseconds>(steady_clock::now() - tick).count();    \
  } while (false)
  // FIXME: refactor (need to re-do unit tests...)
  std::vector<fcomplex> &out = _resources[cpu_thread_idx]->out;
  // perform signal processing
  size_t nsymb = 0;
  PHYTX_TIMED_EXEC(encode_bits(pf.frame, params, cpu_thread_idx), t_code_ns);
  PHYTX_TIMED_EXEC(nsymb = modulate_bits(symbols, cpu_thread_idx), t_mod_ns);
  PHYTX_TIMED_EXEC(interleave(nsymb, cpu_thread_idx), t_inter_ns);
  PHYTX_TIMED_EXEC(spread_symbols(symbols, params, cpu_thread_idx),
                   t_spread_ns);
  PHYTX_TIMED_EXEC(map_subcarriers(symbols, cpu_thread_idx), t_map_ns);
  PHYTX_TIMED_EXEC(shift_upsample(symbols, out, cpu_thread_idx), t_shift_ns);
  PHYTX_TIMED_EXEC(window_cp(symbols, out, cpu_thread_idx), t_cp_ns);
  PHYTX_TIMED_EXEC(freq_shift(out, pf.channel.rotator_phase(), nsamp),
                   t_mix_ns);
  PHYTX_TIMED_EXEC(multiply_const(out, pf.channel.sample_gain, nsamp),
                   t_scale_ns);
  // stream to USRP
  PHYTX_TIMED_EXEC(stream(out, nsamp), t_stream_ns);

  int64_t const l2_tx_time =
      std::chrono::system_clock::now().time_since_epoch().count();

  NotificationCenter::shared.post(
      dll::SentFrameEvent,
      dll::SentFrameEventInfo{.sourceNodeID = pf.frame->sourceNodeID(),
                              .destNodeID = pf.frame->destinationNodeID(),
                              .payloadMCS = pf.frame->payloadMcs(),
                              .payloadSymSeqID = pf.frame->payloadSymSeqID(),
                              .nsamples = (int64_t)params.numTXSamples(),
                              .seqNum = pf.frame->seqNum(),
                              .frameID = pf.frame->frameID(),
                              .numBlocks =
                                  pf.frame->numBlocks(false), // numblocks
                              .txTime = l2_tx_time,
                              .sampleGain = pf.channel.sample_gain});

  for (auto const &s : pf.frame->segments()) {
    nlohmann::json segmentJson = s;
    int64_t const st = s->sourceTime().time_since_epoch().count();
    NotificationCenter::shared.post(
        dll::SentSegmentEvent,
        dll::SentSegmentEventInfo{.flow = s->flowID(),
                                  .sourceNodeID = pf.frame->sourceNodeID(),
                                  .destNodeID = s->destNodeID(),
                                  .seqNum = pf.frame->seqNum(),
                                  .frameID = pf.frame->frameID(),
                                  .sourceTime = st,
                                  .description = segmentJson,
                                  .type = (int)s->type(),
                                  .nbytes = s->length()});
  }
  // post to log
  // FIXME: log t_inter properly
  NotificationCenter::shared.post(
      ModulationEvent,
      ModulationEventInfo{t_code_ns, t_mod_ns, t_spread_ns + t_inter_ns,
                          t_map_ns, t_shift_ns, t_cp_ns, t_mix_ns, t_scale_ns,
                          t_stream_ns, pf.frame->sourceNodeID(),
                          pf.frame->frameID()});
}

void phy_tx::connect(uhd::tx_streamer::sptr stream) {
  std::unique_lock<decltype(_wmtx)> l(_wmtx);
  _stream = stream;
  _connected = true;
}

//
// Synchronization & detection
//

#define is_even(v) (((v) & static_cast<decltype(v)>(1)) == 0)
#define power_of_2(v) ((v) && !((v) & ((v)-1)))

const std::vector<fcomplex> &get_preamble_pilot_symbols(const size_t L,
                                                        size_t const O) {
  // Compute the peamble only once for a given L;
  static std::map<std::pair<size_t, size_t>, std::vector<fcomplex>> p;
  auto &pi = p[{L, O}];
  if (pi.size() > 0) {
    return pi;
  }

  // sync this with get_preamble_td
  const unsigned int M = 3; // gcd(M,N) = 1
  pi = gr::bamofdm::generate_cazac_seq(L, M);

  auto const ntds = O * 2 * L;

  // Insert zeros in CAZAC
  std::vector<fcomplex> cazac0(ntds, fcomplex(0.0, 0.0));
  for (size_t i = 0; i < L / 2; ++i) {
    cazac0[i * 2] = pi[i];
    cazac0[ntds - L + i * 2] = pi[L / 2 + i];
  }

  // Take IFFT of cazac.
  gr::fft::fft_complex ifft(ntds, false);
  memset(ifft.get_inbuf(), 0x00, sizeof(fcomplex) * ifft.inbuf_length());
  std::copy(cazac0.begin(), cazac0.end(), ifft.get_inbuf());
  ifft.execute();

  // Copy out IFFT result
  std::vector<fcomplex> preamble(ntds);
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

const std::vector<fcomplex> &get_preamble_td_oversampled(const size_t L,
                                                         size_t const O) {
  // Compute the peamble only once for a given L;
  static std::map<std::pair<size_t, size_t>, std::vector<fcomplex>> preambles;

  auto &preamble = preambles[{L, O}];

  if (preamble.size() > 0) {
    return preamble;
  }

  // TODO generalize to arbitrary fft_len and set M correctly.

  if (!power_of_2(L)) {
    log::doomsday("get_preamble_td: L not power of 2.", __FILE__, __LINE__);
  }
  const unsigned int M = 3; // gcd(M,N) = 1

  // Generate CAZAC sequence.
  auto const cazac = gr::bamofdm::generate_cazac_seq(L, M);

  auto const ntds = O * 2 * L;

  // Insert zeros in CAZAC
  std::vector<fcomplex> cazac0(ntds, fcomplex(0.0, 0.0));
  for (size_t i = 0; i < L / 2; ++i) {
    cazac0[i * 2] = cazac[i];
    cazac0[ntds - L + i * 2] = cazac[L / 2 + i];
  }

  // Take IFFT of cazac.
  gr::fft::fft_complex ifft(ntds, false);
  memset(ifft.get_inbuf(), 0x00, sizeof(fcomplex) * ifft.inbuf_length());
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
// Frame detection
//

frame_detect_demux::frame_detect_demux(size_t Nc, size_t Ns,
                                       size_t num_pre_header_ofdm_syms,
                                       float preamble_threshold, demodfn demod,
                                       interpret_frame interpret_frame)
    : _interpret_frame(interpret_frame), _state(State::SEARCHING),
      _inbuf(nullptr), _running(false), _waiting(false), _ph_Nc(Nc), _ph_Ns(Ns),
      _num_pre_header_ofdm_syms(num_pre_header_ofdm_syms),
      _threshold(preamble_threshold), _Nc(Nc), _Ns(Ns), _domega(0.0f),
      _phi(1.0f, 0.0f), _searching_skip(0), _pre_header_num_syms_remaining(0),
      _current_frame(nullptr), _num_payload_ofdm_syms(0),
      _payload_num_syms_remaining(0), _current_Md(0.0), _avg_len(5),
      _demod(demod) {}

frame_detect_demux::~frame_detect_demux() {
  if (_running) {
    stop();
  }
}

void frame_detect_demux::start(ChannelOutputBuffer::sptr inbuf,
                               std::string const &name = "fdd") {
  _inbuf = inbuf;
  _running = true;
  _work_thread = std::thread([this, name] {
    bamradio::set_thread_name(name);
    while (_running) {
      auto const navail = _inbuf->items_avail();
      auto const nconsumed =
          work(navail, _inbuf->samples->read_ptr(), _inbuf->Md->read_ptr(),
               _inbuf->Pd->read_ptr());
      _inbuf->consume_each(nconsumed);
    }
  });
}

void frame_detect_demux::stop() {
  _running = false;
  // might not be the best thing to do, but this will work
  if (_state == State::PRE_HEADER_WAIT) {
    notify(nullptr);
  }
  _work_thread.join();
}

void frame_detect_demux::notify(DFTSOFDMFrame::sptr frame) {
  if (_state != State::PRE_HEADER_WAIT) {
    // drop everything and start over in SEARCHING state. this should NEVER
    // happen. in the impossible case that it does, we log.
    std::lock_guard<decltype(_mtx)> l(_mtx);
    _waiting = false;
    _state = State::SEARCHING;
    log::text(
        (boost::format("FDD notified, but _state == %1%") % (int)_state).str(),
        __FILE__, __LINE__);
    return;
  }
  // this is what we really want to do here. set the current frame based on
  // whether we successfully decoded a header or not.
  {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    _waiting = false;
    _current_frame = frame;
  }
  _cv.notify_one();
}

size_t frame_detect_demux::output_ofdm_symbols(fcomplex *out, size_t nout,
                                               fcomplex const *r, size_t nin,
                                               size_t max_num_ofdm_syms) {
  size_t k = 0;

  while (nin >= (_Ns + _Nc) && nout >= _Ns && k < max_num_ofdm_syms) {
    auto const domega = fcomplex(cos(-_domega), sin(-_domega));
    volk_32fc_s32fc_x2_rotator_32fc(out, r, domega, &_phi, _Ns);
    ++k;
    out += _Ns;
    nout -= _Ns;
    r += (_Ns + _Nc);
    nin -= (_Ns + _Nc);
    // Advance phi past the cyclic prefix.
    _phi *= fcomplex(cos(-_domega * _Nc), sin(-_domega * _Nc));
  }

  return k;
}

size_t frame_detect_demux::detect_preamble(float &domega, float &Md_t,
                                           ssize_t &t, float const *Md,
                                           fcomplex const *Pd, size_t const N) {
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

size_t frame_detect_demux::work(int navail, fcomplex const *r, float const *Md,
                                fcomplex const *Pd) {
  auto const Nin = static_cast<size_t>(std::max<int>(
      0, navail - (_state == State::SEARCHING ? (int)_Ns - 1 : 0)));
  if (Nin == 0) {
    return 0;
  }

  size_t nin = Nin;

  auto fdd_consume = [&](auto const n) {
    r += n;
    Md += n;
    Pd += n;
    nin -= n;
  };

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
      fdd_consume(nskip);
      break;
    }
    // else preamble detected to start at t
    if (_Md_history.empty()) {
      for (int i = 0; i < _avg_len; ++i)
        _Md_history.push_back(Md_t);
      _current_Md = Md_t;
    } else {
      _Md_history.push_back(Md_t);
      _current_Md += (_Md_history.back() - _Md_history.front()) / _avg_len;
      _Md_history.pop_front();
    }

    // save SNR and estimated frequency offset to pass on to demodulator
    auto snr =
        10 * std::log10(std::sqrt(_current_Md) / (1 - std::sqrt(_current_Md)));
    _sc_info = SCInfo{.snr = snr, .domega = _domega};

    // when we consume, probably need to consume at a minimum slightly more
    // than t, so remember that minimium amount to consume
    assert(nskip >= (size_t)t);
    _searching_skip = nskip - t;

    // consume up to the trigger
    fdd_consume(t);

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
      _state = State::PRE_HEADER_WAIT;
      _current_frame = nullptr;
      _waiting = true;
      _demod(nullptr, std::move(_hdr_buf), _sc_info, this);
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
  //   - Next state: PAYLOAD
  case State::PRE_HEADER_WAIT: {
    // wait for header decode message
    if (_waiting) {
      std::unique_lock<decltype(_mtx)> l(_mtx);
      _cv.wait(l, [this] { return !_waiting; });
    }
    if (_current_frame == nullptr) {
      // decode bad
      if (nin < _searching_skip) {
        break;
      }
      fdd_consume(_searching_skip);
      _state = State::SEARCHING;
      _searching_skip = 0;
    } else {
      // decode good
      auto const n = _num_pre_header_ofdm_syms * (_Nc + _Ns);
      if (nin < n) {
        // technically this is impossible, but we'll leave it in here...
        break;
      }
      _interpret_frame(_current_frame, _num_payload_ofdm_syms, _Ns, _Nc);
      // post to log
      NotificationCenter::shared.post(
          SynchronizationEvent,
          SynchronizationEventInfo{_sc_info.snr, _sc_info.domega,
                                   _current_frame->sourceNodeID(),
                                   _current_frame->frameID()});
      _searching_skip = 0;
      if (_num_payload_ofdm_syms == 0) {
        // NO payload so skip back to searching.
        fdd_consume(n - _Nc);
        _state = State::SEARCHING;
        _current_frame = nullptr;
        _Nc = _ph_Nc;
        _Ns = _ph_Ns;
        break;
      }
      fdd_consume(n);
      _state = State::PAYLOAD;
      _payload_num_syms_remaining = _num_payload_ofdm_syms;
      _payl_buf.clear();
      _payl_buf.resize(_num_payload_ofdm_syms * _Ns);
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
      fdd_consume(n - _Nc);
    } else {
      fdd_consume(n);
    }
    _payload_num_syms_remaining -= num_syms_produced;
    if (_payload_num_syms_remaining == 0) {
      _demod(_current_frame, std::move(_payl_buf), _sc_info, this);
      _state = State::SEARCHING;
      _current_frame = nullptr;
      _Nc = _ph_Nc;
      _Ns = _ph_Ns;
    }
  } break;
  }
  return Nin - nin;
}

// Interleaver
Interleaver::Interleaver() {
  // initialize with some reasonable number of OFDM symbols and assuming the DFT spread length is '108'
  for (size_t i = 3; i < 300; ++i) {
    operator()(i * 108);
  }
}

std::vector<size_t> const &Interleaver::operator()(size_t n) {
  std::unique_lock<decltype(_mtx)> l(_mtx);
  try {
    return _map.at(n);
  } catch (std::out_of_range) {
    std::mt19937 rng(n);
    std::vector<size_t> v(n);
    int k = 0;
    std::generate(begin(v), end(v), [&k] { return k++; });
    std::shuffle(begin(v), end(v), rng);
    _map.emplace(n, v);
    l.unlock();
    return operator()(n);
  }
}

Interleaver Interleaver::get;

} // namespace ofdm

//
// Control channel PHY
//
namespace controlchannel {

// eternal constants
const std::vector<uint8_t> phy_ctrl_mod::preamble{73, 25, 128, 247,
                                                  8,  95, 134, 4};

const std::vector<fcomplex> phy_ctrl_mod::pulse_shape{[] {
  // Symbol duration is equivalent to 10 samples @ fs=480kHz
  // (10/480e3)/(1/46.08e6)=960
  // Raised cosine pulse shape
  std::vector<fcomplex> pulse(960);
  for (size_t i = 0; i < pulse.size(); ++i) {
    pulse[i] = (1 - std::cos(2 * M_PI * i / pulse.size())) / 2.0;
  }
  return pulse;
}()};

phy_ctrl_mod::phy_ctrl_mod(float sample_rate, float bandwidth, size_t rs_k,
                           size_t npoints, float scale)
    : _sample_rate(sample_rate), _bandwidth(bandwidth), _scale(scale),
      _rs_k(rs_k), _npoints(npoints), _bps(std::log2(npoints)),
      _freq_table_normalized([=] {
        std::vector<float> fdv(npoints);
        auto const start =
            bandwidth * (-1.0 / 2.0 + 2.0 / (2.0 * 10 + 1)) / sample_rate;
        auto const stop =
            bandwidth * (1.0 / 2.0 - 2.0 / (2.0 * 10 + 1)) / sample_rate;
        auto const step = (stop - start) / (fdv.size() - 1);
        size_t k = 0;
        std::generate(begin(fdv), end(fdv),
                      [&] { return start + (step * k++); });
        return fdv;
      }()) {
  if (_rs_k < 1) {
    throw std::runtime_error("this makes no sense.");
  }
  if (sample_rate < bandwidth) {
    throw std::runtime_error("you need to respect harry.");
  }
}

size_t phy_ctrl_mod::mod(CCData::sptr cc_data, float const freq_offset_hz,
                         Resource *r, std::vector<fcomplex> &out) const {
  // check frequency value
  if (freq_offset_hz < (-1.0 * _sample_rate / 2.0) ||
      (_sample_rate / 2.0) < freq_offset_hz) {
    throw std::runtime_error(
        "FSK Tx frequency offset is outside of sample bandwidth");
  }
  // get a fresh control packet
  auto const raw_data_size = cc_data->getNbytes();
  if (r->raw_data.size() < raw_data_size) {
    r->raw_data.resize(raw_data_size);
  }
  cc_data->serializeShortMsg(
      boost::asio::mutable_buffer(r->raw_data.data(), raw_data_size));
  // process the data
  auto const ncrc = _crc32(r);
  auto const ncoded = _encode(ncrc, r);
  auto const nsymb = _unpack(ncoded, r);
  // write to output buffer
  auto const nsamp = _mfsk_mod(nsymb, freq_offset_hz, r, out);
  return nsamp;
}

size_t phy_ctrl_mod::_crc32(Resource *r) const {
  // grow memory if need be
  auto const nbytes = r->raw_data.size();
  if (r->bytes.size() < nbytes + 4) {
    r->bytes.resize(nbytes + 4);
  }
  // copy fresh bytes in first buffer
  memcpy(r->bytes.data(), r->raw_data.data(), nbytes);
  // compute CRC and add to buffer
  r->_crc.reset();
  r->_crc.process_bytes(r->bytes.data(), nbytes);
  *((unsigned int *)(r->bytes.data() + nbytes)) = r->_crc();
  return nbytes + 4;
}

size_t phy_ctrl_mod::_encode(size_t ncrc, Resource *r) const {
  namespace rs = gr::bamfsk::rs::ccsds;
  // grow memory if need be
  auto const nparity = (size_t)std::ceil(((double)ncrc) / (double)_rs_k);
  auto const enc_size = ncrc + nparity * rs::PARITY;
  auto const pre_size = phy_ctrl_mod::preamble.size();
  auto const cwp_size = pre_size + enc_size;
  if (r->coded_with_preamble.size() < cwp_size) {
    r->coded_with_preamble.resize(cwp_size);
    memcpy(r->coded_with_preamble.data(), phy_ctrl_mod::preamble.data(),
           pre_size);
    r->rs_start = r->coded_with_preamble.data() + pre_size;
  }
  if (r->rs_buf.size() < rs::K) {
    r->rs_buf.resize(rs::K);
  }
  // encode bytes. copypasta from the smelly rs_ccsds_encode_bb_impl.cc
  memset(r->rs_start, 0x00, enc_size); // safer to initialize this
  for (size_t i = 0; i < nparity; ++i) {
    auto const kp = std::min(ncrc - i * _rs_k, _rs_k);
    auto const in = r->bytes.data() + i * _rs_k;
    auto out = r->rs_start + i * (_rs_k + rs::PARITY);
    memset(r->rs_buf.data(), 0x00, rs::K);
    memcpy(r->rs_buf.data() + (rs::K - kp), in, kp);
    memcpy(out, in, kp);
    rs::encode(r->rs_buf.data(), out + kp);
  }
  return cwp_size;
}

// copypasta from gr::blocks::repack_bits_bb_impl
size_t phy_ctrl_mod::_unpack(size_t nbytes, Resource *r) const {
  auto const k = 8;
  auto const l = _bps;
  // grow memory if need be
  auto nout = nbytes * k / l;
  if (((nbytes * k) % l) != 0) {
    ++nout;
  }
  if (r->chunks.size() < nout) {
    r->chunks.resize(nout);
  }
  // repack bits to chunks
  size_t nw = 0, nr = 0, i = 0, j = 0;
  while (nw < nout && nr < nbytes) {
    if (j == 0) {
      r->chunks[nw] = 0;
    }
    r->chunks[nw] |= ((r->coded_with_preamble[nr] >> i) & 0x01) << j;
    i = (i + 1) % k;
    j = (j + 1) % l;
    if (i == 0) {
      ++nr;
      i = 0;
    }
    if (j == 0) {
      ++nw;
      j = 0;
    }
  }
  return nout;
}

size_t phy_ctrl_mod::_mfsk_mod(size_t nsymb, float const freq_offset_hz,
                               Resource *r, std::vector<fcomplex> &out) const {
  // grow memory if need be
  auto const nsamp_per_symb = pulse_shape.size();
  auto const nsamp = nsymb * nsamp_per_symb;
  if (out.size() < nsamp) {
    out.resize(nsamp);
  }

  // modulate symbols
  auto wptr = out.data();
  fcomplex phase = 1;
  float const freq_offset_normalized = freq_offset_hz / _sample_rate;
  for (size_t i = 0; i < nsymb; ++i) {
    auto const symb = r->chunks[i];
    float const tone_freq_normalized =
        freq_offset_normalized + _freq_table_normalized[symb];
    auto const phase_incr =
        std::exp(fcomplex(0, 2.0 * M_PI * tone_freq_normalized));
    volk_32fc_s32fc_x2_rotator_32fc(wptr, pulse_shape.data(), phase_incr,
                                    &phase, nsamp_per_symb);
    wptr += nsamp_per_symb;
  }

  return nsamp;
}

} // namespace controlchannel

} // namespace bamradio
