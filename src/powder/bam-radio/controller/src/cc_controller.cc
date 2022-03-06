#include "cc_controller.h"
#include "bandwidth.h"
#include "net.h"
#include "options.h"
#include "segment.h"
#include "util.h"

#include <boost/asio.hpp>

#include <cmath>
#include <random>
#include <string>

// debug print statements
// #define CCC_DEBUG 1
#ifdef CCC_DEBUG
#include <ctime>
#include <iostream>
#endif

using namespace std::string_literals;

namespace bamradio {
namespace controlchannel {

NotificationCenter::Name const CCTuneNotification =
    NotificationCenter::makeName("CC Tune Request");

using namespace boost::asio;
using namespace std::literals::chrono_literals;
using std::chrono::duration_cast;

// FIXME: get rid of options... Phase 3 backlog
CCController::CCController(controlchannel::CCData::sptr ccdata,
                           bamradio::AbstractDataLinkLayer::sptr dll,
                           FSKTransmitter fsk_tx)
    : _ccdata(ccdata), _silent(false), _srnid(options::phy::control::id), _ofdm_dll(dll),
      _t_ofdm(duration_cast<Duration>(std::chrono::duration<float>(
          options::phy::control::ccsegment_interval))),
      // FSK
      _fsk_tx(fsk_tx),
      _m(bam::dsp::sample_rate, options::phy::control::bandwidth,
         options::phy::control::rs_k, options::phy::control::num_fsk_points,
         options::phy::control::atten),
      _fsk_rx(controlchannel::ctrl_ch::make(
          _ccdata, options::phy::max_n_nodes, options::phy::control::id,
          options::phy::control::sample_rate, options::phy::control::t_slot,
          options::phy::control::atten, options::phy::control::num_fsk_points,
          options::phy::control::rs_k, 600)),
      _t_ref(duration_cast<Timepoint::duration>(
          std::chrono::seconds(1485882000))), // FIXME hardcoded
      _t_last_fsk(std::chrono::system_clock::now()),
      _t_slot(duration_cast<Duration>(
          std::chrono::duration<double>(options::phy::control::t_slot))),
      _t_proc(5ms), // FIXME hardcoded again
      _coin_dist(0, options::phy::max_n_nodes - 1),
      // hopping
      _cchop_freq_table(_computeBandEdges(options::phy::bandwidth)),
      _cchop_rnd_dist(0, _cchop_freq_table.size() - 1),
      _t_hop(1s), // FIXME this is hardcoded
      _t_last_hop(std::chrono::system_clock::now()),
      _ios_work(new boost::asio::io_service::work(_ios)), _work_thread([this] {
        bamradio::set_thread_name("cc_controller");
        _ios.run();
      }),
      _timer_ofdm(_ios), _timer_fsk(_ios), _timer_hop(_ios),
      _timer_chan_update(_ios) {
  // enforce rules
  if (_t_proc >= _t_slot) {
    throw std::runtime_error("CCController: t_proc >= t_slot");
  }
  // initialize
  _rng1.seed(std::random_device()());
  // Subscribe to OFDM control segment notification
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<
          std::pair<dll::Segment::sptr, std::shared_ptr<std::vector<uint8_t>>>>(
          std::hash<std::string>{}("New Rx CC Segment"), _ios,
          [this](auto data) {
            auto cc_seg = std::dynamic_pointer_cast<
                controlchannel::ControlChannelSegment>(data.first);
            _ccdata->deserialize(cc_seg->packetContentsBuffer(), false);
          }));
}

void CCController::start() {
  auto now = std::chrono::system_clock::now();

  // OFDM timer
  _timer_ofdm.expires_at(now + _t_ofdm + 500ms); // fudging to avoid overlap
  _timer_ofdm.async_wait([this](auto &e) { this->_sendOFDM(); });

  // FSK timer
  auto const next_fsk = _nextSlotBoundary(_t_slot, _t_last_fsk);
  _timer_fsk.expires_at(next_fsk - _t_proc);
  _timer_fsk.async_wait(
      [this, next_fsk](auto &e) { this->_sendFSK(next_fsk); });

  // Hop timer
  auto const next_hop = _nextSlotBoundary(_t_hop, _t_last_hop);
  _timer_hop.expires_at(next_hop);
  _timer_hop.async_wait([this](auto &e) { this->_updateFrequency(); });

  // Set silent mode
  d_nc_tokens.push_back(
      NotificationCenter::shared
      .subscribe<controlchannel::CCData::OFDMChannelUpdateInfo>(
          controlchannel::CCData::OFDMChannelBandNotification, _ios,
          [this](auto v) {
            if (v.channels.find(_srnid) != v.channels.end()) {
              // schedule channel update
              _timer_chan_update.expires_at(v.t_effective);
              _timer_chan_update.async_wait([=](auto &e) {
                this->_silent = v.channels.at(_srnid).silent;
              });
            }
          }));
}

// re-compute the band edges for the freq_table when the RF bandwidth changes
std::vector<float> CCController::_computeBandEdges(int64_t rf_bandwidth) const {
  decltype(_cchop_freq_table) new_freq_table;
  new_freq_table.reserve(2);
  auto new_center_freq =
      (((float)rf_bandwidth) / 2.0f) - options::phy::control::band_edge_offset;
  new_freq_table.push_back(-1.0 * new_center_freq);
  new_freq_table.push_back(new_center_freq);
  return new_freq_table;
}

void CCController::updateRFBandwidth(int64_t rf_bandwidth) {
  std::lock_guard<decltype(_channalloc_mtx)> l(_channalloc_mtx);
  auto const nft = _computeBandEdges(rf_bandwidth);
  _cchop_freq_table = nft;
}

std::vector<float> CCController::getFreqTable() const {
  std::lock_guard<decltype(_channalloc_mtx)> l(_channalloc_mtx);
  return _cchop_freq_table;
}

size_t CCController::_hopIdx(Timepoint t) {
  auto const seconds =
      duration_cast<std::chrono::seconds>(t.time_since_epoch());
  _rng2.seed(seconds.count());
  return _cchop_rnd_dist(_rng2);
}

void CCController::forceRetune() {
  _ios.post([this] {
    std::lock_guard<decltype(_channalloc_mtx)> l(_channalloc_mtx);
    NotificationCenter::shared.post(CCTuneNotification,
                                    _cchop_freq_table[_cchop_prev_idx]);
  });
}

void CCController::_updateFrequency() {
  auto new_idx = _hopIdx(_timer_hop.expiry());

#ifdef CCC_DEBUG
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      _timer_hop.expiry().time_since_epoch());
  std::cout << "CCCDEBUG: _updateFrequency called at t = " << ms.count()
            << ". index = " << (int)new_idx << std::endl;
#endif

  if (_cchop_prev_idx != new_idx) {
    std::lock_guard<decltype(_channalloc_mtx)> l(_channalloc_mtx);
    NotificationCenter::shared.post(CCTuneNotification,
                                    _cchop_freq_table[new_idx]);
    _cchop_prev_idx = new_idx;
  }
  // Schedule next call
  if (_ios_work) {
    _timer_hop.expires_at(_nextSlotBoundary(_t_hop, _t_last_hop));
    _timer_hop.async_wait([this](auto &e) { this->_updateFrequency(); });
  }
}

CCController::Timepoint CCController::_nextSlotBoundary(Duration t_slot,
                                                        Timepoint &t_last) {
  using namespace std::chrono;
  auto const now = system_clock::now();
  // we use floating point numbers to compute the next time point. this is less
  // precision than the system_clock::duration, so we limit our usage of this to
  // here and the conversion to uhd::time_spec
  using dds = bamradio::double_dur_uhd;
  auto const t_rel = duration_cast<dds>(now - _t_ref);
  auto const d_t_slot = duration_cast<dds>(t_slot);
  auto const t_next_rel =
      dds(std::ceil(t_rel.count() / d_t_slot.count()) * d_t_slot.count());
  Timepoint t_next(duration_cast<Duration>(t_next_rel) + _t_ref);
  // make sure we always return a time point after the last one computed
  while (t_next <= t_last + t_slot / 2) {
    t_next += t_slot;
  }
  t_last = t_next;
  return t_next;
}

void CCController::_sendFSK(Timepoint time) {
  // this buffer holds tx samples
  static std::vector<fcomplex> txbuf;

  // we opportunistically transmit on the successful flip of an num_nodes-sided
  // fair coin
  auto transmitNow = [this]() -> bool { return _coin_dist(_rng1) == 0; };

  // convert the time point to a UHD-friendly format
  auto const time_spec =
      bamradio::duration_to_uhd_time_spec(time.time_since_epoch());

  // tx if can
  if (transmitNow()) {
#ifdef CCC_DEBUG
    auto now = std::chrono::system_clock::now();
    std::cout << "CCCDEBUG: _sendFSK now {" << now.time_since_epoch().count()
              << "} sched {" << time.time_since_epoch().count()
              << "} time_spec {" << time_spec.get_real_secs() << ", "
              << time_spec.get_frac_secs() << "}" << std::endl;
#endif
    _fsk_tx(txbuf,
            _m.mod(_ccdata, _cchop_freq_table[_cchop_prev_idx], &_r, txbuf),
            time_spec);
  }

  // Schedule next call
  if (_ios_work) {
    auto const next_time = _nextSlotBoundary(_t_slot, _t_last_fsk);
    _timer_fsk.expires_at(next_time - _t_proc);
    _timer_fsk.async_wait(
        [this, next_time](auto &e) { this->_sendFSK(next_time); });
  }
}

void CCController::_sendOFDM() {
#ifdef CCC_DEBUG
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      _timer_ofdm.expiry().time_since_epoch());
  std::cout << "CCCDEBUG: _sendOFDM called at t = " << ms.count()
            << " OFDM DLL is "
            << (_ofdm_dll->running() ? "running" : "NOT running") << std::endl;
#endif

  if (_ofdm_dll->running() && !_silent) {
    auto const bs = _ccdata->serialize();
    auto const seg = std::make_shared<ControlChannelSegment>(
        buffer(*bs), std::chrono::system_clock::now());
    try {
      _ofdm_dll->send(seg, bs);
    } catch (std::runtime_error e) {
      log::text("Failed sending control segment on OFDM DLL. "s + e.what());
    }
  }
  // Wait
  if (_ios_work) {
    _timer_ofdm.expires_at(_timer_ofdm.expiry() + _t_ofdm);
    _timer_ofdm.async_wait([this](auto &e) { this->_sendOFDM(); });
  }
}

bool CCController::running() { return d_nc_tokens.size() > 0; }

void CCController::stop() {
  if (running()) {
    d_nc_tokens.clear();
    _timer_ofdm.cancel();
    _timer_fsk.cancel();
    _timer_hop.cancel();
    if (_ios_work) {
      delete _ios_work;
    }
    _ios.stop();
    if (_work_thread.joinable()) {
      _work_thread.join();
    }
  }
}

CCController::~CCController() {
  if (running()) {
    stop();
  }
}

} // namespace controlchannel
} // namespace bamradio
