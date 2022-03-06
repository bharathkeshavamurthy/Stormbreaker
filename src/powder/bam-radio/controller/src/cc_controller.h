// Control channel controller.
//
// Manages Layer 1 and Layer 2 aspects of our FSK control link.
//
// Copyright (c) 2017-2018 Tomohiro Arakawa <tarakawa@purdue.edu>
// Copyright (c) 2017-2018 Dennis Ogbe <dogbe@purdue.edu>

#ifndef CC_CONTROLLER_H_INCLUDED
#define CC_CONTROLLER_H_INCLUDED

#include "cc_data.h"
#include "ctrl_ch.h"
#include "dll.h"
#include "phy.h"
#include "segment.h"

#include <chrono>
#include <random>

namespace bamradio {
namespace controlchannel {

extern NotificationCenter::Name const CCTuneNotification;

class CCController {
public:
  // shared_ptr convention
  typedef std::shared_ptr<CCController> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<CCController>(std::forward<Args>(args)...);
  }

  // attempt to transmit a given number of complex samples at a given time
  typedef std::function<bool(std::vector<fcomplex> const &, size_t,
                             uhd::time_spec_t)>
      FSKTransmitter;

  // tors
  CCController(controlchannel::CCData::sptr ccdata,
               bamradio::AbstractDataLinkLayer::sptr dll,
               FSKTransmitter fsk_tx);
  ~CCController();
  CCController(const CCController &other) = delete;

  // API
  void start();
  void stop();
  bool running();

  void updateRFBandwidth(int64_t rf_bandwidth);
  void forceRetune();
  std::vector<float> getFreqTable() const;
  controlchannel::ctrl_ch::sptr rx() const { return _fsk_rx; }

private:
  typedef std::chrono::system_clock::duration Duration;
  typedef std::chrono::system_clock::time_point Timepoint;

  void _sendOFDM();
  void _sendFSK(Timepoint time);
  void _updateFrequency();
  std::vector<float> _computeBandEdges(int64_t rf_bandwidth) const;
  size_t _hopIdx(Timepoint t);
  Timepoint _nextSlotBoundary(Duration t_slot, Timepoint &t_last);

  // CCData gives us information
  controlchannel::CCData::sptr _ccdata;
  std::vector<NotificationCenter::SubToken> d_nc_tokens;

  // Silent mode
  bool _silent;

  // OFDM DLL/PHY
  NodeID const _srnid;
  bamradio::AbstractDataLinkLayer::sptr _ofdm_dll;
  Duration const _t_ofdm;

  // FSK transmit
  FSKTransmitter _fsk_tx;
  phy_ctrl_mod::Resource _r;
  phy_ctrl_mod _m;

  // FSK receive
  controlchannel::ctrl_ch::sptr _fsk_rx;

  // FSK TDD
  Timepoint const _t_ref; /// Reference time point
  Timepoint _t_last_fsk;  /// Last slot boundary
  Duration const _t_slot; /// Slot duration
  Duration const _t_proc; /// Fudge time for tx processing
  std::uniform_int_distribution<int> _coin_dist;

  // FSK hopping
  std::vector<float> _cchop_freq_table;
  size_t _cchop_prev_idx = 0;
  std::uniform_int_distribution<size_t> _cchop_rnd_dist;
  Duration const _t_hop;
  Timepoint _t_last_hop;

  // io_service / timers
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  std::thread _work_thread; // Single thread

  boost::asio::system_timer _timer_ofdm;        // OFDM Segment generation
  boost::asio::system_timer _timer_fsk;         // FSK Segment generation
  boost::asio::system_timer _timer_hop;         // Frequency hopping
  boost::asio::system_timer _timer_chan_update; // OFDM channel alloc scheduler

  // randomness
  std::mt19937 _rng1;
  std::mt19937 _rng2;

  // threading
  mutable std::mutex _channalloc_mtx;
};
} // namespace controlchannel
} // namespace bamradio

#endif
