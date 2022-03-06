// -*- c++ -*-
// Copyright (c) 2017 Tomohiro Arakawa
// Copyright (c) 2017 Diyu Yang

#ifndef INCLUDED_STATISTICS_H_
#define INCLUDED_STATISTICS_H_
#include "bandwidth.h"
#include "dll_types.h"
#include "events.h"
#include "json.hpp"
#include "notify.h"
#include "util.h"
#include <queue>

#include <algorithm>
#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <thread>

namespace bamradio {
namespace stats {

//
// Notifications & events
//
extern NotificationCenter::Name const RxRateMapNotification;
extern NotificationCenter::Name const OfferedRateNotification;
extern NotificationCenter::Name const DeliveredRateNotification;
extern NotificationCenter::Name const BLERNotification;
extern NotificationCenter::Name const SNRNotification;
extern NotificationCenter::Name const NoiseVarNotification;
extern NotificationCenter::Name const FlowPriorityNotification;
extern NotificationCenter::Name const StatPrintEvent;
extern NotificationCenter::Name const NewFlowNotification;
extern NotificationCenter::Name const FlowPerformanceNotification;
extern NotificationCenter::Name const OutcomesMapNotification;
extern NotificationCenter::Name const DutyCycleNotification;

// Flow information
struct FlowInfo {
  bool available;
  NodeID src;
  NodeID dst;
};

// Flow information
struct FlowPerformance {
  unsigned int mps; // number of measurement periods over which all PTs were met
  unsigned int point_value; // The point value of the flow
  float scalar_performance;
};

/// Helper class for computing sliding average
template <class T> class TrafficStat {
private:
  std::chrono::duration<float> _window_duration;
  typedef std::chrono::time_point<std::chrono::system_clock> const TimeStamp;
  std::deque<std::pair<T, TimeStamp>> _q;
  void _refresh() {
    auto now = std::chrono::system_clock::now();
    while (_q.size() > 0) {
      auto v = _q.front();
      auto diff = now - v.second;
      if (diff > _window_duration) {
        _q.pop_front();
      } else {
        break;
      }
    }
  };

public:
  TrafficStat() : _window_duration(1.0){};
  TrafficStat(float window_sec) : _window_duration(window_sec){};
  void push(T val) {
    auto now = std::chrono::system_clock::now();
    _q.push_back(std::make_pair(val, now));
    _refresh();
  }
  T median() {
    std::deque<T> _qsort;
    for (auto &v : _q) {
      _qsort.push_back(v.first);
    }
    std::sort(_qsort.begin(), _qsort.end());
    return _qsort.at(int(_q.size() / 2));
  }
  T sum() {
    _refresh();
    T sum = 0;
    for (auto const &val : _q) {
      sum += val.first;
    }
    return sum;
  }
  float average() {
    _refresh();
    T sum = 0;
    for (auto const &val : _q) {
      sum += val.first;
    }
    return sum / _window_duration.count();
  }
  float average_elements() {
    _refresh();
    T sum = 0;
    for (auto const &val : _q) {
      sum += val.first;
    }
    if (_q.size() > 0)
      return sum / _q.size();
    else
      return 0;
  }
  size_t size() const { return _q.size(); }
  void flush() { _q.clear(); }
};

template <class T> class TrafficStat5s : public TrafficStat<T> {
public:
  TrafficStat5s() : TrafficStat<T>(5.0){}; // help me
};

// For stdout output similar to phase 1 code
struct StatPrintEventInfo {
  float sum_rate_bps;
  float offered_rate_bps;
  float delivered_rate_bps;

  size_t total_n_frames_transmitted;
  size_t total_n_headers_decoded;
  size_t total_n_segments_sent;
  size_t total_n_segments_rxd;

  std::map<FlowUID, TrafficStat<size_t>> flow_offered_bytes;    // in bytes
  std::map<FlowUID, TrafficStat<size_t>> flow_delivered_bytes;  // in bytes
  std::map<FlowUID, TrafficStat<float>> flow_delivered_latency; // in seconds

  float duty_cycle;
};

enum IMType { C2 = 0, UNKNOWN };
struct IndividualMandate {
  // hold_period
  unsigned int hold_period;
  // point_value
  unsigned int point_value;
  // max_latency_s
  bool has_max_latency_s;
  float max_latency_s;
  // min_throughput_bps
  bool has_min_throughput_bps;
  float min_throughput_bps;
  // file_transfer_deadline_s
  bool has_file_transfer_deadline_s;
  float file_transfer_deadline_s;
  // file_size_bytes
  bool has_file_size_bytes;
  size_t file_size_bytes;
  // goal_set
  IMType im_type;

  static std::map<FlowUID, IndividualMandate> fromJSON(nlohmann::json mandates);
};

struct DutyCycleInfo {
  float duty_cycle;
  std::chrono::system_clock::time_point t;
};

class StatCenter {
public:
  StatCenter();
  ~StatCenter();

  /// Start StatCenter. In batch mode, this should be called when "Start"
  /// command is receved via C2API.
  void start();

  /// trigger publication of StatPrintEvent
  void publishStatPrintEvent();

private:
  typedef std::chrono::system_clock::duration Duration;

  // State
  bool _running;

  // Time when C2API "Start" is received
  std::chrono::system_clock::time_point _t_start;

  // Flows
  std::map<FlowUID, FlowInfo> _flows;

  // Point-to-point link info
  size_t _total_n_frames_transmitted = 0;
  size_t _total_n_segments_sent = 0;
  size_t _total_n_frames_detected = 0;
  size_t _total_n_headers_decoded = 0;
  size_t _total_n_segments_rxd = 0;

  // PHY statistics
  std::map<uint8_t, size_t> _rx_bits;
  std::map<NodeID, TrafficStat5s<size_t>> _rx_nblocks_all;
  std::map<NodeID, TrafficStat5s<size_t>> _rx_nblocks_decoded;

  // internal map for average snr calculation
  std::map<NodeID, TrafficStat5s<double>> _rx_snr;
  std::map<NodeID, TrafficStat5s<double>> _noiseVar;

  // Per-flow traffic info
  std::map<FlowUID, TrafficStat<size_t>> _flow_offered_bytes;    // in bytes
  std::map<FlowUID, TrafficStat<size_t>> _flow_delivered_bytes;  // in bytes
  std::map<FlowUID, TrafficStat<size_t>> _flow_forwarded_bytes;  // in bytes
  std::map<FlowUID, TrafficStat<float>> _flow_delivered_latency; // in seconds

  // Per-link traffic info
  std::map<NodeID, TrafficStat<size_t>> _link_rx;
  std::map<NodeID, TrafficStat<size_t>> _link_offered;

  // Tx nsamples for estimating duty cycle
  TrafficStat<size_t> _tx_nsamples;

  // Mandated outcomes
  enum FileTransferState { SUCCESS = 0, FAILURE, UNKNOWN };
  struct FlowMPInfo {
    FileTransferState file_transfer_state = FileTransferState::UNKNOWN;
    int burst_number = -3;
    bool achieved = false;
    size_t received_nbits = 0;
  };
  typedef std::map<FlowUID, FlowMPInfo> MPInfo;
  int _getMP(std::chrono::system_clock::time_point t)
      const; /// Get index of measurement period with respect to _t_start
  std::map<FlowUID, IndividualMandate> _getIMs(nlohmann::json mandates);
  std::map<FlowUID, IndividualMandate> _ims; // Current IM's
  std::vector<MPInfo> _perf_hist;            // performance history
  int _last_mandate_update_mp;

  // io_service
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  boost::asio::system_timer _timer_broadcast;
  std::thread _work_thread; // Single thread

  // Stats broadcast
  Duration const _t_broadcast;
  void _broadcast();

  // Notification center
  std::vector<NotificationCenter::SubToken> _subTokens;

  void _updatePerformance();
  void _handleRouteDecisionEvent(net::RouteDecisionEventInfo ei);
};

} // namespace stats
} // namespace bamradio

#endif
