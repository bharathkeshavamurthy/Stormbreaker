#include "statistics.h"
#include "c2api.h"
#include "events.h"
#include "options.h"

#include "json.hpp"
#include <algorithm>
#include <boost/bind.hpp>
#include <iostream>
#include <math.h>
#include <vector>

namespace bamradio {
namespace stats {

using Json = nlohmann::json;

//
// Notifications & Events
//
NotificationCenter::Name const RxRateMapNotification =
    NotificationCenter::makeName("Rx Rate Update");
NotificationCenter::Name const OfferedRateNotification =
    NotificationCenter::makeName("Offered Traffic Rate Update");
NotificationCenter::Name const DeliveredRateNotification =
    NotificationCenter::makeName("Delivered Traffic Rate Update");
NotificationCenter::Name const BLERNotification =
    NotificationCenter::makeName("Rx BLER Update");
NotificationCenter::Name const SNRNotification =
    NotificationCenter::makeName("Rx SNR Update");
NotificationCenter::Name const NoiseVarNotification =
    NotificationCenter::makeName("Rx noise variance Update");
NotificationCenter::Name const FlowPriorityNotification =
    NotificationCenter::makeName("Flow Priority Update");
NotificationCenter::Name const StatPrintEvent =
    NotificationCenter::makeName("Print Stats to stdout");
NotificationCenter::Name const NewFlowNotification =
    NotificationCenter::makeName("New Flow Notification");
NotificationCenter::Name const FlowPerformanceNotification =
    NotificationCenter::makeName("Flow Performance Notification");
NotificationCenter::Name const OutcomesMapNotification =
    NotificationCenter::makeName("Outcomes Map Notification");
NotificationCenter::Name const DutyCycleNotification =
    NotificationCenter::makeName("Duty Cycle Notification");

using namespace std::literals::chrono_literals;
using namespace std::string_literals;

StatCenter::StatCenter()
    : _running(false), _ios_work(new boost::asio::io_service::work(_ios)),
      _timer_broadcast(_ios), _t_broadcast(1s), _tx_nsamples(5) {
  // Start thread
  _work_thread = std::thread([this] {
    bamradio::set_thread_name("statistics_work");
    _ios.run();
  });

  // OutcomesUpdateEventInfo
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
          OutcomesUpdateEvent, _ios, [this](auto ei) {
            _ims = this->_getIMs(ei.j);
            _last_mandate_update_mp =
                _running ? this->_getMP(std::chrono::system_clock::now()) : 0;
            // Flush stats. No need to keep stats of old flows.
            _link_offered.clear();
            _flows.clear();
            log::text("StatCenter received MO");
          }));
}

void StatCenter::start() {
  _running = true;

  // Set start time
  _t_start =
      c2api::env.scenarioStart().value_or(std::chrono::system_clock::now());

  _perf_hist.reserve(60 * 60 * 2); // 2 hours of measurement periods
  _perf_hist.push_back(MPInfo());

  //
  // SUBSCRIPTIONS
  //

  // SentFrameEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::SentFrameEventInfo>(
          dll::SentFrameEvent, _ios, [this](auto ei) {
            _total_n_frames_transmitted++;
            _tx_nsamples.push(ei.nsamples);
          }));

  // SentSegmentEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::SentSegmentEventInfo>(
          dll::SentSegmentEvent, _ios,
          [this](auto) { _total_n_segments_sent++; }));

  // ReceivedCompleteSegmentEvent
  _subTokens.push_back(NotificationCenter::shared
                           .subscribe<dll::ReceivedCompleteSegmentEventInfo>(
                               dll::ReceivedCompleteSegmentEvent, _ios,
                               [this](auto) { _total_n_segments_rxd++; }));

  // DetectedFrameEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::DetectedFrameEventInfo>(
          dll::DetectedFrameEvent, _ios, [this](auto) {
            _total_n_frames_detected++;
            _total_n_headers_decoded++;
          }));

  // ReceivedFrameEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::ReceivedFrameEventInfo>(
          dll::ReceivedFrameEvent, _ios, [this](auto ei) {
            // convert from dB to gain then push
            _rx_snr[ei.sourceNodeID].push(pow(10., (ei.snr) / 10.));
            // ei.noiseVar is the noise variance of the frame
            _noiseVar[ei.sourceNodeID].push(ei.noiseVar);

            // blocks
            if (ei.destNodeID != AllNodesID) {
              _rx_nblocks_all[ei.sourceNodeID].push(ei.numBlocks);
              _rx_nblocks_decoded[ei.sourceNodeID].push(ei.numBlocksValid);
            }
          }));

  // InvalidFrameHeaderEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::InvalidFrameHeaderEventInfo>(
          dll::InvalidFrameHeaderEvent, _ios,
          [this](auto) { _total_n_frames_detected++; }));

  // RouteDecisionEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<net::RouteDecisionEventInfo>(
          net::RouteDecisionEvent, _ios,
          [this](auto ei) { this->_handleRouteDecisionEvent(ei); }));

  // FlowTrackerIMEvent
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<dll::FlowTrackerIMEventInfo>(
          dll::FlowTrackerIMEvent, _ios, [this](auto ei) {
            FileTransferState state;
            if (ei.expired) {
              state = FileTransferState::FAILURE;
            } else if (ei.completed) {
              state = FileTransferState::SUCCESS;
            } else {
              // this should not happen
              log::text("Neither expired nor completed");
              return;
            }
            int idx = this->_getMP(ei.sourceTime);
            if (0 <= idx && idx < (int)_perf_hist.size()) {
              auto &flowmpinfo = _perf_hist[idx][ei.flow_uid];
              flowmpinfo.file_transfer_state = state;
              flowmpinfo.burst_number = ei.burstNum;
            }
          }));

  // Start broadcasting stats
  _timer_broadcast.expires_at(_t_start + _t_broadcast);
  _timer_broadcast.async_wait([this](auto &e) {
    if (e == boost::asio::error::operation_aborted) {
      return;
    }
    if (e) {
      panic("statcenter timer broadcast failed "s + e.message());
    }
    this->_broadcast();
  });
}

StatCenter::~StatCenter() {
  _running = false;
  _subTokens.clear();
  delete _ios_work;
  _work_thread.join();
}

int StatCenter::_getMP(std::chrono::system_clock::time_point t) const {
  // n.b. assuming 1 MP = 1 second
  return std::chrono::duration_cast<std::chrono::seconds>(t - _t_start).count();
}

void StatCenter::_handleRouteDecisionEvent(net::RouteDecisionEventInfo ei) {
  // Ignore non-IP segments
  if (ei.type != dll::SegmentType::IPv4 &&
      ei.type != dll::SegmentType::ARQIPv4) {
    return;
  }

  // Flow UID is dest port. See C2API section on help desk website.
  auto const flow_uid = ei.dst_port;

  // Get payload size
  if (ei.packetLength < 28) {
    log::text("StatCenter: too small IPv4 UDP packet");
    return;
  }
  size_t const payload_nbytes =
      ei.packetLength - 28; // IP header (20 bytes) + UDP header (8 bytes)

  // traffic from MGEN
  if (ei.src_srnid == ExtNodeID) {
    if (_link_offered.find(ei.next_hop) == _link_offered.end()) {
      // Use 10s averaging window for estimating offered data rate. We can
      // assume that the data rate is almost constant over a stage. This rate is
      // used by the decision engine; the longer window gives DE a better
      // estimate of the offered rate.
      _link_offered.emplace(ei.next_hop, TrafficStat<size_t>(10));
    }
    _flow_offered_bytes[flow_uid].push(payload_nbytes);
    _link_offered[ei.next_hop].push(payload_nbytes);
  }

  // traffic to MGEN
  else if (ei.next_hop == ExtNodeID) {
    // Calculate latency
    std::chrono::duration<float> latency =
        std::chrono::system_clock::now() - ei.sourceTime;
    _flow_delivered_latency[flow_uid].push(latency.count());
    _flow_delivered_bytes[flow_uid].push(payload_nbytes);
    // Check if we have max_latency_s requirement
    auto ims_itr = _ims.find(flow_uid);
    if (ims_itr != _ims.end() &&
        !ims_itr->second.has_file_transfer_deadline_s &&
        (!ims_itr->second.has_max_latency_s ||
         ims_itr->second.max_latency_s > latency.count())) {
      // update MP
      int mp = this->_getMP(ei.sourceTime);
      if (mp > 0 && mp < (int)_perf_hist.size()) {
        _perf_hist[mp][flow_uid].received_nbits += 8 * payload_nbytes;
      }
    }
  }

  // forward to another node (via RF)
  else {
    _flow_forwarded_bytes[flow_uid].push(payload_nbytes);
  }

  // Received from another node (via RF)
  if (ei.src_srnid != ExtNodeID) {
    _link_rx[ei.src_srnid].push(payload_nbytes);
  }

  // Is this a new traffic flow?
  auto itr = _flows.find(flow_uid);
  if (itr == _flows.end()) {
    // Get SRN IDs from IP addr
    NodeID srcid = ((ei.src_ip.to_ulong() >> 8) & 0xff) - 100;
    NodeID dstid = ((ei.dst_ip.to_ulong() >> 8) & 0xff) - 100;
    // add new flow
    _flows[flow_uid] = FlowInfo{true, srcid, dstid};
    // Send flow info to CCData
    NotificationCenter::shared.post(NewFlowNotification, _flows);
  }
}

void StatCenter::_updatePerformance() {
  std::map<FlowUID, unsigned int> n_mp_im;
  for (auto const &v : _ims) {
    auto const flow_uid = v.first;
    auto const &im = v.second;

    n_mp_im[flow_uid] = 0;
    bool prev_file_im_event_disposition = false;
    bool prev_c2_state = false;
    int last_burst_num = -2;

    for (int mp = _last_mandate_update_mp; mp < (int)(_perf_hist.size() - 1);
         ++mp) {
      auto &mp_perf = _perf_hist[mp][flow_uid];
      if (im.has_min_throughput_bps) {
        // === Stream ===
#warning TODO account for MPs with < min_throughput_bps traffic (hello ARQ seqnums)
        if (im.min_throughput_bps <= mp_perf.received_nbits) {
          // Meeting throughput PT
          n_mp_im[flow_uid] += 1;
          prev_c2_state = true;
        } else {
          // Not meeting throughput PT
          if (im.im_type == IMType::C2 && prev_c2_state == true) {
            // FIXME: C2 Hack
            n_mp_im[flow_uid] += 1;
            prev_c2_state = false;
          } else {
            n_mp_im[flow_uid] = 0;
            prev_c2_state = false;
          }
        }
      } else if (im.has_file_transfer_deadline_s) {
        // === File transfer ===
        if (mp_perf.file_transfer_state == FileTransferState::SUCCESS) {
          if (last_burst_num == -2) {
            // First completed burst this stage.
            n_mp_im[flow_uid] = 1;
          } else if (mp_perf.burst_number == last_burst_num + 1) {
            // In-order next burst completed.
            n_mp_im[flow_uid] += 1;
          } else {
            // A burst is missing because we skipped burst numbers.
            n_mp_im[flow_uid] = 1;
          }
          last_burst_num = mp_perf.burst_number;
          prev_file_im_event_disposition = true;
        } else if (mp_perf.file_transfer_state == FileTransferState::UNKNOWN) {
          if (prev_file_im_event_disposition) {
            // State of file transfer is unknown but the previous burst was
            // successful. Provisionally mark this mp as good.
            n_mp_im[flow_uid] += 1;
          } else {
            // Unknown state but previous burst was bad so this MP is
            // effectively bad.
            n_mp_im[flow_uid] = 0;
          }
        } else if (mp_perf.file_transfer_state == FileTransferState::FAILURE) {
          // File transfer was unsuccessful
          n_mp_im[flow_uid] = 0;
          prev_file_im_event_disposition = false;
          last_burst_num = -2;
        }
      }
    }
  }

  // flow performance map
  std::map<FlowUID, FlowPerformance> perf_map;
  for (auto const &nmp : n_mp_im) {
    auto const flow_uid = nmp.first;
    // Get corresponding individual mandate
    auto im_itr = _ims.find(flow_uid);
    assert(im_itr != _ims.end());
    assert(nmp.second >= 0);
    // Have we achieved the PT more than hold_period?
    FlowPerformance flow_performance;
    flow_performance.mps = nmp.second; // # of MPs
    if (flow_performance.mps == 0) {
      // Do not report performance if I'm not getting any traffic
      continue;
    }
    if (im_itr->second.hold_period == 0) {
      flow_performance.scalar_performance = nmp.second;
    } else {
      flow_performance.scalar_performance =
          nmp.second / (float)im_itr->second.hold_period;
    }
    perf_map[flow_uid] = flow_performance;
  }
  NotificationCenter::shared.post(FlowPerformanceNotification, perf_map);

  // log # of achieved IMs
  size_t n_achieved_ims = 0;
  for (auto const &perf : perf_map) {
    if (perf.second.scalar_performance >= 1.0f) {
      n_achieved_ims++;
    }
  }
  NotificationCenter::shared.post(AchievedIMsUpdateEvent,
                                  AchievedIMsUpdateEventInfo{n_achieved_ims});
}

std::map<FlowUID, IndividualMandate>
StatCenter::_getIMs(nlohmann::json mandates) {
  return IndividualMandate::fromJSON(mandates);
}

std::map<FlowUID, IndividualMandate>
IndividualMandate::fromJSON(nlohmann::json mandates) {
  std::map<FlowUID, IndividualMandate> ims;

  for (auto goal : mandates) {
    IndividualMandate im;

    // UID
    auto flow_uid = goal["flow_uid"];

    // hold_period
    if (goal.count("hold_period")) {
      im.hold_period = goal["hold_period"];
    } else {
      im.hold_period = 0;
    }

    // point_value
    if (goal.count("point_value")) {
      im.point_value = goal["point_value"];
    } else {
      // FIXME: Is this an accurate default value?
      im.point_value = 1;
    }

    // goal_set
    if (goal.count("goal_set") && goal["goal_set"] == "C2") {
      im.im_type = IMType::C2;
    } else {
      im.im_type = IMType::UNKNOWN;
    }

    // get all PTs
    auto req = goal["requirements"];

    // max_latency_s
    if (req.count("max_latency_s")) {
      im.has_max_latency_s = true;
      im.max_latency_s = req["max_latency_s"];
    } else {
      im.has_max_latency_s = false;
    }

    // min_throughput_bps
    if (req.count("min_throughput_bps")) {
      im.has_min_throughput_bps = true;
      im.min_throughput_bps = req["min_throughput_bps"];
    } else {
      im.has_min_throughput_bps = false;
    }

    // file_transfer_deadline_s
    if (req.count("file_transfer_deadline_s")) {
      im.has_file_transfer_deadline_s = true;
      im.file_transfer_deadline_s = req["file_transfer_deadline_s"];
    } else {
      im.has_file_transfer_deadline_s = false;
    }

    // file_size_bytes
    if (req.count("file_size_bytes")) {
      im.has_file_size_bytes = true;
      im.file_size_bytes = req["file_size_bytes"];
    } else {
      im.has_file_size_bytes = false;
    }

    ims[flow_uid] = im;
  }
  return ims;
}

void StatCenter::publishStatPrintEvent() {
  _ios.post([this] {
    // RF Rx
    float sum_rate = 0;
    for (auto &x : _link_rx)
      sum_rate += 8 * x.second.average();
    // Offered
    float offered_rate = 0;
    for (auto &x : _flow_offered_bytes)
      offered_rate += 8 * x.second.average();
    // Delivered
    float delivered_rate = 0;
    for (auto &x : _flow_delivered_bytes)
      delivered_rate += 8 * x.second.average();

    float dc =
        static_cast<float>(_tx_nsamples.average() / bam::dsp::sample_rate);
    StatPrintEventInfo ei{.sum_rate_bps = sum_rate,
                          .offered_rate_bps = offered_rate,
                          .delivered_rate_bps = delivered_rate,
                          .total_n_frames_transmitted =
                              _total_n_frames_transmitted,
                          .total_n_headers_decoded = _total_n_headers_decoded,
                          .total_n_segments_sent = _total_n_segments_sent,
                          .total_n_segments_rxd = _total_n_segments_rxd,
                          .flow_offered_bytes = _flow_offered_bytes,
                          .flow_delivered_bytes = _flow_delivered_bytes,
                          .flow_delivered_latency = _flow_delivered_latency,
                          .duty_cycle = (dc > 1 ? 1.0f : dc)};

    NotificationCenter::shared.post(StatPrintEvent, ei);
  });
}

void StatCenter::_broadcast() {
  // Publish IMs
  // This is necessary because bbcontroller may not be running when the new IMs
  // are received. Therefore, Statcenter keeps posting the up-to-date IMs every
  // time _broadcast() is called.
  NotificationCenter::shared.post(OutcomesMapNotification, _ims);

  // RX rate
  std::map<uint8_t, double> rx_rate_per_link;
  for (auto &v : _link_rx)
    rx_rate_per_link[v.first] = 8 * v.second.average();
  NotificationCenter::shared.post(RxRateMapNotification, rx_rate_per_link);

  // Compute and publish offered traffic rate from ext nodes (i.e. TGEN)
  std::map<uint8_t, size_t> offered_rate_per_link;
  for (auto &x : _link_offered) {
    offered_rate_per_link[x.first] = 8 * x.second.average();
  }
  NotificationCenter::shared.post(OfferedRateNotification,
                                  offered_rate_per_link);

  // Compute and publish delivered traffic rate to ext nodes (i.e. TGEN)
  double delivered_rate = 0;
  for (auto &x : _flow_delivered_bytes)
    delivered_rate += x.second.average();
  NotificationCenter::shared.post(DeliveredRateNotification, delivered_rate);

  // Compute BLER
  std::map<uint8_t, double> rx_bler;
  for (auto &r : _rx_nblocks_all) {
    auto const srn = r.first;
    auto const nblocks_all = r.second.sum();
    auto const nblocks_decoded = _rx_nblocks_decoded[srn].sum();
    if (nblocks_all == 0) {
      // No headers detected
      continue;
    } else {
      rx_bler[srn] = 1.0 - (double)nblocks_decoded / (double)nblocks_all;
    }
  }
  NotificationCenter::shared.post(BLERNotification, rx_bler);

  // Publish SNR
  std::map<NodeID, double> snr_avg;
  for (auto &v : _rx_snr) {
    if (v.second.size()) {
      snr_avg[v.first] = 10. * log10(v.second.average_elements());
    }
  }
  if (!snr_avg.empty()) {
    NotificationCenter::shared.post(SNRNotification, snr_avg);
  }

  // publish filtered noise variance
  std::map<NodeID, double> noise_var_map;
  for (auto &v : _noiseVar) {
    if (v.second.size() > 0) {
      noise_var_map[v.first] = v.second.median();
    }
  }
  NotificationCenter::shared.post(NoiseVarNotification, noise_var_map);

  // publish duty cycle
  float dc = static_cast<float>(_tx_nsamples.average() / bam::dsp::sample_rate);
  NotificationCenter::shared.post(
      DutyCycleNotification,
      DutyCycleInfo{.duty_cycle = (dc > 1 ? 1.0f : dc),
                    .t = std::chrono::system_clock::now()});

  // Check IMs
  _updatePerformance();
  _perf_hist.push_back(MPInfo());

  // Wait and do it all again
  if (_running) {
    _timer_broadcast.expires_at(_timer_broadcast.expires_at() + _t_broadcast);
    _timer_broadcast.async_wait([this](auto &e) {
      if (e == boost::asio::error::operation_aborted) {
        return;
      }
      if (e) {
        panic("statcenter timer broadcast failed "s + e.message());
      }
      this->_broadcast();
    });
  }
}
} // namespace stats
} // namespace bamradio
