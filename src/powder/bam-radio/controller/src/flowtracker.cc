// FlowTracker for ARQ
// Copyright (c) 2018 Tomohiro Arakawa
//  Copyright © 2018 Stephen Larew

#include "flowtracker.h"
#include "events.h"
#include <boost/format.hpp>

using namespace std::chrono_literals;

namespace bamradio {
namespace dll {
void FlowTracker::markReceived(
    ARQBurstInfo burst_info, std::chrono::system_clock::time_point source_time,
    std::chrono::system_clock::time_point now, size_t seglength) {
  // Find flow-burst
  auto burst_it =
      std::find_if(_bursts.begin(), _bursts.end(), [&](auto const &v) {
        return v.flow_uid == burst_info.flow_uid &&
               v.burst_num == burst_info.burst_num;
      });

  if (burst_it != _bursts.end()) {
    // When burst extra goes positive, then the burst size has been fed forward
    // and we can begin tracking the true bytes remaining in the burst.
    if (burst_it->burst_size == 0 && burst_info.extra > 0) {
      burst_it->burst_remaining += burst_info.extra;
      burst_it->burst_size = burst_info.extra;
    }
    // If we haven't received this segment before, account for its length.
    if (burst_info.seq_num > burst_it->last_seq &&
        burst_it->seq_nums.count(burst_info.seq_num) == 0) {
      burst_it->burst_remaining -= (int64_t)seglength;
    }
    burst_it->min_source_time =
        std::min(burst_it->min_source_time, source_time);
    burst_it->max_source_time =
        std::max(burst_it->max_source_time, source_time);
  } else {
    // Create new burst info and add the sequence number
    _bursts.emplace_back(ARQBurstTrackerInfo{
        .flow_uid = burst_info.flow_uid,
        .min_source_time = source_time,
        .max_source_time = source_time,
        .burst_num = burst_info.burst_num,
        .last_seq = 0,
        .burst_size = burst_info.extra,
        .burst_remaining = burst_info.extra - (int64_t)seglength,
        .seq_nums = {},
        .noim_rx = {},
        .lateSeg = false,
        .notified = false});
    burst_it = std::prev(_bursts.end());
  }

  burst_it->trackDelay(_im, source_time, now);

  // Add received sequence number
  if (burst_info.seq_num > burst_it->last_seq) {
    burst_it->seq_nums.insert(burst_info.seq_num);
  }
  _updateLastSeq(*burst_it);

  NotificationCenter::shared.post(
      FlowTrackerStateUpdateEvent,
      FlowTrackerStateUpdateEventInfo{
          burst_it->flow_uid, burst_it->min_source_time, burst_it->burst_num,
          burst_it->last_seq, burst_it->burst_size, burst_it->burst_remaining,
          burst_it->completed()});

  deleteBursts(now);
}

void FlowTracker::ARQBurstTrackerInfo::trackDelay(
    std::map<FlowUID, IndividualMandate> const &imm,
    std::chrono::system_clock::time_point seg_source_time,
    std::chrono::system_clock::time_point gnow) {
  auto imit = imm.find(flow_uid);
  if (imit == imm.end()) {
    noim_rx.insert(gnow - seg_source_time);
    return;
  }
  auto transfer_dur =
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          imit->second.visit<std::chrono::duration<float>>(
              [this](auto const &) {
                log::text(boost::format("expected a file flow %u") % flow_uid);
                return 0s;
              },
              [](IndividualMandate::FilePT const &filept) {
                return filept.transfer_duration;
              }));
  if (gnow - seg_source_time > transfer_dur) {
    lateSeg = true;
  }
}

std::chrono::nanoseconds FlowTracker::ARQBurstTrackerInfo::timeSinceDeadline(
    std::map<FlowUID, IndividualMandate> const &imm,
    std::chrono::system_clock::time_point gnow) {
  auto imit = imm.find(flow_uid);
  if (imit == imm.end()) {
    return 0s;
  }
  auto transfer_dur =
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          imit->second.visit<std::chrono::duration<float>>(
              [this](auto const &) {
                log::text(boost::format("expected a file flow %u") % flow_uid);
                return 0s;
              },
              [](IndividualMandate::FilePT const &filept) {
                return filept.transfer_duration;
              }));
  // FIXME use time of delivery of packet at actual terminal destination
  // node #multihop
  for (auto const delay : noim_rx) {
    if (delay > transfer_dur) {
      lateSeg = true;
      break;
    }
  }
  noim_rx.clear();
  return gnow - max_source_time - transfer_dur;
}

void FlowTracker::deleteBursts(std::chrono::system_clock::time_point gnow) {
  // Two deletion criteria:
  // 1) Burst completed (yay).
  // 2) Burst expired (IM.burstdeadeline) (boo).

  for (auto it = _bursts.begin(); it != _bursts.end();) {
    auto tsd = it->timeSinceDeadline(_im, gnow);
    bool expired = tsd > 0s || it->lateSeg;
    bool completed = it->completed();

#warning fixme arq multihop dont post IM status if not terminal node

    if (!it->notified && (expired || completed)) {
      NotificationCenter::shared.post(
          FlowTrackerIMEvent,
          FlowTrackerIMEventInfo{it->flow_uid, it->max_source_time,
                                 it->burst_num, completed, expired});
      it->notified = true;
    }
    // FIXME magic number:
    if (tsd > 1s) {
      // Don't delete until burst has expired because subsequent retransmitted
      // segments could revive and recreate this burst erroneously.
      it = _bursts.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<ARQBurstInfo> FlowTracker::getLastSeqNums() {
  deleteBursts(std::chrono::system_clock::now());
  // Get the last in-sequence sequence numeber
  std::vector<ARQBurstInfo> bursts;
  if (_bursts.empty()) {
    return bursts;
  }
  ++_lastBurstFedBack;
  size_t const start =
      _lastBurstFedBack >= _bursts.size() ? 0 : _lastBurstFedBack;
  auto i = start;
  do {
    auto &v = _bursts[i];
    if (v.timeSinceDeadline(_im, std::chrono::system_clock::now()) <= 0s) {
      // We ack 1 past the end of a burst to indicate we know the
      // burst is complete.
      uint16_t ackedSeqNum = v.last_seq + (uint16_t)(v.completed() ? 1 : 0);
      bursts.push_back(ARQBurstInfo{.flow_uid = v.flow_uid,
                                    .burst_num = v.burst_num,
                                    .seq_num = ackedSeqNum});
    }
    _lastBurstFedBack = i;
    ++i;
    i = i >= _bursts.size() ? 0 : i;
  } while (i != start && bursts.size() < MaxNumARQFeedback);
  return bursts;
}

void FlowTracker::_updateLastSeq(ARQBurstTrackerInfo &info) {
  // Find the last in-sequence sequence number and update last_seq value
  for (auto it = info.seq_nums.begin(); it != info.seq_nums.end();) {
    if (info.last_seq + 1 == *it) {
      info.last_seq++;
      it = info.seq_nums.erase(it);
    } else {
      return;
    }
  }
}

void FlowTracker::addIndividualMandates(
    std::map<FlowUID, IndividualMandate> const &imm) {
  for (auto const &im : imm) {
    _im.emplace(im.first, im.second);
  }
}

} // namespace dll
} // namespace bamradio
