// FlowTracker for ARQ
// Copyright (c) 2018 Tomohiro Arakawa
//  Copyright © 2018 Stephen Larew

#ifndef FLOWTRACKER_H
#define FLOWTRACKER_H

#include "dll_types.h"
#include "im.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <vector>

namespace bamradio {
namespace dll {

class FlowTracker {
public:
  FlowTracker(){};
  /// Mark the segment as received
  void markReceived(ARQBurstInfo burst_info,
                    std::chrono::system_clock::time_point source_time,
                    std::chrono::system_clock::time_point now,
                    size_t seglength);
  /// Get the vector of burst info. ARQBurstInfo.seq_num is the last in-sequence
  /// received packet.
  std::vector<ARQBurstInfo> getLastSeqNums();

  void addIndividualMandates(std::map<FlowUID, IndividualMandate> const &im);

private:
  void deleteBursts(std::chrono::system_clock::time_point gnow);
  struct ARQBurstTrackerInfo {
    FlowUID flow_uid;
    std::chrono::system_clock::time_point min_source_time, max_source_time;
    uint8_t burst_num;
    uint16_t last_seq;
    int64_t burst_size;
    int64_t burst_remaining;
    std::set<uint16_t> seq_nums;
    std::set<std::chrono::nanoseconds> noim_rx;
    bool lateSeg;
    bool notified;
    bool completed() const {
      assert(burst_size == 0 || burst_remaining >= 0);
      assert(burst_size == 0 || burst_remaining > 0 || seq_nums.empty());
      return burst_size > 0 && burst_remaining <= 0;
    }
    void trackDelay(std::map<FlowUID, IndividualMandate> const &im,
                    std::chrono::system_clock::time_point seg_source_time,
                    std::chrono::system_clock::time_point gnow);
    std::chrono::nanoseconds
    timeSinceDeadline(std::map<FlowUID, IndividualMandate> const &im,
                      std::chrono::system_clock::time_point gnow);
  };
  size_t _lastBurstFedBack = 0;
  std::map<FlowUID, IndividualMandate> _im;
  std::vector<ARQBurstTrackerInfo> _bursts;
  void _updateLastSeq(ARQBurstTrackerInfo &info);
};
} // namespace dll
} // namespace bamradio
#endif
