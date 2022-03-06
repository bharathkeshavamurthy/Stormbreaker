//  Copyright © 2018 Stephen Larew

#ifndef ce58303ba0b3d0f189a4d03a1c0a953642e7ec62
#define ce58303ba0b3d0f189a4d03a1c0a953642e7ec62

#include "dll_types.h"
#include "json.hpp"
#include "util.h"
#include <boost/variant.hpp>
#include <chrono>
#include <map>
#include <vector>

namespace bamradio {

/// Information on a link between a transmitter and receiver.
struct LinkInfo {
  /// Overhead in frame header, DSP, and general frame padding.
  std::chrono::duration<float> frame_overhead;
  /// Bits of overhead in each segment.
  size_t segment_bit_overhead;
  /// Transmitter throughput information rate.
  /// DFT-S-OFDM + MCS + BANDWIDTH
  float throughput_bps;
  /// Receiver throughput information rate.
  /// throughput_bps * [0,1]
  float goodput_bps;
  /// Receiver throughput reduced by ARQ overhead.
  float goodput_arq_bps;
  //The channel bandwidth for ranking and branching metric evaluation in the new prioritized flow scheduling algorithm
  float channel_bandwidth;
  std::string description() const;
};

/// Progress on a file flow.
struct FileFlowProgress {
  /// Deadline before which all bytes of file must be transferred.
  std::chrono::system_clock::time_point deadline;
  /// Bits of file left to transmit.
  size_t bits_remaining;
};

/// A DSC2 individual mandate.
struct IndividualMandate {
  // The point value of the flow for prioritized flow scheduling
  unsigned int point_value;
  /// Performance thresholds for a UDP packet stream.
  struct StreamPT {
    // Throughput is measured as bits per second of IP traffic delivered to
    // the traffic generator sink at the destination node.
    float min_throughput_bps;
    // Packet latency is measured from when the traffic generator source
    // provides a packet to the source node to when the traffic generator
    // sink receives it from the destination node. A packet which is not
    // delivered in less than max_latency_s is considered dropped, decreasing
    // the measured throughput, even if it is eventually delivered.
    // [100ms,∞)
    std::chrono::duration<float> max_latency;
  };
  /// Performance thresholds for a file transfer.
  struct FilePT {
    // A23: File transfers are modeled as a short burst of UDP packets whose
    // aggregate payload is equal to the original file size.
    // Negative size indicates to-be-determined.
    ssize_t size_bytes;
    // File transfer latency is measured from when the traffic generator
    // source provides a packet to the source node to when the traffic
    // generator sink receives it from the destination node. Latency is
    // measured on a per packet basis. All packets offered in a MP must be
    // delivered in order to meet the PT. A packet which is not delivered in
    // less than file_transfer_deadline_s is considered dropped.
    std::chrono::duration<float> transfer_duration;
  };
  /// Destination port of IP packets in flow.
  // FlowUID flowUID;
  // Undocumented value that shows up in the JSON. We ignore it.
  // float max_packet_drop_rate;
  /// Performance thresholds for this IM.
  boost::variant<StreamPT, FilePT> pt;
  template <typename R>
  using pt_visitor = boost::visitor2_ptr_t<StreamPT, FilePT, R>;
  template <typename R>
  inline R visit(typename pt_visitor<R>::visitor1_t streamvis,
                 typename pt_visitor<R>::visitor2_t filevis) const {
    return boost::apply_visitor(pt_visitor<R>(streamvis, filevis), pt);
  }
  std::string description() const;
  /// Return IndividualMandates from JSON description.
  static std::map<FlowUID, IndividualMandate> fromJSON(nlohmann::json const &mandates);
};

/// Information about a flow to be scheduled.
struct FlowInfo {
  IndividualMandate im;
  /// Receiver to which this flow will be transmitted.
  // NodeID next_hop_rx;
  LinkInfo link_info;
  /// Size in bits of segments in this flow (i.e. segment::length()*8).
  size_t bits_per_segment;
  /// Size in bits of the non-payload in a segment (e.g.
  ///   (segment::length() - packetLength())*8).
  size_t segment_payload_bit_overhead;
  /// Minimum number of segments per frame.
  size_t min_segments_per_frame;
  /// Latency scale factor (non-negative, usually in [0,1]).
  float alpha;
  /// Rate scale factor (non-negative, usually in [0,1]).
  float beta;
  /// Relative priority to other flows.
  int priority;
};

struct QuantumSchedule {
  std::map<FlowID, std::chrono::nanoseconds> quantums;
  bool valid;
  std::chrono::duration<float> period, periodlb, periodub;
};

bool operator==(QuantumSchedule const &lhs, QuantumSchedule const &rhs);
bool operator!=(QuantumSchedule const &lhs, QuantumSchedule const &rhs);

typedef std::map<FlowID, FlowInfo const *> FlowInfoMap;

/// Returns a schedule of quantums for the the given flows.
// QuantumSchedule scheduleFlows(FlowInfoMap const &flows,
//                               std::map<FlowUID, FileFlowProgress> const &ffp,
//                               std::chrono::system_clock::time_point now);

/// Changing the signature of this routine for the newly added ranking and branching logic
QuantumSchedule scheduleFlows(std::vector< std::pair< FlowID, std::pair< float, FlowInfo > > > &reorderingVector,
                              std::map<FlowUID, FileFlowProgress> const &ffp,
                              std::chrono::system_clock::time_point now);

enum class MaxFlowsSearch {
  // Remove the flow with the lowest max_latency
  RemoveMinMaxLatency = 1,
  // Remove the flow with the largest quantum
  RemoveMaxQuantum = 2,
  // Create two branches -> One with max_quantum as the metric and the other with min_max_latency as the metric
  RemoveMinMaxLatencyAndMaxQuantum = 3,
  // Remove the flow with the least value/resource
  RemoveMinValue = 4,
  // Create two branches -> One with min_max_latency as the metric and the other with min_value as the metric
  RemoveMinMaxLatencyAndMinValue = 5,
  // Create two branches -> One with max_quantums as the metric and the other with min_value as the metric
  RemoveMaxQuantumAndMinValue = 6,
  // Create three branches -> One with max_quantum as the metric, another with min_max_latency as the metric, and another with min_value as the metric
  RemoveMinMaxLatencyAndMaxQuantumAndMinValue = 7
};

/// Returns the largest number of flows that can be scheduled.
QuantumSchedule scheduleMaxFlows(FlowInfoMap const &flows,
                                 std::map<FlowUID, FileFlowProgress> const &ffp,
                                 std::chrono::system_clock::time_point now,
                                 MaxFlowsSearch search, bool respectPriority);

} // namespace bamradio

/*
 * Loop over (F,S) pairs, calling scheduleFlows(F,S), and if valid schedule then
 * add to T_i.
 *
 * #im -> T_i
 *
 * #im is number of individual mandates
 * T_i is set of (F,S) pairs where i = #im
 * F is a set of flows where |F| = #im
 * S is set of L
 * L is a link defined by (mcs,bw) pair
 *
 * allocate channels chooses center frequency for each link
 */

#endif

