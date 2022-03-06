/* -*- c++ -*- */
// Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>

#ifndef CC_DATA_H_INCLUDED
#define CC_DATA_H_INCLUDED

#include "adaptive_mcs_controller.h"
#include "discrete_channels.h"
#include "frame.h"
#include "networkmap.h"
#include "notify.h"
#include "statistics.h"
#include "util.h"

#include <algorithm>
#include <boost/asio.hpp>
#include <cc_data.pb.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <thread>
#include <tuple>

namespace bamradio {
namespace controlchannel {

// Node Location
struct Location {
  double latitude;
  double longitude;
  double elevation;
};

struct LinkState {
  float error_rate;
  ofdm::MCS::Name mcs;
  ofdm::SeqID::ID seqid;
};

// does an SRN overlap with a competitor's transmission?
typedef std::map<NodeID, bool> OverlapMap;

// Max number of nodes in network
constexpr size_t max_nodes = 10;

class CCData {
public:
  typedef std::shared_ptr<CCData> sptr;
  typedef std::chrono::system_clock::time_point Timepoint;

  CCData(uint8_t id, double slot_duration, bool gateway);
  ~CCData();

  /// Band for the single OFDM channel (center freq: double, bandwidth: double)
  static NotificationCenter::Name const OFDMChannelBandNotification;
  struct OFDMChannelUpdateInfo {
    std::map<NodeID, decisionengine::TransmitAssignment> channels;
    Timepoint t_effective;
    Timepoint t_last_update;
  };
  /// Notify the BBController of freshly deserialized control channel data
  static NotificationCenter::Name const NewCCDataNotification;

  // Overlap information
  static NotificationCenter::Name const OFDMChannelOverlapNotification;
  struct OFDMChannelOverlapInfo {
    std::map<NodeID, bool> overlap_map;
    Timepoint t_last_update;
  };

  /**
   * Change OFDM center frequency and bandwidth
   * @param channels tuple of {center frequency, waveform, attenuation}
   *
   * returns unique identifier for this request
   */
  uint32_t updateOFDMParams(
      std::map<NodeID, decisionengine::TransmitAssignment> channels,
      Timepoint t_effective);
  /// Get size of serialized data
  size_t getNbytes();
  /// Get serialized control packet
  std::shared_ptr<std::vector<uint8_t>> serialize();
  /// Deserialize control packet
  void deserialize(boost::asio::const_buffer data, bool descramble);
  /// Write serialized short control packet to mutable_buffer
  void serializeShortMsg(boost::asio::mutable_buffer data);
  /// Deserialize short control packet
  void deserializeShortMsg(boost::asio::const_buffer data);
  /// Set current location
  void setLocation(double latitude, double longitude, double elevation);
  /// Get SRNID => Location map
  std::map<uint8_t, Location> getLocationMap();
  void setOfferedRate(std::map<uint8_t, size_t> rate);
  double getOfferedDataRate(uint8_t tx_srnid, uint8_t rx_srnid);
  /// Set error rate
  void setFER(uint8_t channel, float fer);
  /// Set MCS request
  void setMCSReq(uint8_t tx_srnid, uint8_t mcs, uint8_t seqid);
  /// Returns map <Tx SRNID, link_state> for given Rx SRNID
  std::map<uint8_t, LinkState> getLinkStateInfo(uint8_t rx_srnid);
  /// Returns elapsed time in seconds since the last update of link state info
  std::map<uint8_t, float> getLSIElapsedTimeSinceLastUpdate();
  /// Set flow information
  void setFlowInfo(std::map<FlowUID, stats::FlowInfo> flow_map);
  /// Get flow information
  stats::FlowInfo getFlowInfo(FlowUID uid);
  /// Set flow performance
  void setFlowPerformance(std::map<FlowUID, stats::FlowPerformance> perf_map);
  /// Get flow performance
  std::map<FlowUID, stats::FlowPerformance> getFlowPerformance();
  /// Get SRNIDs
  std::vector<NodeID> getAllSRNIDs();
  /// Set my active flows
  void addActiveFlow(dll::NewActiveFlowEventInfo flow_info);
  /// Get all active flows
  std::map<NodeID, std::vector<dll::NewActiveFlowEventInfo>> getActiveFlows();
  /// Clear my active flows
  void clearActiveFlows();
  /// Get gateway SRN ID
  NodeID getGatewaySRNID();
  /// Get Tx duty cycle
  std::map<NodeID, stats::DutyCycleInfo> getDutyCycle();
  /// Get the current Tx Assignment map
  std::map<NodeID, decisionengine::TransmitAssignment> getTransmitAssignment();
  /// Set channel overlap info
  uint32_t updateOverlapInfo(OverlapMap overlap_map);

private:
  typedef std::chrono::system_clock::duration Duration;

  uint8_t const d_my_srnid;                      /// My SRNID
  std::set<uint8_t> _all_srnids;                 /// All SRNIDs in network
  bool const _gateway;                           /// True if gateway
  std::shared_ptr<CCDataPb::CCDataMsg> d_pb_msg; /// Protobuf msg
  time_t const d_start_ref_time;
  std::mutex d_mutex; /// Mutex to lock CCData
  std::vector<NotificationCenter::SubToken>
      d_nc_tokens;                       /// Notification center tokens
  std::vector<uint8_t> d_scramble_bytes; /// Byte array to scramble data
  int d_prev_slot;                       /// Previous slot number
  std::map<uint8_t, unsigned int> _timeout_counter; /// timeout counter
  uint64_t _seq_num;

  void _timeout_check();
  void _updateRoute();

  void _publishOFDMChannels();
  void _publishOverlapInfo();
  CCDataPb::CCDataMsg_NodeInfo &_nodeInfo(NodeID srn) {
    return (*d_pb_msg->mutable_node_list())[srn];
  }
  CCDataPb::CCDataMsg_NodeInfo_LinkState &_linkState(NodeID tx_srn) {
    return (*_nodeInfo(d_my_srnid).mutable_link_state())[tx_srn];
  }

  // FSK packet struct
  struct ShortMsgPayload {
    NodeID src_srnid;
    uint16_t seq_num;
    NodeID srnids[max_nodes];
    uint8_t nchan;
    uint32_t channel_last_update;
    uint64_t t_channel_effective;
    decisionengine::TransmitAssignment tx_assign[max_nodes];
    bool overlapped[max_nodes];
    uint32_t overlapped_last_update;
  };

  std::vector<uint8_t> _decompress_tmp;

  // io_service
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  boost::asio::system_timer _timeout_check_timer;
  Duration const _tcheck_interval; // Timeout check interval
  std::thread _work_thread;        // Single thread
};
} // namespace controlchannel
} // namespace bamradio

#endif
