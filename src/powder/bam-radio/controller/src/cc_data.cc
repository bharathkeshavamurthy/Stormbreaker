/* -*- c++ -*- */
// Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>

#include "cc_data.h"
#include "adaptive_mcs_controller.h"
#include "c2api.h"
#include "events.h"
#include "statistics.h"

#include <algorithm>
#include <boost/bind.hpp>
#include <ctime>
#include <iostream>
#include <lz4.h>
#include <memory>
#include <random>
#include <string>
#include <tuple>

// define reference time as Aug 1, 2019. Valid until Dec 10, 2020.
constexpr unsigned long __CC_REF_SECONDS__ = 1564617600;

namespace de = bamradio::decisionengine;

namespace bamradio {
namespace controlchannel {

NotificationCenter::Name const CCData::OFDMChannelBandNotification =
    std::hash<std::string>{}("OFDM Channel Band");
NotificationCenter::Name const CCData::OFDMChannelOverlapNotification =
    std::hash<std::string>{}("OFDM Overlap");
NotificationCenter::Name const CCData::NewCCDataNotification =
    std::hash<std::string>{}("NEW CC DATA");

using namespace boost::asio;

inline uint32_t getIntRelTimeNow() {
  uint64_t milliseconds_from_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  return (uint32_t)(milliseconds_from_epoch / 10.0 - 100 * __CC_REF_SECONDS__);
}

inline CCData::Timepoint timepointFromRelTime(uint32_t reltime) {
  auto const milliseconds_from_epoch =
      10 * (reltime + 100 * __CC_REF_SECONDS__);
  return CCData::Timepoint(std::chrono::milliseconds(milliseconds_from_epoch));
}

CCData::CCData(uint8_t id, double slot_duration, bool gateway)
    : d_my_srnid(id), _gateway(gateway), d_start_ref_time(time(nullptr)),
      d_prev_slot(-1), _seq_num(0),
      _ios_work(new boost::asio::io_service::work(_ios)), _work_thread([this] {
        bamradio::set_thread_name("cc_data");
        _ios.run();
      }),
      _timeout_check_timer(_ios),
      _tcheck_interval(std::chrono::duration_cast<Duration>(
          std::chrono::duration<double>(slot_duration))),
      _decompress_tmp(1 << 16) {
  d_pb_msg = std::make_shared<CCDataPb::CCDataMsg>();
  d_pb_msg->set_src_srnid(id);
  _all_srnids.insert(id);
  this->_nodeInfo(d_my_srnid).set_gateway(gateway);
  // Subscribe to location message
  d_nc_tokens.push_back(NotificationCenter::shared.subscribe<gps::GPSEventInfo>(
      gps::GPSEvent, _ios, [this](auto ei) {
        if (ei.type == gps::GPSEventType::READ_GOOD) {
          this->setLocation(ei.lat, ei.lon, ei.alt);
        }
      }));
  // Subscribe to offered rate notification
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<uint8_t, size_t>>(
          stats::OfferedRateNotification, _ios,
          [this](auto const rate) { this->setOfferedRate(rate); }));
  // Subscribe to FER notification
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<uint8_t, double>>(
          stats::BLERNotification, _ios, [this](auto fer_map) {
            for (auto it = fer_map.cbegin(); it != fer_map.cend(); ++it)
              this->setFER(it->first, it->second);
          }));
  // Subscribe to NewFlowNotification
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<FlowUID, stats::FlowInfo>>(
          stats::NewFlowNotification, _ios,
          [this](auto flow_map) { this->setFlowInfo(flow_map); }));
  // Subscribe to FlowPerformanceNotification
  d_nc_tokens.push_back(
      NotificationCenter::shared
          .subscribe<std::map<FlowUID, stats::FlowPerformance>>(
              stats::FlowPerformanceNotification, _ios,
              [this](auto flow_map) { this->setFlowPerformance(flow_map); }));
  // Subscribe to MCS request notification
  d_nc_tokens.push_back(NotificationCenter::shared.subscribe<ofdm::MCSRequest>(
      ofdm::AdaptiveMCSController::MCSRequestNotification, _ios,
      [this](auto mcs_request) {
        // Only broadcast MCS req for Remote->Local link
        if (mcs_request.dst_srnid == d_my_srnid) {
          this->setMCSReq(mcs_request.src_srnid, mcs_request.mcs,
                          static_cast<uint8_t>(mcs_request.seqid));
        }
      }));
  // Subscribe to NewActiveFlowEvent
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<dll::NewActiveFlowEventInfo>(
          dll::NewActiveFlowEvent, _ios,
          [this](auto flow_info) { this->addActiveFlow(flow_info); }));
  // OutcomesUpdateEventInfo
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
          OutcomesUpdateEvent, _ios, [this](auto ei) {
            // Clear active flows when new mandates are received
            this->clearActiveFlows();
          }));
  // Subscribe to duty cycle update
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<stats::DutyCycleInfo>(
          stats::DutyCycleNotification, _ios, [this](auto dc) {
            this->_nodeInfo(d_my_srnid).set_tx_duty_cycle(dc.duty_cycle);
            uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(
                             dc.t.time_since_epoch())
                             .count();
            this->_nodeInfo(d_my_srnid).set_tx_duty_cycle_t(t);
          }));

  // Initialize scrambler (from frame.cc)
  d_scramble_bytes.resize(getNbytes());
  std::mt19937_64 rng(33);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::generate(begin(d_scramble_bytes), end(d_scramble_bytes),
                [&] { return dist(rng); });
  // Start TTL timer
  auto now = std::chrono::system_clock::now();
  _timeout_check_timer.expires_at(now + _tcheck_interval);
  _timeout_check_timer.async_wait([this](auto &e) { this->_timeout_check(); });
}

CCData::~CCData() {
  delete _ios_work;
  d_nc_tokens.clear();
  _timeout_check_timer.cancel();
  _ios.stop();
  _work_thread.join();
}

size_t CCData::getNbytes() { return sizeof(ShortMsgPayload); }

std::shared_ptr<std::vector<uint8_t>> CCData::serialize() {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Set and increment sequence number
  d_pb_msg->set_seq_num(_seq_num++);

  // Update SRN ID list
  d_pb_msg->clear_srnids();
  for (auto x : _all_srnids)
    d_pb_msg->add_srnids(x);

  // Update timestamp
  _nodeInfo(d_my_srnid).set_last_update(getIntRelTimeNow());

  // Serialize
  size_t pb_size = d_pb_msg->ByteSize();
  std::vector<uint8_t> serialized(pb_size);
  d_pb_msg->SerializeWithCachedSizesToArray(serialized.data());

  // Compress
  size_t const max_dst_size = LZ4_compressBound(pb_size);
  auto buf =
      std::make_shared<std::vector<uint8_t>>(max_dst_size + sizeof(uint16_t));
  int data_size = LZ4_compress_default((char *)serialized.data(),
                                       sizeof(uint16_t) + (char *)buf->data(),
                                       pb_size, max_dst_size);
  if (data_size <= 0) {
    log::text("Failed to compress CCData.");
    return nullptr;
  }
  uint16_t const payload_size = data_size;
  std::memcpy(buf->data(), &payload_size, sizeof(payload_size));
  buf->resize(sizeof(payload_size) + payload_size);

  // Logging
  NotificationCenter::shared.post(
      CCPacketEvent,
      CCPacketEventInfo{d_my_srnid, CCPacketEventType::CCEVENT_TX,
                        CCPacketPHYType::CCPHY_OFDM,
                        (uint32_t)d_pb_msg->seq_num(), 0});

  return buf;
}

void CCData::serializeShortMsg(mutable_buffer data) {
  // buffer must be large enough
  assert(sizeof(ShortMsgPayload) <= buffer_size(data));

  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // packet struct
  ShortMsgPayload msg;

  // Set my SRN ID and sequence number
  msg.src_srnid = d_my_srnid;
  msg.seq_num = _seq_num++;

  // Generate SRN ID and channel list
  size_t n_nodes_with_txassgn = 0;
  size_t n_nodes_without_txassgn = 0;
  std::fill_n(msg.srnids, max_nodes, UnspecifiedNodeID);
  for (size_t i = 0; i < max_nodes; ++i) {
    if (i < d_pb_msg->srnids().size()) {
      // housekeeping
      auto const srnid = d_pb_msg->srnids(i);
      bool const txass_avail = d_pb_msg->transmit_assignment().find(srnid) !=
                               d_pb_msg->transmit_assignment().end();
      size_t info_array_pos = txass_avail
                                  ? n_nodes_with_txassgn
                                  : max_nodes - n_nodes_without_txassgn - 1;
      if (txass_avail) {
        n_nodes_with_txassgn++;
      } else {
        n_nodes_without_txassgn++;
      }

      // SRNID
      msg.srnids[info_array_pos] = srnid;

      // TransmitAssignment
      if (txass_avail) {
        msg.tx_assign[info_array_pos] = de::TransmitAssignment::fromProto(
            d_pb_msg->transmit_assignment().at(srnid));
      } else {
        msg.tx_assign[info_array_pos] = de::TransmitAssignment{.bw_idx = 0,
                                                               .chan_idx = 0,
                                                               .chan_ofst = 0,
                                                               .atten = 0.0f,
                                                               .silent = true};
      }

      // Overlaps
      msg.overlapped[info_array_pos] =
          d_pb_msg->channel_overlapped().find(srnid) !=
                  d_pb_msg->channel_overlapped().end()
              ? d_pb_msg->channel_overlapped().at(srnid)
              : false;
    }
  }
  msg.nchan = n_nodes_with_txassgn;

  // Set channel update timestamp
  msg.channel_last_update = d_pb_msg->channel_last_update();

  // Set channel update effective time
  msg.t_channel_effective = d_pb_msg->t_channel_effective();

  // Set overlap info timestamp
  msg.overlapped_last_update = d_pb_msg->overlapped_last_update();

  // Copy
  auto const dst = buffer_cast<char *>(data);
  std::memcpy(dst, &msg, sizeof(ShortMsgPayload));

  // Scramble
  for (size_t i = 0; i < sizeof(ShortMsgPayload); i++) {
    dst[i] ^= d_scramble_bytes[i];
  }

  // Logging
  NotificationCenter::shared.post(
      CCPacketEvent,
      CCPacketEventInfo{msg.src_srnid, CCPacketEventType::CCEVENT_TX,
                        CCPacketPHYType::CCPHY_FSK, msg.seq_num, 0});
}

void CCData::deserializeShortMsg(const_buffer data) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Descramble
  auto const src = buffer_cast<char const *>(data);
  std::vector<uint8_t> src_desc(buffer_size(data));
  for (size_t i = 0; i < sizeof(ShortMsgPayload); i++) {
    src_desc[i] = src[i] ^ d_scramble_bytes[i];
  }

  ShortMsgPayload msg;
  std::memcpy(&msg, src_desc.data(), buffer_size(data));

  assert(msg.nchan < max_nodes);

  // ignore packet from myself
  if (msg.src_srnid == d_my_srnid) {
    return;
  }

  // Logging
  NotificationCenter::shared.post(
      CCPacketEvent,
      CCPacketEventInfo{msg.src_srnid, CCPacketEventType::CCEVENT_RX,
                        CCPacketPHYType::CCPHY_FSK, msg.seq_num, 0});

  _all_srnids.insert(msg.src_srnid);

  for (size_t i = 0; i < max_nodes; ++i) {
    if (msg.srnids[i] != UnspecifiedNodeID) {
      _all_srnids.insert(msg.srnids[i]);
    }
  }

  // Update Rx list (my node)
  auto tx_list = _nodeInfo(d_my_srnid).mutable_rx_srnids();
  auto tx_itr = std::find_if(tx_list->begin(), tx_list->end(),
                             [&](auto const &v) { return v == msg.src_srnid; });
  if (tx_itr == tx_list->end()) {
    _nodeInfo(d_my_srnid).set_rx_srnids_last_update(getIntRelTimeNow());
    _nodeInfo(d_my_srnid).set_last_update(getIntRelTimeNow());
    _nodeInfo(d_my_srnid).add_rx_srnids(msg.src_srnid);
    // Update route
    this->_updateRoute();
  }

  // Reset timeout
  _timeout_counter[msg.src_srnid] = options::net::route_timeout;

  // Channel updates
  if (d_pb_msg->channel_last_update() < msg.channel_last_update) {
    d_pb_msg->set_t_channel_effective(msg.t_channel_effective);
    d_pb_msg->set_channel_last_update(msg.channel_last_update);
    d_pb_msg->clear_transmit_assignment();
    for (size_t i = 0; i < msg.nchan; ++i) {
      (*d_pb_msg->mutable_transmit_assignment())[msg.srnids[i]] =
          msg.tx_assign[i].toProto();
    }
    // Publish
    _publishOFDMChannels();
  }

  // Overlap info update
  if (d_pb_msg->overlapped_last_update() < msg.overlapped_last_update) {
    d_pb_msg->set_overlapped_last_update(msg.overlapped_last_update);
    d_pb_msg->clear_channel_overlapped();
    for (size_t i = 0; i < max_nodes; ++i) {
      if (msg.srnids[i] != UnspecifiedNodeID) {
        (*d_pb_msg->mutable_channel_overlapped())[msg.srnids[i]] =
            msg.overlapped[i];
      }
    }
    // Publish
    _publishOverlapInfo();
  }

  // Notify bbcontroller
  if (_gateway) {
    NotificationCenter::shared.post(NewCCDataNotification, true);
  }
}

void CCData::deserialize(const_buffer data, bool descramble) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Descramble
  auto const src = buffer_cast<char const *>(data);
  std::vector<uint8_t> src_desc(buffer_size(data));
  if (descramble) {
    for (size_t i = 0; i < buffer_size(data); i++)
      src_desc[i] = src[i] ^ d_scramble_bytes[i];
  } else {
    std::memcpy(src_desc.data(), src, buffer_size(data));
  }

  // Deserialize
  uint16_t payload_nbytes = 0;
  std::memcpy(&payload_nbytes, src_desc.data(), sizeof(payload_nbytes));

  // Decompress
  const int decompressed_size = LZ4_decompress_safe(
      (const char *)(src_desc.data() + sizeof(payload_nbytes)),
      (char *)_decompress_tmp.data(), payload_nbytes, _decompress_tmp.size());
  if (decompressed_size <= 0) {
    log::text("Failed to decompress CCData.");
    return;
  }

  // Parse protobuf
  CCDataPb::CCDataMsg pb_msg;
  if (!pb_msg.ParseFromArray(_decompress_tmp.data(), decompressed_size)) {
    log::text("Failed to deserialize CCData.");
    return;
  }

  if (pb_msg.src_srnid() == d_my_srnid) {
    // no need to decode a packet from myself
    return;
  }

  // Logging
  NotificationCenter::shared.post(
      CCPacketEvent,
      CCPacketEventInfo{pb_msg.src_srnid(), CCPacketEventType::CCEVENT_RX,
                        CCPacketPHYType::CCPHY_OFDM, (uint32_t)pb_msg.seq_num(),
                        0});

  // Update SRN IDs
  _all_srnids.insert(pb_msg.src_srnid());
  for (size_t i = 0; i < pb_msg.srnids_size(); ++i) {
    _all_srnids.insert(pb_msg.srnids(i));
  }

  //
  // if gateway node, notify the controller, if not, call
  // publishOFDMChannels like below
  //
  if (_gateway) {
    NotificationCenter::shared.post(NewCCDataNotification, true);
  } else {
    // Update OFDM frequency and BW
    if (d_pb_msg->channel_last_update() < pb_msg.channel_last_update()) {
      // New channel information received. Update local ch info
      d_pb_msg->set_t_channel_effective(pb_msg.t_channel_effective());
      d_pb_msg->set_channel_last_update(pb_msg.channel_last_update());
      (*d_pb_msg->mutable_transmit_assignment()) = pb_msg.transmit_assignment();
      // Publish
      _publishOFDMChannels();
    }

    // Update channel overlap info
    if (d_pb_msg->overlapped_last_update() < pb_msg.overlapped_last_update()) {
      // New channel overlap information received.
      d_pb_msg->set_overlapped_last_update(pb_msg.overlapped_last_update());
      (*d_pb_msg->mutable_channel_overlapped()) = pb_msg.channel_overlapped();
      // Publish
      _publishOverlapInfo();
    }
  }

  // Flag to indicate routing table  update
  bool req_route_update = false;

  // Update Rx list (other nodes)
  for (auto const &node : pb_msg.node_list()) {
    // Get node id
    NodeID const node_id = node.first;

    // ignore my info
    if (node_id == d_my_srnid) {
      continue;
    }

    // ignore old info
    if (node.second.last_update() <= _nodeInfo(node_id).last_update()) {
      continue;
    }

    // process mcs request
    for (auto tx : node.second.link_state()) {
      if (tx.first != d_my_srnid)
        continue; // We are only interested in Local->Remote link
      if (tx.second.mcs_req() !=
          (*_nodeInfo(node_id).mutable_link_state())[tx.first].mcs_req()) {
        ofdm::MCSRequest mcs_req;
        mcs_req.src_srnid = d_my_srnid;
        mcs_req.dst_srnid = node_id;
        mcs_req.mcs = static_cast<ofdm::MCS::Name>(tx.second.mcs_req());
        mcs_req.seqid = static_cast<ofdm::SeqID::ID>(tx.second.seqid_req());
        NotificationCenter::shared.post(
            ofdm::AdaptiveMCSController::MCSRequestNotification, mcs_req);
      }
    }

    // new rx node?
    if (_nodeInfo(node_id).rx_srnids_last_update() <
        node.second.rx_srnids_last_update()) {
      req_route_update = true;
    }

    // update local info
    _nodeInfo(node_id).CopyFrom(node.second);
  }

  // notify
  if (req_route_update)
    _updateRoute();
}

inline de::ChannelAllocUpdateEventInfo
extractChannelParamsUpdateInfo(std::shared_ptr<CCDataPb::CCDataMsg> msg) {
  auto lmsg = std::make_shared<CCDataPb::ChannelParamsUpdateInfo>();
  auto const env = c2api::env.current();
  for (auto const &tx_assign : msg->transmit_assignment()) {
    lmsg->mutable_channel_info()->AddAllocated(
        de::TransmitAssignment::fromProto(tx_assign.second)
            .toLegacyChannelInfo(env.scenario_rf_bandwidth));
  }
  lmsg->set_channel_last_update(msg->channel_last_update());
  return {lmsg};
}

uint32_t CCData::updateOFDMParams(de::TransmitAssignment::Map channels,
                                  Timepoint t_effective) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);
  if (!_gateway) {
    log::text("Only gateway can call updateOFDMParams().", __FILE__, __LINE__);
    return 0;
  }
  d_pb_msg->clear_transmit_assignment();
  for (auto const &x : channels) {
    (*d_pb_msg->mutable_transmit_assignment())[x.first] = x.second.toProto();
  }
  auto reltime = getIntRelTimeNow();
  d_pb_msg->set_channel_last_update(reltime);

  // Set effective time point
  uint64_t t_effective_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          t_effective.time_since_epoch())
          .count();
  d_pb_msg->set_t_channel_effective(t_effective_ms);

  // publish new info immediately
  _publishOFDMChannels();
  return reltime;
}

uint32_t CCData::updateOverlapInfo(std::map<NodeID, bool> overlap_map) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);
  if (!_gateway) {
    log::text("Only gateway can call updateOverlapInfo().", __FILE__, __LINE__);
    return 0;
  }
  d_pb_msg->clear_channel_overlapped();
  for (auto const &x : overlap_map) {
    (*d_pb_msg->mutable_channel_overlapped())[x.first] = x.second;
  }
  auto reltime = getIntRelTimeNow();
  d_pb_msg->set_overlapped_last_update(reltime);
  return reltime;
}

void CCData::setLocation(double latitude, double longitude, double elevation) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Set location
  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  my_srn_entry.set_latitude(latitude);
  my_srn_entry.set_longitude(longitude);
  my_srn_entry.set_elevation(elevation);

  // Update timestamp
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

std::map<uint8_t, Location> CCData::getLocationMap() {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Generate location map
  std::map<uint8_t, Location> location_map;
  auto &adj_list = d_pb_msg->node_list();
  for (auto node : adj_list) {
    // ignore location information older than 3 sec
    // FIXME: this timeout check is not working properly
    // if ((getIntRelTimeNow() - node.second.location_last_update()) < 300) {
    Location loc;
    loc.latitude = node.second.latitude();
    loc.longitude = node.second.longitude();
    loc.elevation = node.second.elevation();
    location_map[node.first] = loc;
    // }
  }

  return location_map;
}

void CCData::setOfferedRate(std::map<uint8_t, size_t> rate) {
  std::lock_guard<std::mutex> lock(d_mutex);
  // Set rate
  auto &my_srn_entry = (*d_pb_msg->mutable_node_list())[d_my_srnid];
  for (auto rx : rate) {
    auto &channel_entry = (*my_srn_entry.mutable_link_state())[rx.first];
    channel_entry.set_offered_traffic_rate(rx.second);
  }
  // Update timestamp
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

double CCData::getOfferedDataRate(uint8_t tx_srnid, uint8_t rx_srnid) {
  std::lock_guard<std::mutex> lock(d_mutex);
  auto &node_list = d_pb_msg->node_list();
  auto it = node_list.find(tx_srnid);
  if (it != node_list.end()) {
    auto it2 = it->second.link_state().find(rx_srnid);
    if (it2 != it->second.link_state().end())
      return it2->second.offered_traffic_rate();
  }
  return 0.0;
}

void CCData::setFER(uint8_t tx_srnid, float fer) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Set location
  _linkState(tx_srnid).set_fer(fer);

  // Update timestamp
  _nodeInfo(d_my_srnid).set_last_update(getIntRelTimeNow());
}

void CCData::setMCSReq(uint8_t tx_srnid, uint8_t mcs, uint8_t seqid) {
  // Lock
  std::lock_guard<std::mutex> lock(d_mutex);

  // Set MCS and SeqID
  _linkState(tx_srnid).set_mcs_req(mcs);
  _linkState(tx_srnid).set_seqid_req(seqid);

  // Update timestamp
  _nodeInfo(d_my_srnid).set_last_update(getIntRelTimeNow());
}

std::map<uint8_t, LinkState> CCData::getLinkStateInfo(uint8_t rx_srnid) {
  std::lock_guard<std::mutex> lock(d_mutex);

  std::map<uint8_t, LinkState> map;
  auto srn_entry = d_pb_msg->node_list().find(rx_srnid);
  if (srn_entry == d_pb_msg->node_list().end()) {
    return map; // specified SRN not found
  }

  auto channel_entries = srn_entry->second.link_state();
  for (auto x : channel_entries) {
    LinkState ls;
    ls.error_rate = x.second.fer();
    ls.mcs = static_cast<ofdm::MCS::Name>(x.second.mcs_req());
    ls.seqid = static_cast<ofdm::SeqID::ID>(x.second.seqid_req());
    map[x.first] = ls;
  }
  return map;
}

std::map<uint8_t, float> CCData::getLSIElapsedTimeSinceLastUpdate() {
  std::lock_guard<std::mutex> lock(d_mutex);
  std::map<uint8_t, float> map;
  auto &node_list = d_pb_msg->node_list();
  for (auto rx : node_list) {
    if (rx.first == d_my_srnid)
      continue; // don't add myself
    float elapsed_time_sec =
        (getIntRelTimeNow() - rx.second.last_update()) / 100.0;
    map[rx.first] = elapsed_time_sec;
  }
  return map;
}

void CCData::setFlowInfo(std::map<FlowUID, stats::FlowInfo> flow_map) {
  std::lock_guard<std::mutex> lock(d_mutex);

  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  my_srn_entry.clear_offered_flows();
  for (auto const &v : flow_map) {
    if (v.second.src == d_my_srnid) {
      auto &entry = (*my_srn_entry.mutable_offered_flows())[v.first];
      entry.set_src_srnid(v.second.src);
      entry.set_dst_srnid(v.second.dst);
    }
  }
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

stats::FlowInfo CCData::getFlowInfo(FlowUID uid) {
  std::lock_guard<std::mutex> lock(d_mutex);

  for (auto const &node : d_pb_msg->node_list()) {
    auto itr = node.second.offered_flows().find(uid);
    if (itr != node.second.offered_flows().end()) {
      // if the specified FlowUID exists, set 'available' to true and return
      // source and destination SRN ID
      return stats::FlowInfo{true, (NodeID)itr->second.src_srnid(),
                             (NodeID)itr->second.dst_srnid()};
    }
  }
  // otherwise, set 'available' to false
  return stats::FlowInfo{false, UnspecifiedNodeID, UnspecifiedNodeID};
}

void CCData::setFlowPerformance(
    std::map<FlowUID, stats::FlowPerformance> perf_map) {
  std::lock_guard<std::mutex> lock(d_mutex);

  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  my_srn_entry.clear_delivered_flows();
  for (auto const &v : perf_map) {
    auto &entry = (*my_srn_entry.mutable_delivered_flows())[v.first];
    entry.set_scalar_performance(v.second.scalar_performance);
    entry.set_mps(v.second.mps);
    entry.set_point_value(v.second.point_value);
  }

  // Update timestamp
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

std::map<FlowUID, stats::FlowPerformance> CCData::getFlowPerformance() {
  std::lock_guard<std::mutex> lock(d_mutex);

  std::map<FlowUID, stats::FlowPerformance> perf_map;
  for (auto const &node : d_pb_msg->node_list()) {
    if ((getIntRelTimeNow() - node.second.last_update()) > 500) {
      // Ignore node information older than 5 secs
      continue;
    }
    for (auto &perf : node.second.delivered_flows()) {
      stats::FlowPerformance p;
      p.scalar_performance = perf.second.scalar_performance();
      p.mps = perf.second.mps();
      p.point_value = perf.second.point_value();
      perf_map[perf.first] = p;
    }
  }
  return perf_map;
}

std::vector<NodeID> CCData::getAllSRNIDs() {
  std::lock_guard<std::mutex> lock(d_mutex);
  std::vector<NodeID> nodes;
  for (auto const &v : _all_srnids) {
    nodes.push_back(v);
  }
  return nodes;
}

void CCData::addActiveFlow(dll::NewActiveFlowEventInfo flow_info) {
  std::lock_guard<std::mutex> lock(d_mutex);
  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  auto &entry = (*my_srn_entry.mutable_tx_flows())[flow_info.flow_uid];
  entry.set_bits_per_segment(flow_info.bits_per_segment);
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

std::map<NodeID, std::vector<dll::NewActiveFlowEventInfo>>
CCData::getActiveFlows() {
  std::lock_guard<std::mutex> lock(d_mutex);
  std::map<NodeID, std::vector<dll::NewActiveFlowEventInfo>> active_flows;
  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  for (auto const &v : d_pb_msg->node_list()) {
    for (auto const &f : v.second.tx_flows()) {
      active_flows[v.first].push_back(dll::NewActiveFlowEventInfo{
          (FlowUID)f.first, f.second.bits_per_segment()});
    }
  }
  return active_flows;
}

void CCData::clearActiveFlows() {
  std::lock_guard<std::mutex> lock(d_mutex);
  auto &my_srn_entry = _nodeInfo(d_my_srnid);
  my_srn_entry.clear_tx_flows();
  my_srn_entry.set_last_update(getIntRelTimeNow());
}

void CCData::_publishOFDMChannels() {
  // Generate channel vector
  OFDMChannelUpdateInfo ch_info;
  ch_info.t_effective =
      Timepoint(std::chrono::milliseconds(d_pb_msg->t_channel_effective()));
  for (auto const &tx_assign : d_pb_msg->transmit_assignment()) {
    ch_info.channels[tx_assign.first] =
        de::TransmitAssignment::fromProto(tx_assign.second);
  }
  ch_info.t_last_update = timepointFromRelTime(d_pb_msg->channel_last_update());

  // Post
  NotificationCenter::shared.post(OFDMChannelBandNotification, ch_info);
  // Log
  NotificationCenter::shared.post(de::ChannelAllocUpdateEvent,
                                  extractChannelParamsUpdateInfo(d_pb_msg));
}

void CCData::_publishOverlapInfo() {
  OFDMChannelOverlapInfo ol_info;
  for (auto const &ol : d_pb_msg->channel_overlapped()) {
    ol_info.overlap_map[ol.first] = ol.second;
  }
  ol_info.t_last_update =
      timepointFromRelTime(d_pb_msg->overlapped_last_update());

  // Post
  NotificationCenter::shared.post(OFDMChannelOverlapNotification, ol_info);
}

NodeID CCData::getGatewaySRNID() {
  NodeID gw_srn = UnspecifiedNodeID;
  for (auto const &node : d_pb_msg->node_list()) {
    if (node.second.gateway()) {
      gw_srn = node.first;
    }
  }
  // Returns SRN ID of the gateway.
  // If gateway does not exist, return UnspecifiedNodeID.
  return gw_srn;
}

std::map<NodeID, stats::DutyCycleInfo> CCData::getDutyCycle() {
  std::lock_guard<std::mutex> lock(d_mutex);
  std::map<NodeID, stats::DutyCycleInfo> dc;
  for (auto const &node : d_pb_msg->node_list()) {
    auto t =
        Timepoint(std::chrono::milliseconds(node.second.tx_duty_cycle_t()));
    dc[node.first] =
        stats::DutyCycleInfo{.duty_cycle = node.second.tx_duty_cycle(), .t = t};
  }
  return dc;
}

void CCData::_timeout_check() {
  std::lock_guard<std::mutex> lock(d_mutex);
  bool route_update = false;
  for (auto it = _timeout_counter.begin(); it != _timeout_counter.end();) {
    // Decrement TTL
    it->second--;
    // Update Rx list
    if (it->second == 0) {
      auto adj_list = d_pb_msg->mutable_node_list();
      auto &my_srn_entry = (*adj_list)[d_my_srnid];
      auto tx_list = my_srn_entry.mutable_rx_srnids();
      auto tx_list_itr = tx_list->begin();
      while ((tx_list_itr != tx_list->end()) && (*tx_list_itr != it->first))
        ++tx_list_itr;
      if (tx_list_itr != tx_list->end()) {
        my_srn_entry.set_rx_srnids_last_update(getIntRelTimeNow());
        my_srn_entry.mutable_rx_srnids()->erase(tx_list_itr);
        it = _timeout_counter.erase(it);
      } else {
        log::doomsday("Fatal error: inconsistent CC data", __FILE__, __LINE__);
      }
      route_update = true;
    } else {
      ++it;
    }
  }

  if (route_update) {
    _updateRoute();
  }

  // Wait
  _timeout_check_timer.expires_at(_timeout_check_timer.expires_at() +
                                  _tcheck_interval);
  _timeout_check_timer.async_wait([this](auto &e) { this->_timeout_check(); });
}

void CCData::_updateRoute() {
  // Generate netmap and publish
  NetworkMap netmap;
  auto &adj_list = d_pb_msg->node_list();
  for (auto it_rx = adj_list.begin(); it_rx != adj_list.end(); ++it_rx) {
    uint8_t to_id = it_rx->first;
    for (unsigned int i = 0; i < it_rx->second.rx_srnids_size(); ++i) {
      uint8_t from_id = it_rx->second.rx_srnids(i);
      netmap.setLink(from_id, to_id, 1);
    }
  }

  // Publish
  NotificationCenter::shared.post(NetworkMapEvent, NetworkMapEventInfo{netmap});

#ifndef NDEBUG
  // debug output
  auto srns = netmap.getAllSrnIds();
  std::sort(srns.begin(), srns.end());
  std::cout << "== Published Network Map ==\n";
  for (auto const srn_rx : srns) {
    std::cout << (int)srn_rx << " : ";
    for (auto const srn_tx : srns) {
      char val;
      if (srn_tx == srn_rx)
        val = '-';
      else
        val = netmap.isLinkUp(srn_tx, srn_rx) ? '1' : '0';
      std::cout << val << " ";
    }
    std::cout << "\n";
  }
#endif
}

de::TransmitAssignment::Map CCData::getTransmitAssignment() {
  std::lock_guard<decltype(d_mutex)> lock(d_mutex);
  std::map<NodeID, de::TransmitAssignment> o;
  for (auto const tx_assign : d_pb_msg->transmit_assignment()) {
    o[tx_assign.first] = de::TransmitAssignment::fromProto(tx_assign.second);
  }
  return o;
}

} // namespace controlchannel
} // namespace bamradio
