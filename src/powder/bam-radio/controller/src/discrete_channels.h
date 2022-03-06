// -*- c++ -*-
// Copyright (c) 2019 Dennis Ogbe

#ifndef a926e9677b6ea199717ac
#define a926e9677b6ea199717ac

#include "dll_types.h"
#include "lisp.h"
#include "radiocontroller_types.h"
#include <cc_data.pb.h>

#include <cstdint>
#include <map>
#include <vector>

namespace bamradio {
namespace decisionengine {

//
// static channelization table for all possible scenario bandwidths
//
struct Channelization {
  // the highest bandwidth allowed for this rf_bandwidth
  int const max_bw_idx;

  // vector of all possible offsets from 0 Hz for this rf_bandwidth
  std::vector<int64_t> const center_offsets;

  // get the initial assignment of slots to channels after an environment update
  std::vector<uint8_t> initial_assignment() const;

  static const std::map<int64_t, Channelization> table;

  // given a bandwidth, return max_subchannel_idx and center_offsets
  static Channelization const &get(int64_t rf_bandwidth);

  // get the current maximum bandwidth -- use at own risk, this is dirty
  static double get_current_max_bandwidth();
};

cl_object toLisp(Channelization c);

//
// High-level representation of where an SRN is transmitting
//
struct TransmitAssignment {
  /// map from srn id to a transmit assignment
  typedef std::map<NodeID, TransmitAssignment> Map;

  /// index into the bandwidth table. "Which bandwidth am I using?"
  uint8_t bw_idx;
  /// index into the Channelization table. "Which channel am I transmitting on?"
  uint8_t chan_idx;
  /// offset from the channel center frequency. "How far off am I transmitting?"
  int32_t chan_ofst;
  /// Attenuation in dB. "How much am I scaling my signal?"
  float atten;
  /// Flag stopping transmission. "Am I allowed to transmit?
  bool silent;

  /// convert myself (high-level representation) to a radiocontroller channel
  /// (low-level representation, the radiocontroller and below do not need to
  /// know about the discrete channelization)
  bamradio::Channel toRCChannel(int64_t rf_bandwith) const;

  /// convert myself to the format we are using in the GUI log
  CCDataPb::ChannelInfo *toLegacyChannelInfo(int64_t rf_bandwidth) const;

  /// convert myself to protobuf
  CCDataPb::TransmitAssignmentMsg toProto() const;

  /// create from protobuf
  static TransmitAssignment
  fromProto(CCDataPb::TransmitAssignmentMsg const &msg);

  /// create from its lisp representation
  static TransmitAssignment fromLisp(cl_object obj);
};

// convert a transmit assignment to its lisp representation
cl_object toLisp(TransmitAssignment);

//
// data structure to report transmit assignment updates -- here for legacy
// reasons
//
struct TxAssignmentUpdate {
  /// the new transmit assignments
  TransmitAssignment::Map assignment_map;

  /// some bools indicating what was changed
  bool channel_updated;
  bool bandwidth_updated;
  bool atten_updated;

  // re-construct from lisp object
  static TxAssignmentUpdate fromLisp(cl_object obj);
};

} // namespace decisionengine
} // namespace bamradio

#endif // a926e9677b6ea199717ac
