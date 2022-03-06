// -*- c++ -*-
//
// Events for logger.
//
// This file consolidates all of the *_events.h files. If you want to log
// something, add it here!
//
// Copyright (c) 2017-2018 Dennis Ogbe
// Copyright (c) 2017-2018 Stephen Larew
// Copyright (c) 2017-2018 Tomohiro Arakawa

#ifndef a1bf5aba5c5dba61e72d
#define a1bf5aba5c5dba61e72d

#include "cc_data.pb.h"
#include "log.pb.h"
#include "mcs.h"
#include "networkmap.h"
#include "notify.h"
#include "ofdm.h"
#include "segment.h"
#include "dll_types.h"

#include <chrono>
#include <cstdint>
#include <memory>

#include "json.hpp"
#include <boost/format.hpp>
#include <cil.pb.h>
#include <registration.pb.h>
#include <sqlite3.h>
#include <uhd/types/metadata.hpp>
#include <uhd/utils/msg.hpp>

namespace bamradio {

// helper classes
class DBLayout {
public:
  enum Type { INT = 0, REAL, BLOB, TEXT };
  DBLayout(std::string const &name);
  DBLayout &addColumn(std::string const &name, Type type,
                      std::string const &attr = "");
  std::string sql() const;
  int prepare(sqlite3 *db, sqlite3_stmt **s, std::string const &name) const;
  int pos(std::string const &col) const;
  int nCols() const;

  static std::string type2text(Type);

private:
  std::string _name;
  std::vector<std::string> _columns;
  std::vector<Type> _types;
  std::vector<std::string> _attr;
};

typedef int64_t UnixTime; // just to be clear

namespace dll {

/// Frame is being sent.
extern NotificationCenter::Name const SentFrameEvent;
/// Segment is being sent.
extern NotificationCenter::Name const SentSegmentEvent;
/// Frame was detected.
extern NotificationCenter::Name const DetectedFrameEvent;
/// All segments of a frame were received correctly
extern NotificationCenter::Name const ReceivedFrameEvent;
/// Frame was detected but invalid or detection is false-positive.
extern NotificationCenter::Name const InvalidFrameHeaderEvent;
/// Block was decoded.
extern NotificationCenter::Name const ReceivedBlockEvent;
/// Received a complete segment.
extern NotificationCenter::Name const ReceivedCompleteSegmentEvent;
/// Segment delay.
extern NotificationCenter::Name const CoDelDelayEvent;
/// CoDel state.
extern NotificationCenter::Name const CoDelStateEvent;
/// New flow.
extern NotificationCenter::Name const NewFlowEvent;
extern NotificationCenter::Name const FlowQueuePushEvent;
extern NotificationCenter::Name const FlowQueuePopEvent;
extern NotificationCenter::Name const ScheduleUpdateEvent;
extern NotificationCenter::Name const NewActiveFlowEvent;

struct FlowQueuePushEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  size_t numQueued;
  size_t bytesQueued;
  int64_t currentRound;
  std::chrono::nanoseconds balance;
};
void to_json(nlohmann::json &j, FlowQueuePushEventInfo const &ei);

struct FlowQueuePopEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  size_t numQueued;
  size_t bytesQueued;
  int64_t currentRound;
  std::chrono::nanoseconds balance, quantumCredit, dequeueDebit;
};
void to_json(nlohmann::json &j, FlowQueuePopEventInfo const &ei);

struct ScheduleUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, std::map<std::string, sqlite3_stmt *> &stmt,
             std::chrono::system_clock::time_point t,
             std::chrono::steady_clock::time_point ts) const;

  int64_t round;
  std::map<FlowID, std::chrono::nanoseconds> quantums;
  bool valid;
  std::chrono::duration<float> period, periodlb, periodub;
  std::string flowInfos;

  struct FlowQuantum {
    static const std::string Name;
    static const DBLayout Layout;
    int to_sql(sqlite3 *db, std::map<std::string, sqlite3_stmt *> &stmt,
               std::chrono::system_clock::time_point t,
               std::chrono::steady_clock::time_point ts) const;

    int64_t roundepoch;
    decltype(ScheduleUpdateEventInfo::quantums)::const_iterator v;
  };
};
void to_json(nlohmann::json &j, ScheduleUpdateEventInfo const &ei);

struct SentFrameEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  NodeID sourceNodeID, destNodeID;
  ofdm::MCS::Name payloadMCS;
  ofdm::SeqID::ID payloadSymSeqID;
  int64_t nsamples;
  uint16_t seqNum;
  int64_t frameID;
  size_t numBlocks;
  UnixTime txTime;
  float sampleGain;
};

void to_json(nlohmann::json &j, SentFrameEventInfo const &ei);

struct SentSegmentEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  NodeID sourceNodeID, destNodeID;
  uint16_t seqNum;
  int64_t frameID;
  UnixTime sourceTime;
  nlohmann::json description;
  int type;
  size_t nbytes;
};

void to_json(nlohmann::json &j, SentSegmentEventInfo const &ei);

struct ReceivedCompleteSegmentEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  NodeID sourceNodeID, destNodeID;
  uint16_t seqNum;
  int64_t frameID;
  UnixTime rxTime;
  UnixTime sourceTime;
  nlohmann::json description;
  bool queueSuccess;
};

void to_json(nlohmann::json &j, ReceivedCompleteSegmentEventInfo const &ei);

struct DetectedFrameEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  size_t channelIdx;
  NodeID sourceNodeID, destNodeID;
  ofdm::MCS::Name payloadMCS;
  ofdm::SeqID::ID payloadSymSeqID;
  uint16_t seqNum;
  int64_t frameID;
  size_t numBlocks;
  UnixTime rxTime;
  float snr;
};

void to_json(nlohmann::json &j, DetectedFrameEventInfo const &ei);

struct ReceivedFrameEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  NodeID sourceNodeID, destNodeID;
  int64_t frameID;
  int64_t numBlocks;
  int64_t numBlocksValid;
  bool rxSuccess;
  float snr;
  float noiseVar;
};

void to_json(nlohmann::json &j, ReceivedFrameEventInfo const &ei);

struct InvalidFrameHeaderEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  size_t channelIdx;
  UnixTime rxTime;
  float snr;
};

void to_json(nlohmann::json &j, InvalidFrameHeaderEventInfo const &ei);

struct ReceivedBlockEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  size_t channelIdx;
  NodeID sourceNodeID;
  size_t numBits;
  bool valid;
  float snr;
  int64_t frameID;
  uint16_t seqNum;
  uint16_t blockNumber;
};

void to_json(nlohmann::json &j, ReceivedBlockEventInfo const &ei);

struct CoDelDelayEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::chrono::system_clock::time_point sourceTime;
  std::chrono::nanoseconds delay;
  nlohmann::json description;
};

void to_json(nlohmann::json &j, CoDelDelayEventInfo const &ei);

struct NewFlowEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  int64_t currentRound;
  std::string queueType;
};

void to_json(nlohmann::json &j, NewFlowEventInfo const &ei);

struct CoDelStateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  std::chrono::steady_clock::time_point first_above_time;
  std::chrono::steady_clock::time_point drop_next;
  ssize_t drop_count;
  ssize_t last_drop_count;
  bool dropping;
  ssize_t bytes;
  size_t queue_size;
  std::chrono::nanoseconds avg_latency;
};

void to_json(nlohmann::json &j, CoDelStateEventInfo const &ei);

extern NotificationCenter::Name const ReceivedARQFeedbackEvent;

struct ReceivedARQFeedbackEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  int64_t frameID;
  uint16_t flowUID;
  uint8_t burstNum;
  uint16_t lastSeq;
};

void to_json(nlohmann::json &j, ReceivedARQFeedbackEventInfo const &ei);

extern NotificationCenter::Name const FlowQueueResendEvent;

struct FlowQueueResendEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowID flow;
  std::chrono::system_clock::time_point sourceTime;
  uint8_t burstNum;
  uint16_t seqNum;
  int disposition;
  size_t numQueued;
  size_t bytesQueued;
};

void to_json(nlohmann::json &j, FlowQueueResendEventInfo const &ei);

extern NotificationCenter::Name const FlowTrackerStateUpdateEvent;

struct FlowTrackerStateUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowUID flow_uid;
  std::chrono::system_clock::time_point sourceTime;
  uint8_t burstNum;
  uint16_t lastSeq;
  int64_t burstSize;
  int64_t burstRemaining;
  bool completed;
};

void to_json(nlohmann::json &j, FlowTrackerStateUpdateEventInfo const &ei);

extern NotificationCenter::Name const FlowTrackerIMEvent;

struct FlowTrackerIMEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  FlowUID flow_uid;
  std::chrono::system_clock::time_point sourceTime;
  uint8_t burstNum;
  bool completed;
  bool expired;
};

void to_json(nlohmann::json &j, FlowTrackerIMEventInfo const &ei);

struct NewActiveFlowEventInfo {
  FlowUID flow_uid;
  size_t bits_per_segment;
};

} // namespace dll

namespace net {

// Route Decision
extern NotificationCenter::Name const RouteDecisionEvent;

enum RouterAction : uint8_t { FORWARD = 0, WRITE_TO_TUN, DROP_UNKNOWN_PACKET };

struct RouteDecisionEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  RouterAction action;
  NodeID src_srnid;
  NodeID next_hop;
  dll::SegmentType type;
  boost::asio::ip::address_v4 src_ip;
  boost::asio::ip::address_v4 dst_ip;
  uint16_t src_port;
  uint16_t dst_port;
  uint8_t protocol;
  size_t packetLength;
  std::chrono::system_clock::time_point sourceTime;
};

void to_json(nlohmann::json &j, RouteDecisionEventInfo const &ei);

// Routing table update
extern NotificationCenter::Name const RoutingTableUpdateEvent;

struct RoutingTableUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::map<uint8_t, uint8_t> table;
};

void to_json(nlohmann::json &j, RoutingTableUpdateEventInfo const &ei);

} // namespace net

namespace gps {

extern NotificationCenter::Name const GPSEvent;

// GPS Event
enum GPSEventType : uint8_t {
  TRY_CONNECT_GOOD = 0,
  TRY_CONNECT_BAD,
  READ_ERROR,
  READ_NO_FIX,
  READ_NO_DATA,
  READ_GOOD,
  READ_TIMEOUT,
};

struct GPSEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  GPSEventType type;
  double lat;
  double lon;
  double alt;
};

void to_json(nlohmann::json &j, GPSEventInfo const &ei);

} // namespace gps

namespace log {

///////////////////////////////////////////////////////////////////////////////
// Doomsday and generic
///////////////////////////////////////////////////////////////////////////////

//
// Doomsday event. Instead of calling abort(), we use this event to write an
// error message to the log before crashing. Use anywhere where you otherwise
// would have called abort().
//
extern NotificationCenter::Name const DoomsdayEvent;

struct DoomsdayEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::string msg;
  std::string file;
  std::string function;
  int line;
  std::function<void()> judgement_day;
};

void to_json(nlohmann::json &j, DoomsdayEventInfo const &ei);

//
// Use this function to freak out
//
void doomsday(std::string const &msg, std::string const &file = "",
              int line = 0, char const *func = nullptr)
    __attribute__((noreturn));

//
// Generic text logging event. This really should not be used other than for
// quick stdout one-offs.
//
extern NotificationCenter::Name const TextLogEvent;

struct TextLogEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::string msg;
  std::string file;
  int line;
};

void to_json(nlohmann::json &j, TextLogEventInfo const &ei);

//
// Use this function to log some text
//
void text(std::string const &msg, std::string const &file = "", int line = 0);
void text(boost::format const &msg, std::string const &file = "", int line = 0);

// debug for database
extern NotificationCenter::Name const DBEvent;

struct DBEventInfo {
  enum Type : uint8_t { SUCCESS = 0, FAIL, CALLBACK } type;
  std::string msg; // Human-readable message (or return from callback)
  std::string sql; // sql statement this event describes
  int64_t sql_id;  // ID of the sql statement made
};

void to_json(nlohmann::json &j, DBEventInfo const &ei);

// scenario start time event
extern NotificationCenter::Name const ScenarioStartEvent;

struct ScenarioStartEventInfo {
  std::chrono::system_clock::time_point time;
};

} // namespace log

// this is for Stephen
#define panic(msg)                                                             \
  do {                                                                         \
    ::bamradio::log::doomsday((msg), __FILE__, __LINE__, __PRETTY_FUNCTION__); \
  } while (false)
#define unimplemented()                                                        \
  do {                                                                         \
    ::bamradio::log::doomsday("unimplemented feature", __FILE__, __LINE__,     \
                              __PRETTY_FUNCTION__);                            \
  } while (false)

///////////////////////////////////////////////////////////////////////////////
// PHY Events
///////////////////////////////////////////////////////////////////////////////

extern NotificationCenter::Name const BurstSendEvent;

struct BurstSendEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  UnixTime time;
};

void to_json(nlohmann::json &j, BurstSendEventInfo const &ei);

namespace uhdfeedback {
extern NotificationCenter::Name const UHDAsyncEvent;
extern NotificationCenter::Name const UHDMsgEvent;

struct UHDAsyncEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  size_t channel;
  UnixTime time;
  ::uhd::async_metadata_t::event_code_t event_code;
};

void to_json(nlohmann::json &j, UHDAsyncEventInfo const &ei);

const char *uhdAsync2str(::uhd::async_metadata_t::event_code_t event_code);

struct UHDMsgEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  ::uhd::msg::type_t type;
  std::string msg;
};

const char *uhdMsg2str(::uhd::msg::type_t type);
void to_json(nlohmann::json &j, UHDMsgEventInfo const &ei);
} // namespace uhdfeedback

// psd log
namespace psdsensing {
extern NotificationCenter::Name const PSDUpdateEvent;

struct PSDUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::vector<float> psd;
  int64_t time_ns;
};

void to_json(nlohmann::json &j, PSDUpdateEventInfo const &ei);

extern NotificationCenter::Name const PSDRxEvent;

struct PSDRxEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  NodeID node_id;
  std::vector<float> psd;
  int64_t time_ns;
};
void to_json(nlohmann::json &j, PSDRxEventInfo const &ei);
} // namespace psdsensing

///////////////////////////////////////////////////////////////////////////////
// Control Channel Events
///////////////////////////////////////////////////////////////////////////////
namespace controlchannel {

extern NotificationCenter::Name const CCPacketEvent;
extern NotificationCenter::Name const NetworkMapEvent;

enum CCPacketEventType : uint8_t { CCEVENT_TX = 0, CCEVENT_RX };
enum CCPacketPHYType : uint8_t { CCPHY_OFDM = 0, CCPHY_FSK };
struct CCPacketEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  uint32_t src_srnid;
  CCPacketEventType event_type;
  CCPacketPHYType phy_type;
  uint32_t seq_num;
  uint32_t hash;
};
void to_json(nlohmann::json &j, CCPacketEventInfo const &ei);

struct NetworkMapEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  NetworkMap netmap;
};

void to_json(nlohmann::json &j, NetworkMapEventInfo const &ei);

} // namespace controlchannel

///////////////////////////////////////////////////////////////////////////////
// CIL Client Events
///////////////////////////////////////////////////////////////////////////////
namespace collab {

extern NotificationCenter::Name const CollabRxEvent;
extern NotificationCenter::Name const CollabTxEvent;
extern NotificationCenter::Name const ServerRxEvent;
extern NotificationCenter::Name const ServerTxEvent;
extern NotificationCenter::Name const ConnectionEvent;
extern NotificationCenter::Name const StateChangeEvent;
extern NotificationCenter::Name const CollabPeerEvent;
extern NotificationCenter::Name const ErrorEvent;

struct CollabRxEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<sc2::cil::CilMessage> msg;
  int64_t msg_id;
};
void to_json(nlohmann::json &j, CollabRxEventInfo const &ei);

struct CollabTxEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<sc2::cil::CilMessage> msg;
  int64_t msg_id;
  std::vector<boost::asio::ip::address_v4> dst;
};
void to_json(nlohmann::json &j, CollabTxEventInfo const &ei);

struct ServerRxEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<sc2::reg::TellClient> msg;
  int64_t msg_id;
};
void to_json(nlohmann::json &j, ServerRxEventInfo const &ei);

struct ServerTxEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<sc2::reg::TalkToServer> msg;
  int64_t msg_id;
};
void to_json(nlohmann::json &j, ServerTxEventInfo const &ei);

struct ConnectionEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  bool success;
};
void to_json(nlohmann::json &j, ConnectionEventInfo const &ei);

struct StateChangeEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::string from;
  std::string to;
};
void to_json(nlohmann::json &j, StateChangeEventInfo const &ei);

struct CollabPeerEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  bool add;
  boost::asio::ip::address_v4 ip;
};
void to_json(nlohmann::json &j, CollabPeerEventInfo const &ei);

enum ErrorType : uint8_t {
  UNSUPPORTED = 0,
  NOT_CONNECTED,
  SEND_FAIL,
  NO_PEER,
  SERVER_SYNC
};

struct ErrorEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  ErrorType type;
  std::shared_ptr<sc2::cil::CilMessage> msg;
};

void to_json(nlohmann::json &j, ErrorEventInfo const &ei);

} // namespace collab

///////////////////////////////////////////////////////////////////////////////
// Achieved IMs
///////////////////////////////////////////////////////////////////////////////
extern NotificationCenter::Name const AchievedIMsUpdateEvent;

struct AchievedIMsUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  size_t n_achieved_IMs;
};

void to_json(nlohmann::json &j, AchievedIMsUpdateEventInfo const &ei);

///////////////////////////////////////////////////////////////////////////////
// Mandated Outcomes
///////////////////////////////////////////////////////////////////////////////
extern NotificationCenter::Name const OutcomesUpdateEvent;

struct OutcomesUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  nlohmann::json j;
};

void to_json(nlohmann::json &j, OutcomesUpdateEventInfo const &ei);

///////////////////////////////////////////////////////////////////////////////
// Incumbent Protection
///////////////////////////////////////////////////////////////////////////////
extern NotificationCenter::Name const IncumbentAttenuationUpdateEvent;

struct IncumbentAttenuationUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  int32_t incumbent_id;
  float attenuation_dB;
  uint32_t OFDMParamsUpdateID;
};

void to_json(nlohmann::json &j, IncumbentAttenuationUpdateEventInfo const &ei);

///////////////////////////////////////////////////////////////////////////////
// C2API
///////////////////////////////////////////////////////////////////////////////
extern NotificationCenter::Name const C2APIEvent;

enum C2APIEventType : uint8_t { RECEIVED_COMMAND = 0, UPDATED_STATUS };

struct C2APIEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  C2APIEventType type;
  std::string txt;
};

void to_json(nlohmann::json &j, C2APIEventInfo const &ei);

///////////////////////////////////////////////////////////////////////////////
// Environment Updates
///////////////////////////////////////////////////////////////////////////////
extern NotificationCenter::Name const EnvironmentUpdateEvent;

struct EnvironmentUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  enum class CollabNetworkType { Internet = 0, SATCOM, HF, UNSPEC };
  struct IncumbentProtection {
    int64_t center_frequency; // [Hz]
    int64_t rf_bandwidth;     // [Hz]
  };

  CollabNetworkType collab_network_type;
  IncumbentProtection incumbent_protection;

  int64_t scenario_rf_bandwidth;     // [Hz]
  int64_t scenario_center_frequency; // [Hz]
  uint32_t bonus_threshold; //Score Threshold above which we enter the Bonus Zone [New in Phase-3]

  bool has_incumbent;
  int64_t stage_number;
  int64_t timestamp;

  nlohmann::json raw;
};

void to_json(nlohmann::json &j, EnvironmentUpdateEventInfo const &ei);

///////////////////////////////////////////////////////////////////////////////
// OFDM Physical layer processing
///////////////////////////////////////////////////////////////////////////////
namespace ofdm {

extern NotificationCenter::Name const ModulationEvent;
extern NotificationCenter::Name const ChannelEstimationEvent;
extern NotificationCenter::Name const SynchronizationEvent;
extern NotificationCenter::Name const MCSDecisionEvent;

struct ModulationEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  /// all time differences in nanoseconds, i.e. t_code_ns gives the time it
  /// took to do the encoding step at the transmitter

  /// time to encode all bits
  int64_t t_code_ns;
  /// time to modulate the entire frame
  int64_t t_mod_ns;
  /// time to DFT spread all OFDM symbols
  int64_t t_spread_ns;
  /// time to map subcarriers
  int64_t t_map_ns;
  /// time to shift frequency-domain symbols to time domain + upsample
  int64_t t_shift_ns;
  /// time to apply cyclic prefix and windowing
  int64_t t_cp_ns;
  /// time to frequency shift frame to desired center frequency
  int64_t t_mix_ns;
  /// time to multiply by the power control factor
  int64_t t_scale_ns;
  /// time to enqueue the complexPDU
  int64_t t_stream_ns;

  /// identification of the frame. use this to look up segments and other
  /// parameters
  NodeID srcNodeID;
  int64_t frameID;
};

void to_json(nlohmann::json &j, ModulationEventInfo const &ei);

struct ChannelEstimationEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  /// channel estimate
  std::vector<std::complex<float>> chanest;
  /// some room for later?
  std::string desc;
  /// identification of the frame. use this to look up segments and other
  /// parameters
  NodeID srcNodeID;
  int64_t frameID;
};

void to_json(nlohmann::json &j, ChannelEstimationEventInfo const &ei);

struct SynchronizationEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  float snr;
  float domega;

  /// identification of the frame. use this to look up segments and other
  /// parameters
  NodeID srcNodeID;
  int64_t frameID;
};

void to_json(nlohmann::json &j, SynchronizationEventInfo const &ei);

struct MCSDecisionEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  NodeID txNodeID;
  float noiseVar;
  float errorRate;
  bool overlapped;
  MCS::Name payloadMCS;
  SeqID::ID payloadSymSeqID;
};

void to_json(nlohmann::json &j, MCSDecisionEventInfo const &ei);

} // namespace ofdm

///////////////////////////////////////////////////////////////////////////////
// Decision engine events
///////////////////////////////////////////////////////////////////////////////
namespace decisionengine {

extern NotificationCenter::Name const ChannelAllocUpdateEvent;
extern NotificationCenter::Name const ChannelAllocEvent; // don't mix these up!

// n.b. the difference between these two bad boys is that the ChannelAllocUpdate
// Event is logged by every SRN (including the gateway), but the ChannelAlloc
// Event is only logged by the gateway which actually performs the channel
// allocation
struct ChannelAllocUpdateEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<CCDataPb::ChannelParamsUpdateInfo> data;
};

void to_json(nlohmann::json &j, ChannelAllocUpdateEventInfo const &ei);

struct ChannelAllocEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<BAMLogPb::ChannelAllocEventInfo> data;
};

void to_json(nlohmann::json &j, ChannelAllocEventInfo const &ei);

// log the decision engine input lisp object to database
extern NotificationCenter::Name const StepEvent;

struct StepEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<std::vector<uint8_t>> data;
};

void to_json(nlohmann::json &j, StepEventInfo const &ei);

// log the decision engine outputs to database
extern NotificationCenter::Name const StepOutputEvent;

struct StepOutputEventInfo {
  static const std::string Name;
  static const DBLayout Layout;
  int to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t, UnixTime ts) const;

  std::shared_ptr<std::vector<uint8_t>> data;
};

void to_json(nlohmann::json &j, StepOutputEventInfo const &ei);

} // namespace decisionengine

} // namespace bamradio

#endif // a1bf5aba5c5dba61e72d
