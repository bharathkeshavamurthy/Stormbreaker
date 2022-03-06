// -*- c++ -*-
//
// Copyright (c) 2017-2018 Dennis Ogbe
// Copyright (c) 2017-2018 Stephen Larew
// Copyright (c) 2017-2018 Tomohiro Arakawa

#include "events.h"

#include <condition_variable>
#include <mutex>
#include <type_traits>

#include <boost/asio.hpp>
#include <boost/format.hpp>

using boost::asio::ip::address_v4;

namespace bamradio {

///////////////////////////////////////////////////////////////////////////////
// SQL Helpers
///////////////////////////////////////////////////////////////////////////////

DBLayout::DBLayout(std::string const &name) : _name(name) {}

std::string DBLayout::type2text(DBLayout::Type t) {
  switch (t) {
  case Type::INT:
    return "INT";
    break;
  case Type::REAL:
    return "REAL";
    break;
  case Type::BLOB:
    return "BLOB";
    break;
  case Type::TEXT:
    return "TEXT";
    break;
  }
}

DBLayout &DBLayout::addColumn(std::string const &name, Type type,
                              std::string const &attr) {
  _types.push_back(type);
  _attr.push_back(attr);
  _columns.push_back(name);
  return *this;
}

std::string DBLayout::sql() const {
  std::stringstream s;
  s << "CREATE TABLE IF NOT EXISTS " << _name << "(";
  for (size_t i = 0; i < _types.size(); ++i) {
    auto const &name = _columns[i];
    auto const &type = _types[i];
    auto const &attr = _attr[i];
    s << name << " " << type2text(type) << " " << attr << ", ";
  }
  // primary key (auto-increments)
  s << "id INTEGER PRIMARY KEY);";
  return s.str();
}

// compile the insert statement
int DBLayout::prepare(sqlite3 *db, sqlite3_stmt **s,
                      std::string const &name) const {
  std::stringstream ss;
  ss << "INSERT INTO " << name << " (";
  std::vector<std::string> vals;
  for (size_t i = 0; i < _columns.size(); ++i) {
    auto const &c = _columns[i];
    ss << c;
    if (i < _columns.size() - 1) {
      ss << ",";
    }
  }

  ss << ") VALUES (";

  for (size_t i = 0; i < _columns.size(); ++i) {
    ss << "?";
    if (i < _columns.size() - 1) {
      ss << ",";
    }
  }

  ss << ");";

  auto const str = ss.str();
  auto const ret = sqlite3_prepare_v2(db, str.c_str(), -1, s, nullptr);
  return ret;
}

int DBLayout::nCols() const { return _columns.size(); }

// bind functions
inline int bind_int(sqlite3_stmt *s, int n, int64_t val) {
  return sqlite3_bind_int64(s, n, val);
}

inline int bind_string(sqlite3_stmt *s, int n, std::string const &val) {
  return sqlite3_bind_text(s, n, val.c_str(), -1, SQLITE_TRANSIENT);
}

inline int bind_float(sqlite3_stmt *s, int n, double val) {
  return sqlite3_bind_double(s, n, val);
}

template <typename T>
inline int bind_bytes(sqlite3_stmt *s, int n, std::vector<T> const &bytes) {
  return sqlite3_bind_blob64(s, n, (void *)bytes.data(),
                             sizeof(T) * bytes.size(), SQLITE_TRANSIENT);
}

inline int bind_null(sqlite3_stmt *s, int n) { return sqlite3_bind_null(s, n); }

// serialize a protobuf message to byte vector
template <typename T> inline std::vector<uint8_t> bytes(T msg) {
  std::vector<uint8_t> o(msg->ByteSizeLong());
  msg->SerializeToArray(static_cast<void *>(o.data()), o.size());
  return o;
}

using DB = bamradio::DBLayout;

namespace dll {

///////////////////////////////////////////////////////////////////////////////
// Sent Frame
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const SentFrameEvent =
    NotificationCenter::makeName("Sent Frame");

const std::string SentFrameEventInfo::Name = "SentFrame";
const DBLayout SentFrameEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("dstNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("txTime", DB::Type::INT)
        .addColumn("payloadMCS", DB::Type::INT)
        .addColumn("payloadSymSeqID", DB::Type::INT)
        .addColumn("nsamples", DB::Type::INT)
        .addColumn("numBlocks", DB::Type::INT)
        .addColumn("sampleGain", DB::Type::REAL)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);
// n.b. make sure to bind(...) in the exact same order below

int SentFrameEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                               UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)seqNum);
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)destNodeID);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)txTime);
  ret = bind_int(*s, n++, (int64_t)payloadMCS);
  ret = bind_int(*s, n++, (int64_t)payloadSymSeqID);
  ret = bind_int(*s, n++, (int64_t)nsamples);
  ret = bind_int(*s, n++, (int64_t)numBlocks);
  ret = bind_float(*s, n++, sampleGain);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);

  assert(n - 1 == Layout.nCols());
  return ret; // FIXME is only last value
}

void to_json(nlohmann::json &j, SentFrameEventInfo const &ei) {
  j["srcNodeID"] = ei.sourceNodeID;
  j["dstNodeID"] = ei.destNodeID;
  j["txTime"] = ei.txTime;
  j["sampleGain"] = ei.sampleGain;
  j["payloadMCS"] = (int64_t)ei.payloadMCS;
  j["payloadSymSeqID"] = (int64_t)ei.payloadSymSeqID;
  j["nint"] = (int64_t)ei.nsamples;
  j["seqNum"] = ei.seqNum;
  j["frameID"] = ei.frameID;
}

///////////////////////////////////////////////////////////////////////////////
// Sent Segment
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const SentSegmentEvent =
    NotificationCenter::makeName("Sent Segment");

const std::string SentSegmentEventInfo::Name = "SentSegment";

const DBLayout SentSegmentEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("dstNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("description", DB::Type::TEXT)
        .addColumn("type", DB::Type::INT)
        .addColumn("nbytes", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int SentSegmentEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                 UnixTime ts) const {
  std::string const desc = description.dump();
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, (int64_t)seqNum);
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)destNodeID);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)sourceTime);
  ret = bind_string(*s, n++, desc);
  ret = bind_int(*s, n++, (int64_t)type);
  ret = bind_int(*s, n++, (int64_t)nbytes);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);

  assert(n - 1 == Layout.nCols());
  return ret; // FIXME is only last value
}

void to_json(nlohmann::json &j, SentSegmentEventInfo const &ei) {
  j = ei.flow;
  j["description"] = ei.description;
  j["srcNodeID"] = ei.sourceNodeID;
  j["dstNodeID"] = ei.destNodeID;
  j["sourceTime"] = ei.sourceTime;
  j["seqNum"] = ei.seqNum;
  j["frameID"] = ei.frameID;
  j["nbytes"] = ei.nbytes;
  j["type"] = ei.type;
}

///////////////////////////////////////////////////////////////////////////////
// Detected Frame
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const DetectedFrameEvent =
    NotificationCenter::makeName("Detected Frame");

const std::string DetectedFrameEventInfo::Name = "DetectedFrame";

const DBLayout DetectedFrameEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("dstNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("payloadMCS", DB::Type::INT)
        .addColumn("payloadSymSeqID", DB::Type::INT)
        .addColumn("rxTime", DB::Type::INT)
        .addColumn("channelIdx", DB::Type::INT)
        .addColumn("snr", DB::Type::REAL)
        .addColumn("numBlocks", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

void to_json(nlohmann::json &j, DetectedFrameEventInfo const &ei) {
  j["channelIdx"] = ei.channelIdx;
  j["srcNodeID"] = ei.sourceNodeID;
  j["dstNodeID"] = ei.destNodeID;
  j["payloadMCS"] = (int64_t)ei.payloadMCS;
  j["payloadSymSeqID"] = (int64_t)ei.payloadSymSeqID;
  j["rxTime"] = ei.rxTime;
  j["seqNum"] = ei.seqNum;
  j["numBlocks"] = ei.numBlocks;
  j["snr"] = ei.snr;
  j["frameID"] = ei.frameID;
}

int DetectedFrameEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)seqNum);
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)destNodeID);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)payloadMCS);
  ret = bind_int(*s, n++, (int64_t)payloadSymSeqID);
  ret = bind_int(*s, n++, (int64_t)rxTime);
  ret = bind_int(*s, n++, (int64_t)channelIdx);
  ret = bind_float(*s, n++, (double)snr);
  ret = bind_int(*s, n++, (int64_t)numBlocks);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Received Frame
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const ReceivedFrameEvent =
    NotificationCenter::makeName("Received Frame");

const std::string ReceivedFrameEventInfo::Name = "ReceivedFrame";

const DBLayout ReceivedFrameEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("dstNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("numBlocks", DB::Type::INT)
        .addColumn("numBlocksValid", DB::Type::INT)
        .addColumn("rxSuccess", DB::Type::INT)
        .addColumn("snr", DB::Type::REAL)
        .addColumn("noiseVar", DB::Type::REAL)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ReceivedFrameEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)destNodeID);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)numBlocks);
  ret = bind_int(*s, n++, (int64_t)numBlocksValid);
  ret = bind_int(*s, n++, (int64_t)rxSuccess);
  ret = bind_float(*s, n++, snr);
  ret = bind_float(*s, n++, noiseVar);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ReceivedFrameEventInfo const &ei) {
  j["srcNodeID"] = ei.sourceNodeID;
  j["dstNodeID"] = ei.destNodeID;
  j["numBlocks"] = ei.numBlocks;
  j["numBlocksValid"] = ei.numBlocksValid;
  j["rxSuccess"] = ei.rxSuccess;
  j["snr"] = ei.snr;
  j["noiseVar"] = ei.noiseVar;
}

///////////////////////////////////////////////////////////////////////////////
// Invalid FrameHeader
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const InvalidFrameHeaderEvent =
    NotificationCenter::makeName("Invalid Frame Header");

const std::string InvalidFrameHeaderEventInfo::Name = "InvalidFrameHeader";

const DBLayout InvalidFrameHeaderEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("rxTime", DB::Type::INT)
        .addColumn("channelIdx", DB::Type::INT)
        .addColumn("snr", DB::Type::REAL)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int InvalidFrameHeaderEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                        UnixTime t, UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)rxTime);
  ret = bind_int(*s, n++, (int64_t)channelIdx);
  ret = bind_float(*s, n++, snr);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, InvalidFrameHeaderEventInfo const &ei) {
  j["channelIdx"] = ei.channelIdx;
  j["rxTime"] = ei.rxTime;
  j["snr"] = ei.snr;
}

///////////////////////////////////////////////////////////////////////////////
// Received Block
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const ReceivedBlockEvent =
    NotificationCenter::makeName("Received Payload Block");

const std::string ReceivedBlockEventInfo::Name = "ReceivedBlock";

const DBLayout ReceivedBlockEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("channelIdx", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("numBits", DB::Type::INT)
        .addColumn("valid", DB::Type::INT)
        .addColumn("snr", DB::Type::REAL)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("blockNumber", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ReceivedBlockEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  int isValid = valid ? 1 : 0;
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)channelIdx);
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)numBits);
  ret = bind_int(*s, n++, isValid);
  ret = bind_float(*s, n++, (double)snr);
  ret = bind_int(*s, n++, (int64_t)seqNum);
  ret = bind_int(*s, n++, (int64_t)blockNumber);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ReceivedBlockEventInfo const &ei) {
  j["channelIdx"] = ei.channelIdx;
  j["srcNodeID"] = ei.sourceNodeID;
  j["numBits"] = ei.numBits;
  j["valid"] = ei.valid;
  j["snr"] = ei.snr;
  j["seqNum"] = ei.seqNum;
  j["blockNumber"] = ei.blockNumber;
}

///////////////////////////////////////////////////////////////////////////////
// CoDelDelay
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const CoDelDelayEvent =
    NotificationCenter::makeName("Segment Delay");

const std::string CoDelDelayEventInfo::Name = "CoDelDelay";

const DBLayout CoDelDelayEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("delay", DB::Type::INT)
        .addColumn("description", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CoDelDelayEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  std::string const desc = description.dump();
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, sourceTime.time_since_epoch().count());
  ret = bind_int(*s, n++, delay.count());
  ret = bind_string(*s, n++, desc);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);

  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CoDelDelayEventInfo const &ei) {
  j = ei.description;
  j["sourceTime"] = ei.sourceTime.time_since_epoch().count();
  j["delay"] = ei.delay.count();
}

///////////////////////////////////////////////////////////////////////////////
// ReceivedCompleteSegment
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const ReceivedCompleteSegmentEvent =
    NotificationCenter::makeName("Completed Segment");

const std::string ReceivedCompleteSegmentEventInfo::Name =
    "ReceivedCompleteSegment";

const DBLayout ReceivedCompleteSegmentEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("dstNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("rxTime", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("description", DB::Type::TEXT)
        .addColumn("queueSuccess", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ReceivedCompleteSegmentEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                             UnixTime t, UnixTime ts) const {
  std::string const desc = description.dump();
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, (int64_t)seqNum);
  ret = bind_int(*s, n++, (int64_t)sourceNodeID);
  ret = bind_int(*s, n++, (int64_t)destNodeID);
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)rxTime);
  ret = bind_int(*s, n++, (int64_t)sourceTime);
  ret = bind_string(*s, n++, desc);
  ret = bind_int(*s, n++, queueSuccess ? 1 : 0);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);

  assert(n - 1 == Layout.nCols());
  return ret; // FIXME is only last value
}

void to_json(nlohmann::json &j, ReceivedCompleteSegmentEventInfo const &ei) {
  j = ei.flow;
  j["description"] = ei.description;
  j["srcNodeID"] = ei.sourceNodeID;
  j["dstNodeID"] = ei.destNodeID;
  j["seqNum"] = ei.seqNum;
  j["rxTime"] = ei.rxTime;
  j["sourceTime"] = ei.sourceTime;
  j["frameID"] = ei.frameID;
  j["queueSuccess"] = ei.queueSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// FlowQueuePushEventInfo
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const FlowQueuePushEvent =
    NotificationCenter::makeName("Flow Queue Push");

const std::string FlowQueuePushEventInfo::Name = "FlowQueuePush";

const DBLayout FlowQueuePushEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("numQueued", DB::Type::INT)
        .addColumn("bytesQueued", DB::Type::INT)
        .addColumn("currentRound", DB::Type::INT)
        .addColumn("balance", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int FlowQueuePushEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, numQueued);
  ret = bind_int(*s, n++, bytesQueued);
  ret = bind_int(*s, n++, currentRound);
  ret = bind_int(*s, n++, balance.count());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, FlowQueuePushEventInfo const &ei) {
  j = ei.flow;
  j["numQueued"] = ei.numQueued;
  j["bytesQueued"] = ei.bytesQueued;
  j["currentRound"] = ei.currentRound;
  j["balance"] = ei.balance.count();
}

///////////////////////////////////////////////////////////////////////////////
// FlowQueuePopEventInfo
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const FlowQueuePopEvent =
    NotificationCenter::makeName("Flow Queue Pop");

const std::string FlowQueuePopEventInfo::Name = "FlowQueuePop";

const DBLayout FlowQueuePopEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("numQueued", DB::Type::INT)
        .addColumn("bytesQueued", DB::Type::INT)
        .addColumn("currentRound", DB::Type::INT)
        .addColumn("balance", DB::Type::INT)
        .addColumn("quantumCredit", DB::Type::INT)
        .addColumn("dequeueDebit", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int FlowQueuePopEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                  UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, numQueued);
  ret = bind_int(*s, n++, bytesQueued);
  ret = bind_int(*s, n++, currentRound);
  ret = bind_int(*s, n++, balance.count());
  ret = bind_int(*s, n++, quantumCredit.count());
  ret = bind_int(*s, n++, dequeueDebit.count());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, FlowQueuePopEventInfo const &ei) {
  j = ei.flow;
  j["numQueued"] = ei.numQueued;
  j["bytesQueued"] = ei.bytesQueued;
  j["currentRound"] = ei.currentRound;
  j["balance"] = ei.balance.count();
  j["quantumCredit"] = ei.quantumCredit.count();
  j["dequeueDebit"] = ei.dequeueDebit.count();
}

///////////////////////////////////////////////////////////////////////////////
// ScheduleUpdateEventInfo
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const ScheduleUpdateEvent =
    NotificationCenter::makeName("Schedule Update");

const std::string ScheduleUpdateEventInfo::Name = "ScheduleUpdate";
const std::string ScheduleUpdateEventInfo::FlowQuantum::Name = "FlowQuantum";

const DBLayout ScheduleUpdateEventInfo::Layout =
    bamradio::DBLayout(ScheduleUpdateEventInfo::Name)
        .addColumn("round", DB::Type::INT)
        .addColumn("valid", DB::Type::INT)
        .addColumn("period", DB::Type::REAL)
        .addColumn("periodlb", DB::Type::REAL)
        .addColumn("periodub", DB::Type::REAL)
        .addColumn("flowInfos", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);
const DBLayout ScheduleUpdateEventInfo::FlowQuantum::Layout =
    bamradio::DBLayout(ScheduleUpdateEventInfo::FlowQuantum::Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("quantum", DB::Type::INT)
        .addColumn("roundepochid", DB::Type::INT,
                   "REFERENCES ScheduleUpdate(id)");

int ScheduleUpdateEventInfo::to_sql(
    sqlite3 *db, std::map<std::string, sqlite3_stmt *> &stmt,
    std::chrono::system_clock::time_point t,
    std::chrono::steady_clock::time_point ts) const {
  int rc;
  // first call: prepare the statement
  sqlite3_stmt *&s = stmt[Name];
  if (s == nullptr) {
    rc = Layout.prepare(db, &s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  rc = bind_int(s, n++, round);
  rc = bind_int(s, n++, valid);
  rc = bind_float(s, n++, period.count());
  rc = bind_float(s, n++, periodlb.count());
  rc = bind_float(s, n++, periodub.count());
  rc = bind_string(s, n++, flowInfos);
  rc = bind_int(s, n++, t.time_since_epoch().count());
  rc = bind_int(s, n++, ts.time_since_epoch().count());
  assert(n - 1 == Layout.nCols());
  rc = sqlite3_step(s); // step()
  if (rc != SQLITE_DONE) {
    return rc;
  }
  rc = sqlite3_reset(s); // reset()
  if (rc != SQLITE_OK) {
    return rc;
  }
  auto const roundepochid = sqlite3_last_insert_rowid(db);
  for (auto i = quantums.begin(); i != quantums.end(); ++i) {
    FlowQuantum{roundepochid, i}.to_sql(db, stmt, t, ts);
  }
  return rc;
}

int ScheduleUpdateEventInfo::FlowQuantum::to_sql(
    sqlite3 *db, std::map<std::string, sqlite3_stmt *> &stmt,
    std::chrono::system_clock::time_point,
    std::chrono::steady_clock::time_point) const {
  int rc;
  // first call: prepare the statement
  sqlite3_stmt *&s = stmt[Name];
  if (s == nullptr) {
    rc = Layout.prepare(db, &s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  rc = bind_string(s, n++, v->first.srcIPString());
  rc = bind_string(s, n++, v->first.dstIPString());
  rc = bind_int(s, n++, v->first.proto);
  rc = bind_int(s, n++, v->first.srcPort);
  rc = bind_int(s, n++, v->first.dstPort);
  rc = bind_int(s, n++, v->second.count());
  rc = bind_int(s, n++, roundepoch);
  assert(n - 1 == Layout.nCols());
  rc = sqlite3_step(s); // step()
  if (rc != SQLITE_DONE) {
    return rc;
  }
  rc = sqlite3_reset(s); // reset()
  if (rc != SQLITE_OK) {
    return rc;
  }
  return rc;
}

void to_json(nlohmann::json &j, ScheduleUpdateEventInfo const &ei) {
  std::map<FlowID, float> q;
  for (auto const &a : ei.quantums) {
    q.emplace(a.first,
              std::chrono::duration_cast<std::chrono::duration<float>>(a.second)
                  .count());
  }
  j["quantums"] = nlohmann::json(q);
  j["round"] = ei.round;
  j["valid"] = ei.valid;
  j["period"] = ei.period.count();
  j["periodlb"] = ei.periodlb.count();
  j["periodub"] = ei.periodub.count();
  j["flowInfos"] = ei.flowInfos;
}

///////////////////////////////////////////////////////////////////////////////
// New Flow
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const NewFlowEvent =
    NotificationCenter::makeName("New Flow");

const std::string NewFlowEventInfo::Name = "NewFlow";

const DBLayout NewFlowEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("currentRound", DB::Type::INT)
        .addColumn("queueType", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int NewFlowEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                             UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, currentRound);
  ret = bind_string(*s, n++, queueType);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, NewFlowEventInfo const &ei) {
  j = ei.flow;
  j["currentRound"] = ei.currentRound;
  j["queueType"] = ei.queueType;
}

///////////////////////////////////////////////////////////////////////////////
// New Active Flow
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const NewActiveFlowEvent =
    NotificationCenter::makeName("New Active Flow");

///////////////////////////////////////////////////////////////////////////////
// CoDelState
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const CoDelStateEvent =
    NotificationCenter::makeName("CoDel State");

const std::string CoDelStateEventInfo::Name = "CoDelState";

const DBLayout CoDelStateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("first_above_time", DB::Type::INT)
        .addColumn("drop_next", DB::Type::INT)
        .addColumn("drop_count", DB::Type::INT)
        .addColumn("last_drop_count", DB::Type::INT)
        .addColumn("dropping", DB::Type::INT)
        .addColumn("bbytes", DB::Type::INT)
        .addColumn("queue_size", DB::Type::INT)
        .addColumn("avg_latency", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CoDelStateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, first_above_time.time_since_epoch().count());
  ret = bind_int(*s, n++, drop_next.time_since_epoch().count());
  ret = bind_int(*s, n++, (int64_t)drop_count);
  ret = bind_int(*s, n++, (int64_t)last_drop_count);
  ret = bind_int(*s, n++, dropping ? 1 : 0);
  ret = bind_int(*s, n++, (int64_t)bytes);
  ret = bind_int(*s, n++, (int64_t)queue_size);
  ret = bind_int(*s, n++, avg_latency.count());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CoDelStateEventInfo const &ei) {
  j = ei.flow;
  j["first_above_time"] = ei.first_above_time.time_since_epoch().count();
  j["drop_next"] = ei.drop_next.time_since_epoch().count();
  j["drop_count"] = ei.drop_count;
  j["last_drop_count"] = ei.last_drop_count;
  j["dropping"] = ei.dropping;
  j["bytes"] = ei.bytes;
  j["queue_size"] = ei.queue_size;
  j["avg_latency"] = ei.avg_latency.count();
}

///////////////////////////////////////////////////////////////////////////////
// ARQ
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const ReceivedARQFeedbackEvent =
    NotificationCenter::makeName("ReceivedARQFeedbackEvent");

const std::string ReceivedARQFeedbackEventInfo::Name =
    "ReceivedARQFeedbackEvent";

const DBLayout ReceivedARQFeedbackEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("flowUID", DB::Type::INT)
        .addColumn("burstNum", DB::Type::INT)
        .addColumn("lastSeq", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ReceivedARQFeedbackEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                         UnixTime t, UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)frameID);
  ret = bind_int(*s, n++, (int64_t)flowUID);
  ret = bind_int(*s, n++, (int64_t)burstNum);
  ret = bind_int(*s, n++, (int64_t)lastSeq);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ReceivedARQFeedbackEventInfo const &ei) {
  j["frameID"] = ei.frameID;
  j["flowUID"] = ei.flowUID;
  j["burstNum"] = ei.burstNum;
  j["lastSeq"] = ei.lastSeq;
}

NotificationCenter::Name const FlowQueueResendEvent =
    NotificationCenter::makeName("FlowQueueResendEvent");

const std::string FlowQueueResendEventInfo::Name = "FlowQueueResendEvent";

const DBLayout FlowQueueResendEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("burstNum", DB::Type::INT)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("disposition", DB::Type::INT)
        .addColumn("numQueued", DB::Type::INT)
        .addColumn("bytesQueued", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int FlowQueueResendEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                     UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, flow.srcIPString());
  ret = bind_string(*s, n++, flow.dstIPString());
  ret = bind_int(*s, n++, flow.proto);
  ret = bind_int(*s, n++, flow.srcPort);
  ret = bind_int(*s, n++, flow.dstPort);
  ret = bind_int(*s, n++, sourceTime.time_since_epoch().count());
  ret = bind_int(*s, n++, burstNum);
  ret = bind_int(*s, n++, seqNum);
  ret = bind_int(*s, n++, disposition);
  ret = bind_int(*s, n++, numQueued);
  ret = bind_int(*s, n++, bytesQueued);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, FlowQueueResendEventInfo const &ei) {
  j = ei.flow;
  j["sourceTime"] = ei.sourceTime.time_since_epoch().count();
  j["burstNum"] = ei.burstNum;
  j["seqNum"] = ei.seqNum;
  j["disposition"] = ei.disposition;
  j["numQueued"] = ei.numQueued;
  j["bytesQueued"] = ei.bytesQueued;
}

NotificationCenter::Name const FlowTrackerStateUpdateEvent =
    NotificationCenter::makeName("FlowTrackerStateUpdateEvent");

const std::string FlowTrackerStateUpdateEventInfo::Name =
    "FlowTrackerStateUpdateEvent";

const DBLayout FlowTrackerStateUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("flowUID", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("burstNum", DB::Type::INT)
        .addColumn("lastSeq", DB::Type::INT)
        .addColumn("burstSize", DB::Type::INT)
        .addColumn("burstRemaining", DB::Type::INT)
        .addColumn("completed", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int FlowTrackerStateUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                            UnixTime t, UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, flow_uid);
  ret = bind_int(*s, n++, sourceTime.time_since_epoch().count());
  ret = bind_int(*s, n++, burstNum);
  ret = bind_int(*s, n++, lastSeq);
  ret = bind_int(*s, n++, burstSize);
  ret = bind_int(*s, n++, burstRemaining);
  ret = bind_int(*s, n++, completed);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, FlowTrackerStateUpdateEventInfo const &ei) {
  j["flowUID"] = ei.flow_uid;
  j["sourceTime"] = ei.sourceTime.time_since_epoch().count();
  j["burstNum"] = ei.burstNum;
  j["lastSeq"] = ei.lastSeq;
  j["burstSize"] = ei.burstSize;
  j["burstRemaining"] = ei.burstRemaining;
  j["completed"] = ei.completed;
}

NotificationCenter::Name const FlowTrackerIMEvent =
    NotificationCenter::makeName("FlowTrackerIMEvent");

const std::string FlowTrackerIMEventInfo::Name = "FlowTrackerIMEvent";

const DBLayout FlowTrackerIMEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("flowUID", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("burstNum", DB::Type::INT)
        .addColumn("completed", DB::Type::INT)
        .addColumn("expired", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int FlowTrackerIMEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, flow_uid);
  ret = bind_int(*s, n++, sourceTime.time_since_epoch().count());
  ret = bind_int(*s, n++, burstNum);
  ret = bind_int(*s, n++, completed);
  ret = bind_int(*s, n++, expired);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, FlowTrackerIMEventInfo const &ei) {
  j["flowUID"] = ei.flow_uid;
  j["sourceTime"] = ei.sourceTime.time_since_epoch().count();
  j["burstNum"] = ei.burstNum;
  j["completed"] = ei.completed;
  j["expired"] = ei.expired;
}
} // namespace dll

namespace gps {

///////////////////////////////////////////////////////////////////////////////
// GPS
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const GPSEvent =
    NotificationCenter::makeName("GPSEvent");

const std::string GPSEventInfo::Name = "GPSEvent";

const DBLayout GPSEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("type", DB::Type::TEXT)
        .addColumn("latitude", DB::Type::REAL)
        .addColumn("longitude", DB::Type::REAL)
        .addColumn("altidute", DB::Type::REAL)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int GPSEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                         UnixTime ts) const {
  auto msg = [](auto const &t) {
    switch (t) {
    case GPSEventType::TRY_CONNECT_GOOD:
      return "try_connect_good";
    case GPSEventType::TRY_CONNECT_BAD:
      return "try_connect_bad";
    case GPSEventType::READ_ERROR:
      return "read_error";
    case GPSEventType::READ_NO_FIX:
      return "read_no_fix";
    case GPSEventType::READ_NO_DATA:
      return "read_no_data";
    case GPSEventType::READ_GOOD:
      return "read_good";
    case GPSEventType::READ_TIMEOUT:
      return "read_timeout";
    }
  }(type);
  int ret;

  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, msg);
  ret = bind_float(*s, n++, lat);
  ret = bind_float(*s, n++, lon);
  ret = bind_float(*s, n++, alt);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, GPSEventInfo const &ei) {
  switch (ei.type) {
  case GPSEventType::TRY_CONNECT_GOOD:
    j["type"] = "connect";
    j["success"] = true;
    break;
  case GPSEventType::TRY_CONNECT_BAD:
    j["type"] = "connect";
    j["success"] = false;
    break;
  case GPSEventType::READ_ERROR:
    j["type"] = "read";
    j["success"] = false;
    j["reason"] = "error";
    break;
  case GPSEventType::READ_NO_FIX:
    j["type"] = "read";
    j["success"] = false;
    j["reason"] = "no_fix";
    break;
  case GPSEventType::READ_NO_DATA:
    j["type"] = "read";
    j["success"] = false;
    j["reason"] = "no_data";
    break;
  case GPSEventType::READ_GOOD:
    j["type"] = "read";
    j["success"] = true;
    j["latitude"] = ei.lat;
    j["longitude"] = ei.lon;
    j["altitude"] = ei.alt;
    break;
  case GPSEventType::READ_TIMEOUT:
    j["type"] = "read";
    j["success"] = false;
    j["reason"] = "timeout";
    break;
  }
}
} // namespace gps

namespace net {

///////////////////////////////////////////////////////////////////////////////
// Route Decision
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const RouteDecisionEvent =
    NotificationCenter::makeName("Route Decision");

const std::string RouteDecisionEventInfo::Name = "RouteDecision";

const DBLayout RouteDecisionEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("action", DB::Type::TEXT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("nextHop", DB::Type::INT)
        .addColumn("type", DB::Type::INT)
        .addColumn("srcIP", DB::Type::TEXT)
        .addColumn("dstIP", DB::Type::TEXT)
        .addColumn("srcPort", DB::Type::INT)
        .addColumn("dstPort", DB::Type::INT)
        .addColumn("protocol", DB::Type::INT)
        .addColumn("packetLength", DB::Type::INT)
        .addColumn("sourceTime", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int RouteDecisionEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                   UnixTime ts) const {
  auto action_str = [](auto const &t) {
    switch (t) {
    case RouterAction::FORWARD:
      return "Forward";
    case RouterAction::WRITE_TO_TUN:
      return "WriteToTun";
    case RouterAction::DROP_UNKNOWN_PACKET:
      return "DropUnknownPacket";
    }
  }(action);

  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, action_str);
  ret = bind_int(*s, n++, (int64_t)src_srnid);
  ret = bind_int(*s, n++, (int64_t)next_hop);
  ret = bind_int(*s, n++, (int64_t)type);
  ret = bind_string(*s, n++, src_ip.to_string());
  ret = bind_string(*s, n++, dst_ip.to_string());
  ret = bind_int(*s, n++, (int64_t)src_port);
  ret = bind_int(*s, n++, (int64_t)dst_port);
  ret = bind_int(*s, n++, (int64_t)protocol);
  ret = bind_int(*s, n++, (int64_t)packetLength);
  ret = bind_int(*s, n++, (int64_t)sourceTime.time_since_epoch().count());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, RouteDecisionEventInfo const &ei) {
  j["action"] = ei.action;
  j["src"] = ei.src_srnid;
  j["next_hop"] = ei.next_hop;
  j["type"] = (int)ei.type;
  j["srcIP"] = ei.src_ip.to_string();
  j["dstIP"] = ei.dst_ip.to_string();
  j["srcPort"] = ei.src_port;
  j["dstPort"] = ei.dst_port;
  j["protocol"] = ei.protocol;
  j["packetLength"] = ei.packetLength;
}

///////////////////////////////////////////////////////////////////////////////
// Routing table update
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const RoutingTableUpdateEvent =
    NotificationCenter::makeName("Routing Table Update");

const std::string RoutingTableUpdateEventInfo::Name = "RoutingTableUpdate";

const DBLayout RoutingTableUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("routingTable", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int RoutingTableUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                        UnixTime t, UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  nlohmann::json j = *this;
  int n = 1;
  ret = bind_string(*s, n++, j.dump());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, RoutingTableUpdateEventInfo const &ei) {
  for (auto const &ent : ei.table)
    j[std::to_string((int)(ent.first))] = (int)(ent.second);
}

} // namespace net

namespace log {

///////////////////////////////////////////////////////////////////////////////
// Doomsday and Generic
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const DoomsdayEvent =
    NotificationCenter::makeName("666");

const std::string DoomsdayEventInfo::Name = "Doomsday";

const DBLayout DoomsdayEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::TEXT)
        .addColumn("file", DB::Type::TEXT)
        .addColumn("line", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int DoomsdayEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, msg);
  ret = bind_string(*s, n++, file);
  ret = bind_int(*s, n++, line);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, DoomsdayEventInfo const &ei) {
  j["msg"] = ei.msg;
  j["file"] = ei.file;
  j["line"] = ei.line;
}

void doomsday(std::string const &msg, std::string const &file, int line,
              char const *func) {
  if (!func) {
    func = "<unspecified>";
  }
  std::mutex m;
  std::condition_variable cv;
  bool waitabort = true;

  NotificationCenter::shared.post(
      DoomsdayEvent,
      DoomsdayEventInfo{msg, file, func, line, [&m, &cv, &waitabort] {
                          std::unique_lock<std::mutex> l(m);
                          waitabort = false;
                          l.unlock();
                          cv.notify_all();
                        }});

  // Wait (don't actually waitabort) for event handler to signal us before
  // aborting.
  std::unique_lock<std::mutex> l(m);
  if (!cv.wait_for(l, std::chrono::seconds(10), [&] { return !waitabort; })) {
    std::cout << "doomsday wait timed out" << std::endl;
  }
  // Abort from calling thread for nicer backtrace.
  abort();
}

NotificationCenter::Name const TextLogEvent =
    NotificationCenter::makeName("Log this text");

const std::string TextLogEventInfo::Name = "Text";

const DBLayout TextLogEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::TEXT)
        .addColumn("file", DB::Type::TEXT)
        .addColumn("line", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int TextLogEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                             UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, msg);
  ret = bind_string(*s, n++, file);
  ret = bind_int(*s, n++, line);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, TextLogEventInfo const &ei) {
  j["msg"] = ei.msg;
  j["file"] = ei.file;
  j["line"] = ei.line;
}

void text(std::string const &msg, std::string const &file, int line) {
  NotificationCenter::shared.post(TextLogEvent,
                                  TextLogEventInfo{msg, file, line});
}

void text(boost::format const &msg, std::string const &file, int line) {
  text(msg.str(), file, line);
}

///////////////////////////////////////////////////////////////////////////////
// Database events
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const DBEvent =
    NotificationCenter::makeName("Database Event");

void to_json(nlohmann::json &j, DBEventInfo const &ei) {
  switch (ei.type) {
  case DBEventInfo::Type::SUCCESS:
    j["type"] = "success";
    break;
  case DBEventInfo::Type::FAIL:
    j["type"] = "fail";
    break;
  case DBEventInfo::Type::CALLBACK:
    j["type"] = "callback";
    break;
  }
  j["msg"] = ei.msg;
  j["sql"] = ei.sql;
  j["sql_id"] = ei.sql_id;
}

///////////////////////////////////////////////////////////////////////////////
// ScenarioStartEvent
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const ScenarioStartEvent =
    NotificationCenter::makeName("Scenario Start Event");

} // namespace log

///////////////////////////////////////////////////////////////////////////////
// PHY Events
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const BurstSendEvent =
    NotificationCenter::makeName("BurstSend");

const std::string BurstSendEventInfo::Name = "BurstSendEvent";

const DBLayout BurstSendEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("burstTime", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int BurstSendEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                               UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, time);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, BurstSendEventInfo const &ei) {
  j["time"] = ei.time;
}

namespace uhdfeedback {

// UHD Async Metadats
const char *uhdAsync2str(::uhd::async_metadata_t::event_code_t event_code) {
  using md = ::uhd::async_metadata_t;
  switch (event_code) {
  case md::EVENT_CODE_BURST_ACK:
    return "burst ack";
  case md::EVENT_CODE_UNDERFLOW:
    return "underflow";
  case md::EVENT_CODE_SEQ_ERROR:
    return "seq error";
  case md::EVENT_CODE_TIME_ERROR:
    return "time error";
  case md::EVENT_CODE_UNDERFLOW_IN_PACKET:
    return "underflow in packet";
  case md::EVENT_CODE_SEQ_ERROR_IN_BURST:
    return "seq error in burst";
  case md::EVENT_CODE_USER_PAYLOAD:
    return "user payload";
  default:
    return "";
  }
}

NotificationCenter::Name const UHDAsyncEvent =
    NotificationCenter::makeName("UHD Async");

const std::string UHDAsyncEventInfo::Name = "UHDAsyncEvent";

const DBLayout UHDAsyncEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("channel", DB::Type::INT)
        .addColumn("timespec", DB::Type::INT)
        .addColumn("type", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int UHDAsyncEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)channel);
  ret = bind_int(*s, n++, time);
  ret = bind_string(*s, n++, uhdAsync2str(event_code));
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, UHDAsyncEventInfo const &ei) {
  j["channel"] = ei.channel;
  j["timespec"] = ei.time;
  j["type"] = uhdAsync2str(ei.event_code);
}

// UHD messages
const char *uhdMsg2str(::uhd::msg::type_t type) {
  switch (type) {
  case uhd::msg::type_t::status:
    return "status";
  case uhd::msg::type_t::warning:
    return "warning";
  case uhd::msg::type_t::error:
    return "error";
  case uhd::msg::type_t::fastpath:
    return "fast";
  default:
    return "";
  }
}

NotificationCenter::Name const UHDMsgEvent =
    NotificationCenter::makeName("UHD Msg");

const std::string UHDMsgEventInfo::Name = "UHDMsgEvent";

const DBLayout UHDMsgEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("type", DB::Type::TEXT)
        .addColumn("msg", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int UHDMsgEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                            UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, uhdMsg2str(type));
  ret = bind_string(*s, n++, msg);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, UHDMsgEventInfo const &ei) {
  j["type"] = uhdMsg2str(ei.type);
}

} // namespace uhdfeedback

namespace psdsensing {
NotificationCenter::Name const PSDUpdateEvent =
    NotificationCenter::makeName("PSD Update");

const std::string PSDUpdateEventInfo::Name = "PSDUpdateEvent";

const DBLayout PSDUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("psd", DB::Type::BLOB)
        .addColumn("time_ns", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int PSDUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                               UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_bytes(*s, n++, psd);
  ret = bind_int(*s, n++, time_ns);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, PSDUpdateEventInfo const &ei) {
  // we don't need PSD in JSON, just write timestamp (this function should never
  // be called)
  j["time_ns"] = ei.time_ns;
}

NotificationCenter::Name const PSDRxEvent =
    NotificationCenter::makeName("Received PSD Event");

const std::string PSDRxEventInfo::Name = "PSDRxEvent";

const DBLayout PSDRxEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("psd", DB::Type::BLOB)
        .addColumn("time_ns", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int PSDRxEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                           UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, node_id);
  ret = bind_bytes(*s, n++, psd);
  ret = bind_int(*s, n++, time_ns);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, PSDRxEventInfo const &ei) {
  // we don't need PSD in JSON, just write timestamp (this function should never
  // be called)
  j["time_ns"] = ei.time_ns;
}
} // namespace psdsensing

///////////////////////////////////////////////////////////////////////////////
// Control Channel Events
///////////////////////////////////////////////////////////////////////////////
namespace controlchannel {

// CC Packet Event
NotificationCenter::Name const CCPacketEvent =
    NotificationCenter::makeName("CCPacketEvent");

const std::string CCPacketEventInfo::Name = "CCPacketEvent";

const DBLayout CCPacketEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("srcNodeID", DB::Type::TEXT)
        .addColumn("eventType", DB::Type::TEXT)
        .addColumn("phyType", DB::Type::TEXT)
        .addColumn("seqNum", DB::Type::INT)
        .addColumn("hash", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CCPacketEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  auto event_type_str = [](auto const &t) {
    switch (t) {
    case CCPacketEventType::CCEVENT_RX:
      return "RX";
    case CCPacketEventType::CCEVENT_TX:
      return "TX";
    }
  }(event_type);
  auto phy_type_str = [](auto const &t) {
    switch (t) {
    case CCPacketPHYType::CCPHY_FSK:
      return "FSK";
    case CCPacketPHYType::CCPHY_OFDM:
      return "OFDM";
    }
  }(phy_type);

  int ret;

  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, src_srnid);
  ret = bind_string(*s, n++, event_type_str);
  ret = bind_string(*s, n++, phy_type_str);
  ret = bind_int(*s, n++, seq_num);
  ret = bind_int(*s, n++, hash);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CCPacketEventInfo const &ei) {
  // TODO: Implement
  j["ccevent"] = true;
}

const NotificationCenter::Name NetworkMapEvent =
    NotificationCenter::makeName("Network Map update");

const std::string NetworkMapEventInfo::Name = "NetworkMap";
const DBLayout NetworkMapEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("data", DB::Type::BLOB)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int NetworkMapEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  // need to serialize the networkmap to a vector of bytes
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_bytes(*s, n++, netmap.serializeToVector());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
};

void to_json(nlohmann::json &, NetworkMapEventInfo const &) {}

} // namespace controlchannel

namespace collab {
///////////////////////////////////////////////////////////////////////////////
// CIL Client Events
///////////////////////////////////////////////////////////////////////////////

// CIL RX
NotificationCenter::Name const CollabRxEvent =
    NotificationCenter::makeName("Collab Rx");

const std::string CollabRxEventInfo::Name = "CollabCILRx";

const DBLayout CollabRxEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::BLOB)
        .addColumn("msg_id", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CollabRxEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (msg) {
    ret = bind_bytes(*s, n++, bytes(msg));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, msg_id);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CollabRxEventInfo const &ei) {
  if (ei.msg)
    j["msg"] = ei.msg->ShortDebugString();
  j["msg_id"] = ei.msg_id;
}

// CIL TX
NotificationCenter::Name const CollabTxEvent =
    NotificationCenter::makeName("Collab Tx");

const std::string CollabTxEventInfo::Name = "CollabCILTx";

const DBLayout CollabTxEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::BLOB)
        .addColumn("msg_id", DB::Type::INT)
        .addColumn("broadcast", DB::Type::INT)
        .addColumn("dstIP", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CollabTxEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (msg) {
    ret = bind_bytes(*s, n++, bytes(msg));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, msg_id);
  ret = bind_int(*s, n++, dst.size() > 1 ? 1 : 0);
  if (dst.size() > 0) {
    ret = bind_int(*s, n++, (int64_t)(dst[0].to_uint()));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CollabTxEventInfo const &ei) {
  if (ei.msg)
    j["msg"] = ei.msg->ShortDebugString();
  j["msg_id"] = ei.msg_id;
  j["broadcast"] = ei.dst.size() > 1 ? true : false;
  if (ei.dst.size() > 0)
    j["dst"] = ei.dst[0].to_uint();
}

// Reg Rx
NotificationCenter::Name const ServerRxEvent =
    NotificationCenter::makeName("Reg Rx");

const std::string ServerRxEventInfo::Name = "CollabServerRx";

const DBLayout ServerRxEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::BLOB)
        .addColumn("msg_id", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ServerRxEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (msg) {
    ret = bind_bytes(*s, n++, bytes(msg));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, msg_id);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ServerRxEventInfo const &ei) {
  if (ei.msg)
    j["msg"] = ei.msg->ShortDebugString();
  j["msg_id"] = ei.msg_id;
}

// Reg TX
NotificationCenter::Name const ServerTxEvent =
    NotificationCenter::makeName("Reg Tx");

const std::string ServerTxEventInfo::Name = "CollabServerTx";

const DBLayout ServerTxEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("msg", DB::Type::BLOB)
        .addColumn("msg_id", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ServerTxEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                              UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (msg) {
    ret = bind_bytes(*s, n++, bytes(msg));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, msg_id);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ServerTxEventInfo const &ei) {
  if (ei.msg)
    j["msg"] = ei.msg->ShortDebugString();
  j["msg_id"] = ei.msg_id;
}

// Connection attempt
NotificationCenter::Name const ConnectionEvent =
    NotificationCenter::makeName("Collab Conn");

const std::string ConnectionEventInfo::Name = "CollabConnectionAttempt";

const DBLayout ConnectionEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("success", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ConnectionEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_int(*s, n++, success ? 1 : 0);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ConnectionEventInfo const &ei) {
  j["success"] = ei.success;
}

// StateChange
NotificationCenter::Name const StateChangeEvent =
    NotificationCenter::makeName("Collab State Change");

const std::string StateChangeEventInfo::Name = "CollabStateChange";

const DBLayout StateChangeEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("fromState", DB::Type::TEXT)
        .addColumn("toState", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int StateChangeEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                 UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_string(*s, n++, from);
  ret = bind_string(*s, n++, to);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, StateChangeEventInfo const &ei) {
  j["from"] = ei.from;
  j["to"] = ei.to;
}

// Peers
NotificationCenter::Name const CollabPeerEvent =
    NotificationCenter::makeName("Collab Peer");

const std::string CollabPeerEventInfo::Name = "CollabPeerEvent";

const DBLayout CollabPeerEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("isAdd", DB::Type::INT)
        .addColumn("ip", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int CollabPeerEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_int(*s, n++, add ? 1 : 0);
  ret = bind_int(*s, n++, ip.to_uint());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, CollabPeerEventInfo const &ei) {
  j["add"] = ei.add;
  j["ip"] = ei.ip.to_string();
}

// Error
NotificationCenter::Name const ErrorEvent =
    NotificationCenter::makeName("CollabError");

inline std::string collabErr2Str(ErrorType e) {
  switch (e) {
  case ErrorType::UNSUPPORTED:
    return "unsupported message";
  case ErrorType::NOT_CONNECTED:
    return "cil client not connected";
  case ErrorType::SEND_FAIL:
    return "cil message send fail";
  case ErrorType::NO_PEER:
    return "peer not available";
  case ErrorType::SERVER_SYNC:
    return "lost sync w/ cil server";
  }
}

const std::string ErrorEventInfo::Name = "CollabError";

const DBLayout ErrorEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("type", DB::Type::TEXT)
        .addColumn("msg", DB::Type::BLOB)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ErrorEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                           UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_string(*s, n++, collabErr2Str(type));
  if (msg) {
    ret = bind_bytes(*s, n++, bytes(msg));
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ErrorEventInfo const &ei) {
  j["type"] = collabErr2Str(ei.type);
  if (ei.msg) {
    j["msg"] = ei.msg->ShortDebugString();
  }
}
} // namespace collab

///////////////////////////////////////////////////////////////////////////////
// Achieved IMs
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const AchievedIMsUpdateEvent =
    NotificationCenter::makeName("Achieved IMs Update");

const std::string AchievedIMsUpdateEventInfo::Name = "AchievedIMsUpdate";

const DBLayout AchievedIMsUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("nAchievedIMs", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int AchievedIMsUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                       UnixTime t, UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_int(*s, n++, n_achieved_IMs);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, AchievedIMsUpdateEventInfo const &ei) {
  j["nAchievedIMs"] = ei.n_achieved_IMs;
}

///////////////////////////////////////////////////////////////////////////////
// Mandated Outcomes
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const OutcomesUpdateEvent =
    NotificationCenter::makeName("Mandated Outcomes Update");

const std::string OutcomesUpdateEventInfo::Name = "MandatedOutcomeUpdate";

const DBLayout OutcomesUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("json", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int OutcomesUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                    UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_string(*s, n++, j.dump());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, OutcomesUpdateEventInfo const &ei) { j = ei.j; }

///////////////////////////////////////////////////////////////////////////////
// Incumbent Protection
///////////////////////////////////////////////////////////////////////////////
NotificationCenter::Name const IncumbentAttenuationUpdateEvent =
    NotificationCenter::makeName("Incumbent Protection");

const std::string IncumbentAttenuationUpdateEventInfo::Name =
    "IncumbentAttenuationUpdateEvent";

const DBLayout IncumbentAttenuationUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("incumbent_id", DB::Type::INT)
        .addColumn("attenuation", DB::Type::REAL)
        .addColumn("OFDMParamsUpdateID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int IncumbentAttenuationUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                                UnixTime t, UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  ret = bind_int(*s, n++, incumbent_id);
  ret = bind_float(*s, n++, attenuation_dB);
  ret = bind_int(*s, n++, OFDMParamsUpdateID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, IncumbentAttenuationUpdateEventInfo const &ei) {
  j["incumbent_id"] = ei.incumbent_id;
  j["attenuation"] = ei.attenuation_dB;
  j["OFDMParamsUpdateID"] = ei.OFDMParamsUpdateID;
}

///////////////////////////////////////////////////////////////////////////////
// C2API
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const C2APIEvent =
    NotificationCenter::makeName("C2APIEvent");

const std::string C2APIEventInfo::Name = "C2APIEvent";

const DBLayout C2APIEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("type", DB::Type::TEXT)
        .addColumn("txt", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int C2APIEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                           UnixTime ts) const {
  auto msg = [](auto const &t) {
    switch (t) {
    case C2APIEventType::RECEIVED_COMMAND:
      return "received_command";
    case C2APIEventType::UPDATED_STATUS:
      return "updated_status";
    }
  }(type);
  int ret;

  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_string(*s, n++, msg);
  ret = bind_string(*s, n++, txt);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, C2APIEventInfo const &ei) {
  switch (ei.type) {
  case C2APIEventType::RECEIVED_COMMAND:
    j["type"] = "received_command";
    break;
  case C2APIEventType::UPDATED_STATUS:
    j["type"] = "updated_status";
    break;
  }
  j["txt"] = ei.txt;
}

///////////////////////////////////////////////////////////////////////////////
// Environment Updates
///////////////////////////////////////////////////////////////////////////////

NotificationCenter::Name const EnvironmentUpdateEvent =
    NotificationCenter::makeName("EnvironmentUpdateEvent");

const std::string EnvironmentUpdateEventInfo::Name = "EnvironmentUpdateEvent";

const DBLayout EnvironmentUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("collab_network_type", DB::Type::INT)
        .addColumn("incumbent_protection_center_frequency", DB::Type::INT)
        .addColumn("incumbent_protection_rf_bandwidth", DB::Type::INT)
        .addColumn("scenario_rf_bandwidth", DB::Type::INT)
        .addColumn("scenario_center_frequency", DB::Type::INT)
        .addColumn("has_incumbent", DB::Type::INT)
        .addColumn("stage_number", DB::Type::INT)
        .addColumn("timestamp", DB::Type::INT)
        .addColumn("raw", DB::Type::TEXT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int EnvironmentUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                       UnixTime t, UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(
      *s, n++,
      static_cast<std::underlying_type_t<decltype(collab_network_type)>>(
          collab_network_type));
  ret = bind_int(*s, n++, incumbent_protection.center_frequency);
  ret = bind_int(*s, n++, incumbent_protection.rf_bandwidth);
  ret = bind_int(*s, n++, scenario_rf_bandwidth);
  ret = bind_int(*s, n++, scenario_center_frequency);
  ret = bind_int(*s, n++, has_incumbent ? 1 : 0);
  ret = bind_int(*s, n++, stage_number);
  ret = bind_int(*s, n++, timestamp);
  ret = bind_string(*s, n++, raw.dump());
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, EnvironmentUpdateEventInfo const &ei) {
  j["collab_network_type"] =
      static_cast<std::underlying_type_t<decltype(ei.collab_network_type)>>(
          ei.collab_network_type);
  j["incumbent_protection_center_frequency"] =
      ei.incumbent_protection.center_frequency;
  j["incumbent_protection_rf_bandwidth"] = ei.incumbent_protection.rf_bandwidth;
  j["scenario_rf_bandwidth"] = ei.scenario_rf_bandwidth;
  j["scenario_center_frequency"] = ei.scenario_center_frequency;
  j["has_incumbent"] = ei.has_incumbent;
  j["stage_number"] = ei.stage_number;
  j["timestamp"] = ei.timestamp;
  j["raw"] = ei.raw;
}

///////////////////////////////////////////////////////////////////////////////
// OFDM Physical layer processing
///////////////////////////////////////////////////////////////////////////////
namespace ofdm {

NotificationCenter::Name const ModulationEvent =
    NotificationCenter::makeName("ModulationEvent");

const std::string ModulationEventInfo::Name = "ModulationEvent";

const DBLayout ModulationEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("t_code_ns", DB::Type::INT)
        .addColumn("t_mod_ns", DB::Type::INT)
        .addColumn("t_spread_ns", DB::Type::INT)
        .addColumn("t_map_ns", DB::Type::INT)
        .addColumn("t_shift_ns", DB::Type::INT)
        .addColumn("t_cp_ns", DB::Type::INT)
        .addColumn("t_mix_ns", DB::Type::INT)
        .addColumn("t_scale_ns", DB::Type::INT)
        .addColumn("t_stream_ns", DB::Type::INT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ModulationEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, t_code_ns);
  ret = bind_int(*s, n++, t_mod_ns);
  ret = bind_int(*s, n++, t_spread_ns);
  ret = bind_int(*s, n++, t_map_ns);
  ret = bind_int(*s, n++, t_shift_ns);
  ret = bind_int(*s, n++, t_cp_ns);
  ret = bind_int(*s, n++, t_mix_ns);
  ret = bind_int(*s, n++, t_scale_ns);
  ret = bind_int(*s, n++, t_stream_ns);
  ret = bind_int(*s, n++, (int64_t)srcNodeID);
  ret = bind_int(*s, n++, frameID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ModulationEventInfo const &ei) {
  j["t_code_ns"] = ei.t_code_ns;
  j["t_mod_ns"] = ei.t_mod_ns;
  j["t_spread_ns"] = ei.t_spread_ns;
  j["t_map_ns"] = ei.t_map_ns;
  j["t_shift_ns"] = ei.t_shift_ns;
  j["t_cp_ns"] = ei.t_cp_ns;
  j["t_mix_ns"] = ei.t_mix_ns;
  j["t_scale_ns"] = ei.t_scale_ns;
  j["t_stream_ns"] = ei.t_stream_ns;
  j["srcNodeID"] = ei.srcNodeID;
  j["frameID"] = ei.frameID;
}

NotificationCenter::Name const ChannelEstimationEvent =
    NotificationCenter::makeName("ChannelEstimationEvent");

const std::string ChannelEstimationEventInfo::Name = "ChannelEstimationEvent";

const DBLayout ChannelEstimationEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("chanest", DB::Type::BLOB)
        .addColumn("desc", DB::Type::TEXT)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ChannelEstimationEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                       UnixTime t, UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_bytes(*s, n++, chanest);
  ret = bind_string(*s, n++, desc);
  ret = bind_int(*s, n++, (int64_t)srcNodeID);
  ret = bind_int(*s, n++, frameID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ChannelEstimationEventInfo const &ei) {
  // I'm just here so I won't get fined
  j["desc"] = ei.desc;
  j["srcNodeID"] = ei.srcNodeID;
  j["frameID"] = ei.frameID;
}

NotificationCenter::Name const SynchronizationEvent =
    NotificationCenter::makeName("SynchronizationEvent");

const std::string SynchronizationEventInfo::Name = "SynchronizationEvent";

const DBLayout SynchronizationEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("snr", DB::Type::REAL)
        .addColumn("domega", DB::Type::REAL)
        .addColumn("srcNodeID", DB::Type::INT)
        .addColumn("frameID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int SynchronizationEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                     UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_float(*s, n++, snr);
  ret = bind_float(*s, n++, domega);
  ret = bind_int(*s, n++, (int64_t)srcNodeID);
  ret = bind_int(*s, n++, frameID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, SynchronizationEventInfo const &ei) {
  // I'm just here so I won't get fined
  j["snr"] = ei.snr;
  j["domega"] = ei.domega;
  j["frameID"] = ei.frameID;
}

NotificationCenter::Name const MCSDecisionEvent =
    NotificationCenter::makeName("MCSDecisionEvent");

const std::string MCSDecisionEventInfo::Name = "MCSDecisionEvent";

const DBLayout MCSDecisionEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("txNodeID", DB::Type::INT)
        .addColumn("noiseVar", DB::Type::REAL)
        .addColumn("errorRate", DB::Type::REAL)
        .addColumn("overlapped", DB::Type::INT)
        .addColumn("payloadMCS", DB::Type::INT)
        .addColumn("payloadSymSeqID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int MCSDecisionEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                 UnixTime ts) const {
  int ret;
  // first call: prepare the statement
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  // subsequent calls: bind
  int n = 1;
  ret = bind_int(*s, n++, (int64_t)txNodeID);
  ret = bind_float(*s, n++, noiseVar);
  ret = bind_float(*s, n++, errorRate);
  ret = bind_int(*s, n++, overlapped ? 1 : 0);
  ret = bind_int(*s, n++, (int64_t)payloadMCS);
  ret = bind_int(*s, n++, (int64_t)payloadSymSeqID);
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, MCSDecisionEventInfo const &ei) {
  j["txNodeID"] = ei.txNodeID;
  j["noiseVar"] = ei.noiseVar;
  j["errorRate"] = ei.errorRate;
  j["overlapped"] = ei.overlapped;
  j["payloadMCS"] = (int)ei.payloadMCS;
  j["payloadSymSeqID"] = (int)ei.payloadSymSeqID;
}
} // namespace ofdm

///////////////////////////////////////////////////////////////////////////////
// Decision engine events
///////////////////////////////////////////////////////////////////////////////
namespace decisionengine {

NotificationCenter::Name const ChannelAllocUpdateEvent =
    NotificationCenter::makeName("Channel Allocation Update");

const std::string ChannelAllocUpdateEventInfo::Name = "ChannelAllocUpdate";
const DBLayout ChannelAllocUpdateEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("data", DB::Type::BLOB)
        // this should really be just "id" ¯\_(ツ)_/¯
        .addColumn("OFDMParamsUpdateID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ChannelAllocUpdateEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s,
                                        UnixTime t, UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (data) {
    ret = bind_bytes(*s, n++, bytes(data));
    // copy this out for easier SQL
    ret = bind_int(*s, n++, (int64_t)(data->channel_last_update()));
  } else {
    ret = bind_null(*s, n++);
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ChannelAllocUpdateEventInfo const &ei) {
  if (ei.data) {
    j["data"] = ei.data->ShortDebugString();
  }
}

// Channel Allocation attempt
NotificationCenter::Name const ChannelAllocEvent =
    NotificationCenter::makeName("Channel Allocation Attempt at gateway");

const std::string ChannelAllocEventInfo::Name = "ChannelAlloc";
const DBLayout ChannelAllocEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("data", DB::Type::BLOB)
        .addColumn("OFDMParamsUpdateID", DB::Type::INT)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int ChannelAllocEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                  UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (data) {
    ret = bind_bytes(*s, n++, bytes(data));
    ret = bind_int(*s, n++, (int64_t)(data->ofdm_params_update_id()));
  } else {
    ret = bind_null(*s, n++);
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, ChannelAllocEventInfo const &ei) {
  if (ei.data) {
    j["data"] = ei.data->ShortDebugString();
  }
}

// decision engine step. we got a serialized lisp object that we write to the
// database.

NotificationCenter::Name const StepEvent =
    NotificationCenter::makeName("Decision Engine Step");

const std::string StepEventInfo::Name = "DecisionEngineStep";
const DBLayout StepEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("data", DB::Type::BLOB)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int StepEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                          UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (data) {
    ret = bind_bytes(*s, n++, *data);
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, StepEventInfo const &ei) {
  j["data"] = "¯\\_(ツ)_/¯";
}

NotificationCenter::Name const StepOutputEvent =
    NotificationCenter::makeName("Decision Engine Step Output");

const std::string StepOutputEventInfo::Name = "DecisionEngineStepOutput";
const DBLayout StepOutputEventInfo::Layout =
    bamradio::DBLayout(Name)
        .addColumn("data", DB::Type::BLOB)
        .addColumn("time", DB::Type::INT)
        .addColumn("timeSteady", DB::Type::INT);

int StepOutputEventInfo::to_sql(sqlite3 *db, sqlite3_stmt **s, UnixTime t,
                                UnixTime ts) const {
  int ret;
  if (*s == nullptr) {
    ret = Layout.prepare(db, s, Name);
  }
  int n = 1;
  if (data) {
    ret = bind_bytes(*s, n++, *data);
  } else {
    ret = bind_null(*s, n++);
  }
  ret = bind_int(*s, n++, t);
  ret = bind_int(*s, n++, ts);
  assert(n - 1 == Layout.nCols());
  return ret;
}

void to_json(nlohmann::json &j, StepOutputEventInfo const &ei) {
  j["data"] = "¯\\_(ツ)_/¯";
}

} // namespace decisionengine
} // namespace bamradio
