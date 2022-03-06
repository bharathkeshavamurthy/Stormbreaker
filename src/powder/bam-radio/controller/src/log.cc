// Logging.
//
// Copyright (c) 2018 Dennis Ogbe
// Copyright (c) 2018 Tomohiro Arakawa
// Copyright (c) 2018 Stephen Larew
// Copyright (c) 2018 Diyu Yang

#include "log.h"
#include "build_info.h"
#include "bandwidth.h"
#include "events.h"
#include "statistics.h"
#include "util.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <boost/format.hpp>

namespace bamradio {
namespace log {

void Logger::setStartTime(std::chrono::system_clock::time_point t) {
  _scenario_start_time = t;
  printWithTime("[Logger]: Setting start time...");
}

inline void printWithTimeNoStartTime(std::string const &str) {
  auto const now = std::chrono::system_clock::now();
  auto const in_time_t = std::chrono::system_clock::to_time_t(now);
  // FIXME?
  if (str.back() == '\n') {
    std::cout << std::put_time(std::localtime(&in_time_t), "[%H:%M:%S]") << ": "
              << str.substr(0, str.size() - 1) << std::endl;
  } else {
    std::cout << std::put_time(std::localtime(&in_time_t), "[%H:%M:%S]") << ": "
              << str << std::endl;
  }
}

inline void
printWithTimeStartTime(std::string const &str,
                       std::chrono::system_clock::time_point const &t) {
  using namespace std::chrono;
  auto const reltime =
      duration_cast<milliseconds>(std::chrono::system_clock::now() - t);
  // clang-format off
  std::cout << (boost::format("[%1$.2f]: %2%")
                % ((double)(reltime.count()) / 1000.0) %
                (str.back() == '\n' ? str.substr(0, str.size() - 1) : str))
            << std::endl;
  // clang-format on
}

inline void Logger::printWithTime(std::string const &str) {
  if (_scenario_start_time) {
    printWithTimeStartTime(str, *_scenario_start_time);
  } else {
    printWithTimeNoStartTime(str);
  }
}

using Json = nlohmann::json;

// basic subscribe means log to JSON and to SQL. Anything more thath that,
// define your own lambda and act accordingly. See below.
template <typename EventInfo>
void Logger::_basic_subscribe(NotificationCenter::Name event) {
  _st.push_back(NotificationCenter::shared.subscribe<EventInfo>(
      event, _ioctx, [self = this](auto ei) {
        auto const now = self->_getCurrentTime();
        auto const nowSteady = self->_getCurrentTimeSteady();
        if (self->backendActive(Backend::JSON)) {
          Json j = ei;
          j["eventTime"] = now;
          j["eventTimeSteady"] = nowSteady;
          j["event"] = ei.Name;
          self->_writeJson(j);
        }
        if (self->backendActive(Backend::SQL)) {
          self->_db->insert(ei, now, nowSteady);
        }
      }));
}

Logger::Logger()
    : _work(boost::asio::make_work_guard(_ioctx)),
      _work_thread(std::thread([this] {
        // bamradio::set_thread_name("logger")
        pthread_setname_np(pthread_self(), "logger");
        _ioctx.run();
      })),
      _db(nullptr), _scenario_start_time(boost::none) {

  //
  // DLL Events
  //

  _basic_subscribe<dll::SentFrameEventInfo>(dll::SentFrameEvent);
  _basic_subscribe<dll::SentSegmentEventInfo>(dll::SentSegmentEvent);
  _basic_subscribe<dll::DetectedFrameEventInfo>(dll::DetectedFrameEvent);
  _basic_subscribe<dll::ReceivedFrameEventInfo>(dll::ReceivedFrameEvent);
  _basic_subscribe<dll::InvalidFrameHeaderEventInfo>(
      dll::InvalidFrameHeaderEvent);
  _basic_subscribe<dll::ReceivedBlockEventInfo>(dll::ReceivedBlockEvent);
  _basic_subscribe<dll::ReceivedCompleteSegmentEventInfo>(
      dll::ReceivedCompleteSegmentEvent);
  _basic_subscribe<dll::CoDelDelayEventInfo>(dll::CoDelDelayEvent);
  _basic_subscribe<dll::CoDelStateEventInfo>(dll::CoDelStateEvent);
  _basic_subscribe<dll::NewFlowEventInfo>(dll::NewFlowEvent);
  _basic_subscribe<dll::FlowQueuePushEventInfo>(dll::FlowQueuePushEvent);
  _basic_subscribe<dll::FlowQueuePopEventInfo>(dll::FlowQueuePopEvent);
  _basic_subscribe<dll::ReceivedARQFeedbackEventInfo>(
      dll::ReceivedARQFeedbackEvent);
  _basic_subscribe<dll::FlowTrackerStateUpdateEventInfo>(
      dll::FlowTrackerStateUpdateEvent);
  _basic_subscribe<dll::FlowTrackerIMEventInfo>(dll::FlowTrackerIMEvent);
  _basic_subscribe<dll::FlowQueueResendEventInfo>(dll::FlowQueueResendEvent);

  // TODO how to templatize this like _basic_subscribe. some templating magic?
  _st.push_back(
      NotificationCenter::shared.subscribe<dll::ScheduleUpdateEventInfo>(
          dll::ScheduleUpdateEvent, _ioctx, [this](auto ei) {
            auto const now = this->_getCurrentTime();
            auto const nowSteady = this->_getCurrentTimeSteady();
            if (this->backendActive(Backend::JSON)) {
              Json j = ei;
              j["eventTime"] = now;
              j["eventTimeSteady"] = nowSteady;
              j["event"] = ei.Name;
              this->_writeJson(j);
            }
            if (this->backendActive(Backend::SQL)) {
              this->_db->insert_raw(
                  ei,
                  std::chrono::system_clock::time_point(
                      std::chrono::system_clock::duration(now)),
                  std::chrono::steady_clock::time_point(
                      std::chrono::steady_clock::duration(nowSteady)));
            }
          }));

  //
  // Statistics
  //

  _st.push_back(NotificationCenter::shared.subscribe<stats::StatPrintEventInfo>(
      stats::StatPrintEvent, _ioctx, [self = this](auto ei) {
        auto const now = self->_getCurrentTime();
        auto const nowSteady = self->_getCurrentTimeSteady();
        if (self->backendActive(Backend::STDOUT)) {
          self->printWithTime("Traffic Stats Update:");
          std::cout.setf(std::ios::fixed, std::ios::floatfield);
          // HACK: put thousands separators
          std::cout.imbue(std::locale("en_US.UTF-8"));

          std::cout << "  Tx segments: " << ei.total_n_segments_sent
                    << "  Rx decoded headers: " << ei.total_n_headers_decoded
                    << "  Rx segments: " << ei.total_n_segments_rxd
                    << std::endl;

          // per-flow traffic information
          for (auto &x : ei.flow_offered_bytes) {
            int rate = 8 * x.second.average();
            if (rate > 0) {
              std::cout << "  [ offered flow ]  UID: " << x.first
                        << "  Avg rate: " << rate << " bps" << std::endl;
            }
          }
          for (auto &x : ei.flow_delivered_bytes) {
            int rate = 8 * x.second.average();
            if (rate > 0) {
              std::cout << "  [delivered flow]  UID: " << x.first
                        << "  Avg rate: " << rate << " bps,  Avg latency: "
                        << (int)(1000.0 * ei.flow_delivered_latency[x.first]
                                              .average_elements())
                        << " ms" << std::endl;
            }
          }
          std::cout << "Tx Duty Cycle: " << ei.duty_cycle << std::endl;
        }
      }));

  //
  // Collab Events
  //

  _basic_subscribe<collab::CollabRxEventInfo>(collab::CollabRxEvent);
  _basic_subscribe<collab::CollabTxEventInfo>(collab::CollabTxEvent);
  _basic_subscribe<collab::ServerRxEventInfo>(collab::ServerRxEvent);
  _basic_subscribe<collab::ServerTxEventInfo>(collab::ServerTxEvent);
  _basic_subscribe<collab::ConnectionEventInfo>(collab::ConnectionEvent);
  _basic_subscribe<collab::StateChangeEventInfo>(collab::StateChangeEvent);
  _basic_subscribe<collab::CollabPeerEventInfo>(collab::CollabPeerEvent);
  _basic_subscribe<collab::ErrorEventInfo>(collab::ErrorEvent);

  //
  // PHY Events
  //

  _basic_subscribe<BurstSendEventInfo>(BurstSendEvent);
  _basic_subscribe<psdsensing::PSDUpdateEventInfo>(psdsensing::PSDUpdateEvent);
  _basic_subscribe<psdsensing::PSDRxEventInfo>(psdsensing::PSDRxEvent);
  _basic_subscribe<ofdm::ModulationEventInfo>(ofdm::ModulationEvent);
  _basic_subscribe<ofdm::ChannelEstimationEventInfo>(
      ofdm::ChannelEstimationEvent);
  _basic_subscribe<ofdm::SynchronizationEventInfo>(ofdm::SynchronizationEvent);
  _basic_subscribe<ofdm::MCSDecisionEventInfo>(ofdm::MCSDecisionEvent);
  _st.push_back(
      NotificationCenter::shared.subscribe<uhdfeedback::UHDMsgEventInfo>(
          uhdfeedback::UHDMsgEvent, _ioctx, [self = this](auto ei) {
            auto const now = self->_getCurrentTime();
            auto const nowSteady = self->_getCurrentTimeSteady();
            if (self->backendActive(Backend::JSON)) {
              Json j = ei;
              j["eventTime"] = now;
              j["eventTimeSteady"] = nowSteady;
              self->_writeJson(j);
            }
            if (self->backendActive(Backend::SQL)) {
              self->_db->insert(ei, now, nowSteady);
            }
            if (self->backendActive(Backend::STDOUT)) {
              self->printWithTime((boost::format("UHD(%1%): %2%") %
                                   uhdfeedback::uhdMsg2str(ei.type) % ei.msg)
                                      .str());
            }
          }));

  _st.push_back(
      NotificationCenter::shared.subscribe<uhdfeedback::UHDAsyncEventInfo>(
          uhdfeedback::UHDAsyncEvent, _ioctx, [self = this](auto ei) {
            auto const now = self->_getCurrentTime();
            auto const nowSteady = self->_getCurrentTimeSteady();
            if (self->backendActive(Backend::JSON)) {
              Json j = ei;
              j["eventTime"] = now;
              j["eventTimeSteady"] = nowSteady;
              self->_writeJson(j);
            }
            if (self->backendActive(Backend::SQL)) {
              self->_db->insert(ei, now, nowSteady);
            }
            if (self->backendActive(Backend::STDOUT)) {
              if (ei.event_code !=
                  ::uhd::async_metadata_t::EVENT_CODE_BURST_ACK) {
                self->printWithTime((boost::format("UHD: %1% {%2%}") %
                                     uhdfeedback::uhdAsync2str(ei.event_code) %
                                     ei.time)
                                        .str());
              }
            }
          }));

  //
  // NET Events
  //

  // Route Decision
  _basic_subscribe<net::RouteDecisionEventInfo>(net::RouteDecisionEvent);

  // Routing Table Update
  _st.push_back(
      NotificationCenter::shared.subscribe<net::RoutingTableUpdateEventInfo>(
          net::RoutingTableUpdateEvent, _ioctx, [self = this](auto ei) {
            auto const now = self->_getCurrentTime();
            auto const nowSteady = self->_getCurrentTimeSteady();
            Json j = ei;
            std::stringstream s;
            s << "Routing Table Update:\n" << std::setw(2) << j;
            if (self->backendActive(Backend::JSON)) {
              j["eventTime"] = now;
              j["eventTimeSteady"] = nowSteady;
              self->_writeJson(j);
            }
            if (self->backendActive(Backend::SQL)) {
              self->_db->insert(ei, now, nowSteady);
            }
            if (self->backendActive(Backend::STDOUT)) {
              self->printWithTime(s.str());
            }
          }));

  //
  // Misc
  //

  _basic_subscribe<controlchannel::CCPacketEventInfo>(
      controlchannel::CCPacketEvent);
  _basic_subscribe<controlchannel::NetworkMapEventInfo>(
      controlchannel::NetworkMapEvent);
  _basic_subscribe<gps::GPSEventInfo>(gps::GPSEvent);
  _basic_subscribe<AchievedIMsUpdateEventInfo>(AchievedIMsUpdateEvent);
  _basic_subscribe<OutcomesUpdateEventInfo>(OutcomesUpdateEvent);
  _basic_subscribe<C2APIEventInfo>(C2APIEvent);
  _basic_subscribe<EnvironmentUpdateEventInfo>(EnvironmentUpdateEvent);
  _basic_subscribe<IncumbentAttenuationUpdateEventInfo>(
      IncumbentAttenuationUpdateEvent);
  _basic_subscribe<decisionengine::ChannelAllocUpdateEventInfo>(
      decisionengine::ChannelAllocUpdateEvent);
  _basic_subscribe<decisionengine::ChannelAllocEventInfo>(
      decisionengine::ChannelAllocEvent);
  _basic_subscribe<decisionengine::StepEventInfo>(decisionengine::StepEvent);
  _basic_subscribe<decisionengine::StepOutputEventInfo>(
      decisionengine::StepOutputEvent);

  // Doomsday Event
  _st.push_back(NotificationCenter::shared.subscribe<DoomsdayEventInfo>(
      DoomsdayEvent, _ioctx, [self = this](auto ei) {
        // first log as normal
        auto const now = self->_getCurrentTime();
        auto const nowSteady = self->_getCurrentTimeSteady();
        if (self->backendActive(Backend::JSON)) {
          Json j = ei;
          j["eventTime"] = now;
          j["eventTimeSteady"] = nowSteady;
          j["event"] = ei.Name;
          self->_writeJson(j);
        }
        if (self->backendActive(Backend::SQL)) {
          self->_db->insert(ei, now, nowSteady);
        }
        if (self->backendActive(Backend::STDOUT)) {
          self->printWithTime("!!!DOOMSDAY " + ei.msg + " DOOMSDAY!!!");
        }
        // then cleanly close the logs before last judgement
        self->_st.clear();
        if (self->_work.owns_work()) {
          self->_work.reset();
        }
        self->_ioctx.stop();
        self->_closeFiles();
        ei.judgement_day();
      }));

  // Just log some text
  _st.push_back(NotificationCenter::shared.subscribe<TextLogEventInfo>(
      TextLogEvent, _ioctx, [self = this](auto ei) {
        auto const now = self->_getCurrentTime();
        auto const nowSteady = self->_getCurrentTimeSteady();
        if (self->backendActive(Backend::JSON)) {
          Json j = ei;
          j["eventTime"] = now;
          j["eventTimeSteady"] = nowSteady;
          self->_writeJson(j);
        }
        if (self->backendActive(Backend::SQL)) {
          self->_db->insert(ei, now, nowSteady);
        }
        if (self->backendActive(Backend::STDOUT)) {
          self->printWithTime(ei.msg);
        }
      }));

  // Database interaction events (we really should not have to log this)
  _st.push_back(NotificationCenter::shared.subscribe<DBEventInfo>(
      DBEvent, _ioctx, [self = this](auto ei) {
        auto const now = self->_getCurrentTime();
        auto const nowSteady = self->_getCurrentTimeSteady();
        if (self->backendActive(Backend::JSON)) {
          Json j = ei;
          j["eventTime"] = now;
          j["eventTimeSteady"] = nowSteady;
          j["event"] = "Database";
          self->_writeJson(j);
        }
      }));

  // start time notification
  _st.push_back(NotificationCenter::shared.subscribe<ScenarioStartEventInfo>(
      ScenarioStartEvent,
      _ioctx, [self = this](auto ei) { self->setStartTime(ei.time); }));

} // END SUBSCRIPTIONS

//
// LOGGER INTERNALS
//

void Logger::_closeFiles() {
  if (_ofs.is_open()) {
    _ofs.close();
  }
  if (_db != nullptr) {
    _db->close();
    _db = nullptr;
  }
}

Logger::~Logger() { shutdown(); }

void Logger::shutdown() {
  _st.clear();
  if (_work.owns_work()) {
    _work.reset();
  }
  _ioctx.stop();
  if (_work_thread.joinable()) {
    _work_thread.join();
  }
  _closeFiles();
}

void Logger::enableBackend(Backend b) {
  // not the best interface, but they don't pay me for UX!
  if (Backend::STDOUT == b) {
    _filename[b] = "stdout";
  } else {
    throw std::runtime_error(
        "File name needs to be specified for JSON and SQL backends.");
  }
}

void Logger::enableBackend(Backend b, std::string const &filename,
                           bool append) {
  if (backendActive(b)) {
    return;
  }
  switch (b) {
  case Backend::JSON: {
    _ofs.open(filename, (append ? std::ofstream::out | std::ofstream::app
                                : std::ofstream::out));
    Json j = {{"event", "logOpened"},
              {"eventTime", _getCurrentTime()},
              {"eventTimeSteady", _getCurrentTimeSteady()}};
    _ofs << j.dump() << std::endl;
    _filename[b] = filename;
  } break;
  case Backend::SQL: {
    _db = database::make();
    _db->open(filename, true);
    _db->insert_start(_getCurrentTime());
    _filename[b] = filename;
  } break;
  case Backend::STDOUT: {
    _filename[b] = "stdout";
  } break;
  default: { std::cout << "Backend " << b << " does not exist."; }
  }
}

void Logger::disableBackend(Backend b) {
  if (!backendActive(b)) {
    return;
  }
  switch (b) {
  case Backend::JSON: {
    Json j = {{"event", "logClosed"}};
    _ofs << j.dump() << std::endl;
    _ofs.close();
    _filename.erase(b);
  } break;
  case Backend::SQL: {
    if (_db != nullptr) {
      _db->close();
      _db = nullptr;
    }
    _filename.erase(b);
  }
  case Backend::STDOUT: {
    _filename.erase(b);
  }
  }
}

bool Logger::backendActive(Backend b) const {
  return not(_filename.find(b) == _filename.end());
}

int64_t Logger::_getCurrentTime() const {
  return std::chrono::system_clock::now().time_since_epoch().count();
}
int64_t Logger::_getCurrentTimeSteady() const {
  return std::chrono::steady_clock::now().time_since_epoch().count();
}

void Logger::_writeJson(nlohmann::json const &j) {
  _ofs << j.dump() << std::endl;
}

// SQL interface
inline void logError(int error_code, char const *file, int line) {
  NotificationCenter::shared.post(
      DBEvent,
      DBEventInfo{DBEventInfo::Type::FAIL,
                  (boost::format("%1%:%2%: SQL Error! Error Code %3%") % file %
                   line % error_code)
                      .str()});
}

// FIXME what is this for again?
static int callback(void *p, int argc, char **argv, char **azColName) {
  std::stringstream s;
  auto self = (database *)p;
  for (int i = 0; i < argc; i++) {
    s << boost::format("%1% = %2%") % azColName[i] %
             (argv[i] ? argv[i] : "NULL")
      << std::endl;
  }
  NotificationCenter::shared.post(
      DBEvent, DBEventInfo{DBEventInfo::Type::CALLBACK, s.str()});
  return 0;
}

database::database() : db(nullptr), _sql_id(0){};

database::~database() {
  if (is_open()) {
    close();
  }
}

bool database::_exec(std::string const &sql) {
  if (!is_open()) {
    return false;
  }
  char *zErrMsg;
  auto rc = sqlite3_exec(db, sql.c_str(), callback, this, &zErrMsg);
  if (rc != SQLITE_OK) {
    NotificationCenter::shared.post(
        DBEvent, DBEventInfo{DBEventInfo::Type::FAIL,
                             (boost::format("SQL Error: %1%") % zErrMsg).str(),
                             sql, _sql_id});
    sqlite3_free(zErrMsg);
  } else {
    NotificationCenter::shared.post(
        DBEvent, DBEventInfo{DBEventInfo::Type::SUCCESS, "", sql, _sql_id});
  }
  _sql_id++;
  return rc == SQLITE_OK;
}

// FIXME figure out what append means for this
void database::open(std::string const &filename, bool append) {
  _filename = filename;
  // Open database
  if (sqlite3_open(_filename.c_str(), &db) != SQLITE_OK) {
    NotificationCenter::shared.post(
        DBEvent, DBEventInfo{DBEventInfo::Type::FAIL,
                             (boost::format("Can't open database: %1%\n") %
                              sqlite3_errmsg(db))
                                 .str(),
                             "", -1});
    return;
  } else {
    NotificationCenter::shared.post(
        DBEvent, DBEventInfo{DBEventInfo::Type::SUCCESS,
                             "Opened database successfully", "", -1});
  }

  // insert build information
  [this] {
    using namespace buildinfo;
    int ret;
    std::string const binfo_name = "BuildInfo";
    auto binfo_layout = DBLayout(binfo_name)
                            .addColumn("commithash", DBLayout::Type::TEXT)
                            .addColumn("buildtime", DBLayout::Type::TEXT)
                            .addColumn("cilproto", DBLayout::Type::TEXT)
                            .addColumn("regproto", DBLayout::Type::TEXT)
                            .addColumn("ccdataproto", DBLayout::Type::TEXT)
                            .addColumn("logproto", DBLayout::Type::TEXT);
    _exec(binfo_layout.sql());
    sqlite3_stmt *st;
    ret = binfo_layout.prepare(db, &st, binfo_name);
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
    ret = sqlite3_bind_text(st, 1, commithash.c_str(), -1, SQLITE_STATIC);
    ret = sqlite3_bind_text(st, 2, buildtime.c_str(), -1, SQLITE_STATIC);
    ret = sqlite3_bind_text(st, 3, cilproto.c_str(), -1, SQLITE_STATIC);
    ret = sqlite3_bind_text(st, 4, regproto.c_str(), -1, SQLITE_STATIC);
    ret = sqlite3_bind_text(st, 5, ccdataproto.c_str(), -1, SQLITE_STATIC);
    ret = sqlite3_bind_text(st, 6, logproto.c_str(), -1, SQLITE_STATIC);
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
    ret = sqlite3_step(st); // step()
    if (ret != SQLITE_DONE)
      logError(ret, __FILE__, __LINE__);
    ret = sqlite3_reset(st); // reset()
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
    ret = sqlite3_finalize(st); // finalize()
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
  }();

  // insert waveform ID table (FIXME maybe refactor all of these extra tables
  // out?)
  [this] {
    int ret;
    std::string const wft_name = "Waveform";
    auto wft_layout = DBLayout(wft_name)
                          .addColumn("waveformID", DBLayout::Type::INT)
                          .addColumn("edge", DBLayout::Type::REAL);
    _exec(wft_layout.sql());
    sqlite3_stmt *st;
    ret = wft_layout.prepare(db, &st, wft_name);
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
    for (size_t i = 0; i < bam::dsp::SubChannel::table().size(); ++i) {
      auto waveform = bam::dsp::SubChannel::table()[i];
      ret = sqlite3_bind_int64(st, 1, (int64_t)i);
      ret = sqlite3_bind_double(st, 2, 1 / (double)waveform.os);
      if (ret != SQLITE_OK)
        logError(ret, __FILE__, __LINE__);
      ret = sqlite3_step(st); // step()
      if (ret != SQLITE_DONE)
        logError(ret, __FILE__, __LINE__);
      ret = sqlite3_reset(st); // reset()
      if (ret != SQLITE_OK)
        logError(ret, __FILE__, __LINE__);
    }
    ret = sqlite3_finalize(st); // finalize()
    if (ret != SQLITE_OK)
      logError(ret, __FILE__, __LINE__);
  }();

  // create tables
  _exec(dll::SentFrameEventInfo::Layout.sql());
  _exec(dll::SentSegmentEventInfo::Layout.sql());
  _exec(dll::DetectedFrameEventInfo::Layout.sql());
  _exec(dll::ReceivedFrameEventInfo::Layout.sql());
  _exec(dll::InvalidFrameHeaderEventInfo::Layout.sql());
  _exec(dll::ReceivedBlockEventInfo::Layout.sql());
  _exec(dll::ReceivedCompleteSegmentEventInfo::Layout.sql());
  _exec(dll::CoDelDelayEventInfo::Layout.sql());
  _exec(dll::CoDelStateEventInfo::Layout.sql());
  _exec(dll::NewFlowEventInfo::Layout.sql());
  _exec(dll::FlowQueuePushEventInfo::Layout.sql());
  _exec(dll::FlowQueuePopEventInfo::Layout.sql());
  _exec(dll::ScheduleUpdateEventInfo::Layout.sql());
  _exec(dll::ScheduleUpdateEventInfo::FlowQuantum::Layout.sql());
  _exec(dll::ReceivedARQFeedbackEventInfo::Layout.sql());
  _exec(dll::FlowTrackerStateUpdateEventInfo::Layout.sql());
  _exec(dll::FlowTrackerIMEventInfo::Layout.sql());
  _exec(dll::FlowQueueResendEventInfo::Layout.sql());
  _exec(gps::GPSEventInfo::Layout.sql());
  _exec(net::RouteDecisionEventInfo::Layout.sql());
  _exec(net::RoutingTableUpdateEventInfo::Layout.sql());
  _exec(BurstSendEventInfo::Layout.sql());
  _exec(uhdfeedback::UHDAsyncEventInfo::Layout.sql());
  _exec(uhdfeedback::UHDMsgEventInfo::Layout.sql());
  _exec(controlchannel::CCPacketEventInfo::Layout.sql());
  _exec(controlchannel::NetworkMapEventInfo::Layout.sql());
  _exec(collab::CollabRxEventInfo::Layout.sql());
  _exec(collab::CollabTxEventInfo::Layout.sql());
  _exec(collab::ServerRxEventInfo::Layout.sql());
  _exec(collab::ServerTxEventInfo::Layout.sql());
  _exec(collab::ConnectionEventInfo::Layout.sql());
  _exec(collab::StateChangeEventInfo::Layout.sql());
  _exec(collab::CollabPeerEventInfo::Layout.sql());
  _exec(collab::ErrorEventInfo::Layout.sql());
  _exec(AchievedIMsUpdateEventInfo::Layout.sql());
  _exec(OutcomesUpdateEventInfo::Layout.sql());
  _exec(C2APIEventInfo::Layout.sql());
  _exec(EnvironmentUpdateEventInfo::Layout.sql());
  _exec(DoomsdayEventInfo::Layout.sql());
  _exec(TextLogEventInfo::Layout.sql());
  _exec(psdsensing::PSDUpdateEventInfo::Layout.sql());
  _exec(psdsensing::PSDRxEventInfo::Layout.sql());
  _exec(ofdm::ModulationEventInfo::Layout.sql());
  _exec(ofdm::ChannelEstimationEventInfo::Layout.sql());
  _exec(ofdm::SynchronizationEventInfo::Layout.sql());
  _exec(ofdm::MCSDecisionEventInfo::Layout.sql());
  _exec(IncumbentAttenuationUpdateEventInfo::Layout.sql());
  _exec(decisionengine::ChannelAllocUpdateEventInfo::Layout.sql());
  _exec(decisionengine::ChannelAllocEventInfo::Layout.sql());
  _exec(decisionengine::StepEventInfo::Layout.sql());
  _exec(decisionengine::StepOutputEventInfo::Layout.sql());
}

bool database::is_open() const { return db != nullptr; }

void database::close() {
  auto ldb = db;
  db = nullptr;
  for (auto const &c : _stmt) {
    sqlite3_finalize(c.second);
  }
  _stmt.clear();
  sqlite3_close(ldb);
}

// Start Event
bool database::insert_start(unsigned long long const TimeStamp) {
  int ret;
  // start time
  _exec(DBLayout("Start").addColumn("time_init", DBLayout::Type::INT).sql());
  std::stringstream s;
  s << "insert into Start (time_init) values(" << TimeStamp << ");";
  ret = _exec(s.str());
  return ret;
}

template <typename EventInfo>
bool database::insert(EventInfo const &ei, unsigned long long t,
                      unsigned long long ts) {
  if (!is_open()) {
    return false;
  }
  int rc;
  if (_stmt.find(ei.Name) == _stmt.end()) {
    _stmt[ei.Name] = nullptr; // initialize stmt
  }
  auto s = _stmt.at(ei.Name);
  rc = ei.to_sql(db, &s, t, ts); // bind()
  if (rc != SQLITE_OK) {
    logError(rc, __FILE__, __LINE__);
    return false;
  }
  rc = sqlite3_step(s); // step()
  if (rc != SQLITE_DONE) {
    logError(rc, __FILE__, __LINE__);
    return false;
  }
  rc = sqlite3_reset(s); // reset()
  if (rc != SQLITE_OK) {
    logError(rc, __FILE__, __LINE__);
    return false;
  }
  NotificationCenter::shared.post(
      DBEvent, DBEventInfo{DBEventInfo::Type::SUCCESS, ei.Name, "", _sql_id});
  _sql_id++;
  return true;
}

template <typename EventInfo>
bool database::insert_raw(EventInfo const &ei,
                          std::chrono::system_clock::time_point t,
                          std::chrono::steady_clock::time_point ts) {
  if (!is_open()) {
    return false;
  }
  int rc;
  rc = ei.to_sql(db, _stmt, t, ts);
  if (rc != SQLITE_OK) {
    logError(rc, __FILE__, __LINE__);
    return false;
  }
  NotificationCenter::shared.post(
      DBEvent, DBEventInfo{DBEventInfo::Type::SUCCESS, ei.Name, "", _sql_id});
  _sql_id++;
  return true;
}

// clang-format off
const std::string asciiLogo = "\n"
"                                                                                              ``\n"
"                                                                                        ```---:::--..`\n"
"                                                                                  `--/ooyyyyyyyyyyyyyyss+/:``\n"
"                                                                                --:oosyyhhhyyyyyyyyhhhhhyss//-..\n"
"                                                                             `..hhdmms++-....------..-::syymmy++.````\n"
"  -:::::::::::::::::::::..`               .--:::::::::::::-..           ``.::ossyys--://yhhmmmmmmmmddyss:-://smm+////:--`          ..-////:::-\n"
"``+yyhhhhhhhhhhhhhhhhhhhyy/--        ``.ooshhhhhhhhhhhhhhhhyy:--        ::+hhyyy///++syysssoo++++++oosyyyys//oyyhddhhs++`         `ooshhhhhyy+\n"
"``sNNMMMMMMMMmmmNNNMMMMMMMhoo`       .-:NNNMMMMMNNNNNMMMMMMMMyo+        ++sMMdhh--/hhhhh+//--------::+ooddh++ossNMMMMdoo.         `yydMMMMMddo\n"
"``smmMMMMMmmm+++ssdNNMMMMMMMM:..     oyyMMMMMMMMddyssmNMMMMMMMMN..`     //sMMMMMyyyoo:``++sddmNNmmmyy+----:yymMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhss.``../ooNMMMMMMM/--   ``hdmMMMMMNNm--...+osMMMMMMMM--.     //sMMMMMNNmssooohhy++/:://+sshddo+oNNNMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMyoo     .::NMMMMMMM/:-   ``hmNMMMMMdhy     -:/MMMMMMMM::.     //sMMMMMMMNddhss++/...````.::ohhhhdMMMMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhoo     `..NMMMMMMM/:-   ``hmNMMMMMyoo     ..-MMMMMMMM::.     //sMMMMMMMMMMdhh..`        ``:ooNMMMMMMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhoo     .--NMMMMMMM/:-   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMMMMMMMMMMMMyyo``     `++hMMMMMMMMMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhoo     :++NMMMMMMM:-.   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMMMMMMMMMMMMNNd++-  ::+mmNMMMMMMMMMMMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMdhh///++shhMMMMMmhh.``   ``hmNMMMMMdhy/////++oMMMMMMMM::.     //sMMNNNNNMMMMMMMMNdds++yyhMMMMMMMNddmNNMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMMMMddmNNMMMMMMMMy::`     ``hmNMMMMMMMMdddddmmmMMMMMMMM::.     //sMMmmmmmmMMMMMMMMMMMMMMMMMMMMMMNmoohNNMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMMMMMMMMMMMMMMMNNo``      ``hmNMMMMMMMMMMMMMMMMMMMMMMMM::.     //sMMy++``/NNNMMMMMMMMMMMMMMMMMMo+/  +mmMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMdhh+++++ymmNNNMMdyy.     ``hmNMMMMMdhy+++++ssyMMMMMMMM::.     //sMMyoo` .//hNNMMMMMMMMMMMMMdss-..``+mmMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhss.``../ssmNNMMMNN-.`   ``hmNMMMMMyso`````::/MMMMMMMM::.     //sMMyoo   ``+yyNMMMMMMMMMNddo::`  ``ommMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMyoo     .--NMMMMMMM/:-   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo     `..mNNMMMMMMMm++-     ``ommMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMhoo     `..NMMMMMMM/:-   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo        ../NNMMMo+/        ``ommMMMMMdoo`\n"
"``smmMMMMMhoo     .--NMMMMMMM/:-   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo        ``.++oss-..        ``ommMMMMMdoo`          ::/sssss++:\n"
"``smmMMMMMhoo`  ``:++NMMMMMMM/:-   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo           ``---           ``ommMMMMMdoo`         `ooyNNNNmhh+\n"
"``smmMMMMMhss.``..oddMMMMMMMM/--   ``hmNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo                           ``ommMMMMMdoo`         `yydMMMMMddo\n"
"``sNNMMMMMMMMNNNNNMMMMMMMMmhh.     ``hNNMMMMMso+     ..-MMMMMMMM::.     //sMMyoo                           ``ommMMMMMdoo`         `yydMMMMMddo\n"
"``smmMMMMMMMMMMMMMMMMMMNhho//`     ``hmmMMMMMso+     ..-MMMMMMMM::.     //sMMyoo                           ``ommMMMMMh++`         `yydMMMMMddo\n"
"  +yyhhhhhhhhhhhhhhhhhhy//-          oyyhhhhh+/:     `..hhhhhhhh--.     ::+hho//                           ``/yyhhhhho::`         `ooshhhhhss/\n"
"                                                              ://..`                          `..::-\n"
"                                                             ://..`                          .::syo\n"
"                                     `..--.  ...--.  ---..`  /++...``-::..-::.``...:::----.  -++mmh   ``------::.``  `..-----.``   ..------``\n"
"                                     //+mmo  oyydd+``ymmoo/``ymm::-..hNNhhdNNo::sshNNNmmmmy``-//ddy  .--mNNmmmNN+/:  :ssmNNmmy::` `sshNNmmm::.\n"
"                                     //+mm+  oyyddo``ymmoo/``yNm::-..hMMsss++:..mmh++///NMd--://ddy  -//mNd--ohhhhs``+mmdhh::---``.mmdhh+::--.\n"
"                                     :/+dd+  oyyddo``ymmoo:``yNm::-..hMM//:...``mNdoo+++mmh--://ddy  -//mNm//ohhyys``/yyddd++/..` `yyhdds++..`\n"
"                                     :/+dd+  oyydd+``ymm++:``yNm::-..hNN```   ``mNmhhyyyyyo--://ddy  -//mMNyyyyyoo+  .::hmmNNy--`  ::ommNNN--.\n"
"                                     :/+dd+  oyydd+``hNm::-``yNm::-..hNN```  ```mNd++-``::-``://ddy  -++mNd.....:/:``-//--:NNhoo.``//:--hNNoo:\n"
"                                     :/+NNdssmNNNNmhhsso`````yNm::-..hNN```     ssyddhyyyyo``-//hdy  .--dNmhhhhh/::  :oohddNNy::` `ooyddmNN::.\n"
"                                     --:yysooyyyyyyyy/::     +ss.....+yy```     ::+yysssoo/  .--oo+  ```ossyyyyy:-.  -//syyyy/``   //oyyyys```\n"
"\n\n\n";
// clang-format on

} // namespace log
} // namespace bamradio
