// C2API.
//
// Copyright (c) 2018 Dennis Ogbe
// Copyright (c) 2018 Tomohiro Arakawa

#include "c2api.h"
#include "notify.h"
#include "watchdog.h"

#include <chrono>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

namespace bamradio {
namespace c2api {

// beware of hacks
int64_t const INIT_ENV_MAGIC = 33333333;
constexpr int32_t T_REF = 1485882000;

Client::Client(zmq::context_t &ctx, int port, std::string const &status_path,
               std::string const &env_recovery_path,
               std::string const &mo_recovery_path)
    : _port(port), _status_path(status_path),
      _env_recovery_path(env_recovery_path),
      _mo_recovery_path(mo_recovery_path), _state(State::Stopped), _ctx(ctx),
      _sock(_ctx, ZMQ_PULL), _zpi({.socket = static_cast<void *>(_sock),
                                   .fd = 0,
                                   .events = ZMQ_POLLIN,
                                   .revents = 0}) {
  // bind to socket / setsockopt
  int i = 5;
  while (i-- > 0) {
    try {
      _sock.bind((boost::format("tcp://%1%:%2%") % "127.0.0.1" % _port).str());
      break;
    } catch (...) {
      ; // connection failed, retry until we can't no more
    }
  }
  if (i <= 0) {
    throw std::runtime_error("Connection Timeout");
  }
  int linger = 0;
  _sock.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

  // initial status is "Booting"
  updateStatus(Status::Booting);
}

Client::~Client() {
  _updateState(State::Stopped);
  if (_thread.joinable()) {
    _thread.join();
  }
  _sock.close();
}

void Client::_updateState(State state) {
  std::unique_lock<decltype(_mtx)> l(_mtx);
  _state = state;
  _cv.notify_all();
}

Client::State Client::_getState() {
  std::unique_lock<decltype(_mtx)> l(_mtx);
  return _state;
}

void Client::start() {
  if (_getState() == State::Stopped) {
    _updateState(State::Running);
    _thread = std::thread([this] {
      while (true) {
        if (_poll()) {
          try {
            _handle(_recv());
          } catch (...) {
            ; // if exception is thrown, JSON couldn't parse the bytes or
              // _handle(...) couldn't classify the message
          }
        }
        if (_getState() == State::Stopped) {
          break;
        }
      }
    });
  }
}

void Client::stop() {
  if (_getState() != State::Stopped) {
    _updateState(State::Stopped);
  }
  if (_thread.joinable()) {
    _thread.join();
  }
}

void Client::waitStart() {
  if (_getState() == State::Stopped) {
    start();
  }
  if (Watchdog::shared.haveCrashed()) {
    return;
  }
  _updateState(State::WaitStart);
  std::unique_lock<decltype(_mtx)> l(_mtx);
  _cv.wait(l, [this] { return _state != State::WaitStart; });
}

void Client::waitStop() {
  if (_getState() == State::Stopped) {
    start();
  }
  _updateState(State::WaitStop);
  std::unique_lock<decltype(_mtx)> l(_mtx);
  _cv.wait(l, [this] { return _state != State::WaitStop; });
}

Status Client::status() const { return _status; }

void Client::updateStatus(Status status) {
  auto s2str = [](auto const &s) -> const char * {
    switch (s) {
    case Status::Booting:
      return "BOOTING";
      break;
    case Status::Ready:
      return "READY";
      break;
    case Status::Active:
      return "ACTIVE";
      break;
    case Status::Stopping:
      return "STOPPING";
      break;
    case Status::Finished:
      return "FINISHED";
      break;
    case Status::Error:
      return "ERROR";
      break;
    }
  };

  std::string const str(s2str(status));
  std::ofstream f(_status_path);
  if (!f.fail()) {
    f << str;
    f.close();
  }

  NotificationCenter::shared.post(
      C2APIEvent, C2APIEventInfo{C2APIEventType::UPDATED_STATUS, str});
  _status = status;
}

bool Client::_poll() {
  int const timeout = 500; // [ms] hardcoded polling timeout. this is fine.
  try {
    zmq::poll(&_zpi, 1, timeout);
    if (_zpi.revents & ZMQ_POLLIN) {
      return true;
    } else {
      return false;
    }
  } catch (zmq::error_t &ex) {
    if (ex.num() != EINTR)
      throw;
    return false;
  }
}

// throws if can't parse
nlohmann::json Client::_recv() {
  zmq::message_t msg;
  _sock.recv(&msg);
  std::string msg_str(static_cast<char *>(msg.data()), msg.size());
  auto j = nlohmann::json::parse(msg_str);
  return j;
}

// classify and handle the message
void Client::_handle(nlohmann::json msg) {
  // if we can find our "magic" key in the JSON object, we are confident that
  // this is a start or stop message. if we are in one of the "waiting" states,
  // we now go back to the "running" state
  static std::string const magic = "BAM_INTERNAL_MAGIC";
  if (msg.find(magic) != msg.end()) {
    std::string cmd = msg.at("CMD");
    if ((cmd == "START") && (_getState() == State::WaitStart)) {
      auto const now = std::chrono::system_clock::now();
      _updateState(State::Running);
      Watchdog::shared.setStartTime(now);
      NotificationCenter::shared.post(log::ScenarioStartEvent,
                                      log::ScenarioStartEventInfo{now});
      env._scenario_start_time = now;
    } else if ((cmd == "STOP") && (_getState() == State::WaitStop)) {
      _updateState(State::Running);
    }
    NotificationCenter::shared.post(
        C2APIEvent, C2APIEventInfo{C2APIEventType::RECEIVED_COMMAND, cmd});
    return;
  }

  // if we cannot find our magic string in the JSON object, the message is
  // either an environment update or an outcome update. We can detect a mandated
  // outcome update by the existance of the key "flow_uid" in any of the objects
  if (msg.is_array() && (msg.at(0).find("flow_uid") != msg.at(0).end())) {
    NotificationCenter::shared.post(OutcomesUpdateEvent,
                                    OutcomesUpdateEventInfo{msg});
    return;
  }

  // if we get to here, we try to detect an environment update as last resort.
  if (msg.is_array() && (msg.at(0).find("environment") != msg.at(0).end())) {
    env._from_json(msg);
  }

  if (msg.is_array() &&
      (msg.at(0).find("scenario_center_frequency") != msg.at(0).end())) {
    env._from_json(msg);
  }
}

void Client::init_env(int64_t center_frequency, int64_t rf_bandwidth) {
  env._init(center_frequency, rf_bandwidth, _env_recovery_path,
            _mo_recovery_path);
}

// the global environment object. use this to get access to the colosseum
// environment from anywhere in the program
EnvironmentManager env;

EnvironmentManager::EnvironmentManager()
    : _scenario_start_time(
          std::chrono::duration_cast<std::chrono::system_clock::duration>(
              std::chrono::seconds(T_REF))) {
  // start with a conservative estimate
  _env.push_back(Environment{
      .collab_network_type = Environment::CollabNetworkType::UNSPEC,
      .incumbent_protection = Environment::IncumbentProtection{0, 0},
      .scenario_rf_bandwidth = (int64_t)40e6,
      .scenario_center_frequency = (int64_t)1000e6,
      .bonus_threshold = 0,
      .has_incumbent = false,
      .stage_number = -1,
      .timestamp = INIT_ENV_MAGIC});
}

// Schema available here:
// https://sc2colosseum.freshdesk.com/support/solutions/articles/22000233318-environment-json-schema
void EnvironmentManager::_from_json(nlohmann::json j) {
  auto cnt2enum = [](std::string const &cnt) {
    if (cnt == "INTERNET") {
      return Environment::CollabNetworkType::Internet;
    } else if (cnt == "SATCOM") {
      return Environment::CollabNetworkType::SATCOM;
    } else if (cnt == "HF") {
      return Environment::CollabNetworkType::HF;
    } else {
      return Environment::CollabNetworkType::UNSPEC;
    }
  };

  auto parse_env_json = [&cnt2enum](Environment &ee,
                                    nlohmann::json const &jenv) {
    if (jenv.find("incumbent_protection") != jenv.end()) {
      ee.incumbent_protection = Environment::IncumbentProtection{
          .center_frequency =
              jenv.at("incumbent_protection").at(0).at("center_frequency"),
          .rf_bandwidth =
              jenv.at("incumbent_protection").at(0).at("rf_bandwidth")};
      ee.has_incumbent = true;
    } else {
      ee.has_incumbent = false;
    }
    ee.collab_network_type = cnt2enum(jenv.at("collab_network_type"));
    ee.scenario_rf_bandwidth = jenv.at("scenario_rf_bandwidth");
    ee.scenario_center_frequency = jenv.at("scenario_center_frequency");
    ee.bonus_threshold = jenv.at("scoring_point_threshold");
  };

  Environment e;
  e.raw = j;

  try {
    // this is the version according to the schema. unlikely to work as of
    // <2018-10-25 Thu>.
    auto jenv = j.at(0).at("environment").at(0); // FML
    parse_env_json(e, jenv);
    e.timestamp = j.at(0).at("timestamp");
    e.stage_number = j.at(0).at("stage_number");
  } catch (...) {
    // this is the format that we have seen since scrimmage 6
    try {
      auto jenv = j.at(0);
      parse_env_json(e, jenv);
      e.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
      e.stage_number = _env.back().stage_number + 1;
    } catch (...) {
      // rethrow, the loop will catch item
      throw;
    }
  }

  // if we haven't left this disaster of a function, we have successfully parsed
  // the environment file.
  {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    _env.push_back(e);
  }
  NotificationCenter::shared.post(EnvironmentUpdateEvent, e);
}

// first "Environment" after parsing options
void EnvironmentManager::_init(int64_t center_frequency, int64_t rf_bandwidth,
                               std::string const &env_recovery_path,
                               std::string const &mo_recovery_path) {
  using boost::filesystem::exists;
  using boost::filesystem::path;
  auto json_load = [](auto const &p) {
    std::ifstream i(p);
    nlohmann::json j;
    i >> j;
    return j;
  };

  // if we have crashed, we attempt to recover the environment and the current
  // mandated outcomes from disk
  if (Watchdog::shared.haveCrashed()) {
    // first see whether there are any mandated outcomes
    if (exists(path(mo_recovery_path))) {
      try {
        auto j = json_load(mo_recovery_path);
        NotificationCenter::shared.post(OutcomesUpdateEvent,
                                        OutcomesUpdateEventInfo{j});
      } catch (...) {
        ; // we can't read it, we don't have outcomes
      }
    }
    // now check for the environment. if we get it, we are done.
    if (exists(path(env_recovery_path))) {
      try {
        _from_json(json_load(env_recovery_path));
        return;
      } catch (...) {
        ; // just default to the options file
      }
    }
  }

  // if we haven't crashed, we initialize the environment with the given center
  // frequency and bandwidth
  _env.push_back(Environment{
      .collab_network_type = Environment::CollabNetworkType::UNSPEC,
      .incumbent_protection = Environment::IncumbentProtection{0, 0},
      .scenario_rf_bandwidth = rf_bandwidth,
      .scenario_center_frequency = center_frequency,
      .bonus_threshold = 0,
      .has_incumbent = false,
      .stage_number = 0,
      .timestamp = INIT_ENV_MAGIC});

  {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    NotificationCenter::shared.post(EnvironmentUpdateEvent, _env.back());
  }
}

EnvironmentManager::Environment EnvironmentManager::current() const {
  std::lock_guard<decltype(_mtx)> l(_mtx);
  return _env.back();
}

boost::optional<EnvironmentManager::Environment>
EnvironmentManager::previous() const {
  std::lock_guard<decltype(_mtx)> l(_mtx);
  if (1 >= _env.size()) {
    return boost::none;
  } else {
    return _env[_env.size() - 2];
  }
}

boost::optional<std::chrono::system_clock::time_point>
EnvironmentManager::scenarioStart() const {
  using namespace std::chrono;
  if (duration_cast<seconds>(_scenario_start_time.time_since_epoch()) ==
      seconds(T_REF)) {
    return boost::none;
  } else {
    return _scenario_start_time;
  }
}

#ifndef NDEBUG
void EnvironmentManager::setEnvironment(EnvironmentManager::Environment env) {
  std::lock_guard<decltype(_mtx)> l(_mtx);
  _env.push_back(env);
  NotificationCenter::shared.post(EnvironmentUpdateEvent, _env.back());
}
#endif

} // namespace c2api
} // namespace bamradio
