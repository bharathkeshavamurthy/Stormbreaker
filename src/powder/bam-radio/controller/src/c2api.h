// C2API.
//
// Implements
// https://sc2colosseum.freshdesk.com/support/solutions/articles/22000220460-radio-command-and-control-c2-api
//
// Copyright (c) 2018 Dennis Ogbe
// Copyright (c) 2018 Tomohiro Arakawa

#ifndef b011c956f30b516d6acaa
#define b011c956f30b516d6acaa

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "events.h"

#include "json.hpp"
#include <boost/optional.hpp>
#include <zmq.hpp>

namespace bamradio {
namespace c2api {

// see the wiki for the different possible SRN stati
enum class Status { Booting, Ready, Active, Stopping, Finished, Error };

//
// The C2API client
//
class Client {
public:
  Client(zmq::context_t &ctx, int port, std::string const &status_path,
         std::string const &env_recovery_path,
         std::string const &mo_recovery_path);
  ~Client();
  Client(Client const &other) = delete;

  void start();
  void stop();

  // this needs to be called after options are initialized.
  void init_env(int64_t center_frequency, int64_t rf_bandwidth);

  // wait for start/stop command
  void waitStart();
  void waitStop();

  // status handling
  void updateStatus(Status status);
  Status status() const;

protected:
  // constants
  int const _port;
  std::string const _status_path;
  std::string const _env_recovery_path;
  std::string const _mo_recovery_path;

  // a "simple" state machine for this
  enum class State { Running, Stopped, WaitStart, WaitStop };
  State _state;
  std::condition_variable _cv;
  std::mutex _mtx;

  void _updateState(State state);
  State _getState();

  // don't confuse _state with _status
  std::atomic<Status> _status;

  // background thread
  std::thread _thread;

  // message handling
  bool _poll();
  nlohmann::json _recv();
  void _handle(nlohmann::json msg);

  // zmq
  zmq::context_t &_ctx;
  zmq::socket_t _sock;
  zmq::pollitem_t _zpi;
};

//
// Globally available Environment information
//
class EnvironmentManager {
public:
  typedef bamradio::EnvironmentUpdateEventInfo Environment;

  Environment current() const;
  boost::optional<Environment> previous() const;
  boost::optional<std::chrono::system_clock::time_point> scenarioStart() const;

  EnvironmentManager();

#ifndef NDEBUG
  // for easier unit testing
  void setEnvironment(Environment env);
#endif

protected:
  friend class Client; // c2api client can call private setter
  // everything here is mutex protected
  mutable std::mutex _mtx;
  // update environment given JSON from colosseum
  void _from_json(nlohmann::json j);
  // initial environment from colosseum
  void _init(int64_t center_frequency, int64_t rf_bandwidth,
             std::string const &env_recovery_path,
             std::string const &mo_recovery_path);
  // the environment
  std::vector<Environment> _env;
  std::chrono::system_clock::time_point _scenario_start_time;
};

extern EnvironmentManager env;

// this is the "timestamp" of the initial environment. Use this to detect
// whether we have received an environment up date or not... fix this hack
// later.q
extern int64_t const INIT_ENV_MAGIC;

} // namespace c2api
} // namespace bamradio

#endif // b011c956f30b516d6acaa
