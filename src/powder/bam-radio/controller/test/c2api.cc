// C2API Test suite
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE c2api
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "c2api.h"
#include "events.h"
#include "notify.h"
#include "watchdog.h"

#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#include "json.hpp"
#include <zmq.hpp>

///////////////////////////////////////////////////////////////////////////////
// Namespaces
///////////////////////////////////////////////////////////////////////////////

namespace fs = boost::filesystem;
namespace utf = boost::unit_test;
namespace timing = std::chrono;

using json = nlohmann::json;
using namespace bamradio;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

int const PORT = 9999;
int const CFREQ = 10;
int const BANDWIDTH = 20;
auto BASEPATH = [] {
  auto p = fs::temp_directory_path();
  p += "/";
  p += fs::unique_path();
  return p;
}();
auto mkpath = [](auto base, auto p) {
  base += p;
  return base;
};
auto const ENVPATH = mkpath(BASEPATH, "/environment.json");
auto const MOPATH = mkpath(BASEPATH, "/mo.json");
auto const STATUSPATH = mkpath(BASEPATH, "/status.json");

// manage a temp directory
struct TempDirManage {
  TempDirManage() { fs::create_directories(BASEPATH); }
  ~TempDirManage() {
    if (fs::exists(BASEPATH)) {
      fs::remove_all(BASEPATH);
    }
  }
};

BOOST_GLOBAL_FIXTURE(TempDirManage);

///////////////////////////////////////////////////////////////////////////////
// JSON data for fuzzing
///////////////////////////////////////////////////////////////////////////////

json START_MSG =
    "{\"BAM_INTERNAL_MAGIC\":\"DRAGON ENERGY\", \"CMD\": \"START\"}"_json;
json STOP_MSG =
    "{\"BAM_INTERNAL_MAGIC\":\"DRAGON ENERGY\", \"CMD\": \"STOP\"}"_json;
json RANDOM_JSON =
    "[{\"This is random data\": 155324234, \"package\": \"dennis\"}, 15, 20, 167]"_json;
json EXAMPLE_MANDATED_OUTCOMES_S1 =
    "[{\"flow_uid\":15001,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15002,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}},{\"flow_uid\":15003,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15004,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15005,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}}]"_json;
json EXAMPLE_MANDATED_OUTCOMES_S2 =
    "[{\"flow_uid\":15001,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15002,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}},{\"flow_uid\":15003,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15004,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15005,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.37,\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":40560.0}},{\"flow_uid\":15006,\"goal_type\":\"Traffic\",\"requirements\":{\"file_size_bytes\":655360,\"file_transfer_deadline_s\":10.0,\"max_packet_drop_rate\":0.0}},{\"flow_uid\":15007,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}},{\"flow_uid\":15008,\"goal_type\":\"Traffic\",\"requirements\":{\"max_packet_drop_rate\":0.1,\"min_throughput_bps\":937440.0}},{\"flow_uid\":15009,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":0.12,\"max_packet_drop_rate\":0.0,\"max_throughput_bps\":12480.0}},{\"flow_uid\":15010,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}},{\"flow_uid\":15011,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}},{\"flow_uid\":15012,\"goal_type\":\"Traffic\",\"requirements\":{\"max_latency_s\":1.0,\"max_packet_drop_rate\":0.5,\"min_throughput_bps\":520.0}}]"_json;
json EXAMPLE_ENVIRONMENT_UPDATE =
    "[{\"scenario_center_frequency\": 1000000000, \"collab_network_type\": \"INTERNET\", \"scenario_rf_bandwidth\": 40000000, \"scoring_point_threshold\": 20, \"incumbent_protection\": [{\"rf_bandwidth\": 40000000, \"center_frequency\": 1000000000}]}]"_json;
std::string MALFORMED_JSON =
    "[{\"This is malformed\": 155324234, \"package\": dennis\"}, 15.5, 20, 167";
std::string RANDOM_BYTES = "a034f70c31880c0657a25afe8571628521a8c40ac6947cf903e"
                           "4c6c71e89c264f22147c789650f1c";

///////////////////////////////////////////////////////////////////////////////
// Test classes
///////////////////////////////////////////////////////////////////////////////

class c2api_test_server {
public:
  c2api_test_server() : ctx(1), sock(ctx, ZMQ_PUSH) {
    // bind to socket / setsockopt
    int i = 5;
    while (i-- > 0) {
      try {
        sock.connect(
            (boost::format("tcp://%1%:%2%") % "127.0.0.1" % PORT).str());
        break;
      } catch (...) {
        ; // connection failed, retry until we can't no more
      }
    }
    if (i <= 0) {
      throw std::runtime_error("Connection Timeout");
    }
    int linger = 0;
    sock.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
  }

  ~c2api_test_server() { sock.close(); }

  bool send(std::string const &bytes) {
    zmq::message_t msg(bytes.size());
    memcpy(msg.data(), bytes.data(), bytes.size());
    bool sent = false;
    while (!sent) {
      try {
        sent = sock.send(msg);
      } catch (zmq::error_t &ex) {
        if (ex.num() != EINTR)
          throw;
        sent = false;
      }
    }
    return sent;
  }

  // zmq
  zmq::context_t ctx;
  zmq::socket_t sock;
};

class c2api_test_client : public c2api::Client {
public:
  typedef c2api::Client::State State;
  c2api_test_client(zmq::context_t &ctx)
      : Client(ctx, PORT, STATUSPATH.native(), ENVPATH.native(),
               MOPATH.native()) {}
  c2api::Client::State getState() { return _getState(); }
  void updateState(c2api::Client::State state) { _updateState(state); }
};

auto write_file(std::string const &str, fs::path file) -> void {
  std::ofstream f(file.native());
  f << str;
  f.close();
}

///////////////////////////////////////////////////////////////////////////////
// Test cases
///////////////////////////////////////////////////////////////////////////////

//
// START/STOP messages. in bamradio.cc, we need to wait for start and stop
// messages. these tests look at this
//

void wait_StartStop(c2api_test_client::State const expected) {
  using State = c2api_test_client::State;

  Watchdog::shared.debug_set_havecrashed(false);

  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);

  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start(); // non-blocking
  BOOST_REQUIRE(cl.getState() == State::Running);

  // have the waiting thread wait for the start command. make sure we are in the
  // correct state
  std::thread wait([&] {
    // blocking
    if (expected == State::WaitStart) {
      cl.waitStart();
    } else {
      cl.waitStop();
    }
  });
  std::this_thread::sleep_for(
      timing::milliseconds(100)); // wait until the thread starts
  BOOST_REQUIRE(cl.getState() == expected);

  // wait 1 second and send the corresponding message
  std::this_thread::sleep_for(timing::seconds(1));
  if (expected == State::WaitStart) {
    srv.send(START_MSG.dump());
  } else {
    srv.send(STOP_MSG.dump());
  }

  // wait until we are done, then should be in Running state again
  wait.join();
  BOOST_REQUIRE(cl.getState() == State::Running);

  // test stop as well
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

BOOST_AUTO_TEST_CASE(wait_start, *utf::timeout(5)) {
  wait_StartStop(c2api_test_client::State::WaitStart);
}

BOOST_AUTO_TEST_CASE(wait_stop, *utf::timeout(5)) {
  wait_StartStop(c2api_test_client::State::WaitStop);
}

// if we recover from crash, we need to skip the waitStart waiting.
BOOST_AUTO_TEST_CASE(test_waitstart_have_crashed, *utf::timeout(5)) {
  using State = c2api_test_client::State;

  Watchdog::shared.debug_set_havecrashed(true);

  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);

  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start(); // non-blocking
  BOOST_REQUIRE(cl.getState() == State::Running);

  std::thread wait([&] { cl.waitStart(); });
  std::this_thread::sleep_for(timing::milliseconds(100));
  BOOST_REQUIRE(cl.getState() == State::Running);
  wait.join();
  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

// What happens when we send messages other than START or STOP while waiting?
void wait_StartStop_fuzz(c2api_test_client::State const expected,
                         std::string const &msg) {
  using State = c2api_test_client::State;

  Watchdog::shared.debug_set_havecrashed(false);

  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);

  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start(); // non-blocking
  BOOST_REQUIRE(cl.getState() == State::Running);

  // have the waiting thread wait for the start command. make sure we are in the
  // correct state
  std::thread wait([&] {
    // blocking
    if (expected == State::WaitStart) {
      cl.waitStart();
    } else {
      cl.waitStop();
    }
  });
  std::this_thread::sleep_for(
      timing::milliseconds(100)); // wait until the thread starts
  BOOST_REQUIRE(cl.getState() == expected);

  // wait a little and send the random message. we should not have changed state
  std::this_thread::sleep_for(timing::milliseconds(50));
  srv.send(msg);
  std::this_thread::sleep_for(timing::milliseconds(50));
  BOOST_REQUIRE(cl.getState() == expected);

  // send the expected message
  if (expected == State::WaitStart) {
    srv.send(START_MSG.dump());
  } else {
    srv.send(STOP_MSG.dump());
  }

  wait.join();
  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

// try a bunch of inputs for both the waitStart and waitStop functions.
BOOST_AUTO_TEST_CASE(wait_startstop_fuzz, *utf::timeout(60)) {
  using State = c2api_test_client::State;
  for (auto const &s : {State::WaitStart, State::WaitStop}) {
    auto othermsg = s == State::WaitStart ? STOP_MSG : START_MSG;
    for (auto const &msg :
         {RANDOM_JSON.dump(), EXAMPLE_MANDATED_OUTCOMES_S1.dump(),
          EXAMPLE_MANDATED_OUTCOMES_S2.dump(), othermsg.dump(),
          EXAMPLE_ENVIRONMENT_UPDATE.dump(), MALFORMED_JSON, RANDOM_BYTES}) {
      wait_StartStop_fuzz(s, msg);
    }
  }
}

//
// Status. Test whether the status file is written correctly
//

void test_status(c2api::Status const status) {
  using Status = c2api::Status;
  using State = c2api_test_client::State;

  auto s2str = [&](auto const &s) -> const char * {
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

  auto check_status_file = [&]() -> bool {
    std::ifstream f(STATUSPATH.native());
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();
    return buffer.str() == s2str(status);
  };

  Watchdog::shared.debug_set_havecrashed(true);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);

  cl.updateStatus(status);
  BOOST_REQUIRE(cl.status() == status);
  BOOST_REQUIRE(check_status_file());

  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

BOOST_AUTO_TEST_CASE(status) {
  using Status = c2api::Status;
  for (auto const &s : {Status::Booting, Status::Ready, Status::Active,
                        Status::Stopping, Status::Finished, Status::Error}) {
    test_status(s);
  }
}

//
// Mandated outcomes. Test whether mandated outcomes are parsed/provided
// correctly
//

void test_mandated_outcome(json const &mo, int n) {
  using State = c2api_test_client::State;

  boost::asio::io_context ioctx;
  int nrx = 0;

  Watchdog::shared.debug_set_havecrashed(false);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);

  auto st = NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
      OutcomesUpdateEvent, ioctx, [&nrx, &mo](auto ei) {
        ++nrx;
        BOOST_REQUIRE(ei == mo);
      });

  for (int i = 0; i < n; ++i) {
    srv.send(mo.dump());
    // fuzz a little
    srv.send(MALFORMED_JSON);
    srv.send(RANDOM_JSON.dump());
    srv.send(START_MSG.dump());
  }

  // wait a little for everything to get where it needs to go
  std::this_thread::sleep_for(timing::milliseconds(n * 50));

  // receive (hopefully) all of the MO updates
  ioctx.run();
  BOOST_REQUIRE_EQUAL(nrx, n);

  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

BOOST_AUTO_TEST_CASE(mandated_outcome_1) {
  test_mandated_outcome(EXAMPLE_MANDATED_OUTCOMES_S1, 10);
}

BOOST_AUTO_TEST_CASE(mandated_outcome_2) {
  test_mandated_outcome(EXAMPLE_MANDATED_OUTCOMES_S2, 10);
}

//
// Environment updates. test whether those are parsed and distributed correctly
//

// C++ is pretty now
template <typename Environment>
auto compare_env(Environment const &e1, Environment const &e2) -> bool {
  bool ret = true;
  ret = e1.collab_network_type == e2.collab_network_type;
  ret = e1.incumbent_protection.center_frequency ==
        e2.incumbent_protection.center_frequency;
  ret = e1.incumbent_protection.rf_bandwidth ==
        e2.incumbent_protection.rf_bandwidth;
  ret = e1.scenario_rf_bandwidth == e2.scenario_rf_bandwidth;
  ret = e1.scenario_center_frequency == e2.scenario_center_frequency;
  ret = e1.has_incumbent == e2.has_incumbent;
  // intentionally not comparing stage number for now
  return ret;
}

BOOST_AUTO_TEST_CASE(env_update) {
  using State = c2api_test_client::State;
  using Environment = c2api::EnvironmentManager::Environment;
  boost::asio::io_context ioctx;
  int const n = 10; // number of env updates to send
  int nrx = 0;

  // the "reference" environment (see the constants above for this)
  auto e = [] {
    Environment e;
    e.collab_network_type = Environment::CollabNetworkType::Internet;
    e.incumbent_protection = Environment::IncumbentProtection{
        .center_frequency = 1000000000, .rf_bandwidth = 40000000};
    e.scenario_rf_bandwidth = 40000000;
    e.scenario_center_frequency = 1000000000;
    e.has_incumbent = true;
    e.stage_number = 0; // be careful comparing this
    e.timestamp = 0;    // no point comparing this
    return e;
  }();

  Watchdog::shared.debug_set_havecrashed(false);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);

  // environment should be initialized to something other than what we have
  // above
  cl.init_env(CFREQ, BANDWIDTH);
  BOOST_REQUIRE(cl.getState() == State::Running);
  BOOST_REQUIRE(!compare_env(e, c2api::env.current()));

  // test both the global environment manager as well as the notification
  auto st = NotificationCenter::shared.subscribe<EnvironmentUpdateEventInfo>(
      EnvironmentUpdateEvent, ioctx, [&e, &nrx](auto ei) {
        ++nrx;
        BOOST_REQUIRE(compare_env(e, ei));
      });

  for (int i = 0; i < n; ++i) {
    srv.send(MALFORMED_JSON);
    srv.send(START_MSG.dump());
    srv.send(EXAMPLE_ENVIRONMENT_UPDATE.dump()); // this is the one
    srv.send(RANDOM_JSON.dump());
    srv.send(MALFORMED_JSON);
    srv.send(EXAMPLE_MANDATED_OUTCOMES_S2.dump());
  }

  std::this_thread::sleep_for(timing::milliseconds(n * 50));
  ioctx.run();
  BOOST_REQUIRE_EQUAL(nrx, n);

  // test the global state
  BOOST_REQUIRE(compare_env(e, c2api::env.current()));

  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

//
// Crash recovery. test whether mandated outcomes and environment are read when
// recovering from crash
//

BOOST_AUTO_TEST_CASE(env_init_no_crash) {
  using State = c2api_test_client::State;
  using Environment = c2api::EnvironmentManager::Environment;

  // the "reference" environment for the case when we call init and have not
  // crashed.
  auto e = [] {
    Environment e;
    e.collab_network_type = Environment::CollabNetworkType::UNSPEC;
    e.incumbent_protection = Environment::IncumbentProtection{0, 0};
    e.scenario_rf_bandwidth = BANDWIDTH;
    e.scenario_center_frequency = CFREQ;
    e.has_incumbent = false;
    e.stage_number = 0; // be careful comparing this
    e.timestamp = 0;    // no point comparing this
    return e;
  }();

  Watchdog::shared.debug_set_havecrashed(false);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);

  cl.init_env(CFREQ, BANDWIDTH);
  BOOST_REQUIRE(cl.getState() == State::Running);
  BOOST_REQUIRE(compare_env(e, c2api::env.current()));
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

BOOST_AUTO_TEST_CASE(env_init_crash) {
  using State = c2api_test_client::State;
  using Environment = c2api::EnvironmentManager::Environment;
  boost::asio::io_context ioctx;

  // write example MO and Environment files
  auto write_json_file = [](auto const &json, auto const &path) {
    write_file(json.dump(), path);
  };

  write_json_file(EXAMPLE_MANDATED_OUTCOMES_S2, MOPATH);
  write_json_file(EXAMPLE_ENVIRONMENT_UPDATE, ENVPATH);

  // the "reference" environment (see the constants above for this)
  auto e = [] {
    Environment e;
    e.collab_network_type = Environment::CollabNetworkType::Internet;
    e.incumbent_protection = Environment::IncumbentProtection{
        .center_frequency = 1000000000, .rf_bandwidth = 40000000};
    e.scenario_rf_bandwidth = 40000000;
    e.scenario_center_frequency = 1000000000;
    e.has_incumbent = true;
    e.stage_number = 0; // be careful comparing this
    e.timestamp = 0;    // no point comparing this
    return e;
  }();

  bool mo_seen = false;
  auto st1 = NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
      OutcomesUpdateEvent, ioctx, [&mo_seen](auto ei) {
        BOOST_REQUIRE(ei == EXAMPLE_MANDATED_OUTCOMES_S2);
        mo_seen = true;
      });
  bool env_seen = false;
  auto st2 = NotificationCenter::shared.subscribe<EnvironmentUpdateEventInfo>(
      EnvironmentUpdateEvent, ioctx, [&env_seen, &e](auto ei) {
        BOOST_REQUIRE(compare_env(e, ei));
        env_seen = true;
      });

  Watchdog::shared.debug_set_havecrashed(true);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);
  BOOST_REQUIRE(!compare_env(e, c2api::env.current()));
  cl.init_env(CFREQ, BANDWIDTH);
  BOOST_REQUIRE(cl.getState() == State::Running);

  // fuzzy fuzz
  for (int i = 0; i < 22; ++i) {
    srv.send(MALFORMED_JSON);
    srv.send(START_MSG.dump());
    srv.send(RANDOM_JSON.dump());
    srv.send(MALFORMED_JSON);
  }

  ioctx.run();
  BOOST_REQUIRE(mo_seen);
  BOOST_REQUIRE(env_seen);
  BOOST_REQUIRE(compare_env(e, c2api::env.current()));

  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}

//
// bash interaction. use system(...) or equivalent to call the radio_api
// scripts and test functionality
//

//#define ZMQ_DEBUG_EXT // if we want to wait for external input instead
BOOST_AUTO_TEST_CASE(bash_env) {
  using State = c2api_test_client::State;
  using Environment = c2api::EnvironmentManager::Environment;

  boost::asio::io_context ioctx;

  // the "reference" environment (see the constants above for this)
  auto e = [] {
    Environment e;
    e.collab_network_type = Environment::CollabNetworkType::Internet;
    e.incumbent_protection = Environment::IncumbentProtection{
        .center_frequency = 1000000000, .rf_bandwidth = 40000000};
    e.scenario_rf_bandwidth = 40000000;
    e.scenario_center_frequency = 1000000000;
    e.has_incumbent = true;
    e.stage_number = 0; // be careful comparing this
    e.timestamp = 0;    // no point comparing this
    return e;
  }();

  Watchdog::shared.debug_set_havecrashed(false);
  c2api_test_server srv;
  c2api_test_client cl(srv.ctx);
  BOOST_REQUIRE(cl.getState() == State::Stopped);
  cl.start();
  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.init_env(CFREQ, BANDWIDTH);
  BOOST_REQUIRE(!compare_env(e, c2api::env.current()));

  bool env_seen = false;
  auto st2 = NotificationCenter::shared.subscribe<EnvironmentUpdateEventInfo>(
      EnvironmentUpdateEvent, ioctx, [&env_seen, &e](auto ei) {
        BOOST_REQUIRE(compare_env(e, ei));
        env_seen = true;
      });

#ifndef ZMQ_DEBUG_EXT
  // fuzzy fuzz
  for (int i = 0; i < 22; ++i) {
    srv.send(MALFORMED_JSON);
    srv.send(START_MSG.dump());
    srv.send(RANDOM_JSON.dump());
    srv.send(MALFORMED_JSON);
  }
#endif
  // write the above and the example environment file to disk
  // auto env_src_file = mkpath(BASEPATH, "/env.json");
  auto env_src_file = mkpath(fs::path("/home/dogbe/test1"), "/env.json");
  write_file(EXAMPLE_ENVIRONMENT_UPDATE.dump(), env_src_file);

  // shell out to python via bash and push to the c2api socket
  auto cmd =
      boost::format("bash -c \'echo \'%1%\' | python -c \"import zmq, sys; "
                    "ctx = zmq.Context(); sock = ctx.socket(zmq.PUSH); "
                    "sock.connect(\\\"tcp://localhost:%2%\\\"); "
                    "sock.send_string(sys.stdin.read())\"\'") %
      env_src_file.native() % PORT;
#ifndef ZMQ_DEBUG_EXT
  auto rc = std::system(cmd.str().c_str());
  BOOST_REQUIRE_EQUAL(rc, 0);
#else
  std::cout << "waiting for external input..." << std::endl;
  std::this_thread::sleep_for(timing::seconds(10));
#endif
  ioctx.run();
  std::this_thread::sleep_for(timing::milliseconds(100));

  BOOST_REQUIRE(env_seen);
  BOOST_REQUIRE(compare_env(e, c2api::env.current()));
  BOOST_REQUIRE(cl.getState() == State::Running);
  cl.stop();
  BOOST_REQUIRE(cl.getState() == State::Stopped);
}
