// DE Tests.
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE de
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <random>

#include "bandwidth.h"
#include "de.h"
#include "log.h"
#include "notify.h"
#include "test_extra.h"

using namespace bamradio;
using namespace bamradio::decisionengine;
using namespace std::chrono_literals;

class fake_radiocontroller : public AbstractRadioController {
public:
  fake_radiocontroller()
      : _ccData(std::make_shared<controlchannel::CCData>(0, 0.1, true)) {}

  std::vector<Channel> ctrlChannelAlloc() { return _ctrlChannelAlloc; };

  controlchannel::CCData::sptr ccData() const { return _ccData; }

  ofdm::DataLinkLayer::sptr ofdmDll() const { return nullptr; }

  void start(){};
  void stop(){};

private:
  std::vector<Channel> _ctrlChannelAlloc;
  controlchannel::CCData::sptr _ccData;
};

class test_decision_engine : public DecisionEngine {
public:
  typedef std::shared_ptr<test_decision_engine> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<test_decision_engine>(std::forward<Args>(args)...);
  }
  test_decision_engine(zmq::context_t &ctx,
                       collab::CollabClient::ConnectionParams ccparams,
                       AbstractRadioController::sptr radioCtrl,
                       Options const &opts)
      : DecisionEngine(ctx, ccparams, radioCtrl, opts) {
    lisp.run([&ccparams] { decisionengine::populateTables(ccparams); });
  }
  // why did I make these methods protected in the first place?
  void step(Trigger t) {
    lisp.run([this, &t] {
      _lispStep(t, std::chrono::system_clock::now(), _stepCount++);
    });
  }
  void fakeStepSetup() {
    _currentTrigger = Trigger::PeriodicStep;
    _currentTriggerTime = std::chrono::system_clock::now();
    _channel_updated = false;
  }
  NetworkInfo &addPeer(NetworkID id, NetworkType type) {
    return _addPeer(id, type);
  }
  void trackSRN(SRNID id) { _trackSRN(id); };
  decltype(_peers) &peers() { return _peers; }
  decltype(_srns) &srns() { return _srns; }
  decltype(_myNetwork) &myNetwork() { return _myNetwork; }

  decltype(_offeredMandates) &offeredMandates() { return _offeredMandates; }
  decltype(_environment) &environment() { return _environment; }
  decltype(_mandatePerformance) &mandatePerformance() {
    return _mandatePerformance;
  }
  decltype(_collabClient) collabClient() { return _collabClient; }
};

// some testing infrastructure
zmq::context_t *zctx = nullptr;
log::Logger *logger = nullptr;
test_decision_engine::sptr de = nullptr;

// spit out a fresh engine
test_decision_engine::sptr makeEngine() {
  using ip = boost::asio::ip::address_v4;
  collab::CollabClient::ConnectionParams collab{
      .server_ip = ip::from_string("172.30.0.111"),
      .client_ip = ip::from_string("172.30.0.2"),
      .server_port = 6000,
      .client_port = 6001,
      .peer_port = 6002};
  test_decision_engine::Options opt{.gatewayID = NodeID(0),
                                    .step_period = 500ms,
                                    .channel_alloc_delay = 100ms,
                                    .cil_broadcast_period = 5s,
                                    .data_tx_gain = 20,
                                    .control_tx_gain = 20,
                                    .sample_rate = bam::dsp::sample_rate,
                                    .guard_band = 100e3,
                                    .max_wf = "DFT_S_OFDM_128_500K",
                                    .psd_hist_params =
                                        psdsensing::PSDSensing::HistParams{
                                            .bin_size = 0.2,
                                            .empty_bin_thresh = 2,
                                            .sn_gap_bins = 30,
                                            .avg_len = 5,
                                            .noise_floor = -70,
                                        }};
  auto rc = std::make_shared<fake_radiocontroller>();
  return test_decision_engine::make(*zctx, collab, rc, opt);
}

struct test_infrastructure {
  test_infrastructure() {
    using Environment = c2api::EnvironmentManager::Environment;
    auto fake_env = Environment{
        .collab_network_type = Environment::CollabNetworkType::Internet,
        .incumbent_protection =
            Environment::IncumbentProtection{(int64_t)2.9e9, (int64_t)2e6},
        .scenario_rf_bandwidth = (int64_t)25e6,
        .scenario_center_frequency = (int64_t)2.4e9,
        .bonus_threshold = 10,
        .has_incumbent = true,
        .stage_number = 1,
        .timestamp = 100,
        .raw = {"json", "raw"}};
    c2api::env.setEnvironment(fake_env);
    zctx = new zmq::context_t(1);
    logger = new log::Logger();
    logger->enableBackend(log::Backend::STDOUT);
    logger->setStartTime(std::chrono::system_clock::now());
    de = makeEngine();
  }
  ~test_infrastructure() {
    de = nullptr;
    delete zctx;
    delete logger;
  }
};

BOOST_GLOBAL_FIXTURE(test_infrastructure);

// simple functionality (start up, don't crash)
BOOST_AUTO_TEST_CASE(de_start) {
  log::text("*** Testing construction + a few periodic steps...");
  de->start();
  std::this_thread::sleep_for(3s);
  de->stop();
}

// test moving some POD to lisp
BOOST_AUTO_TEST_CASE(pod_to_lisp) {
  log::text("*** Testing basic data conversion...");
  // convert a location to lisp and print it
  auto loc = Location{10.0, 20.0, 30.0};
  de->lisp.run([&] { lisp::Print(toLisp(loc)); });
  std::cout << std::endl << std::endl;
}

// log a string from lisp using logger framework
BOOST_AUTO_TEST_CASE(log_from_lisp) {
  log::text("*** Testing FFI...");
  // call a LISP function (see SAY-HELLO indebug.lisp) that in turn calls our
  // logging framework
  de->lisp.run([] { lisp::Funcall(lisp::Symbol("say-hello", "bam-radio")); });
}

// log binary data to the data base
BOOST_AUTO_TEST_CASE(log_binary) {
  log::text("*** Testing binary logging + advanced conversion");

  using decisionengine::toLisp;
  using lisp::Funcall;
  using lisp::Keyword;
  using lisp::Symbol;
  using lisp::toLisp;

  auto now = std::chrono::system_clock::now();

  // set the environment to something
  using Environment = c2api::EnvironmentManager::Environment;
  auto fake_env = Environment{
      .collab_network_type = Environment::CollabNetworkType::Internet,
      .incumbent_protection =
          Environment::IncumbentProtection{(int64_t)2.9e9, (int64_t)2e6},
      .scenario_rf_bandwidth = (int64_t)25e6,
      .scenario_center_frequency = (int64_t)2.4e9,
      .bonus_threshold = 10,
      .has_incumbent = true,
      .stage_number = 1,
      .timestamp = 100,
      .raw = {"json", "raw"}};

  // get some fake mandates + performance report
  auto fake_mp = [] {
    std::map<FlowUID, MandateInfo> mp;
    mp[1] = MandateInfo{
        .mandate =
            stats::IndividualMandate{.hold_period = 100,
                                     .point_value = 200,
                                     .has_max_latency_s = true,
                                     .max_latency_s = 10,
                                     .has_min_throughput_bps = true,
                                     .min_throughput_bps = 1e6,
                                     .has_file_transfer_deadline_s = false,
                                     .file_transfer_deadline_s = -1.0f,
                                     .has_file_size_bytes = false,
                                     .file_size_bytes = 0,
                                     .im_type = stats::IMType::UNKNOWN},
        .activeSRNs = {},
        .performance = stats::FlowPerformance{.mps = 1000,
                                              .point_value = 50,
                                              .scalar_performance = 0.7},
        .endpoints = stats::FlowInfo{.available = true, .src = 1, .dst = 0}};
    mp[2] = MandateInfo{
        .mandate =
            stats::IndividualMandate{.hold_period = 200,
                                     .point_value = 400,
                                     .has_max_latency_s = true,
                                     .max_latency_s = 20,
                                     .has_min_throughput_bps = true,
                                     .min_throughput_bps = 2e6,
                                     .has_file_transfer_deadline_s = false,
                                     .file_transfer_deadline_s = -2.0f,
                                     .has_file_size_bytes = false,
                                     .file_size_bytes = 0,
                                     .im_type = stats::IMType::UNKNOWN},
        .activeSRNs = {},
        .performance = stats::FlowPerformance{.mps = 2000,
                                              .point_value = 100,
                                              .scalar_performance = 1.4},
        .endpoints = stats::FlowInfo{.available = true, .src = 2, .dst = 0}};
    return mp;
  }();

  // write these results to a database
  std::string const dbfn = "bamradio-de-tests.db";
  test::deletefile(dbfn);
  logger->enableBackend(log::Backend::SQL, dbfn, true);

  auto trigger = Trigger::PeriodicStep;
  // print the current time stamp
  if (false) {
    using namespace std::chrono;
    nanoseconds const NSSinceEpoch(now.time_since_epoch());
    seconds const fullSecSinceEpoch(
        duration_cast<seconds>(now.time_since_epoch()));
    nanoseconds const fracNS(NSSinceEpoch - nanoseconds(fullSecSinceEpoch));
    std::cout << "now: {sec: " << fullSecSinceEpoch.count()
              << " nsec: " << fracNS.count() << "}" << std::endl;
  };

  de->fakeStepSetup();

  // make a dummy input object and run a step, similar to _lispStep
  de->lisp.run([&] {
    // get a fake passive incumbent
    auto const fake_incumbent = [&fake_env, &now] {
      auto const last_message = toLisp(PassiveIncumbentMessage{
          .type = PassiveIncumbentMessage::Type::Report,
          .incumbentID = 10,
          .reportTime = now,
          .power = 20,
          .threshold = 100,
          .centerFreq = fake_env.incumbent_protection.center_frequency,
          .bandwidth = fake_env.incumbent_protection.rf_bandwidth,
          .thresholdExceeded = false});
      // clang-format off
    return Funcall(
        Symbol("make-instance"), BRSymbol("passive-incumbent"),
        Keyword("offset"), toLisp(fake_env.incumbent_protection.center_frequency
				  - fake_env.scenario_center_frequency),
        Keyword("bandwidth"), toLisp(fake_env.incumbent_protection.rf_bandwidth),
        Keyword("last-message"), last_message);
      // clang-format on
    }();

    // get a fake node
    auto const fake_node = [](auto id, auto chan) {
      auto const loc = toLisp(controlchannel::Location{30.0, 40.0, 50.0});
      auto const tx_assignment =
          toLisp(TransmitAssignment{.bw_idx = 3,
                                    .chan_idx = (uint8_t)chan,
                                    .chan_ofst = 0,
                                    .atten = 20.0f,
                                    .silent = false});
      std::vector<float> psddd = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
      auto const psdd = psdsensing::PSDData{
          .src_srnid = 0,
          .time_ns = 1,
          .psd = std::make_shared<decltype(psddd)>(psddd),
      };
      auto const real_psd = toLisp(psdd);
      auto const thresh_psd =
          threshPSDtoLisp(psdd, de->options.psd_hist_params);
      // clang-format off
      auto const lobj = Funcall(Symbol("make-instance"), BRSymbol("internal-node"),
                                Keyword("id"), toLisp(id),
                                Keyword("location"), loc,
                                Keyword("tx-assignment"), tx_assignment,
                                Keyword("est-duty-cycle"), toLisp(0.6f),
                                Keyword("real-psd"), real_psd,
                                Keyword("thresh-psd"), thresh_psd);
      // clang-format on
      return lobj;
    };

    auto data =
        Funcall(Symbol("make-instance"), BRSymbol("decision-engine-input"),
                // the trigger
                Keyword("trigger"), toLisp(trigger),
                // the current time
                Keyword("time-stamp"), toLisp(now),
                // the environment
                Keyword("env"), toLisp(fake_env),
                // mandate information
                Keyword("mandates"), toLisp(fake_mp),
                // node information
                Keyword("nodes"),
                lisp::List(fake_node(0, 1), fake_node(1, 2), fake_node(2, 3)),
                // competitor information
                Keyword("collaborators"), lisp::nil, // TODO
                // incumbent information
                Keyword("incumbent"), fake_incumbent);
    Funcall(BRSymbol("decision-engine-step"), data);
  });

  std::this_thread::sleep_for(1s);
  // the above result should have ended up in the data base.
  logger->disableBackend(log::Backend::SQL);
  log::text(boost::format("Check the database %1%") % dbfn);
}

BOOST_AUTO_TEST_CASE(crash, *boost::unit_test::disabled()) {
  log::text("*** Testing lisp crash + restart");

  auto printMPs = [](auto const &mps) {
    std::cout << "{" << std::endl;
    for (auto const &mm : mps) {
      auto const id = mm.first;
      auto const m = mm.second.mandate;
      auto const p = mm.second.performance;
      auto const ep = mm.second.endpoints;
      // clang-format off
      std::cout << " "
                << " id: " << id
                << " hp: " << m.hold_period
                << " pv: " << m.point_value
                << " hml: " << m.has_max_latency_s
                << " hmt: " << m.has_min_throughput_bps
                << " mt: " << m.min_throughput_bps
                << " hftd: " << m.has_file_transfer_deadline_s
                << " ftd: " << m.file_transfer_deadline_s
                << " hfs: " << m.has_file_size_bytes
                << " fs: " << m.file_size_bytes
                << " mps: " << p->mps
                << " pvv: " << p->point_value
                << " sp: " << p->scalar_performance
                << " src: "<< (int)ep->src // todo: make sure this is no funny business
                << " dst: " << (int)ep->dst << std::endl;
      // clang-format on
    }
    std::cout << "}" << std::endl;
  };

  using decisionengine::toLisp;
  using lisp::Funcall;
  using lisp::Keyword;
  using lisp::Symbol;
  using lisp::toLisp;

  // write 10000 random objects to the database
  auto nstep = 10000;

  std::mt19937 rng1(33);
  std::mt19937 rng2(100);
  std::mt19937 rng3(103204134);
  std::uniform_int_distribution<int64_t> int_dist(0, (uint64_t)5e9);
  std::uniform_real_distribution<double> float_dist(0, 5e9);
  std::uniform_real_distribution<float> sf_dist(0, 20);
  std::uniform_real_distribution<double> bool_dist(0, 1);

  auto random_int = [&] { return int_dist(rng1); };
  auto random_float = [&] { return float_dist(rng2); };
  auto random_small_sf = [&] { return sf_dist(rng2); };
  auto random_bool = [&] { return bool_dist(rng3) > 0.5; };

  // write these results to a database
  std::string const dbfn = "bamradio-de-tests.db";
  test::deletefile(dbfn);
  logger->enableBackend(log::Backend::SQL, dbfn, true);

  for (size_t istep = 0; istep < nstep; ++istep) {

    auto now = std::chrono::system_clock::now();

    // set the environment to something
    using Environment = c2api::EnvironmentManager::Environment;

    auto fake_env = Environment{
        .collab_network_type = Environment::CollabNetworkType::Internet,
        .incumbent_protection =
            Environment::IncumbentProtection{random_int(), random_int()},
        .scenario_rf_bandwidth = (int64_t)20e6,
        .scenario_center_frequency = random_int(),
        .bonus_threshold = (uint32_t)random_int(),
        .has_incumbent = random_bool(),
        .stage_number = random_int(),
        .timestamp = random_int(),
        .raw = {"json", "raw"}};

    // get some fake mandates + performance report
    auto fake_mp = [&] {
      std::map<FlowUID, MandateInfo> mp;
      for (size_t i = 0; i < 20; ++i) {
        mp[i] = MandateInfo{
            .mandate =
                stats::IndividualMandate{
                    .hold_period = (unsigned int)random_int(),
                    .point_value = (unsigned int)random_int(),
                    .has_max_latency_s = random_bool(),
                    .max_latency_s = (float)random_float(),
                    .has_min_throughput_bps = random_bool(),
                    .min_throughput_bps = (float)random_float(),
                    .has_file_transfer_deadline_s = random_bool(),
                    .file_transfer_deadline_s = (float)random_float(),
                    .has_file_size_bytes = random_bool(),
                    .file_size_bytes = (size_t)random_int(),
                    .im_type = stats::IMType::UNKNOWN},
            .activeSRNs = {},
            .performance =
                stats::FlowPerformance{
                    .mps = (unsigned int)random_int(),
                    .point_value = (unsigned int)random_int(),
                    .scalar_performance = (float)random_float()},
            .endpoints = stats::FlowInfo{.available = true,
                                         .src = (NodeID)random_int(),
                                         .dst = (NodeID)random_int()}};
      }
      return mp;
    }();
    auto trigger = Trigger::PeriodicStep;

    auto incmsg = PassiveIncumbentMessage{
        .type = PassiveIncumbentMessage::Type::Report,
        .incumbentID = (int32_t)random_int(),
        .reportTime = now,
        .power = random_float(),
        .threshold = random_float(),
        .centerFreq = fake_env.incumbent_protection.center_frequency,
        .bandwidth = fake_env.incumbent_protection.rf_bandwidth,
        .thresholdExceeded = random_bool()};

    de->fakeStepSetup();

    // make a dummy input object and run a step, similar to _lispStep
    de->lisp.run([&] {
      // get a fake passive incumbent
      auto const fake_incumbent = [&incmsg, &fake_env] {
        auto const last_message = toLisp(incmsg);
        auto const offset = fake_env.incumbent_protection.center_frequency -
                            fake_env.scenario_center_frequency;
        // clang-format off
        return Funcall(Symbol("make-instance"), BRSymbol("passive-incumbent"),
                       Keyword("offset"), toLisp(offset),
                       Keyword("bandwidth"), toLisp(fake_env.incumbent_protection.rf_bandwidth),
                       Keyword("last-message"), last_message);
        // clang-format on
      }();

      // get a fake node
      auto const fake_nodes = [&] {
        auto list = lisp::List();
        for (size_t i = 0; i < 10; ++i) {
          auto const id = toLisp(random_int());
          auto const loc = toLisp(controlchannel::Location{
              random_float(), random_float(), random_float()});
          auto const tx_assignment =
              toLisp(TransmitAssignment{.bw_idx = 3,
                                        .chan_idx = 0,
                                        .chan_ofst = (int32_t)random_int(),
                                        .atten = (float)random_float(),
                                        .silent = false});
          std::vector<float> psddd = {random_small_sf(), random_small_sf(),
                                      random_small_sf(), random_small_sf(),
                                      random_small_sf(), random_small_sf()};
          auto const psdd = psdsensing::PSDData{
              .src_srnid = (NodeID)random_int(),
              .time_ns = (uint64_t)random_int(),
              .psd = std::make_shared<decltype(psddd)>(psddd),
          };
          auto const real_psd = toLisp(psdd);
          auto const thresh_psd =
              threshPSDtoLisp(psdd, de->options.psd_hist_params);
          // clang-format off
          lisp::Push(Funcall(Symbol("make-instance"), BRSymbol("internal-node"),
                             Keyword("id"), id,
                             Keyword("location"), loc,
                             Keyword("tx-assignment"), tx_assignment,
                             Keyword("est-duty-cycle"), toLisp(0.6f),
                             Keyword("real-psd"), real_psd,
                             Keyword("thresh-psd"), thresh_psd),
                     list);
          // clang-format on
        }
        return list;
      }();

      printMPs(fake_mp);
      auto data =
          Funcall(Symbol("make-instance"), BRSymbol("decision-engine-input"),
                  // the trigger
                  Keyword("trigger"), toLisp(trigger),
                  // the current time
                  Keyword("time-stamp"), toLisp(now),
                  // the environment
                  Keyword("env"), toLisp(fake_env),
                  // mandate information
                  // Keyword("mandates"), lisp::nil,
                  Keyword("mandates"), toLisp(fake_mp),
                  // node information
                  Keyword("nodes"), fake_nodes,
                  // competitor information
                  Keyword("collaborators"), lisp::nil, // TODO
                  // incumbent information
                  Keyword("incumbent"), fake_incumbent);

      std::cout << "istep: " << istep << std::endl;
      Funcall(BRSymbol("decision-engine-step"), data);
    });
  }
  logger->disableBackend(log::Backend::SQL);
}

BOOST_AUTO_TEST_CASE(bw_table) {
  using lisp::Funcall;
  using lisp::Symbol;
  using lisp::Value;
  using lisp::fromInt;
  using lisp::toLisp;
  de->lisp.run([] {
    auto tbl = bam::dsp::SubChannel::table();
    for (size_t i = 0; i < tbl.size(); ++i) {
      auto lbw = Funcall(Symbol("gethash"), toLisp(i),
                         Value(BRSymbol("*bandwidths*")));
      BOOST_REQUIRE_EQUAL(tbl[i].bw(), fromInt(lbw));
    }
  });
}

BOOST_AUTO_TEST_CASE(full_step) {
  std::mt19937 rng1(33);
  std::mt19937 rng2(100);
  std::mt19937 rng3(103204134);
  std::uniform_int_distribution<int64_t> int_dist(0, (uint64_t)5e9);
  std::uniform_real_distribution<double> float_dist(0, 5e9);
  std::uniform_real_distribution<float> sf_dist(0, 20);
  std::uniform_real_distribution<double> bool_dist(0, 1);

  auto random_int = [&] { return int_dist(rng1); };
  auto random_float = [&] { return float_dist(rng2); };
  auto random_small_sf = [&] { return sf_dist(rng2); };
  auto random_bool = [&] { return bool_dist(rng3) > 0.5; };

  // write these results to a database
  std::string const dbfn = "bamradio-de-tests.db";
  test::deletefile(dbfn);
  logger->enableBackend(log::Backend::SQL, dbfn, true);

  auto const env = c2api::env.current();

  size_t nstep = 100;
  for (size_t istep = 0; istep < nstep; ++istep) {
    // track some fake data and do some steps
    auto const newPeer = boost::asio::ip::address_v4(random_int());
    auto ni = de->addPeer(newPeer, NetworkType::Competitor);
    SpectrumUsage su;
    // add some srns and add some frequency bands for them
    std::vector<CompetitorSRNID> srns(8);
    std::generate(begin(srns), end(srns), [&] {
      auto const id = random_int();
      ni.addSRN(id);
      return id;
    });
    std::uniform_int_distribution<size_t> idx(0, srns.size() - 1);
    std::uniform_int_distribution<size_t> howm(1, srns.size());
    std::uniform_real_distribution<double> freq(
        (double)(env.scenario_center_frequency - env.scenario_rf_bandwidth / 2),
        (double)(env.scenario_center_frequency +
                 env.scenario_rf_bandwidth / 2));
    for (size_t i = 0; i < 10; ++i) {
      // random frequency band
      auto const b1 = freq(rng1);
      auto const b2 = freq(rng2);
      FrequencyBand fb{.lower = std::min(b1, b2), .upper = std::max(b1, b2)};
      // random transmitter rand receivers
      CompetitorSRNID tx = srns[idx(rng3)];
      decltype(srns) rx;
      for (size_t j = 0; j < howm(rng1); ++j) {
        rx.push_back(srns[idx(rng2)]);
      }
      auto const now = std::chrono::system_clock::now();
      su.push_back(FrequencyBandUsage{
          .band = fb,
          .transmitter = tx,
          .receivers = rx,
          .txPowerdB = random_float(),
          .start = now - 10s,
          .end = now + 50s,
      });
    }
    de->peers().at(newPeer).spectrumUsage.track(su);
    de->step(Trigger::PeriodicStep);
  }

  logger->disableBackend(log::Backend::SQL);
}

// some transmitassignment sanity checks

bool tx_assignment_eql(TransmitAssignment const &a,
                       TransmitAssignment const &b) {
  BOOST_REQUIRE_EQUAL(a.bw_idx, b.bw_idx);
  BOOST_REQUIRE_EQUAL(a.chan_idx, b.chan_idx);
  BOOST_REQUIRE_EQUAL(a.chan_ofst, b.chan_ofst);
  BOOST_REQUIRE(a.atten == b.atten);
  BOOST_REQUIRE_EQUAL(a.silent, b.silent);
}

BOOST_AUTO_TEST_CASE(transmit_assignment, *boost::unit_test::tolerance(0.01)) {
  // our test struct
  TransmitAssignment const test{.bw_idx = 2,
                                .chan_idx = 1,
                                .chan_ofst = 13,
                                .atten = 2.0f,
                                .silent = false};
  // (1) to and from protobuf
  auto const proto = test.toProto();
  auto const from_proto = TransmitAssignment::fromProto(proto);
  tx_assignment_eql(from_proto, test);

  // (2) to and from lisp
  de->lisp.run([&] {
    auto const lisp = toLisp(test);
    auto const from_lisp = TransmitAssignment::fromLisp(lisp);
    tx_assignment_eql(from_lisp, test);
  });
}

// test the channelization table conversion
BOOST_AUTO_TEST_CASE(channelization_table) {
  using namespace lisp;
  de->lisp.run([] {
    for (auto const &t : Channelization::table) {
      // the true values
      auto const true_env_bw = t.first;
      auto const true_max_bw = t.second.max_bw_idx;
      auto const true_cfreqs = t.second.center_offsets;
      // the lisp values
      auto pair = Funcall(Symbol("gethash"), toLisp(true_env_bw),
                          Value(BRSymbol("*channelization*")));
      auto const lisp_max_bw = fromInt(Car(pair));
      auto const lisp_cfreqs = map(Cdr(pair), fromInt);
      // compare
      BOOST_REQUIRE_EQUAL(true_max_bw, lisp_max_bw);
      BOOST_REQUIRE(true_cfreqs == lisp_cfreqs);
    }
  });
}

BOOST_AUTO_TEST_CASE(ip_address) {
  using namespace lisp;
  de->lisp.run([] {
    auto const expected = de->collabClient()->client_ip();
    auto const saved_string =
        fromString(Cdr(Value(BRSymbol("*my-network-id*"))));
    auto const saved_ip =
        boost::asio::ip::address_v4::from_string(saved_string);
    BOOST_REQUIRE(expected == saved_ip);
  });
}

// test TxAssignmentUpdate
BOOST_AUTO_TEST_CASE(transmit_assignment_update,
                     *boost::unit_test::tolerance(0.01)) {
  using namespace lisp;
  // test data
  TransmitAssignment::Map const test{
      {1,
       {.bw_idx = 2,
        .chan_idx = 1,
        .chan_ofst = 13,
        .atten = 2.0f,
        .silent = false}},
      {5,
       {.bw_idx = 2,
        .chan_idx = 5,
        .chan_ofst = 14,
        .atten = 25.0f,
        .silent = true}},
  };
  bool const channel_updated_test = false;
  bool const bandwidth_updated_test = true;
  bool const atten_updated_test = true;

  bool const cu_test = channel_updated_test || bandwidth_updated_test;

  de->lisp.run([&] {
    // clang-format off
    auto lobj = Funcall(
        Symbol("make-instance"), BRSymbol("tx-assignment-update"),
        Keyword("assignment-map"), toLisp(test),
        Keyword("channel-updated?"), channel_updated_test ? lisp::t : lisp::nil,
        Keyword("bandwidth-updated?"), bandwidth_updated_test ? lisp::t : lisp::nil,
        Keyword("atten-updated?"), atten_updated_test ? lisp::t : lisp::nil
      );
    // clang-format on
    lisp::Print(lobj);

    // convert back to c++ data
    auto from_lisp = TxAssignmentUpdate::fromLisp(lobj);
    auto cu_from_lisp =
        from_lisp.channel_updated || from_lisp.bandwidth_updated;

    // compare to test data
    BOOST_REQUIRE_EQUAL(channel_updated_test, from_lisp.channel_updated);
    BOOST_REQUIRE_EQUAL(bandwidth_updated_test, from_lisp.bandwidth_updated);
    BOOST_REQUIRE_EQUAL(atten_updated_test, from_lisp.atten_updated);
    BOOST_REQUIRE_EQUAL(test.size(), from_lisp.assignment_map.size());
    BOOST_REQUIRE_EQUAL(cu_test, cu_from_lisp);
    for (auto const &elem : from_lisp.assignment_map) {
      auto srn_id = elem.first;
      tx_assignment_eql(test.at(srn_id), elem.second);
    }
  });
}

// attempt to reproduce a bug
BOOST_AUTO_TEST_CASE(fixnum) {
  using decisionengine::toLisp;
  using lisp::Funcall;
  using lisp::Keyword;
  using lisp::Symbol;
  using lisp::toLisp;

  de->lisp.run([] {
    auto const fake_node = [] {
      std::vector<float> psddd = {
          2.2330925e-4, 2.8578745e-4, 3.1367666e-4, 2.7532584e-4, 1.8204688e-4,
          1.9065816e-4, 2.5976004e-4, 2.0062926e-4, 1.8328881e-4, 2.1442848e-4,
          1.9569774e-4, 2.2476392e-4, 2.2980463e-4, 2.8336275e-4, 2.7232504e-4,
          2.1651383e-4, 1.4357983e-4, 1.9176812e-4, 2.670379e-4,  3.5253583e-4,
          3.6074943e-4, 2.9646672e-4, 2.930558e-4,  3.6371418e-4, 3.3123232e-4,
          2.2369258e-4, 3.539719e-4,  4.2532978e-4, 4.3681782e-4, 4.010351e-4,
          3.8524764e-4, 3.553971e-4,  2.895862e-4,  2.8927435e-4, 3.0333002e-4,
          2.822955e-4,  4.0010532e-4, 3.399436e-4,  2.9558007e-4, 3.2078582e-4,
          3.8612942e-4, 3.6194944e-4, 2.6357383e-4, 3.67206e-4,   4.1568856e-4,
          3.4546718e-4, 3.5254852e-4, 3.199479e-4,  2.5270783e-4, 2.6471616e-4,
          2.9503228e-4, 3.808852e-4,  5.0837814e-4, 5.590174e-4,  5.3364906e-4,
          4.0796777e-4, 2.798801e-4,  2.5490634e-4, 4.4879504e-4, 6.129193e-4,
          5.3680444e-4, 4.9597875e-4, 6.189859e-4,  6.254158e-4,  4.451666e-4,
          5.0380273e-4, 5.322586e-4,  4.3000208e-4, 3.3895054e-4, 4.1903683e-4,
          5.588196e-4,  5.6366547e-4, 5.9088663e-4, 5.8536185e-4, 4.0599392e-4,
          3.0985376e-4, 4.287185e-4,  6.140189e-4,  8.2121853e-4, 8.290294e-4,
          6.1711343e-4, 5.652656e-4,  5.6107406e-4, 6.303482e-4,  5.7479355e-4,
          3.7591855e-4, 4.1193012e-4, 5.570309e-4,  7.392523e-4,  8.872402e-4,
          7.9732167e-4, 5.5775914e-4, 3.3700524e-4, 3.2320985e-4, 4.5007464e-4,
          5.072245e-4,  4.1452964e-4, 4.0213694e-4, 4.4596478e-4, 5.5096205e-4,
          5.157745e-4,  3.5084912e-4, 3.3972724e-4, 3.4350136e-4, 3.0637658e-4,
          3.532101e-4,  4.0905143e-4, 4.973807e-4,  5.2788144e-4, 4.5707935e-4,
          5.9104315e-4, 8.277295e-4,  8.0860226e-4, 7.842546e-4,  7.412638e-4,
          6.598861e-4,  4.905097e-4,  4.136215e-4,  3.5682737e-4, 4.846085e-4,
          6.251415e-4,  4.1983812e-4, 3.3900593e-4, 4.8939e-4,    5.3653296e-4,
          4.6889018e-4, 4.8721393e-4, 6.7521055e-4, 5.4294255e-4, 4.5145405e-4,
          5.328341e-4,  5.8222533e-4, 7.2665734e-4, 6.698551e-4,  4.99848e-4,
          3.5442036e-4, 5.5299327e-4, 8.714481e-4,  8.084188e-4,  6.2830676e-4,
          5.118353e-4,  6.441496e-4,  6.5100135e-4, 6.5903866e-4, 7.3017925e-4,
          6.9437025e-4, 4.5085108e-4, 2.3651616e-4, 2.7205652e-4, 5.0081295e-4,
          9.5487654e-4, 9.571262e-4,  6.3259574e-4, 4.964195e-4,  5.965256e-4,
          7.4939366e-4, 7.0752477e-4, 5.286768e-4,  3.4336976e-4, 3.665308e-4,
          6.1573606e-4, 7.0308306e-4, 7.146583e-4,  7.565504e-4,  7.7147916e-4,
          6.1006e-4,    5.978017e-4,  6.3999905e-4, 5.383911e-4,  4.2193657e-4,
          5.430719e-4,  7.8685896e-4, 7.181093e-4,  5.135189e-4,  4.0126411e-4,
          4.023072e-4,  4.3429845e-4, 4.6325015e-4, 6.846483e-4,  6.849888e-4,
          4.1438383e-4, 4.3571598e-4, 7.0121855e-4, 5.253013e-4,  4.0351477e-4,
          5.207102e-4,  6.988016e-4,  7.6505746e-4, 7.1501767e-4, 6.7102845e-4,
          4.068159e-4,  2.8401535e-4, 3.324961e-4,  3.58427e-4,   4.0721692e-4,
          3.971159e-4,  4.194219e-4,  5.5951404e-4, 5.020662e-4,  2.8873456e-4,
          4.4858048e-4, 7.556662e-4,  7.685455e-4,  6.4307795e-4, 5.589e-4,
          4.9469445e-4, 4.3237058e-4, 5.3788326e-4, 6.363059e-4,  5.4109085e-4,
          5.550873e-4,  5.915278e-4,  7.728263e-4,  8.354121e-4,  7.448468e-4,
          7.152861e-4,  5.983027e-4,  5.143613e-4,  4.6336814e-4, 4.5074703e-4,
          6.0016365e-4, 6.6517026e-4, 5.375798e-4,  4.893645e-4,  5.501186e-4,
          5.956401e-4,  4.6171594e-4, 2.9538103e-4, 3.0558705e-4, 3.0447374e-4,
          4.1208093e-4, 5.138142e-4,  5.4691924e-4, 4.3609962e-4, 3.7496042e-4,
          4.5325444e-4, 6.3176104e-4, 6.5544556e-4, 3.8975844e-4, 3.0514784e-4,
          2.680726e-4,  4.950464e-4,  7.1786426e-4, 5.9758045e-4, 4.2739525e-4,
          5.1532686e-4, 8.5436396e-4, 8.7178947e-4, 5.2741246e-4, 4.4929155e-4,
          4.931207e-4,  6.8617705e-4, 7.356466e-4,  6.3529774e-4, 6.9199485e-4,
          5.6019e-4,    3.850328e-4,  2.599903e-4,  2.892825e-4,  5.224152e-4,
          5.474566e-4,  5.362209e-4,  6.497665e-4,  5.86833e-4,   5.74841e-4,
          4.5039452e-4, 5.8467727e-4, 7.971188e-4,  9.017444e-4,  7.437554e-4,
          4.7867192e-4, 4.6829786e-4, 5.527253e-4,  5.3934066e-4, 6.705999e-4,
          9.3670335e-4, 8.257094e-4,  5.139628e-4,  3.46833e-4,   3.8664287e-4,
          7.696157e-4,  0.0010313978, 5.74118e-4,   2.582905e-4,  2.5205637e-4,
          3.680782e-4,  5.5514224e-4, 4.9118954e-4, 4.6225404e-4, 4.3389454e-4,
          3.3244563e-4, 5.05903e-4,   8.4982987e-4, 7.4907194e-4, 6.2445377e-4,
          5.5256044e-4, 3.9303512e-4, 3.3312384e-4, 3.5464385e-4, 3.4577976e-4,
          3.6399128e-4, 3.911542e-4,  4.6620372e-4, 6.460646e-4,  8.110426e-4,
          7.902738e-4,  7.263263e-4,  7.003661e-4,  7.434171e-4,  7.31777e-4,
          6.513905e-4,  7.310852e-4,  8.601988e-4,  6.949954e-4,  3.5893323e-4,
          3.912842e-4,  6.247034e-4,  8.7293907e-4, 0.0012922605, 0.0011355707,
          6.727616e-4,  4.6671662e-4, 4.577841e-4,  4.0815555e-4, 3.4603983e-4,
          2.945225e-4,  5.4919434e-4, 7.7455956e-4, 5.75511e-4,   5.9804675e-4,
          6.659019e-4,  4.3402723e-4, 3.434968e-4,  3.255779e-4,  3.212145e-4,
          3.032939e-4,  4.0153076e-4, 6.6598284e-4, 6.2648865e-4, 4.27278e-4,
          3.4962888e-4, 4.4094268e-4, 5.47176e-4,   4.3657722e-4, 4.5374286e-4,
          4.0745284e-4, 3.482449e-4,  4.1783e-4,    6.31868e-4,   7.8219e-4,
          8.005043e-4,  7.442354e-4,  6.032367e-4,  5.088021e-4,  4.7797692e-4,
          4.273417e-4,  5.2268646e-4, 5.0587376e-4, 4.1376989e-4, 3.8615824e-4,
          4.0867706e-4, 4.793796e-4,  5.964446e-4,  9.122627e-4,  7.5209734e-4,
          5.572166e-4,  7.9507264e-4, 6.667219e-4,  2.7670086e-4, 3.165941e-4,
          6.5738754e-4, 8.608519e-4,  6.182569e-4,  3.9151617e-4, 4.704668e-4,
          6.563548e-4,  6.2146166e-4, 6.748509e-4,  6.570657e-4,  5.136111e-4,
          3.969599e-4,  5.5833574e-4, 5.748153e-4,  4.120321e-4,  4.502356e-4,
          5.772037e-4,  5.551181e-4,  4.3908358e-4, 3.9757544e-4, 4.795488e-4,
          5.191031e-4,  4.406643e-4,  3.9674272e-4, 4.6943422e-4, 5.371412e-4,
          6.958222e-4,  5.904482e-4,  3.9979088e-4, 4.7678873e-4, 5.976264e-4,
          6.3334295e-4, 7.265479e-4,  6.7387515e-4, 5.486996e-4,  4.080581e-4,
          4.6733936e-4, 7.024013e-4,  7.0938515e-4, 6.0721993e-4, 0.003767417,
          0.042917345,  0.056796163,  0.19986251,   0.15744776,   0.06010871,
          0.032991283,  0.0025381013, 7.0043467e-4, 5.9658726e-4, 4.064993e-4,
          4.4007442e-4, 4.4849853e-4, 3.468992e-4,  5.693784e-4,  7.817513e-4,
          7.667869e-4,  5.5633555e-4, 4.5080722e-4, 4.2339737e-4, 5.2069174e-4,
          6.249243e-4,  5.4120645e-4, 6.9565675e-4, 8.0330577e-4, 5.734693e-4,
          5.2460347e-4, 7.841337e-4,  9.803376e-4,  7.208679e-4,  6.673378e-4,
          8.412837e-4,  6.73285e-4,   5.2378257e-4, 3.6171233e-4, 4.288427e-4,
          6.246195e-4,  7.46233e-4,   6.62324e-4,   5.7731883e-4, 7.592438e-4,
          9.896934e-4,  8.3373376e-4, 6.6441396e-4, 5.421418e-4,  5.726322e-4,
          6.767314e-4,  6.662158e-4,  6.3497195e-4, 5.8277237e-4, 5.2971183e-4,
          5.366392e-4,  4.8199538e-4, 7.590204e-4,  9.499123e-4,  5.460632e-4,
          5.243968e-4,  7.2738365e-4, 6.1877456e-4, 5.815167e-4,  5.929667e-4,
          6.764574e-4,  7.6931634e-4, 7.3846825e-4, 7.066758e-4,  7.039808e-4,
          7.1263855e-4, 4.6449038e-4, 4.859089e-4,  7.2332943e-4, 8.489909e-4,
          9.378188e-4,  8.839005e-4,  7.3804683e-4, 7.105343e-4,  7.2675024e-4,
          7.321362e-4,  6.8483944e-4, 6.0574856e-4, 5.2762404e-4, 6.2146055e-4,
          5.870063e-4,  4.6585314e-4, 5.092067e-4,  4.8548536e-4, 5.0969765e-4,
          4.0535664e-4, 4.207205e-4,  5.7048135e-4, 6.2639907e-4, 6.606551e-4,
          7.5471215e-4, 7.6960074e-4, 8.272687e-4,  9.94821e-4,   9.429923e-4,
          8.4828335e-4, 8.822188e-4,  7.99126e-4,   7.0602697e-4, 7.251902e-4,
          7.759552e-4,  8.026114e-4,  8.6483324e-4, 6.9382845e-4, 6.6590926e-4,
          8.551276e-4,  8.2662486e-4, 6.064846e-4,  3.9846083e-4, 3.012532e-4,
          4.9509585e-4, 5.697699e-4,  5.823572e-4,  5.9480476e-4, 5.015334e-4,
          4.6374672e-4, 7.3871936e-4, 7.60936e-4,   8.040452e-4,  9.568556e-4,
          0.001159664,  0.0012780863, 9.0660306e-4, 7.5894507e-4, 8.0542336e-4,
          9.956403e-4,  7.9996366e-4, 6.8283634e-4, 9.212457e-4,  0.0010061609,
          7.583125e-4,  4.569079e-4,  5.349567e-4,  6.593291e-4,  5.816281e-4,
          5.9828156e-4, 6.403842e-4,  6.661398e-4,  6.120183e-4,  5.157483e-4,
          5.134463e-4,  5.8534514e-4, 7.264745e-4,  7.490634e-4,  7.073742e-4,
          7.8671245e-4, 8.0693094e-4, 6.86902e-4,   6.772069e-4,  6.444863e-4,
          7.6194823e-4, 7.0879876e-4, 5.8773067e-4, 6.2926457e-4, 7.566719e-4,
          7.83377e-4,   7.293378e-4,  6.4822496e-4, 6.338008e-4,  5.941053e-4,
          5.4186495e-4, 5.795842e-4,  7.9947733e-4, 0.0011420934, 0.0011544859,
          7.1092823e-4, 6.585876e-4,  6.69402e-4,   4.3605187e-4, 3.54556e-4,
          5.347077e-4,  6.985443e-4,  7.020391e-4,  8.015466e-4,  9.4049115e-4,
          9.515262e-4,  8.3543954e-4, 6.5909023e-4, 5.978572e-4,  5.541469e-4,
          3.7869695e-4, 3.728041e-4,  5.305653e-4,  6.7437644e-4, 7.1646465e-4,
          6.891695e-4,  8.303525e-4,  7.2694605e-4, 4.0231532e-4, 4.344859e-4,
          5.3571665e-4, 6.117453e-4,  6.864889e-4,  8.227882e-4,  7.640601e-4,
          5.18075e-4,   5.6948484e-4, 6.432206e-4,  5.2617514e-4, 5.477769e-4,
          5.012915e-4,  4.7238465e-4, 5.515041e-4,  5.4233946e-4, 5.6567707e-4,
          6.962159e-4,  6.689685e-4,  6.580928e-4,  5.521885e-4,  5.5019546e-4,
          6.202656e-4,  4.7217665e-4, 4.0536455e-4, 3.7855873e-4, 3.012705e-4,
          2.1644207e-4, 5.35776e-4,   6.9067924e-4, 5.8048236e-4, 7.4483483e-4,
          7.0819486e-4, 8.030564e-4,  8.4060716e-4, 6.84704e-4,   6.417832e-4,
          5.706529e-4,  6.739502e-4,  8.7361154e-4, 8.01845e-4,   7.2294084e-4,
          5.5683195e-4, 4.6653312e-4, 4.987137e-4,  6.129238e-4,  7.545103e-4,
          5.522382e-4,  3.57283e-4,   4.2178805e-4, 4.7393504e-4, 5.784695e-4,
          6.393129e-4,  6.566063e-4,  6.3891785e-4, 6.855422e-4,  7.0597563e-4,
          5.4829597e-4, 4.5736143e-4, 6.128966e-4,  7.561262e-4,  9.754392e-4,
          0.0011243664, 0.0010105578, 8.524776e-4,  7.690146e-4,  7.5405795e-4,
          9.0563274e-4, 8.545413e-4,  7.5359776e-4, 8.2484604e-4, 7.6709385e-4,
          6.219326e-4,  7.093089e-4,  6.5573177e-4, 5.4108415e-4, 6.3753227e-4,
          6.046295e-4,  5.105026e-4,  7.0025196e-4, 8.812325e-4,  7.620573e-4,
          5.413337e-4,  3.9169143e-4, 5.243731e-4,  7.613397e-4,  4.8136644e-4,
          2.6423048e-4, 4.3417476e-4, 6.7550736e-4, 6.036891e-4,  4.9187953e-4,
          8.393164e-4,  0.0010802452, 7.89501e-4,   5.0863554e-4, 5.2083994e-4,
          6.5506174e-4, 6.436091e-4,  7.027085e-4,  6.9839787e-4, 4.5090553e-4,
          4.380106e-4,  5.75931e-4,   8.405985e-4,  8.403963e-4,  7.391893e-4,
          6.208656e-4,  4.941672e-4,  4.155676e-4,  3.8292198e-4, 4.1926536e-4,
          5.531576e-4,  6.984246e-4,  7.7146385e-4, 7.6739537e-4, 5.9343065e-4,
          4.5524014e-4, 4.8532558e-4, 6.4710114e-4, 5.4302136e-4, 4.224927e-4,
          7.741681e-4,  9.406974e-4,  5.986557e-4,  4.3220381e-4, 3.6558526e-4,
          3.7536063e-4, 6.1477005e-4, 6.873645e-4,  5.689625e-4,  7.3737075e-4,
          0.001021656,  8.4746676e-4, 7.5265026e-4, 0.0010154401, 0.0010750346,
          9.511901e-4,  9.171477e-4,  7.8145595e-4, 6.1192864e-4, 6.806934e-4,
          7.524183e-4,  6.034026e-4,  7.849281e-4,  6.909686e-4,  5.4405717e-4,
          4.861891e-4,  4.162317e-4,  5.19298e-4,   7.517978e-4,  6.214522e-4,
          4.7931212e-4, 5.30419e-4,   5.574515e-4,  5.195217e-4,  3.590525e-4,
          2.5781075e-4, 3.013085e-4,  4.6902613e-4, 4.479479e-4,  7.3159474e-4,
          0.0011807093, 9.6055056e-4, 6.652724e-4,  5.58846e-4,   4.593131e-4,
          5.042208e-4,  4.5943353e-4, 4.5695892e-4, 5.7787535e-4, 6.858511e-4,
          6.919276e-4,  7.4339774e-4, 8.552044e-4,  7.2552584e-4, 5.271117e-4,
          5.1915395e-4, 7.982097e-4,  0.0010978102, 9.048475e-4,  7.542242e-4,
          9.3750504e-4, 9.7597623e-4, 9.064301e-4,  9.64282e-4,   0.0010995334,
          0.0013163176, 8.258764e-4,  4.974934e-4,  4.9201824e-4, 3.943123e-4,
          5.134916e-4,  3.9198823e-4, 4.383406e-4,  7.1050937e-4, 0.001004406,
          9.0941694e-4, 6.6585984e-4, 6.187506e-4,  4.495678e-4,  4.69228e-4,
          7.099422e-4,  9.6538774e-4, 0.0010152162, 7.920909e-4,  5.995645e-4,
          5.171482e-4,  5.753286e-4,  5.023605e-4,  4.96249e-4,   7.072754e-4,
          5.185659e-4,  3.5938664e-4, 4.8291776e-4, 5.7580456e-4, 7.919105e-4,
          8.4081694e-4, 4.7773443e-4, 4.8295985e-4, 7.8178453e-4, 7.2898826e-4,
          6.3330313e-4, 8.0528134e-4, 8.390413e-4,  8.6389756e-4, 8.190073e-4,
          4.5463874e-4, 2.2501525e-4, 2.7629518e-4, 3.5232428e-4, 4.7553313e-4,
          6.1824406e-4, 4.8446338e-4, 5.409657e-4,  3.7545327e-4, 2.9294085e-4,
          7.000469e-4,  7.174066e-4,  5.790413e-4,  5.230408e-4,  6.741431e-4,
          7.4817956e-4, 8.5815915e-4, 9.893246e-4,  8.529158e-4,  6.862437e-4,
          6.530469e-4,  5.879375e-4,  5.0360756e-4, 7.1651756e-4, 8.1265956e-4,
          7.116391e-4,  8.554062e-4,  7.435415e-4,  5.0112954e-4, 4.0296422e-4,
          3.9842367e-4, 3.4669248e-4, 5.658332e-4,  6.043565e-4,  4.5227003e-4,
          4.2585607e-4, 5.308282e-4,  6.737374e-4,  7.136055e-4,  5.7920767e-4,
          7.1641046e-4, 0.0010574054, 0.0010744368, 8.6636294e-4, 7.4291276e-4,
          5.5455056e-4, 5.971827e-4,  5.0076126e-4, 4.5138336e-4, 9.08828e-4,
          9.268623e-4,  4.947667e-4,  3.0721974e-4, 3.5881434e-4, 4.4656056e-4,
          7.018065e-4,  8.274777e-4,  8.4979e-4,    8.443898e-4,  6.823784e-4,
          7.207542e-4,  7.717436e-4,  6.4400927e-4, 6.479861e-4,  7.006241e-4,
          7.594318e-4,  6.599464e-4,  3.9304575e-4, 4.4426922e-4, 5.264577e-4,
          6.4157747e-4, 8.3007605e-4, 6.997757e-4,  7.5786014e-4, 0.0010645662,
          8.4320176e-4, 4.8978213e-4, 4.4105403e-4, 5.545745e-4,  4.6977715e-4,
          7.3308725e-4, 0.0010661335, 7.6996704e-4, 5.4809183e-4, 6.109267e-4,
          8.254729e-4,  7.612621e-4,  5.9713196e-4, 5.899189e-4,  8.364227e-4,
          9.041367e-4,  6.1248714e-4, 3.866713e-4,  4.827067e-4,  6.384013e-4,
          7.765562e-4,  7.501844e-4,  4.916071e-4,  5.6525186e-4, 7.7483326e-4,
          5.369978e-4,  2.825196e-4,  3.999915e-4,  6.852752e-4,  7.055425e-4,
          6.49353e-4,   5.4724875e-4, 4.473553e-4,  7.333409e-4,  7.930594e-4,
          6.757613e-4,  6.632661e-4,  6.2635547e-4, 7.22836e-4,   8.5338805e-4,
          9.475256e-4,  8.6261757e-4, 8.8689587e-4, 8.0865785e-4, 6.062857e-4,
          5.2096514e-4, 4.316783e-4,  6.3255825e-4, 8.673196e-4,  9.6239464e-4,
          8.0671394e-4, 4.893405e-4,  4.9016165e-4, 6.0467154e-4, 5.621871e-4,
          4.9803435e-4, 5.8649323e-4, 7.228876e-4,  8.562238e-4,  9.581588e-4,
          8.618035e-4,  9.052283e-4,  8.3588844e-4, 7.749542e-4,  6.3341507e-4,
          5.804359e-4,  5.4945826e-4, 5.226821e-4,  5.27644e-4,   7.519656e-4,
          8.4414287e-4, 5.705712e-4,  4.518191e-4,  4.7927356e-4, 5.239481e-4,
          4.4408249e-4, 5.653487e-4,  7.3578325e-4, 5.7897845e-4, 4.587248e-4,
          4.5763483e-4, 4.6716738e-4, 3.9438612e-4, 4.288192e-4,  6.221906e-4,
          5.614253e-4,  5.222114e-4,  5.7353906e-4, 4.8388526e-4, 6.0619396e-4,
          5.8815617e-4, 3.841348e-4,  3.6565049e-4, 5.172333e-4,  6.414733e-4,
          6.640648e-4,  7.0112123e-4, 6.584271e-4,  7.882196e-4,  8.0361625e-4,
          6.12969e-4,   4.7760594e-4, 4.6051672e-4, 4.2279746e-4, 2.607699e-4,
          2.421128e-4,  2.5925445e-4, 2.6867967e-4, 2.324953e-4,  2.6971192e-4,
          3.836642e-4,  3.865254e-4,  4.2800524e-4, 5.050678e-4,  3.6797082e-4,
          2.544811e-4,  2.6115155e-4, 2.4670406e-4, 2.4308686e-4, 2.8833447e-4,
          3.5226534e-4, 4.266265e-4,  3.444413e-4,  2.1203546e-4,
      };
      auto const psdd = psdsensing::PSDData{
          .src_srnid = 0,
          .time_ns = 1,
          .psd = std::make_shared<decltype(psddd)>(psddd),
      };
      auto const real_psd = toLisp(psdd);
      auto const thresh_psd =
          threshPSDtoLisp(psdd, de->options.psd_hist_params);
      // clang-format off
      auto const lobj = Funcall(Symbol("make-instance"), BRSymbol("internal-node"),
                                Keyword("id"), 59,
                                Keyword("location"), lisp::nil,
                                Keyword("tx-assignment"), lisp::nil,
                                Keyword("est-duty-cycle"), lisp::nil,
                                Keyword("real-psd"), real_psd,
                                Keyword("thresh-psd"), thresh_psd);
      // clang-format on
      return lisp::List(lobj);
    }();
    using Environment = c2api::EnvironmentManager::Environment;

    auto fake_env = Environment{
        .collab_network_type = Environment::CollabNetworkType::UNSPEC,
        .incumbent_protection =
            Environment::IncumbentProtection{(int64_t)2.9e9, (int64_t)2e6},
        .scenario_rf_bandwidth = (int64_t)10e6,
        .scenario_center_frequency = (int64_t)1e9,
        .bonus_threshold = 0,
        .has_incumbent = false,
        .stage_number = 0,
        .timestamp = 100,
        .raw = {"json", "raw"}};

    auto rates = lisp::List(
        lisp::Funcall(Symbol("make-instance"), BRSymbol("offered-traffic-rate"),
                      Keyword("src"), toLisp(59), Keyword("dst"), toLisp(59),
                      Keyword("bps"), toLisp(0.0)));

    auto data =
        Funcall(Symbol("make-instance"), BRSymbol("decision-engine-input"),
                // the trigger
                Keyword("trigger"), toLisp(Trigger::PeriodicStep),
                // the current time
                Keyword("time-stamp"), toLisp(std::chrono::system_clock::now()),
                // the environment
                Keyword("env"), toLisp(fake_env),
                // mandate information
                Keyword("mandates"), lisp::nil,
                // node information
                Keyword("nodes"), fake_node,
                // competitor information
                Keyword("collaborators"), lisp::nil,
                // offered traffic
                Keyword("offered-traffic-rates"), rates,
                // incumbent information
                Keyword("incumbent"), lisp::nil);
    de->fakeStepSetup();
    Funcall(BRSymbol("decision-engine-step"), data);

  });
}

BOOST_AUTO_TEST_CASE(nodeid) {
  auto const ip = boost::asio::ip::address_v4::from_string("192.168.138.1");
  NodeID const expected = 38;
  NodeID const out = ((ip.to_ulong() >> 8) & 0xff) - 100;
  NodeID const wrong = ((htonl((ip.to_ulong())) >> 8) & 0xff) - 100;
  std::cout << "ex: " << (int)expected << " my: " << (int)out
            << " wr: " << (int)wrong << std::endl;
  BOOST_REQUIRE_EQUAL(expected, out);
  BOOST_REQUIRE_NE(expected, wrong);
  BOOST_REQUIRE_NE(out, wrong);
}

BOOST_AUTO_TEST_CASE(multithread, *boost::unit_test::disabled()) {
  de->lisp.run([] { lisp::Funcall(BRSymbol("do-something-multithread")); });
  std::this_thread::sleep_for(std::chrono::seconds(8));
  de->lisp.run([] { lisp::Funcall(BRSymbol("print-processes")); });
}
