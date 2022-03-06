/*
 * FreqAlloc unit test
 * Mai Zhang
 */

#include "../src/cc_data.h"
#include "../src/freq_alloc.h"
#include "../src/events.h"
#include "../src/c2api.h"
#include "../src/log.h"

#include <boost/asio.hpp>
#include <vector>
#include <iostream>
#include <thread>

int main(int argc, char const *argv[]) {
  using namespace bamradio;
  options::init(argc, argv);

  const auto d_1s = std::chrono::seconds(1);
  const auto d_6s = std::chrono::seconds(6);

  log::Logger log;
  log.enableBackend(log::Backend::STDOUT);

  // Number of nodes to simulate
  size_t const n_nodes = 5;
  assert(n_nodes > 1);
  // Generate CCData (Node ID 0 to n_nodes - 1)
  std::vector<controlchannel::CCData::sptr> cc_data_vec;
  for (size_t i = 0; i < n_nodes; ++i) {
    // Node 0 is the gateway
    bool const is_gateway = (i == 0);
    // Instantiate CCData
    auto cc_data = std::make_shared<controlchannel::CCData>(i, 1, is_gateway);
    // Push
    cc_data_vec.push_back(cc_data);
  }
  // Modify CCData contents
  // for instance...
  cc_data_vec[0]->setLocation(30.300344, -97.739655, 0.825000);
  cc_data_vec[1]->setLocation(30.300346, -97.739647, 2.497000);
  cc_data_vec[2]->setLocation(30.300303, -97.739632, 1.349000);
  cc_data_vec[3]->setLocation(30.300352, -97.739609, 4.751000);
  cc_data_vec[4]->setLocation(30.300303, -97.739632, 3.950000);

  std::map<uint16_t, stats::FlowInfo> flow_map;
  flow_map[5001] = stats::FlowInfo{ true, 0, 1 };
  flow_map[5002] = stats::FlowInfo{ true, 0, 2 };
  flow_map[5003] = stats::FlowInfo{ true, 0, 3 };
  flow_map[5004] = stats::FlowInfo{ true, 1, 4 };
  flow_map[5005] = stats::FlowInfo{ true, 5, 1 };
  cc_data_vec[0]->setFlowInfo(flow_map);

  // Send all non-gateway CCData info to the gateway
  for (size_t i = 1; i < n_nodes; ++i) {
    auto const serialized_data = cc_data_vec[i]->serialize();
    auto cbuf =
      boost::asio::buffer(serialized_data->data(), serialized_data->size());
    cc_data_vec[0]->deserialize(cbuf, false);
  }
  // For covenience...
  auto ccdata_gw = cc_data_vec[0];
  // Now it's ready to use the gateway's CCData
  auto srnids = ccdata_gw->getAllSRNIDs();
  for (auto &v : srnids) {
    std::cout << "Node ID: " << (int)v << std::endl;
  }

  // ios
  boost::asio::io_service ios;
  auto *ios_work = new boost::asio::io_service::work(ios);
  std::thread work_thread([&] {
    ios.run();
    std::cout << "thread exiting" << std::endl;
  });

  // FreqAlloc object
  psdsensing::FreqAlloc _freq_alloc(ios, ccdata_gw, 168, 256);

  // location map
  std::map<psdsensing::nodeid_t, controlchannel::Location> their_locmap;
  their_locmap[psdsensing::nodeid_t{ 101, 11 }] = controlchannel::Location{ 30.304238, -97.741573, 1.744000 };
  their_locmap[psdsensing::nodeid_t{ 101, 12 }] = controlchannel::Location{ 30.304228, -97.741580, 3.703000 };
  their_locmap[psdsensing::nodeid_t{ 101, 13 }] = controlchannel::Location{ 30.304217, -97.741515, 4.244000 };
  their_locmap[psdsensing::nodeid_t{ 101, 14 }] = controlchannel::Location{ 30.304215, -97.741520, 1.094000 };
  their_locmap[psdsensing::nodeid_t{ 101, 15 }] = controlchannel::Location{ 30.304218, -97.741522, 2.067000 };
  their_locmap[psdsensing::nodeid_t{ 102, 21 }] = controlchannel::Location{ 30.303518, -97.739873, 1.499000 };
  their_locmap[psdsensing::nodeid_t{ 102, 22 }] = controlchannel::Location{ 30.303518, -97.739875, 2.214000 };
  their_locmap[psdsensing::nodeid_t{ 102, 23 }] = controlchannel::Location{ 30.303512, -97.739858, 1.273000 };
  their_locmap[psdsensing::nodeid_t{ 102, 24 }] = controlchannel::Location{ 30.303495, -97.739813, 3.630000 };
  their_locmap[psdsensing::nodeid_t{ 102, 25 }] = controlchannel::Location{ 30.303487, -97.739797, 3.846000 };
  NotificationCenter::shared.post(
    psdsensing::FreqAlloc::NodeLocationNotification, their_locmap);
  // tx map
  std::map<psdsensing::nodeid_t, std::vector<Channel>> channels_map;
  channels_map[psdsensing::nodeid_t{ 101, 11 }].push_back(Channel(1e6, -9.5e6, 20e6));  // bw, cf, samprate
  channels_map[psdsensing::nodeid_t{ 101, 11 }].push_back(Channel(1e6, -7.5e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 101, 12 }].push_back(Channel(1e6, -6.5e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 101, 13 }].push_back(Channel(1e6, -5.5e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 101, 14 }].push_back(Channel(1e6, 0, 20e6));
  channels_map[psdsensing::nodeid_t{ 102, 21 }].push_back(Channel(2e6, 1e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 102, 22 }].push_back(Channel(1e6, 3e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 102, 23 }].push_back(Channel(1e6, 4e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 102, 24 }].push_back(Channel(1e6, 5e6, 20e6));
  channels_map[psdsensing::nodeid_t{ 102, 25 }].push_back(Channel(1e6, 8e6, 20e6));
  NotificationCenter::shared.post(psdsensing::FreqAlloc::TxBandNotification,
    channels_map);
  std::map<psdsensing::nodeid_t, double> tx_power;
  for (auto const& cm : channels_map) {
    tx_power.emplace(cm.first, 15);
  }
  NotificationCenter::shared.post(psdsensing::FreqAlloc::TxPowerNotification,
    tx_power);
  // peer performance
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(100, 3));  // network 100 achieved 3
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(101, 2));
  // mandated outcomes
  nlohmann::json j = {
    { { "goal_type", "Traffic" }, { "flow_uid", 5001 },
      { "requirements", { { "max_latency_s", 0.37 }, { "min_throughput_bps", 36504.0 } } } },
    { { "goal_type", "Traffic" }, { "flow_uid", 5002 },
      { "requirements", { { "max_latency_s", 0.37 }, { "min_throughput_bps", 36504.0 } } } },
    { { "goal_type", "Traffic" }, { "flow_uid", 5003 },
      { "requirements", { { "max_latency_s", 1.0 }, { "min_throughput_bps", 260.0 } } } },
    { { "goal_type", "Traffic" }, { "flow_uid", 5004 },
      { "requirements", { { "max_latency_s", 0.37 }, { "min_throughput_bps", 36504.0 } } } },
    { { "goal_type", "Traffic" }, { "flow_uid", 5005 },
      { "requirements", { { "max_latency_s", 0.37 }, { "min_throughput_bps", 36504.0 } } } }
  };
  OutcomesUpdateEventInfo ei;
  ei.j = j;
  NotificationCenter::shared.post(OutcomesUpdateEvent, ei);
  // our performance
  std::map<uint16_t, stats::FlowPerformance> perf_map;
  /* XXX DO: this is broken
  perf_map[5001] = stats::FlowPerformance{ 0, 0.9 };
  perf_map[5002] = stats::FlowPerformance{ 0, 0.8 };
  perf_map[5003] = stats::FlowPerformance{ 0, 1.2 };
  perf_map[5004] = stats::FlowPerformance{ 0, 0.9 };
  perf_map[5005] = stats::FlowPerformance{ 0, 1.0 };
  */
  NotificationCenter::shared.post(stats::FlowPerformanceNotification, perf_map);

  std::this_thread::sleep_for(d_1s);
  // # mo achieved: we 2; peers 3,2
  // 0. first allocation
  int counter = 0;
  std::cout << "==" << counter++ << std::endl;
  std::vector<Channel> ctrl_alloc = {Channel(500e3, 9.75e6, 20e6)};
  std::vector<waveform::ID> channel_waveform(n_nodes);
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2);
  auto chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  auto print_alloc = [&](void) {
    for (auto const& ch : chan_vec.channels) {
      double bw = waveform::get(ch.waveform).bw(options::phy::data::sample_rate);
      double cf = ch.cfreq;
      std::cout << "[" << cf - bw / 2 << "," << cf + bw / 2 << "] " << cf << " " << bw << std::endl;
    }
  };
  _freq_alloc.print_debug_info();
  print_alloc();

  // 1. nothing changed, should remain the same allocation
  std::this_thread::sleep_for(d_6s);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2 + 1);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();

  // # mo: we 2, peers 3,5
  // 2. should reallocate
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(100, 3));
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(101, 5));
  std::this_thread::sleep_for(d_6s);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2 + 2);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();

  // # mo: we 5, peers 3,5
  // 3. should reduce bandwidth
  /* XXX DO: this is broken
  perf_map[5001] = stats::FlowPerformance{ 0, 1.0 };
  perf_map[5002] = stats::FlowPerformance{ 0, 1.8 };
  perf_map[5003] = stats::FlowPerformance{ 0, 1.2 };
  perf_map[5004] = stats::FlowPerformance{ 0, 1.0 };
  perf_map[5005] = stats::FlowPerformance{ 0, 1.0 };
  */
  NotificationCenter::shared.post(stats::FlowPerformanceNotification, perf_map);
  std::this_thread::sleep_for(d_6s);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2 + 3);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();

  // # mo: we 5, peers 6,6
  // 4. time hold too short, should remain the same
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(100, 6));
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(101, 6));
  std::this_thread::sleep_for(d_1s);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();

  // # mo: we 5, peers 4,6
  // 5. should remain the same
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(100, 4));
  NotificationCenter::shared.post(psdsensing::FreqAlloc::PerformanceNotification,
    std::make_pair<psdsensing::networkid_t, int>(101, 6));
  std::this_thread::sleep_for(d_6s);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2 + 1);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();

  // 6. change of environment, should reallocate
#ifndef NDEBUG
  c2api::EnvironmentManager::Environment env;
  env.scenario_rf_bandwidth = (int64_t)5e6;
  c2api::env.setEnvironment(env);
  std::cout << "==" << counter++ << std::endl;
  for (int i = 0; i < n_nodes; ++i)
    channel_waveform[i] = static_cast<waveform::ID>(i % 2);
  chan_vec = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc);
  _freq_alloc.print_debug_info();
  print_alloc();
#endif

  // 7. change of peer spectrum usage
  channels_map.clear();
  channels_map[psdsensing::nodeid_t{ 102, 25 }].push_back(Channel(1e6, 8e6, 20e6));
  NotificationCenter::shared.post(psdsensing::FreqAlloc::TxBandNotification,
    channels_map);
  std::this_thread::sleep_for(d_1s);
  _freq_alloc.print_debug_info();

  std::cout << "exit" << std::endl;
  return 0;
}
