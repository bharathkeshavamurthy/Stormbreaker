#define BOOST_TEST_MODULE im
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#ifndef BOOST_ASIO_DISABLE_EPOLL
#error "BOOST_ASIO_DISABLE_EPOLL not defined"
#endif
#include "../src/im.h"
#include <algorithm>
#include <array>
#include <boost/asio/ip/network_v4.hpp>
#include <boost/format.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <thread>

using namespace bamradio;
using boost::format;
namespace asio = boost::asio;
using namespace std::chrono_literals;
namespace chrono = std::chrono;
using namespace nlohmann;
#include "im_mo.inc"

typedef std::chrono::duration<float, chrono::seconds::period> fsec;

BOOST_AUTO_TEST_CASE(imfromjson) {
  IndividualMandate::fromJSON(mo1);
  IndividualMandate::fromJSON(mo2);
  IndividualMandate::fromJSON(mo3);
}

BOOST_AUTO_TEST_CASE(scheduleflows) {
  auto const imm = IndividualMandate::fromJSON(mo3);

  for (auto const &im : imm) {
    std::cout << format("FlowUID %u ") % im.first;
    im.second.visit<void>(
        [](IndividualMandate::StreamPT const &streampt) {
          std::cout << format("stream %f bps %f s\n") %
                           streampt.min_throughput_bps %
                           streampt.max_latency.count();
        },
        [&](IndividualMandate::FilePT const &filept) {
          std::cout << format("file %ld B %f s\n") % filept.size_bytes %
                           filept.transfer_duration.count();
        });
  }

  /* TODO
   * file size from fifonodropqueue size
   * file start time from?
   * packet size from first packet
   * linkinfo
   */

  FlowInfoMap allflows;
  std::map<FlowUID, FileFlowProgress> ffp;
  auto const flowStartTime = chrono::system_clock::now();
  float bps = 1e6;
  LinkInfo li{1ms, 8 * 7, bps, .95f * bps, .9f * bps};
  for (auto const &im : imm) {
    allflows.emplace(FlowID{0, 0, 0, 0, im.first},
                     new FlowInfo{im.second, li, 200, 64, 10, 0.25, 0.9, 1});
    im.second.visit<void>(
        [](auto) {},
        [&](IndividualMandate::FilePT const &filept) {
          auto const t = flowStartTime + filept.transfer_duration;
          ffp[im.first] = FileFlowProgress{
              chrono::time_point_cast<chrono::nanoseconds>(t), 1000};
        });
  }

  for (auto search :
       {MaxFlowsSearch::RemoveMinMaxLatency, MaxFlowsSearch::RemoveMaxQuantum,
        MaxFlowsSearch::RemoveMinMaxLatencyAndMaxQuantum}) {

    auto const qs = scheduleMaxFlows(allflows, ffp, flowStartTime, search, false);

    if (qs.valid) {
      std::cout << "\nScheduled " << qs.quantums.size() << " search "
                << (int)search << ":\n";
      for (auto const &quantump : qs.quantums) {
        auto const fid = quantump.first;
        auto const quantum = quantump.second;
        std::cout << format("FlowUID %u %f ") % fid.flowUID() %
                         fsec(quantum).count();
        allflows[fid]->im.visit<void>(
            [](IndividualMandate::StreamPT const &streampt) {
              std::cout << format("stream %f bps %f s\n") %
                               streampt.min_throughput_bps %
                               streampt.max_latency.count();
            },
            [&](IndividualMandate::FilePT const &filept) {
              std::cout << format("file %ld B %f s\n") % filept.size_bytes %
                               filept.transfer_duration.count();
            });
      }
      std::cout << format("period(s): %f\n") %
                       fsec(std::accumulate(
                                begin(qs.quantums), end(qs.quantums),
                                chrono::nanoseconds::zero(),
                                [](auto a, auto b) { return a + b.second; }))
                           .count();
    }
  }
}
