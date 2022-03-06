// FlowTracker Unit Test
// Copyright (c) 2018 Tomohiro Arakawa

#define BOOST_TEST_MODULE flowtracker
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/flowtracker.h"
#include <unistd.h>
using namespace bamradio;

using namespace std::chrono_literals;

BOOST_AUTO_TEST_CASE(flowtracker) {
  using namespace dll;

  FlowTracker tracker;
  FlowUID uid = 5001;

  uint16_t numsegs = 10;
  size_t seglength = 5;

  for (uint16_t i = 1; i <= numsegs; ++i) {
    if (i == 4) {
      continue;
    }
    uint32_t extra = i > 2 ? numsegs * seglength : 0;
    tracker.markReceived(
        ARQBurstInfo{
            .flow_uid = uid, .burst_num = 1, .seq_num = i, .extra = extra},
        std::chrono::system_clock::now() - 10ms, std::chrono::system_clock::now(), seglength);
  }

  // The last seq_num is 3 because we dropped 4
  BOOST_REQUIRE(tracker.getLastSeqNums()[0].seq_num == 3);

  tracker.markReceived(
      ARQBurstInfo{.flow_uid = uid, .burst_num = 1, .seq_num = 4},
        std::chrono::system_clock::now() - 10ms, std::chrono::system_clock::now(), seglength);

  // tracker should have deleted the burst because it's complete.
  BOOST_REQUIRE(tracker.getLastSeqNums().empty());

  ++uid;
  std::map<FlowUID, IndividualMandate> im;
  im[uid] = IndividualMandate{
      .point_value = 1,
      .pt = IndividualMandate::FilePT{-1, std::chrono::duration<float>(3.0f)}};
  tracker.addIndividualMandates(im);

  for (uint16_t i = 1; i <= numsegs; ++i) {
    if (i == 4) {
      continue;
    }
    uint32_t extra = i > 2 ? numsegs * seglength : 0;
    tracker.markReceived(
        ARQBurstInfo{
            .flow_uid = uid, .burst_num = 1, .seq_num = i, .extra = extra},
        std::chrono::system_clock::now() - 10ms, std::chrono::system_clock::now(), seglength);
  }

  // The last seq_num is 3 because we dropped 4
  BOOST_REQUIRE(tracker.getLastSeqNums()[0].seq_num == 3);

  sleep(4);

  // tracker should have deleted the burst because it's expired
  BOOST_REQUIRE(tracker.getLastSeqNums().empty());
}
