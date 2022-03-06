// DB unit tests
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE db
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "events.h"
#include "log.h"
#include "notify.h"
#include "router.h"
#include "test_extra.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "json.hpp"
#include <boost/asio.hpp>

using namespace bamradio;
using namespace std::chrono_literals;
const std::string dbfn("sql.log");
const std::string jsfn("json.log");

template <typename EventInfo> void post_nots(NotificationCenter::Name event) {
  int n = 5;
  for (int i = 0; i < n; ++i) {
    NotificationCenter::shared.post(event, EventInfo{});
  }
}

BOOST_AUTO_TEST_CASE(db_basic) {
  // delete any pre-existing DBs
  test::deletefile(dbfn);
  test::deletefile(jsfn);

  // run a thread and print DB events
  boost::asio::io_context ios;
  auto work = boost::asio::make_work_guard(ios);
  std::thread printer([&] { ios.run(); });
  std::vector<NotificationCenter::SubToken> st;
  size_t count = 0;
  st.push_back(NotificationCenter::shared.subscribe<log::DBEventInfo>(
      log::DBEvent, ios, [&](auto ei) {
        nlohmann::json j = ei;
        // pretty-printed
        // std::cout << ++count << ": " << std::setw(4) << j << std::endl; //
        // one-line
        std::cout << ++count << ": " << j << std::endl;
        // none of these are allowed to fail
        BOOST_REQUIRE(ei.type != log::DBEventInfo::FAIL);
      }));

  // create logger and enable db
  log::Logger l;
  l.enableBackend(log::Backend::STDOUT);
  l.enableBackend(log::Backend::JSON, jsfn, true);
  l.enableBackend(log::Backend::SQL, dbfn, false);

  // generate 5 fake events for every event
  post_nots<dll::SentFrameEventInfo>(dll::SentFrameEvent);
  post_nots<dll::SentSegmentEventInfo>(dll::SentSegmentEvent);
  post_nots<dll::DetectedFrameEventInfo>(dll::DetectedFrameEvent);
  post_nots<dll::ReceivedFrameEventInfo>(dll::ReceivedFrameEvent);
  post_nots<dll::InvalidFrameHeaderEventInfo>(dll::InvalidFrameHeaderEvent);
  post_nots<dll::ReceivedBlockEventInfo>(dll::ReceivedBlockEvent);
  post_nots<dll::ReceivedCompleteSegmentEventInfo>(
      dll::ReceivedCompleteSegmentEvent);
  post_nots<dll::CoDelDelayEventInfo>(dll::CoDelDelayEvent);
  post_nots<dll::CoDelStateEventInfo>(dll::CoDelStateEvent);
  post_nots<dll::NewFlowEventInfo>(dll::NewFlowEvent);
  post_nots<dll::FlowQueuePushEventInfo>(dll::FlowQueuePushEvent);
  post_nots<dll::FlowQueuePopEventInfo>(dll::FlowQueuePopEvent);
  post_nots<dll::ScheduleUpdateEventInfo>(
      dll::ScheduleUpdateEvent);
  std::array<FlowID,5> fid = {
    FlowID{1, 2, 3, 4, 1},
    FlowID{1, 2, 3, 4, 2},
    FlowID{1, 2, 3, 4, 3},
    FlowID{1, 2, 3, 4, 4},
    FlowID{1, 2, 3, 4, 5},
  };
  for (int i = 0; i < 5; ++i) {
    NotificationCenter::shared.post(dll::ScheduleUpdateEvent, dll::ScheduleUpdateEventInfo{i, {{fid[0], 1s}, {fid[1], 2s}}, true, 1s, 0.9s, 1.1s});
  }
  post_nots<collab::CollabRxEventInfo>(collab::CollabRxEvent);
  post_nots<collab::CollabTxEventInfo>(collab::CollabTxEvent);
  post_nots<collab::ServerRxEventInfo>(collab::ServerRxEvent);
  post_nots<collab::ServerTxEventInfo>(collab::ServerTxEvent);
  post_nots<collab::ConnectionEventInfo>(collab::ConnectionEvent);
  post_nots<collab::StateChangeEventInfo>(collab::StateChangeEvent);
  post_nots<collab::CollabPeerEventInfo>(collab::CollabPeerEvent);
  post_nots<collab::ErrorEventInfo>(collab::ErrorEvent);
  post_nots<BurstSendEventInfo>(BurstSendEvent);
  post_nots<uhdfeedback::UHDAsyncEventInfo>(uhdfeedback::UHDAsyncEvent);
  post_nots<uhdfeedback::UHDMsgEventInfo>(uhdfeedback::UHDMsgEvent);
  post_nots<net::RouteDecisionEventInfo>(net::RouteDecisionEvent);
  post_nots<net::RoutingTableUpdateEventInfo>(net::RoutingTableUpdateEvent);
  post_nots<controlchannel::NetworkMapEventInfo>(
      controlchannel::NetworkMapEvent);
  post_nots<gps::GPSEventInfo>(gps::GPSEvent);
  post_nots<OutcomesUpdateEventInfo>(OutcomesUpdateEvent);
  post_nots<log::TextLogEventInfo>(log::TextLogEvent);
  post_nots<C2APIEventInfo>(C2APIEvent);
  post_nots<EnvironmentUpdateEventInfo>(EnvironmentUpdateEvent);
  post_nots<psdsensing::PSDUpdateEventInfo>(psdsensing::PSDUpdateEvent);
  post_nots<ofdm::ModulationEventInfo>(ofdm::ModulationEvent);
  post_nots<ofdm::ChannelEstimationEventInfo>(ofdm::ChannelEstimationEvent);
  post_nots<ofdm::SynchronizationEventInfo>(ofdm::SynchronizationEvent);
  post_nots<decisionengine::ChannelAllocUpdateEventInfo>(
      decisionengine::ChannelAllocUpdateEvent);
  post_nots<decisionengine::ChannelAllocEventInfo>(
      decisionengine::ChannelAllocEvent);
  post_nots<decisionengine::StepEventInfo>(
      decisionengine::StepEvent);
  post_nots<decisionengine::StepOutputEventInfo>(
      decisionengine::StepOutputEvent);


  l.shutdown();
  work.reset();
  st.clear();
  printer.join();
}
