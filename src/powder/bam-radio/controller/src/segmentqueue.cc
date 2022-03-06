//  Copyright © 2017-2018 Stephen Larew

#include "segmentqueue.h"
#include "notify.h"
#include "util.h"

// Disable preconditions when the dust has settled.
#if 0
#ifdef NDEBUG
#undef NOBAMPRECONDITION
#define NOBAMPRECONDITION
#endif
#endif
// Explicitly include bamassert after other project headers.
#include "bamassert.h"

#include <iostream>

#include <boost/format.hpp>

using namespace std::chrono_literals;
using boost::format;

//#define SEGQDEBUG

namespace bamradio {

NodeID
SegmentQueue::effectiveFrameDestNodeID(std::vector<QueuedSegment> const &qsv) {
  if (qsv.empty()) {
    panic("empty segment vector");
  }
  // Some segments have already been selected for the frame.
  // Determine the effective frame destination NodeID.
  NodeID frameDestNodeID = qsv.front().segment->destNodeID();
  for (QueuedSegment const &qs : qsv) {
    NodeID const dnid = qs.segment->destNodeID();
    // If the frame already contains segments destined to all nodes
    // OR the frame contains segments destined to different unicast
    // NodeIDs, then the frame dest is AllNodesID.
    if (dnid == AllNodesID ||   // Frame is broadcast
        dnid != frameDestNodeID // Frame is mixed unicast
    ) {
      frameDestNodeID = AllNodesID;
      break;
    }
  }
  return frameDestNodeID;
}

std::pair<size_t, bool>
DRRFQSegQ::FlowQueue::numQueuedEndogenous(global_clock::time_point) {
  return {numQueued(), false};
}

DRRFQSegQ::DRRFQSegQ(uint64_t purgeInterval_, FlowQueueMaker makeFlowQueue,
                     ScheduleMaker makeSchedule_)
    : purgeInterval(purgeInterval_), makeSchedule(makeSchedule_),
      _currentRound(0), _nextPurge(purgeInterval_), _numQueued(0),
      _makeFlowQueue(makeFlowQueue) {}

std::vector<QueuedSegment>
DRRFQSegQ::pop(std::chrono::nanoseconds const minDesiredValue,
               bool const allowMixedUnicastDestNodes,
               bool const allowMixedBroadcastUnicastDestNodes) {
  auto const gnow = FlowQueue::global_clock::now();
  auto const lnow = FlowQueue::local_clock::now();

  std::vector<QueuedSegment> vqs;

  SchedulerFlowQueue::value frameDebits = 0ns, roundDebits;

  // Reactivate flow queues that went idle and had endogenous enqueues.
  std::vector<SchedulerFlowQueue *> toreset;
  for (auto &sfqp : _flowQueues) {
    auto &sfq = sfqp.second;
    if (sfq.idle(_currentRound)) {
      auto nqe = sfq.q->numQueuedEndogenous(gnow);
      if (nqe.first > 0 && nqe.second) {
        sfq.activate();
        _newFlows.emplace_back(sfqp.first);
        toreset.push_back(&sfqp.second);
      }
    }
  }
  bool forceMakeSchedule = true;
  for (auto sfq : toreset) {
    sfq->reset(
        makeSchedule(gnow, _currentRound, sfq->q->flowId, forceMakeSchedule)
            .quantumCredit);
    forceMakeSchedule = false;
    ;
  }

  // Outer loop over rounds.
  do {
    // Reset debits for this round. We will bail if this round yields nothing.
    roundDebits = 0s;

    // Accumulate number of queued segments in the filtered subset of oldFlows.
    auto const oldFlowsTotalNumQueued = std::accumulate(
        begin(_oldFlows), end(_oldFlows), (size_t)0, [this](auto v, auto fid) {
          return v + _flowQueues.at(fid).q->numQueued();
        });

    // If no active and ready new or old flows are queued, then move in the out
    // queues to start a new round.
    if (_newFlows.empty() &&
        (_oldFlows.empty() ||
         // Even if _oldFlows isn't empty, all of its contained flowqueues
         // might be empty and thus _oldFlows isn't ready. If so, then we start
         // a new round if its a new frame to avoid starving flowqueues.
         (vqs.empty() && oldFlowsTotalNumQueued == 0))) {

      // Starting new round.
      ++_currentRound;

#ifdef SEGQDEBUG
      std::cerr << "SEGQDEBUG new round: " << _currentRound << std::endl;
#endif

      // Move empty old flows out.
      std::copy(begin(_oldFlows), end(_oldFlows),
                std::back_inserter(_oldFlowsOut));
      _oldFlows.clear();

      // Start new round with non-idle out flows.
      std::copy_if(begin(_oldFlowsOut), end(_oldFlowsOut),
                   std::back_inserter(_oldFlows), [this](auto const &fid) {
                     auto const moveIn =
                         !_flowQueues.at(fid).idle(_currentRound);
#ifdef SEGQDEBUG
                     if (moveIn) {
                       std::cerr << "SEGQDEBUG moveIn flow "
                                 << fid.description() << std::endl;
                     } else {
                       std::cerr << "SEGQDEBUG bye flow " << fid.description()
                                 << std::endl;
                     }
#endif
                     return moveIn;
                   });
      _oldFlowsOut.clear();

      // Add credits.
      for (FlowID const &fid : _oldFlows) {
        auto const s = makeSchedule(gnow, _currentRound, fid, false);
        SchedulerFlowQueue &sfq = _flowQueues.at(fid);
        sfq.debit(s.quantumCredit);
        sfq.cap(s.quantumCredit);
      }

      if (_currentRound == _nextPurge) {
        purgeExpiredFlowQueues(gnow);
        _nextPurge += purgeInterval;
      }
    }

#ifdef SEGQDEBUG
    std::cerr << "SEGQDEBUG preservice flows new\n";
    for (auto const &fid : _newFlows) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
    std::cerr << "SEGQDEBUG preservice flows old" << std::endl;
    for (auto const &fid : _oldFlows) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
    std::cerr << "preservice flows out" << std::endl;
    for (auto const &fid : _oldFlowsOut) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
#endif

    /// Visit the fq and service it until it runs a deficit,
    /// it's empty, or the frame is full.
    auto const serviceFlowQueue = [&](SchedulerFlowQueue &sfq) {
      SchedulerFlowQueue::value flowDebits = 0s;
      auto const s = makeSchedule(gnow, _currentRound, sfq.q->flowId, false);

      while (frameDebits < minDesiredValue && sfq.balance() > 0s &&
             sfq.q->numQueued() > 0) {
        // We're visiting the fq so mark it as active this round.
        sfq.setLastActiveRound(_currentRound);
        int64_t num_dropped = 0;
        QueuedSegment qs;
        auto const dequeued = sfq.q->dequeue(qs, num_dropped, gnow, lnow);
        _numQueued -= num_dropped;
        assert(_numQueued >= 0);
        NotificationCenter::shared.post(
            dll::FlowQueuePopEvent,
            dll::FlowQueuePopEventInfo{
                sfq.q->flowId, sfq.q->numQueued(), sfq.q->bytesQueued(),
                _currentRound, sfq.balance(), s.quantumCredit, s.dequeueDebit});
        if (!dequeued) {
          continue;
        }
        --_numQueued;
        assert(_numQueued >= 0);
        sfq._lastDestNodeID = qs.segment->destNodeID(); // we're friends
        vqs.push_back(qs);
        sfq.credit(s.dequeueDebit);
        frameDebits += s.dequeueDebit;
        flowDebits += s.dequeueDebit;
      }

      return flowDebits;
    };

    auto const servicesFlows = [&](std::deque<FlowID> &flows, bool newFlows) {
      // Flows that are filtered out are pushed on this stack to be
      // requeued later.
      std::stack<FlowID> requeue;

      // To avoid infinite loops over _oldFlows when all filtered flows in
      // _oldFlows are empty, put empty flows here temporarily.
      std::vector<FlowID> oldFlowsEmpty;

      while (frameDebits < minDesiredValue && !flows.empty()) {
        FlowID fid = flows.front();
        flows.pop_front();
        SchedulerFlowQueue &sfq = _flowQueues.at(fid);

        // Special case: If the fq is an HoldQueue, then try to make new flow
        // queue and transfer to it if not a HoldQueue.  If new flow queue is
        // still a HoldQueue, then skip the flow.
        auto const phq = std::dynamic_pointer_cast<HoldQueue>(sfq.q);
        if (phq) {
          auto const nq = _makeFlowQueue(phq->flowId);
          if (std::dynamic_pointer_cast<HoldQueue>(nq)) {
            // We don't want the holdqueue to go inactive so touch it.
            // sfq.setLastActiveRound(_currentRound);
            //
            // Nevermind. when a holdqueue is created in push(), we call
            // sfq.activate() to activate the sfq.  sfq will then be active
            // (non-idle) until setLastActiveRound() is called on it. However,
            // we will never call setLastActiveRound on a holdqueue and thus
            // the sfq always remains active. Even during the next round...
            assert(!sfq.idle(_currentRound + 1));
            requeue.push(fid);
            continue;
          } else {
            _numQueued += phq->transferTo(nq);
            assert(_numQueued >= 0);
            sfq.q = nq;
          }
        }

#if 1
        // Let dequeue() do any necessary dropping.
#else
        // Preemptively drop bursts.
        auto const pfbq = std::dynamic_pointer_cast<FIFOBurstQueue>(sfq.q);
        if (pfbq) {
          while (pfbq->numQueued() > 0 &&
                 pfbq->headBurstBytesRemaining(gnow) < 0) {
            pfbq->dropHeadBurst();
          }
        }
#endif

        // Skip flow if nothing queued.
        if (sfq.q->numQueued() == 0) {
          oldFlowsEmpty.push_back(fid);
          continue;
        }

        // Skip flow (and requeue later) if frame destNodeID filtering needed.
        if ((!allowMixedUnicastDestNodes ||
             !allowMixedBroadcastUnicastDestNodes) &&
            !vqs.empty()) {
          NodeID const frameDestNodeID =
              SegmentQueue::effectiveFrameDestNodeID(vqs);
          NodeID const flowDestNodeID = sfq.destNodeID();

          if ((!allowMixedUnicastDestNodes &&
               frameDestNodeID != AllNodesID &&  // Unicast frame
               flowDestNodeID != AllNodesID &&   // Flow is also unicast
               flowDestNodeID != frameDestNodeID // Frame would be mixed unicast
               ) ||
              (!allowMixedBroadcastUnicastDestNodes &&
               ((frameDestNodeID == AllNodesID && // Frame is broadcast
                 flowDestNodeID != AllNodesID     // Flow is unicast
                 ) ||                             // check other way around
                (frameDestNodeID != AllNodesID && // Frame is unicast
                 flowDestNodeID == AllNodesID     // Flow is broadcast
                 )))) {
            requeue.push(fid);
            continue;
          }
        }

        roundDebits += serviceFlowQueue(sfq);

        // New flows get move to old[Out] always.
        // Old flows may remain in oldFlows if they retain a balance.
        (sfq.balance() > 0s ? _oldFlows : _oldFlowsOut).push_back(fid);
      }

      // Requeue the filtered flowqueues.
      while (!requeue.empty()) {
        flows.push_front(requeue.top());
        requeue.pop();
      }

      // And restore old empty flowqueues.
      assert(!newFlows || oldFlowsEmpty.empty());
      for (FlowID const &fid : oldFlowsEmpty) {
        flows.push_back(fid);
      }
    };

    // Service new flows before old flows.
    servicesFlows(_newFlows, true);
    servicesFlows(_oldFlows, false);
#ifdef SEGQDEBUG
    std::cerr << "postservice flows new" << std::endl;
    for (auto const &fid : _newFlows) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
    std::cerr << "postservice flows old" << std::endl;
    for (auto const &fid : _oldFlows) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
    std::cerr << "postservice flows out" << std::endl;
    for (auto const &fid : _oldFlowsOut) {
      std::cerr << "\tflow " << fid.description() << std::endl;
    }
#endif

    // If the frame is full or the round produced nothing, we're done.
  } while (frameDebits < minDesiredValue && roundDebits > 0s);

  return vqs;
}

void DRRFQSegQ::push(QueuedSegment const &qs) {
  // Identify flow
  auto const flowId = qs.segment->flowID();

  // Find exist flowqueue or make new one.
  auto flowQueueIt = _flowQueues.find(flowId);

  if (flowQueueIt == _flowQueues.end()) {
    // No existing flowqueue so make new one.
    FlowQueue::sptr q = _makeFlowQueue(flowId);

    flowQueueIt =
        _flowQueues.emplace(flowId, SchedulerFlowQueue(q, -10000)).first;

    auto &qid = *q.get();

    NotificationCenter::shared.post(
        dll::NewFlowEvent,
        dll::NewFlowEventInfo{q->flowId, _currentRound, typeid(qid).name()});
  } else {
    // Special case: If the fq is a HoldQueue, then try to make new flow queue
    // and transfer to it if not a HoldQueue.
    auto const phq =
        std::dynamic_pointer_cast<HoldQueue>(flowQueueIt->second.q);
    if (phq) {
      auto const nq = _makeFlowQueue(flowId);
      if (!std::dynamic_pointer_cast<HoldQueue>(nq)) {
        _numQueued += phq->transferTo(nq);
        assert(_numQueued >= 0);
        flowQueueIt->second.q = nq;
      }
    }
  }

  SchedulerFlowQueue &sfq = flowQueueIt->second;

  // Enqueue
  int64_t num_dropped = 0;
  sfq.q->enqueue(qs, num_dropped);
  _numQueued += 1 - num_dropped;
  assert(_numQueued >= 0);

  // Activate and endow credit if necessary.
  if (sfq.idle(_currentRound)) {
#ifndef NDEBUG
    for (auto const &fid : _oldFlows) {
      assert(flowId != fid);
    }
    for (auto const &fid : _oldFlowsOut) {
      assert(flowId != fid);
    }
#endif
    sfq.activate();
    sfq.reset(makeSchedule(std::chrono::system_clock::now(), _currentRound,
                           flowId, true)
                  .quantumCredit);
    _newFlows.emplace_back(flowId);
  }

  NotificationCenter::shared.post(
      dll::FlowQueuePushEvent,
      dll::FlowQueuePushEventInfo{sfq.q->flowId, sfq.q->numQueued(),
                                  sfq.q->bytesQueued(), _currentRound,
                                  sfq.balance()});
}

#if 0
size_t DRRFQSegQ::numReady() const {
  auto const gnow = FlowQueue::global_clock::now();
  size_t nr = 0;
  for (auto const &flowList : {_newFlows, _oldFlows, _oldFlowsOut}) {
    for (auto const &fid : flowList) {
      auto const s = makeSchedule(gnow, _currentRound, fid, false);
      if (s.quantumCredit > 0s) {
        // FlowID fid is scheduled so count its numqueued:
        nr += _flowQueues.at(fid).q->numQueuedEndogenous();
      }
    }
  }
  return nr;
}
#endif

std::map<FlowID, DRRFQSegQ::SchedulerFlowQueue> const &DRRFQSegQ::flowQueues() {
  return _flowQueues;
}

void DRRFQSegQ::purgeExpiredFlowQueues(
    FlowQueue::global_clock::time_point gnow) {
  for (auto it = _flowQueues.begin(); it != _flowQueues.end();) {
    auto nqe = it->second.q->numQueuedEndogenous(gnow);
    if (it->second.idle(_currentRound, purgeInterval) && nqe.first == 0 &&
        !nqe.second) {
      bamassert((std::cerr << "deleteing flowqueue " << it->first.description()
                           << std::endl,
                 true));
      it = _flowQueues.erase(it);
    } else {
      ++it;
    }
  }
}

void DRRFQSegQ::purgeFlowQueues(std::set<FlowID> const &fids) {
  // Trick time dropping queues to think it's the future.
  auto const gnow = FlowQueue::global_clock::now() + 10000000s;
  auto const lnow = FlowQueue::local_clock::now() + 10000000s;

  // Delete the sfqs.
  for (auto const &fid : fids) {
    auto sfqit = _flowQueues.find(fid);
    if (sfqit == _flowQueues.end()) {
      log::text(format("purgeFlowQueues failed deleting unknown flow %s") %
                fid.description());
      continue;
    }
    auto &sfq = sfqit->second;
    auto hqp = std::dynamic_pointer_cast<HoldQueue>(sfq.q);
    if (hqp) {
      // Hold queues lie on enqueue and claim to drop everything.
      // -> No adjustment of _numQueued needed.
    } else {
      while (sfq.q->numQueued() > 0) {
        int64_t num_dropped = 0;
        QueuedSegment qs;
        auto const dequeued = sfq.q->dequeue(qs, num_dropped, gnow, lnow);
        _numQueued -= num_dropped + (dequeued ? 1 : 0);
        assert(_numQueued >= 0);
      }
    }
    _flowQueues.erase(sfqit);
  }
  // Remove from lists.
  for (std::deque<FlowID> *flowList : std::array<std::deque<FlowID> *, 3>{
           &_newFlows, &_oldFlows, &_oldFlowsOut}) {
    std::stack<FlowID> requeue;
    while (!flowList->empty()) {
      auto &fid = flowList->back();
      if (fids.count(fid) == 0) {
        requeue.emplace(fid);
      }
      flowList->pop_back();
    }
    while (!requeue.empty()) {
      flowList->emplace_back(requeue.top());
      requeue.pop();
    }
  }
}

void DRRFQSegQ::FlowQueue::trackSegment(dll::Segment::sptr s) {
  if (!s)
    return;
  auto const l = s->length();
  auto const pl = s->packetLength();
  _stats.minLength = std::min(_stats.minLength, l);
  _stats.maxLength = std::max(_stats.maxLength, l);
  _stats.minPayloadLength = std::min(_stats.minPayloadLength, pl);
  _stats.maxPayloadLength = std::max(_stats.maxPayloadLength, pl);
}

DRRFQSegQ::HoldQueue::HoldQueue(FlowID id) : FlowQueue(id) {}
void DRRFQSegQ::HoldQueue::enqueue(QueuedSegment const &v, int64_t &num_dropped,
                                   local_clock::time_point now) {
  trackSegment(v.segment);
  // Enqueued held packets must not be counted as "queued" in the DRRFQSegQ so
  // "drop" here.  See transferTo() for other part of this accounting.
  num_dropped = 1;
  auto const l = v.segment->length();
  _stats.bytesDropped += l;
  _segments.emplace(HeldSegment{v, now});
}
bool DRRFQSegQ::HoldQueue::dequeue(QueuedSegment &, int64_t &,
                                   global_clock::time_point,
                                   local_clock::time_point) {
  panic("Never dequeue from HoldQueue.");
  return false;
}
NodeID DRRFQSegQ::HoldQueue::headDestNodeID() const {
  if (_segments.empty()) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _segments.front().qs.segment->destNodeID();
}
size_t DRRFQSegQ::HoldQueue::numQueued() const { return 0; }
size_t DRRFQSegQ::HoldQueue::bytesQueued() const { return 0; }
void DRRFQSegQ::HoldQueue::handleARQInfo(dll::ARQBurstInfo,
                                         local_clock::time_point,
                                         global_clock::time_point) {}
int64_t DRRFQSegQ::HoldQueue::transferTo(FlowQueue::sptr q) {
  // All segments will now finally be enqueued except those that enqueue()
  // drops.
  int64_t nq = _segments.size();
  while (!_segments.empty()) {
    auto const &hs = _segments.front();
    int64_t nd = 0;
    q->enqueue(hs.qs, nd, hs.enqueueTime);
    nq -= nd;
    _segments.pop();
  }
  _stats.bytesEnqueued = 0;
  assert(nq >= 0);
  return nq;
}

DRRFQSegQ::DropAllQueue::DropAllQueue(FlowID id) : FlowQueue(id) {}
void DRRFQSegQ::DropAllQueue::enqueue(QueuedSegment const &v,
                                      int64_t &num_dropped,
                                      local_clock::time_point) {
  trackSegment(v.segment);
  num_dropped = 1;
  _stats.bytesDropped += v.segment->length();
}
bool DRRFQSegQ::DropAllQueue::dequeue(QueuedSegment &, int64_t &num_dropped,
                                      global_clock::time_point,
                                      local_clock::time_point) {
  num_dropped = 0;
  return false;
}
NodeID DRRFQSegQ::DropAllQueue::headDestNodeID() const {
  panic("No head destNodeID because no packets queued for flow.");
}
size_t DRRFQSegQ::DropAllQueue::numQueued() const { return 0; }
size_t DRRFQSegQ::DropAllQueue::bytesQueued() const { return 0; }
void DRRFQSegQ::DropAllQueue::handleARQInfo(dll::ARQBurstInfo,
                                            local_clock::time_point,
                                            global_clock::time_point) {}

DRRFQSegQ::DestCoalesceQueue::DestCoalesceQueue(FlowID id) : FlowQueue(id) {}

/// Enqueue segment v at time t.
void DRRFQSegQ::DestCoalesceQueue::enqueue(QueuedSegment const &v,
                                           int64_t &num_dropped,
                                           local_clock::time_point) {
  trackSegment(v.segment);
  if (_segment.valid()) {
    num_dropped = 1;
    _stats.bytesDropped += _segment.segment->length();
  } else {
    num_dropped = 0;
  }
  _stats.bytesEnqueued += v.segment->length();
  _segment = v;
}

/// Return true if a segment is dequeued into result.
bool DRRFQSegQ::DestCoalesceQueue::dequeue(QueuedSegment &result,
                                           int64_t &num_dropped,
                                           global_clock::time_point,
                                           local_clock::time_point) {
  num_dropped = 0;
  if (!_segment.valid()) {
    return false;
  }
  result = _segment;
  _stats.bytesDequeued += result.segment->length();
  _segment = QueuedSegment{};
  return true;
}

NodeID DRRFQSegQ::DestCoalesceQueue::headDestNodeID() const {
  if (!_segment.valid()) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _segment.segment->destNodeID();
}

size_t DRRFQSegQ::DestCoalesceQueue::numQueued() const {
  return _segment.valid() ? 1 : 0;
}

size_t DRRFQSegQ::DestCoalesceQueue::bytesQueued() const {
  return _segment.valid() ? _segment.segment->length() : 0;
}
void DRRFQSegQ::DestCoalesceQueue::handleARQInfo(dll::ARQBurstInfo,
                                                 local_clock::time_point,
                                                 global_clock::time_point) {}

DRRFQSegQ::FIFONoDropQueue::FIFONoDropQueue(FlowID id)
    : FlowQueue(id), _numBytes(0) {}

/// Enqueue segment v at time t.
void DRRFQSegQ::FIFONoDropQueue::enqueue(QueuedSegment const &v,
                                         int64_t &num_dropped,
                                         local_clock::time_point) {
  trackSegment(v.segment);
  num_dropped = 0;
  auto const l = v.segment->length();
  _stats.bytesEnqueued += l;
  _segments.push(v);
  _numBytes += l;
}

/// Return true if a segment is dequeued into result.
bool DRRFQSegQ::FIFONoDropQueue::dequeue(QueuedSegment &result,
                                         int64_t &num_dropped,
                                         global_clock::time_point,
                                         local_clock::time_point) {
  num_dropped = 0;
  if (_segments.empty()) {
    return false;
  }
  result = _segments.front();
  auto const l = result.segment->length();
  _stats.bytesDequeued += l;
  _segments.pop();
  _numBytes -= l;
  return true;
}

NodeID DRRFQSegQ::FIFONoDropQueue::headDestNodeID() const {
  if (_segments.empty()) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _segments.front().segment->destNodeID();
}

size_t DRRFQSegQ::FIFONoDropQueue::numQueued() const {
  return _segments.size();
}

size_t DRRFQSegQ::FIFONoDropQueue::bytesQueued() const { return _numBytes; }
void DRRFQSegQ::FIFONoDropQueue::handleARQInfo(dll::ARQBurstInfo,
                                               local_clock::time_point,
                                               global_clock::time_point) {}

DRRFQSegQ::FIFOBurstQueue::FIFOBurstQueue(
    FlowID id, boost::asio::io_context &ioctx,
    std::function<void(std::function<void()>)> runInSegQCtx,
    std::function<void()> onEndogenousEnqueue,
    std::chrono::nanoseconds burstdur, std::chrono::nanoseconds resendTimeout_,
    std::chrono::nanoseconds maxInterSegmentArrivalDur,
    int newBurstInterSegmentArrivalRatio_)
    : FlowQueue(id), resendTimeout(resendTimeout_),
      newBurstInterSegmentArrivalRatio(newBurstInterSegmentArrivalRatio_),
      _burstdur(burstdur), _numBytes(0), _retxNumDropped(0),
      _rtoDur(resendTimeout), _ioctx(ioctx), _runInSegQCtx(runInSegQCtx),
      _onEndogenousEnqueue(onEndogenousEnqueue),
      _maxInterSegmentArrivalDur(maxInterSegmentArrivalDur), _lastArrivalST(),
      _lastArrivalDur(-1s), _burstReset(true), _burstIdx(0),
      _nextBurstSeqNum(1) {}

/// Enqueue segment v at time t.
void DRRFQSegQ::FIFOBurstQueue::enqueue(QueuedSegment const &vgiven,
                                        int64_t &num_dropped,
                                        local_clock::time_point) {
  auto v = vgiven;

  // Stamp segments needing ARQ with tracking info. If segment is already
  // stamped, then read burst info and update stamp for this leg of the
  // multihop.

  auto arqsegment =
      std::dynamic_pointer_cast<net::ARQIP4PacketSegment>(v.segment);
  auto ip4segment = std::dynamic_pointer_cast<net::IP4PacketSegment>(v.segment);

  if (!ip4segment) {
    panic("FIFOBurstQueue only works with IP4PacketSegment.");
  }
  if (!arqsegment) {
    arqsegment = std::make_shared<net::ARQIP4PacketSegment>(*ip4segment);
    v.segment = std::static_pointer_cast<dll::Segment>(arqsegment);
  }

  trackSegment(v.segment);
  num_dropped = _retxNumDropped;
  _retxNumDropped = 0;
  auto const l = v.segment->length();

  if (arqsegment->arqDataSet() ||
      arqsegment->destNodeID() != flowId.dstIPNodeID()) {
    // The segment is being enqueued again or came through multihop or is
    // multihop. Drop it for now. :( #multihop
    ++num_dropped;
    _stats.bytesDropped += l;
    return;
  } else {
    // This segment has not been stamped with ARQ info before.
    //
    // Do burst detection.
    //
    // Add current burst index or next burst index and set seqnum (starting at
    // 1, not 0).

    // Do burst detection based on inter segment arrival times.
    auto const thisArrivalST = v.segment->sourceTime();
    assert(thisArrivalST != decltype(_lastArrivalST)());
    auto thisArrivalDur = thisArrivalST - _lastArrivalST;
    _lastArrivalST = thisArrivalST;

    if (thisArrivalDur < 0s && !_burstReset) {
      log::text(
          "WARNING: Unstamped segment enqueued with out-of-order source time.");
      thisArrivalDur = 1ms;
    }

    if (_burstReset ||
        // New burst criterion: roughly order of magnitude difference in
        // intervals.
        (_lastArrivalDur > 0s && thisArrivalDur / _lastArrivalDur >=
                                     newBurstInterSegmentArrivalRatio) ||
        (thisArrivalDur > _maxInterSegmentArrivalDur)) {
      // New burst.

      if (!_burstReset) {
        // It's a new burst and the burst index counter hasn't been
        // incremented/reset yet. So do it now.

        if (!_trackedBursts.empty()) {
          auto const &tb = _trackedBursts.back();
          auto &bh = _burstHistory[tb.burstIdx];
          if (!bh.burstSizeKnown()) {
            bh.setBurstSize(tb.sizeBytes);
            bamassert((bh.eobAckedBeforeBurstSizeKnown &&
                           (log::text(format("FlowUID %u burst %u "
                                             "eobAckedBeforeBurstSizeKnown") %
                                      flowId.flowUID() % (int)tb.burstIdx),
                            true),
                       true));
            bamassert((log::text(format("Estimated burst size for FlowUID "
                                        "%u is %ld (enqueue)") %
                                 flowId.flowUID() % bh.sizeBytes),
                       true));
          }
        }

        _burstReset = true;
        _burstIdx += 1;
        _nextBurstSeqNum = 1;
      }

      // Remove previous bursts that have nothing left in them.
      while (!_trackedBursts.empty() &&
             _trackedBursts.front().burstRemaining <= 0) {
        bamprecondition(_trackedBursts.front().burstRemaining == 0);
        _trackedBursts.pop_front();
      }

      _trackedBursts.emplace_back(
          TrackedBurst{_burstIdx, (int64_t)l, (int64_t)l});
      bamassert((std::cerr << "new tb 1 " << flowId.flowUID() << " "
                           << (int)_burstIdx << std::endl,
                 true));
      _burstHistory[_burstIdx] = BurstHistory();
      _lastArrivalDur = -1s;
    } else {
      // This is the second or later segment of the burst.
      _lastArrivalDur = thisArrivalDur;
      auto &tb = _trackedBursts.back();
      bamprecondition(tb.sizeBytes > 0);
      tb.sizeBytes += l;
      tb.burstRemaining += l;
    }

    // Stamp the segment with its arq data. arqExtra will be set at pop time.
    arqsegment->setArqData(net::ARQIP4PacketSegment::ARQData{
        .arqExtra = 0, .seqNum = _nextBurstSeqNum, .burstNum = _burstIdx});

    _burstHistory[_burstIdx].numSegments = _nextBurstSeqNum;
    ++_nextBurstSeqNum;

    // This burstIdx is being used now:
    _burstReset = false;
  }

  _stats.bytesEnqueued += l;
  _segments.push_back(v);
  _numBytes += l;
}

/// Return true if a segment is dequeued into result.
bool DRRFQSegQ::FIFOBurstQueue::dequeue(QueuedSegment &result,
                                        int64_t &num_dropped,
                                        global_clock::time_point gnow,
                                        local_clock::time_point lnow) {
  if (!_burstAcksForResend.empty()) {
    (void)enqueueResends(gnow);
  }

  num_dropped = _retxNumDropped;
  _retxNumDropped = 0;

  while (!_segments.empty()) {
    auto candidate = _segments.front();
    _segments.pop_front();

    // Account for this segment's length.
    auto const l = candidate.segment->length();
    _numBytes -= l;
    bamprecondition(!_trackedBursts.empty());
    auto &tb = _trackedBursts.front();
    tb.burstRemaining -= l;

    auto const thisBurstIdx = tb.burstIdx;
    auto &bh = _burstHistory[thisBurstIdx];

    // If there are more bursts in the queue and this burst has 0 bytes
    // remaining, then this burst is over.
    if (tb.burstRemaining <= 0 && _trackedBursts.size() > 1) {
      bamprecondition(tb.burstRemaining == 0);
      bamprecondition(tb.sizeBytes > 0);
      bamprecondition(bh.burstSizeKnown());
      bamassert((std::cerr << "bye tb 1 " << flowId.flowUID() << " "
                           << (int)tb.burstIdx << std::endl,
                 true));
      _trackedBursts.pop_front();
    }
    // Otherwise, if the current burst is all that's in the queue, do end of
    // burst detection.
    else if (_trackedBursts.size() == 1 && /*NEWSTAMP &&*/
             // must have enqueued at least 1 packet in the current burst
             !_burstReset &&
             // 1+ packet in current burst was enqueued, do time based burst
             // detection.
             (
                 // Is ratio of inter segment arrivals past threshold?
                 (_lastArrivalDur > 0s &&
                  (gnow - _lastArrivalST) / _lastArrivalDur >=
                      newBurstInterSegmentArrivalRatio) ||
                 //  Is the time since last arrival greater than absolute
                 //  threshold?
                 ((gnow - _lastArrivalST) > _maxInterSegmentArrivalDur))) {
      // Right now, it's been long enough since the last enqueue arrival
      // source time that we can declare the burst has been completely sourced
      // to us.
      if (!bh.burstSizeKnown()) {
        bh.setBurstSize(tb.sizeBytes);
        bamassert(
            (bh.eobAckedBeforeBurstSizeKnown &&
                 (log::text(
                      format(
                          "FlowUID %u burst %u eobAckedBeforeBurstSizeKnown") %
                      flowId.flowUID() % (int)tb.burstIdx),
                  true),
             true));
        bamassert((log::text(format("Estimated burst size for FlowUID "
                                    "%u is %ld (dequeue)") %
                             flowId.flowUID() % bh.sizeBytes),
                   true));
      }
      if (tb.burstRemaining <= 0) {
        // Detected end of burst and this burst has nothing left. delete it.
        bamassert((std::cerr << "bye tb 2 " << flowId.flowUID() << " "
                             << (int)_trackedBursts.front().burstIdx
                             << std::endl,
                   true));
        _trackedBursts.pop_front();
      }
      _lastArrivalDur = -1s;
      _burstReset = true;
      _burstIdx += 1;
      _nextBurstSeqNum = 1;
    }
    // Finally, if the current burst is all that's in the queue and we
    // previously detected an end of burst, then check to see if we need to
    // finish this burst.
    else if (_trackedBursts.size() == 1 && _burstReset &&
             tb.burstRemaining <= 0) {
      // An end-of-burst was detected (because burstReset is still true) and the
      // burst has nothing left in it so end it now.
      bamprecondition(tb.burstRemaining == 0);
      bamprecondition(tb.sizeBytes > 0);
      bamprecondition(bh.burstSizeKnown());
      bamprecondition(_segments.empty());
      bamassert((std::cerr << "bye tb 3 " << flowId.flowUID() << " "
                           << (int)tb.burstIdx << std::endl,
                 true));
      _trackedBursts.pop_front();
    }

    auto arqsegment =
        std::static_pointer_cast<net::ARQIP4PacketSegment>(candidate.segment);
    auto arqdata = arqsegment->arqData();
    bamprecondition(arqdata.burstNum == thisBurstIdx);

    // Drop if acknowledged.
    if (bh.ackedSeqnum >= arqdata.seqNum &&
        // Except don't drop if we need to force resend the last segment in the
        // burst.
        !(bh.eobAckedBeforeBurstSizeKnown &&
          arqdata.seqNum == bh.numSegments)) {
      bamassert((std::cerr << "dropping acked " << arqdata.seqNum << std::endl,
                 true));
      continue;
    }

    // Tag the burst size here, just before popping off so we know the
    // latest burst size estimate is used.
    bh.maxSeqNumPopped = std::max(bh.maxSeqNumPopped, arqdata.seqNum);
    auto const thisBurstSize = bh.sizeBytes;
    arqdata.arqExtra =
        thisBurstSize < 0
            ? 0
            : thisBurstSize <= net::ARQIP4PacketSegment::ARQData::MaxFileSize
                  ? thisBurstSize
                  : net::ARQIP4PacketSegment::ARQData::MaxFileSize;
    // FIXME what to do about bursts too large?
    arqsegment->setArqData(arqdata);

    // Use it or lose it: drop if this segment is too late.
    if (gnow - candidate.segment->sourceTime() < _burstdur) {
      retainSegment(candidate, lnow);
      result = std::move(candidate);
      _stats.bytesDequeued += l;
      startRTO(thisBurstIdx, lnow, gnow, false);
      return true;
    } else {
      ++num_dropped;
      _stats.bytesDropped += l;
      bamassert((std::cerr << "dropping late " << flowId.flowUID() << " "
                           << (int)arqdata.burstNum << " " << arqdata.seqNum
                           << std::endl,
                 true));
      unretainSegment(candidate);
    }
  }
  return false;
}

void DRRFQSegQ::FIFOBurstQueue::doNonEnqueueBurstDetection(
    global_clock::time_point gnow) {
  if (_trackedBursts.size() != 1) {
    // We only do burst detection for the most recent burst (i.e. largest burst
    // index).  If there is more than one burst, then burst detection is only
    // performed in enqueue() because either 1) the burst will be detected by
    // enqueue() while the number of bursts is two or more or 2) dequeue() will
    // reduce the number of tracked bursts down to one and then dequeue() and
    // this function can both do burst detction on the front (and earliest)
    // burst.
    bamassert((log::text(format("doNonEnqueueBurstDetection %u no") %
                         flowId.flowUID()),
               true));
    return;
  }
  auto const &tb = _trackedBursts.front();

  // The current burst is all that's in the queue, do end of burst detection.
  // must have enqueued at least 1 packet in the current burst
  if (!_burstReset &&
      // 1+ packet in current burst was enqueued, do time based burst
      // detection.
      (
          // Is ratio of inter segment arrivals past threshold?
          (_lastArrivalDur > 0s && (gnow - _lastArrivalST) / _lastArrivalDur >=
                                       newBurstInterSegmentArrivalRatio) ||
          //  Is the time since last arrival greater than absolute
          //  threshold?
          ((gnow - _lastArrivalST) > _maxInterSegmentArrivalDur))) {
    // Right now, it's been long enough since the last enqueue arrival
    // source time that we can declare the burst has been completely sourced
    // to us.
    bamassert(
        (log::text(format("doNonEnqueueBurstDetection %u burst %u size known") %
                   flowId.flowUID() % (int)tb.burstIdx),
         true));
    auto &bh = _burstHistory[tb.burstIdx];
    if (!bh.burstSizeKnown()) {
      bh.setBurstSize(tb.sizeBytes);
      bamassert(
          (bh.eobAckedBeforeBurstSizeKnown &&
               (log::text(
                    format("FlowUID %u burst %u eobAckedBeforeBurstSizeKnown") %
                    flowId.flowUID() % (int)tb.burstIdx),
                true),
           true));
      bamassert((log::text(format("Estimated burst size for FlowUID "
                                  "%u is %ld (dequeue)") %
                           flowId.flowUID() % bh.sizeBytes),
                 true));
    }
    if (tb.burstRemaining <= 0) {
      // Detected end of burst and this burst has nothing left. delete it.
      bamassert((std::cerr << "bye tb 2 " << flowId.flowUID() << " "
                           << (int)_trackedBursts.front().burstIdx << std::endl,
                 true));
      _trackedBursts.pop_front();
    }
    _lastArrivalDur = -1s;
    _burstReset = true;
    _burstIdx += 1;
    _nextBurstSeqNum = 1;
  }
}

void DRRFQSegQ::FIFOBurstQueue::startRTO(uint8_t burstIdx,
                                         local_clock::time_point lorigin,
                                         global_clock::time_point gorigin,
                                         bool restart) {
  // Create or find the timer.
  auto nit = _rtoTimer.emplace(burstIdx, boost::asio::steady_timer(_ioctx));
  bool timerExists = !nit.second;
  auto &rtoTimer = nit.first->second;

  if (timerExists) {
    if (!restart) {
      // Don't restart the timer cause it already exists.
      bamassert((std::cerr << "flowuid " << flowId.flowUID()
                           << " timer exists, not restarting\n",
                 true));
      return;
    } else {
      // Timer already existed. Cancel outstanding waits.
      bamassert((std::cerr << "flowuid " << flowId.flowUID()
                           << " timer exists, cancel and reset\n",
                 true));
    }
  } else {
    bamassert(
        (std::cerr << "flowuid " << flowId.flowUID() << " timer created\n",
         true));
  }

  auto const rtoLTimeout = lorigin + _rtoDur;
  auto const rtoGTimeout = gorigin + _rtoDur;

  bamassert((std::cerr << "it's now " << lorigin.time_since_epoch().count()
                       << " " << rtoLTimeout.time_since_epoch().count()
                       << std::endl,
             true));

  rtoTimer.expires_at(rtoLTimeout);

  rtoTimer.async_wait([this, burstIdx, rtoLTimeout,
                       rtoGTimeout](auto const &error) {
    if (error == boost::asio::error::operation_aborted) {
      bamassert(
          (std::cerr << "timer cancelled " << flowId.flowUID() << "\n", true));
      return;
    } else if (error) {
      log::text(format("RTO timer error FlowUID %u burst %u %s") %
                flowId.flowUID() % (uint64_t)burstIdx % error.message());
      // I guess just ignore the error and try to resend anyway?
    }

    _runInSegQCtx([this, burstIdx, rtoLTimeout, rtoGTimeout] {
      // Done with this timer so erase it.
      _rtoTimer.erase(burstIdx);
      bamassert(
          (std::cerr << "timer erased " << flowId.flowUID() << "\n", true));

      auto const &bh = _burstHistory[burstIdx];

      bamassert((std::cerr << "RTO fired " << flowId.flowUID() << " "
                           << (int)burstIdx << " " << bh.ackedSeqnum
                           << std::endl,
                 true));

      if (bh.burstCompletedAcked()) {
        return;
      }

      this->doNonEnqueueBurstDetection(rtoGTimeout);

      if (bh.eobAckedBeforeBurstSizeKnown) {
        // Resend the last segment (forced).
        _burstAcksForResend[burstIdx] = {bh.numSegments - 1, rtoLTimeout};
      } else if (!bh.burstCompletedAcked()) {
        _burstAcksForResend[burstIdx] = {bh.ackedSeqnum, rtoLTimeout};
      }

      auto ttw = this->enqueueResends(rtoGTimeout, burstIdx);

      if (ttw > 0s) {
        bamassert((std::cerr << "nothing enqueued " << ttw.count() << std::endl,
                   true));
        // FIXME magic numbers:
        ttw = -_rtoDur + std::max(ttw + 500us, std::chrono::nanoseconds(2ms));
        this->startRTO(burstIdx, rtoLTimeout + ttw, rtoGTimeout + ttw, true);
      }
    });

    _onEndogenousEnqueue();
  });
}

NodeID DRRFQSegQ::FIFOBurstQueue::headDestNodeID() const {
  if (_segments.empty()) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _segments.front().segment->destNodeID();
}

size_t DRRFQSegQ::FIFOBurstQueue::numQueued() const { return _segments.size(); }

#if 0
size_t DRRFQSegQ::FIFOBurstQueue::numQueued(uint16_t burstIdx) const {
  size_t c = 0;
  for (auto const &qs : _segments) {
    auto arqseg = std::static_pointer_cast<net::ARQIP4PacketSegment>(qs.segment);
    auto ad = arqseg->arqData();
    if (ad.burstNum < burstIdx) {
      continue;
    } else if (ad.burstNum == burstIdx) {
      ++c;
    }
    else { 
      break;
    }
  }
  return c;
}
#endif

std::pair<size_t, bool>
DRRFQSegQ::FIFOBurstQueue::numQueuedEndogenous(global_clock::time_point gnow) {
  doNonEnqueueBurstDetection(gnow);
  if (!_burstAcksForResend.empty()) {
    (void)enqueueResends(gnow);
  }
  bool waitForBurstDetection =
      _trackedBursts.size() == 1 &&
      !_burstHistory[_trackedBursts.back().burstIdx].burstSizeKnown();
  return {numQueued(),
          _retxNumDropped != 0 || !_rtoTimer.empty() || waitForBurstDetection};
}

size_t DRRFQSegQ::FIFOBurstQueue::bytesQueued() const { return _numBytes; }

void DRRFQSegQ::FIFOBurstQueue::handleARQInfo(
    dll::ARQBurstInfo const abi, local_clock::time_point const feedbackLTime,
    global_clock::time_point const feedbackGTime) {
  bamprecondition(abi.flow_uid == flowId.flowUID());

  auto fuid = abi.flow_uid;
  auto const bn = abi.burst_num;
  auto const sn = abi.seq_num;
  auto &bh = _burstHistory[abi.burst_num];

  // Seqnum acks should monotonically increase.
  // bamprecondition(sn >= bh.ackedSeqnum);
  if (sn < bh.ackedSeqnum) {
    // Ignore (stale? delayed?) ack.
    bamassert((log::text(format("FlowUID %u rx arqf burst %u seqnum %u < %u") %
                         fuid % (int)bn % sn % bh.ackedSeqnum),
               true));
    return;
  }

  if (sn > bh.maxSeqNumPopped) {
    // Ignore impossible (future) ack.
    bamassert((log::text(format("FlowUID %u rx arqf burst %u seqnum %u > %u") %
                         fuid % (int)bn % sn % bh.maxSeqNumPopped),
               true));
    return;
  }

  bool newAck = sn > bh.ackedSeqnum;

  bh.ackedSeqnum = sn;

  doNonEnqueueBurstDetection(feedbackGTime);

  // Determine the seqnum to resend.

  if (bh.eobAckedBeforeBurstSizeKnown) {
    if (bh.burstCompletedAcked()) {
      bh.eobAckedBeforeBurstSizeKnown = false;
      _burstAcksForResend.erase(bn);
    } else {
      // Resend the last segment.
      _burstAcksForResend[bn] = {bh.numSegments - 1, feedbackLTime};
    }
  } else if (!bh.burstCompletedAcked()) {
    _burstAcksForResend[bn] = {sn, feedbackLTime};
  } else {
    _burstAcksForResend.erase(bn);
  }

  if (!bh.eobAcked() && newAck) {
    bamassert((log::text(format("FlowUID %u handlarq advance burst %u seq %u") %
                         fuid % (int)bn % bh.ackedSeqnum),
               true));
    startRTO(bn, feedbackLTime, feedbackGTime, true);
  } else if (bh.burstCompletedAcked()) {
    bamassert((log::text(format("FlowUID %u eob %u seq %u %u") % fuid %
                         (int)bn % bh.ackedSeqnum % bh.burstCompletedAcked()),
               true));
    _rtoTimer.erase(bn);
  }
}

std::chrono::nanoseconds
DRRFQSegQ::FIFOBurstQueue::enqueueResends(global_clock::time_point gnow,
                                          uint8_t reportRetxForBurstIdx) {
  bamprecondition(!_burstAcksForResend.empty());

  bool reportZero = false;
  bool allCaughtUp = true;
  std::chrono::nanoseconds minttw = _rtoDur;

  // Start with largest (most recent) acknowledged burst.
  for (auto bait = _burstAcksForResend.rbegin();
       bait != _burstAcksForResend.rend(); ++bait) {
    auto const burst_num = bait->first;
    auto const seq_num = bait->second.first;
    auto const lastARQFeedbackTime = bait->second.second;

    uint16_t maxSeqNumPopped = _burstHistory[burst_num].maxSeqNumPopped;

    // Forget about acknowledged segments (except most recent in case it's the
    // last segment in the burst and needed to send an eob bust size).
    for (auto sn = ((int)seq_num) - 1; sn > 0; --sn) {
      if (_retainedSegment.erase(BurstID{burst_num, sn}) == 0) {
        break;
      }
    }

    // Iterate over popped/retained segments after lastInSeqNum. If the segment
    // was retained more than resendTimeout before teh feedbackTime, then it's
    // time to retransmit the segment.
    // Start most recent so that oldest segments are last to be added to top of
    // segment queue and thus are first to be retransmitted.
    for (auto sn = maxSeqNumPopped; sn > seq_num; --sn) {
      auto rsit = _retainedSegment.find(BurstID{burst_num, sn});
      if (rsit == _retainedSegment.end()) {
        // We don't have the segment anymore?
        bamassert(
            (log::text(format("Did not find %u/%u in retained segment for "
                              "FlowUID %u (burst or segment expired?)") %
                       (uint32_t)burst_num % (uint32_t)sn % flowId.flowUID()),
             true)); // <- https://en.wikipedia.org/wiki/Comma_operator :)
        continue;
      }

      RetainedSegment const &rs = rsit->second;

      // Use it or lose it: drop if this segment is too late.
      if (gnow - rs.qs.segment->sourceTime() >= _burstdur) {
        bamassert((std::cerr << "dropping late " << flowId.flowUID() << " "
                             << (int)burst_num << " " << sn << std::endl,
                   true));
        _retainedSegment.erase(rsit);
        continue;
      }

      if (burst_num == reportRetxForBurstIdx) {
        // We will try to resend so we're not caught up for the reporting burst.
        allCaughtUp = false;
      }

      // Compute round trip time from retain to feedback.
      auto const arqrtt = lastARQFeedbackTime - rs.retainTime;

      if (arqrtt >= resendTimeout) {
        // We are going to retransmit this segment.
        auto didEnqueue = retransmit(rs.qs);

        if (didEnqueue && burst_num == reportRetxForBurstIdx) {
          reportZero = true;
        }

        NotificationCenter::shared.post(
            dll::FlowQueueResendEvent,
            dll::FlowQueueResendEventInfo{flowId, rs.qs.segment->sourceTime(),
                                          burst_num, sn, didEnqueue ? 1 : 0,
                                          numQueued(), bytesQueued()});
      } else {
        auto ttw = resendTimeout - arqrtt;
        bamprecondition(ttw > 0s);
        minttw = std::min(minttw, ttw);
        /*
        NotificationCenter::shared.post(
            dll::FlowQueueResendEvent,
            dll::FlowQueueResendEventInfo{flowId, rs.qs.segment->sourceTime(),
                                          burst_num, sn, -(int)arqrtt.count(),
                                          numQueued(), bytesQueued()});*/
      }
    }
  }

  _burstAcksForResend.clear();

  return (allCaughtUp || reportZero) ? 0s : minttw;
}

bool DRRFQSegQ::FIFOBurstQueue::retransmit(QueuedSegment const &qs) {
  // skip tracksegment()
  auto const l = qs.segment->length();

  auto const thisad =
      std::static_pointer_cast<net::ARQIP4PacketSegment>(qs.segment)->arqData();
  auto const thisBurstIdx = thisad.burstNum;
  auto const thisSeqnum = thisad.seqNum;

  // Find the spot in _segments to insert qs.
  auto segit = _segments.begin();
  while (segit != _segments.end()) {
    auto ad = std::static_pointer_cast<net::ARQIP4PacketSegment>(segit->segment)
                  ->arqData();
    if (ad.burstNum > thisBurstIdx) {
      break;
    } else if (ad.burstNum < thisBurstIdx) {
      ++segit;
    } else if (ad.seqNum > thisSeqnum) {
      break;
    } else if (ad.seqNum < thisSeqnum) {
      ++segit;
    } else {
      // Segment is already enqueued.
      return false;
    }
  }
  _numBytes += l;
  _segments.emplace(segit, qs);
  _retxNumDropped -= 1;
  _stats.bytesEnqueued += l;

  // Find the tracked burst to update (or create if needed).
  if (_trackedBursts.empty() || _trackedBursts.back().burstIdx < thisBurstIdx) {
    bamprecondition(_burstHistory[thisBurstIdx].burstSizeKnown());
    _trackedBursts.emplace_back(TrackedBurst{
        thisBurstIdx, _burstHistory[thisBurstIdx].sizeBytes, (int64_t)l});
    bamassert((std::cerr << "new tb 2 " << flowId.flowUID() << " "
                         << (int)thisBurstIdx << std::endl,
               true));
  } else {
    auto tbit = _trackedBursts.begin();
    while (tbit != _trackedBursts.end()) {
      if (tbit->burstIdx > thisBurstIdx) {
        bamprecondition(_burstHistory[thisBurstIdx].burstSizeKnown());
        _trackedBursts.emplace(
            tbit,
            TrackedBurst{thisBurstIdx, _burstHistory[thisBurstIdx].sizeBytes,
                         (int64_t)l});
        bamassert((std::cerr << "new tb 3 " << flowId.flowUID() << " "
                             << (int)thisBurstIdx << std::endl,
                   true));
        break;
      } else if (tbit->burstIdx < thisBurstIdx) {
        ++tbit;
      } else {
        tbit->burstRemaining += (int64_t)l;
        break;
      }
    }
  }

  return true;
}

void DRRFQSegQ::FIFOBurstQueue::retainSegment(
    QueuedSegment const &qs, local_clock::time_point retainTime) {
  auto arqsegment =
      std::static_pointer_cast<net::ARQIP4PacketSegment>(qs.segment);
  auto ad = arqsegment->arqData();
  auto const bn = ad.burstNum;
  auto const sn = ad.seqNum;
  try {
    auto &rs = _retainedSegment.at(BurstID{bn, sn});
    rs.qs = qs;
    rs.retainTime = retainTime;
    bamassert(
        (log::text(format("retaining (update) %u %u %u %u") % flowId.flowUID() %
                   (int)bn % sn % rs.retainTime.time_since_epoch().count()),
         true));
  } catch (std::out_of_range) {
    auto a = _retainedSegment.emplace(BurstID{bn, sn},
                                      RetainedSegment{qs, retainTime});
    bamprecondition(a.second);
    auto &rs = a.first->second;
    bamassert(
        (log::text(format("retaining (insert) %u %u %u %u") % flowId.flowUID() %
                   (int)bn % sn % rs.retainTime.time_since_epoch().count()),
         true));
  }
}

void DRRFQSegQ::FIFOBurstQueue::unretainSegment(QueuedSegment const &qs) {
  auto arqsegment =
      std::static_pointer_cast<net::ARQIP4PacketSegment>(qs.segment);
  auto ad = arqsegment->arqData();
  auto const bn = ad.burstNum;
  auto const sn = ad.seqNum;
  _retainedSegment.erase(BurstID{bn, sn});
}

ssize_t DRRFQSegQ::FIFOBurstQueue::headBurstBytesRemaining(
    global_clock::time_point gnow) const {
  if (_segments.empty()) {
    panic("No head burst bytes because no packets queued for flow.");
  }
  if (gnow - _segments.front().segment->sourceTime() >= _burstdur) {
    return -1;
  }
  // FIXME track head burst bytes across multihop.
  auto &tb = _trackedBursts.front();
  auto &bh = _burstHistory.at(tb.burstIdx);
  if (bh.burstSizeKnown()) {
    // Wait until the burst size is known before returning a positive burst
    // remaining count.
    return tb.burstRemaining;
  } else {
    return std::max(tb.burstRemaining, tb.sizeBytes);
  }
}

DRRFQSegQ::FlowQueue::global_clock::time_point
DRRFQSegQ::FIFOBurstQueue::headBurstDeadline() const {
  if (_segments.empty()) {
    panic("No head burst deadline because no packets queued for flow.");
  }
  return _segments.front().segment->sourceTime() + _burstdur;
}

DRRFQSegQ::LIFODelayDropQueue::LIFODelayDropQueue(
    FlowID id, std::chrono::nanoseconds maxDelay_)
    : FlowQueue(id), maxDelay(maxDelay_), _numBytes(0) {}

/// Enqueue segment v at time t.
void DRRFQSegQ::LIFODelayDropQueue::enqueue(QueuedSegment const &v,
                                            int64_t &num_dropped,
                                            local_clock::time_point) {
  trackSegment(v.segment);
  num_dropped = 0;
  auto const l = v.segment->length();
  _stats.bytesEnqueued += l;
  _segments.push(v);
  _numBytes += l;
}

/// Return true if a segment is dequeued into result.
bool DRRFQSegQ::LIFODelayDropQueue::dequeue(QueuedSegment &result,
                                            int64_t &num_dropped,
                                            global_clock::time_point gnow,
                                            local_clock::time_point) {
  num_dropped = 0;
  while (!_segments.empty()) {
    auto candidate = _segments.top();
    _segments.pop();
    auto const l = candidate.segment->length();
    _numBytes -= l;
    if (gnow - candidate.segment->sourceTime() < maxDelay) {
      result = std::move(candidate);
      _stats.bytesDequeued += l;
      return true;
    } else {
      ++num_dropped;
      _stats.bytesDropped += l;
    }
  }
  return false;
}

NodeID DRRFQSegQ::LIFODelayDropQueue::headDestNodeID() const {
  if (_segments.empty()) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _segments.top().segment->destNodeID();
}

size_t DRRFQSegQ::LIFODelayDropQueue::numQueued() const {
  return _segments.size();
}

size_t DRRFQSegQ::LIFODelayDropQueue::bytesQueued() const { return _numBytes; }
void DRRFQSegQ::LIFODelayDropQueue::handleARQInfo(dll::ARQBurstInfo,
                                                  local_clock::time_point,
                                                  global_clock::time_point) {}

DRRFQSegQ::CoDelQueue::CoDelQueue(
    FlowID id, local_clock::duration target_, local_clock::duration interval_,
    size_t min_queue_bytes_, std::chrono::nanoseconds flow_latency_min,
    std::chrono::nanoseconds flow_latency_max,
    std::chrono::nanoseconds flow_avg_delay_window)
    : FlowQueue(id), delay_target(target_), estimator_interval(interval_),
      min_queue_bytes(min_queue_bytes_), flow_lmin(flow_latency_min),
      flow_lmax(flow_latency_max), flow_lwindow(flow_avg_delay_window),
      _first_above_time(), _drop_next(local_clock::duration(0)), _drop_count(0),
      _last_drop_count(0), _dropping(false), _bytes(0),
      _lavg(std::chrono::nanoseconds(), 0) {
  if (delay_target >= flow_lmax) {
    throw std::invalid_argument(
        "CoDel TARGET must be less than max flow latency.");
  }
  if (flow_lmax <= flow_lmin) {
    throw std::invalid_argument("Max flow latency must be greater than min.");
  }
}

void DRRFQSegQ::CoDelQueue::enqueue(QueuedSegment const &p,
                                    int64_t &num_dropped,
                                    local_clock::time_point t) {
  trackSegment(p.segment);
  num_dropped = 0;
  _stats.bytesEnqueued += p.segment->length();
  _bytes += p.segment->length();
  _queue.emplace(t, p);
}

bool DRRFQSegQ::CoDelQueue::dequeue(QueuedSegment &result, int64_t &num_dropped,
                                    global_clock::time_point gnow,
                                    local_clock::time_point lnow) {
  num_dropped = 0;
  if (_queue.empty()) {
    _first_above_time = local_clock::time_point();
    updateDelayTracking(gnow);
    return false;
  }

  auto r = doDequeue(lnow);

  if (_dropping) {
    // STATE = DROPPING

    if (!r.ok_to_drop) {
      // TRANSITION STATE: DROPPING -> CODEL
      _dropping = false;
    }

    while (lnow >= _drop_next && _dropping) {
      // drop()
      trackDelay(r.segment.segment, flow_lmax);
      _stats.bytesDropped += r.segment.segment->length();
      ++num_dropped;
      ++_drop_count;

      r = doDequeue(lnow);

      if (!r.ok_to_drop) {
        // TRANSITION STATE: DROPPING -> CODEL
        _dropping = false;
      } else {
        _drop_next = controlLaw(_drop_next, _drop_count);
      }
    }
  } else if (r.ok_to_drop) {
    // entering drop state

    // drop()
    trackDelay(r.segment.segment, flow_lmax);
    _stats.bytesDropped += r.segment.segment->length();
    ++num_dropped;

    r = doDequeue(lnow);

    // TRANSITION STATE: CODEL -> DROPPING
    _dropping = true;

    auto const delta = _drop_count - _last_drop_count;
    if (delta > 1 && lnow - _drop_next < 16 * estimator_interval) {
      _drop_count = delta;
    } else {
      _drop_count = 1;
    }

    _drop_next = controlLaw(lnow, _drop_count);
    _last_drop_count = _drop_count;
  }

  // Compute global delay of dequeued packet.
  auto const delay = gnow - r.segment.segment->sourceTime();

  // Drop near max latency packets.
  // FIXME: improve near max latency determination
  if (delay > flow_lmax - (flow_lmax - flow_lmin) / 40) {
    // drop()
    trackDelay(r.segment.segment, flow_lmax + std::chrono::nanoseconds(1));
    _stats.bytesDropped += r.segment.segment->length();
    ++num_dropped;
    // FIXME: remove recursion. this is probably wrong
    return dequeue(result, num_dropped, gnow, lnow);
  }

  trackDelay(r.segment.segment, delay);
  updateDelayTracking(gnow);
  result = r.segment;
  _stats.bytesDequeued += result.segment->length();
  return true;
}

NodeID DRRFQSegQ::CoDelQueue::headDestNodeID() const {
  if (numQueued() == 0) {
    panic("No head destNodeID because no packets queued for flow.");
  }
  return _queue.front().second.segment->destNodeID();
}
void DRRFQSegQ::CoDelQueue::handleARQInfo(dll::ARQBurstInfo,
                                          local_clock::time_point,
                                          global_clock::time_point) {}

std::chrono::nanoseconds DRRFQSegQ::CoDelQueue::averageDelay() const {
  return _lavg.second == 0 ? std::chrono::nanoseconds(0)
                           : _lavg.first / _lavg.second;
}

DRRFQSegQ::CoDelQueue::dodequeue_result
DRRFQSegQ::CoDelQueue::doDequeue(local_clock::time_point now) {
  assert(!_queue.empty());

  // Dequeue next segment.
  auto const p = _queue.front();
  _queue.pop();

  // Update accounting.
  _bytes -= p.second.segment->length();

  // Check if it's OK to drop.  If the minimum sojourn time of this packets
  // during the estimator interval exceeds the delay target, then it's OK to
  // drop.
  bool ok_to_drop = false;
  auto const sojourn_time = now - p.first;

  if (sojourn_time < delay_target || _bytes <= min_queue_bytes ||
      _queue.empty()) {
    // Reset to NOT OK to drop if soujourn time drops below delay target or if
    // the queue size is simply too small.
    _first_above_time = local_clock::time_point();
  } else {
    // Sojourn time exceeds the target. Time to think about controlling delay.
    if (_first_above_time == local_clock::time_point()) {
      // Now is the first soujourn time above the target so set the end of the
      // estimation window.
      _first_above_time = now + estimator_interval;
    } else if (now >= _first_above_time) {
      // Soujourn time has persistently exceeded the target so start dropping.
      ok_to_drop = true;
    }
  }
  return dodequeue_result{p.first, p.second, ok_to_drop};
}

DRRFQSegQ::CoDelQueue::local_clock::time_point
DRRFQSegQ::CoDelQueue::controlLaw(local_clock::time_point t,
                                  ssize_t count) const {
  return t + std::chrono::duration_cast<local_clock::duration>(
                 estimator_interval / std::sqrt(count));
}

void DRRFQSegQ::CoDelQueue::trackDelay(dll::Segment::sptr const &s,
                                       std::chrono::nanoseconds delay) {
  // Add segment to deque of segments in the moving average window.
  _segtimes.emplace_back(s->sourceTime(), delay);
  // Update local delay average.
  _lavg.first += delay;
  ++_lavg.second;
  NotificationCenter::shared.post(
      dll::CoDelDelayEvent,
      dll::CoDelDelayEventInfo{s->sourceTime(), delay, nlohmann::json(s)});
}

void DRRFQSegQ::CoDelQueue::updateDelayTracking(global_clock::time_point gnow) {
  if (_segtimes.empty()) {
    return;
  }
  // Remove segments that are outside of the moving average delay window.
  auto st = _segtimes.front();
  while (st.first + flow_lwindow < gnow) {
    _lavg.first -= st.second;
    --_lavg.second;
    _segtimes.pop_front();
    if (_segtimes.empty()) {
      break;
    }
    st = _segtimes.front();
  }
  postStateEvent();
}

void DRRFQSegQ::CoDelQueue::postStateEvent() const {
  NotificationCenter::shared.post(
      dll::CoDelStateEvent,
      dll::CoDelStateEventInfo{flowId, _first_above_time, _drop_next,
                               _drop_count, _last_drop_count, _dropping, _bytes,
                               _queue.size(), averageDelay()});
}
} // namespace bamradio
