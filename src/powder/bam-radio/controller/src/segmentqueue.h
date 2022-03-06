// -*-c++-*-
//  Copyright © 2017-2018 Stephen Larew

#ifndef aa18d956f21ef53aa1
#define aa18d956f21ef53aa1

#include "dll_types.h"
#include "events.h"
#include "segment.h"

// Explicitly include bamassert after other project headers.
#include "bamassert.h"

#include <array>
#include <deque>
#include <queue>
#include <set>
#include <stack>

namespace bamradio {

/// Abstract segment queue.
class SegmentQueue {
public:
  typedef std::shared_ptr<SegmentQueue> sptr;
  /// Pop the next queued segments to send. (not thread safe)
  virtual std::vector<QueuedSegment>
  pop(std::chrono::nanoseconds minDesiredValue, bool allowMixedUnicastDestNodes,
      bool allowMixedBroadcastUnicastDestNodes) = 0;
  /// Push a segment. (not thread safe)
  virtual void push(QueuedSegment const &qs) = 0;
  /// Return the number of queued segments that are available to be popped off.
  // virtual size_t numReady() const = 0;

  static NodeID effectiveFrameDestNodeID(std::vector<QueuedSegment> const &qsv);
};

class DRRFQSegQ : public SegmentQueue {
  // private:
public:
  /// Abstract base flow queue.
  class FlowQueue {
  public:
    typedef std::shared_ptr<FlowQueue> sptr;
    typedef std::chrono::steady_clock local_clock;
    typedef std::chrono::system_clock global_clock;

    /// Flow identifier.
    FlowID const flowId;

    /// Enqueue segment v at time t.
    virtual void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                         local_clock::time_point t = local_clock::now()) = 0;

    /// Return true if a segment is dequeued into result.
    ///
    /// Let nq = fq.numQueued().  If nq > 0 and dequeue() is called,
    /// then numQueued() < nq regardless of result of dequeue().
    /// In other words, dequeue() is guaranteed to decrease number
    /// of queued segments if any are queued.
    virtual bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                         global_clock::time_point gnow,
                         local_clock::time_point lnow)
        __attribute__((warn_unused_result)) = 0;

    /// Returns the destination NodeID of the head segment.
    ///
    /// The returned NodeID may not be the same NodeID of a segment obtained
    /// from subsequent dequeue().
    virtual NodeID headDestNodeID() const = 0;

    /// Return number of queued segments.
    virtual size_t numQueued() const = 0;

    /// Return number of queued segments including endogenous enqueues.
    /// pair.second is true if there have been endogenous enqueues since the
    /// last enqueue or dequeue call.
    virtual std::pair<size_t, bool>
    numQueuedEndogenous(global_clock::time_point gnow);

    /// Return number of queued bytes.
    virtual size_t bytesQueued() const = 0;

    struct Stats {
      size_t bytesEnqueued = 0, bytesDequeued = 0, bytesDropped = 0;
      size_t minLength = std::numeric_limits<size_t>::max(), maxLength = 0;
      size_t minPayloadLength = std::numeric_limits<size_t>::max(),
             maxPayloadLength = 0;
      size_t midLength() const { return (minLength + maxLength) / 2; }
      size_t midPayloadLength() const {
        return (minPayloadLength + maxPayloadLength) / 2;
      }
      bool valid() { return minLength < std::numeric_limits<size_t>::max(); }
    };

    Stats stats() const { return _stats; }

    /// Handle feedback from receive dest node for given seqnum
    virtual void
    handleARQInfo(dll::ARQBurstInfo abi, local_clock::time_point feedbackTime,
                  global_clock::time_point const feedbackGTime) = 0;

  protected:
    explicit FlowQueue(FlowID id) : flowId(id) {}
    Stats _stats;
    void trackSegment(dll::Segment::sptr s);
  };

  /// Flow queue to hold segments until a non HoldQueue is made.
  class HoldQueue : public FlowQueue {
  public:
    explicit HoldQueue(FlowID id);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackTime,
                       global_clock::time_point const feedbackGTime);

    /// Return num dropped segments after transferring held segments to q.
    int64_t transferTo(FlowQueue::sptr q);

  private:
    struct HeldSegment {
      QueuedSegment qs;
      FlowQueue::local_clock::time_point enqueueTime;
    };
    std::queue<HeldSegment> _segments;
  };

  /// Flow queue that drops all.
  class DropAllQueue : public FlowQueue {
  public:
    explicit DropAllQueue(FlowID id);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);
  };

  /// Flow queue that drops all segments except the most recent.
  class DestCoalesceQueue : public FlowQueue {
  public:
    explicit DestCoalesceQueue(FlowID id);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);

  private:
    QueuedSegment _segment;
  };

  /// Flow queue that is FIFO and never drops.
  class FIFONoDropQueue : public FlowQueue {
  public:
    FIFONoDropQueue(FlowID id);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);

  private:
    size_t _numBytes;
    std::queue<QueuedSegment> _segments;
  };

  /// Flow queue for bursty delay-bounded flows.
  class FIFOBurstQueue : public FlowQueue {
  public:
    FIFOBurstQueue(FlowID id, boost::asio::io_context &ioctx,
                   std::function<void(std::function<void()>)> runInSegQCtx,
                   std::function<void()> onEndogenousEnqueue,
                   std::chrono::nanoseconds burst_transfer_duration,
                   std::chrono::nanoseconds resendTimeout,
                   std::chrono::nanoseconds maxInterSegmentArrivalDur,
                   int newBurstInterSegmentArrivalRatio);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    std::pair<size_t, bool> numQueuedEndogenous(global_clock::time_point gnow);
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);

    /// Returns number of bytes remaining in head-of-line burst or -1 if burst
    /// is overdue.
    ssize_t headBurstBytesRemaining(global_clock::time_point gnow) const;

    /// Deadline time for head burst.
    global_clock::time_point headBurstDeadline() const;

    /// Drop all segments of the head-of-line burst. Returns number of dropped
    /// segments.
    // size_t dropHeadBurst();

    std::chrono::nanoseconds resendTimeout;
    int newBurstInterSegmentArrivalRatio;

  private:
    std::chrono::nanoseconds const _burstdur;
    size_t _numBytes;
    std::deque<QueuedSegment> _segments;
    // size_t numQueued(uint16_t burstIdx) const;

    // ARQ

    void retainSegment(QueuedSegment const &qs,
                       local_clock::time_point retainTime);
    void unretainSegment(QueuedSegment const &qs);
    bool retransmit(QueuedSegment const &qs);
    struct RetainedSegment {
      QueuedSegment qs;
      local_clock::time_point retainTime;
    };
    typedef std::pair<int32_t, int32_t> BurstID;
    std::map<BurstID, RetainedSegment> _retainedSegment;

    struct BurstHistory {
      BurstHistory()
          : numSegments(0), maxSeqNumPopped(0), ackedSeqnum(0), sizeBytes(-1),
            eobAckedBeforeBurstSizeKnown(false) {}
      uint16_t numSegments;
      uint16_t maxSeqNumPopped;
      uint16_t ackedSeqnum;
      int64_t sizeBytes;
      inline bool burstSizeKnown() const { return sizeBytes > 0; }
      inline void setBurstSize(int64_t s) {
        bamprecondition(sizeBytes <= 0);
        bamprecondition(s > 0);
        sizeBytes = s;
        if (eobAcked()) {
          eobAckedBeforeBurstSizeKnown = true;
        }
      }
      bool eobAckedBeforeBurstSizeKnown;
      inline bool burstCompletedAcked() const {
        return sizeBytes > 0 && ackedSeqnum > numSegments;
      }
      inline bool eobAcked() const {
        return sizeBytes > 0 && ackedSeqnum >= numSegments;
      }
    };
    std::array<BurstHistory, 1 << (sizeof(uint8_t) * 8)> _burstHistory;
    int64_t _retxNumDropped;
    std::map<uint8_t, std::pair<uint16_t, local_clock::time_point>>
        _burstAcksForResend;
    std::chrono::nanoseconds enqueueResends(global_clock::time_point gnow,
                                            uint8_t reportRetxForBurstIdx = 255)
        __attribute__((warn_unused_result));
    std::chrono::nanoseconds _rtoDur;
    boost::asio::io_context &_ioctx;
    std::map<uint8_t, boost::asio::steady_timer> _rtoTimer;
    void startRTO(uint8_t burstIdx, local_clock::time_point,
                  global_clock::time_point, bool restart);
    std::function<void(std::function<void()>)> _runInSegQCtx;
    /// callback for endogenous enqueues outside of a enqueue, dequeue, or
    /// numQueuedEndogenous call. Call it from inside the _ioctx.
    std::function<void()> _onEndogenousEnqueue;

    // Burst tracking

    struct TrackedBurst {
      uint8_t burstIdx;
      int64_t sizeBytes, burstRemaining;
      // FIXME track numqueued in burst here
    };
    std::deque<TrackedBurst> _trackedBursts;

    // Burst detection

    // this function needs to be renamed
    void doNonEnqueueBurstDetection(global_clock::time_point gnow);

    std::chrono::nanoseconds const _maxInterSegmentArrivalDur;

    global_clock::time_point _lastArrivalST;
    std::chrono::nanoseconds _lastArrivalDur;

    bool _burstReset;
    uint8_t _burstIdx;
    uint16_t _nextBurstSeqNum;
  };

  /// Flow queue that is LIFO and drops delayed segments.
  class LIFODelayDropQueue : public FlowQueue {
  public:
    LIFODelayDropQueue(FlowID id, std::chrono::nanoseconds maxDelay);
    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point);
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point gnow, local_clock::time_point);
    NodeID headDestNodeID() const;
    size_t numQueued() const;
    size_t bytesQueued() const;
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);
    std::chrono::nanoseconds maxDelay;

  private:
    size_t _numBytes;
    std::stack<QueuedSegment> _segments;
  };

  class CoDelQueue : public FlowQueue {
  public:
    /// Construct a CoDel queue with given constants.
    CoDelQueue(FlowID id, local_clock::duration target,
               local_clock::duration interval, size_t min_queue_bytes,
               std::chrono::nanoseconds flow_latency_min,
               std::chrono::nanoseconds flow_latency_max,
               std::chrono::nanoseconds flow_avg_delay_window);

    void enqueue(QueuedSegment const &v, int64_t &num_dropped,
                 local_clock::time_point t = local_clock::now());
    bool dequeue(QueuedSegment &result, int64_t &num_dropped,
                 global_clock::time_point gnow, local_clock::time_point lnow);
    NodeID headDestNodeID() const;
    size_t numQueued() const { return _queue.size(); }
    size_t bytesQueued() const { return _bytes; }
    void handleARQInfo(dll::ARQBurstInfo abi,
                       local_clock::time_point feedbackLTime,
                       global_clock::time_point feedbackGTime);

    // CoDel parameters.

    /// CoDel delay target.
    local_clock::duration const delay_target;
    /// CoDel estimator interval.
    local_clock::duration const estimator_interval;
    /// CoDel link utilization and starvation bound.
    // TODO: switch from min_queue bytes to time.
    ssize_t const min_queue_bytes;

    /// Return moving average of delay upon dequeue.
    std::chrono::nanoseconds averageDelay() const;

    // Latency parameters.

    /// Minimum delay of flow.
    std::chrono::nanoseconds const flow_lmin;
    /// Maximum delay of flow.
    std::chrono::nanoseconds const flow_lmax;
    /// Window length for average flow delay tracking.
    std::chrono::nanoseconds const flow_lwindow;

  private:
    // CoDel State

    /// First time sojourn time is above the delay target.
    local_clock::time_point _first_above_time;
    /// Next time to drop a segment.
    local_clock::time_point _drop_next;
    /// Number of dropped packets in most recent DROPPING state.
    ssize_t _drop_count;
    /// Initial drop_count at the last state transition CODEL -> DROPPING.
    ssize_t _last_drop_count;
    /// True if in DROPPING state else in CODEL state.
    bool _dropping;
    /// Number of queued bytes.
    ssize_t _bytes;
    /// FIFO queue with enqueue time tags.
    std::queue<std::pair<local_clock::time_point, QueuedSegment>> _queue;

    struct dodequeue_result {
      local_clock::time_point enqueue_time;
      QueuedSegment segment;
      bool ok_to_drop;
    };

    dodequeue_result doDequeue(local_clock::time_point now);

    local_clock::time_point controlLaw(local_clock::time_point t,
                                       ssize_t count) const;

    /// Moving average delay (sum delay, count)
    std::pair<std::chrono::nanoseconds, int64_t> _lavg;

    /// Segment time & delay tracking (source time, delay)
    std::deque<std::pair<global_clock::time_point, std::chrono::nanoseconds>>
        _segtimes;

    /// Track delay of dequeued segments.
    void trackDelay(dll::Segment::sptr const &s,
                    std::chrono::nanoseconds delay);

    /// Update the moving average delay estimate.
    void updateDelayTracking(global_clock::time_point gnow);

    void postStateEvent() const;
  };

public:
  typedef std::shared_ptr<DRRFQSegQ> sptr;

  struct FlowSchedule {
    std::chrono::nanoseconds quantumCredit;
    std::chrono::nanoseconds dequeueDebit;
  };

  typedef std::function<FlowQueue::sptr(FlowID)> FlowQueueMaker;
  typedef std::function<FlowSchedule(std::chrono::system_clock::time_point,
                                     int64_t, FlowID, bool)>
      ScheduleMaker; // 🙈

  // set purgeInterval==-1 to disable auto purge
  DRRFQSegQ(uint64_t purgeInterval_, FlowQueueMaker makeFlowQueue,
            ScheduleMaker makeSchedule);

  std::vector<QueuedSegment> pop(std::chrono::nanoseconds minDesiredValue,
                                 bool allowMixedUnicastDestNodes,
                                 bool allowMixedBroadcastUnicastDestNodes);
  void push(QueuedSegment const &qs);
  // size_t numReady() const;

  struct SchedulerFlowQueue {
  public:
    typedef std::chrono::nanoseconds value;

    SchedulerFlowQueue(FlowQueue::sptr q_, int64_t lastActiveRound)
        : q(q_), _lastActiveRound(lastActiveRound), _balance(0),
          _activated(false),
          _lastDestNodeID(q->numQueued() > 0 ? q->headDestNodeID()
                                             : UnspecifiedNodeID) {}

    FlowQueue::sptr q;
    /// Balance of the asset.
    value balance() const { return _balance; }
    /// Decrease the asset value.
    void credit(value v) { _balance -= v; }
    /// Increae the asset value.
    void debit(value v) { _balance += v; }
    void reset(value v) { _balance = v; }
    void cap(value v) { _balance = std::min(_balance, v); }
    /// Activate the sfq
    void activate() { _activated = true; }
    /// Return true if the queue is idle.
    bool idle(int64_t const currentRound, int64_t const numRounds = 1) const {
      return !_activated && (currentRound - _lastActiveRound > numRounds);
    }
    void setLastActiveRound(int64_t const roundNumber) {
      _activated = false;
      _lastActiveRound = roundNumber;
    }
    NodeID destNodeID() const {
      return q->numQueued() > 0 ? q->headDestNodeID() : _lastDestNodeID;
    }

  private:
    int64_t _lastActiveRound;
    value _balance;
    bool _activated;
    NodeID _lastDestNodeID;
    friend DRRFQSegQ;
  };

  std::map<FlowID, SchedulerFlowQueue> const &flowQueues();

  int64_t const purgeInterval;

  ScheduleMaker makeSchedule;

  void purgeFlowQueues(std::set<FlowID> const &fids);

private:
  /// Purge flow queues that are inactive and have expired.
  void purgeExpiredFlowQueues(FlowQueue::global_clock::time_point gnow);

  std::map<FlowID, SchedulerFlowQueue> _flowQueues;

  /// New and old flow lists for DRR. out holds flows running deficit.
  std::deque<FlowID> _newFlows, _oldFlows, _oldFlowsOut;

  int64_t _currentRound;
  int64_t _nextPurge;
  int64_t _numQueued;

  std::function<FlowQueue::sptr(FlowID)> const _makeFlowQueue;
};
} // namespace bamradio

#endif
