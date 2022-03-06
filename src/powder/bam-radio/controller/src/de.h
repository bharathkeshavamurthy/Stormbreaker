// DE.
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef b4f3fb1d82addc3
#define b4f3fb1d82addc3

#include "c2api.h"
#include "cc_data.h"
#include "collab.h"
#include "discrete_channels.h"
#include "dll_types.h"
#include "lisp.h"
#include "notify.h"
#include "psd.h"
#include "radiocontroller.h"
#include "statistics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <thread>
#include <vector>

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <zmq.hpp>

namespace bamradio {
namespace decisionengine {

///////////////////////////////////////////////////////////////////////////////
// Miscellania
///////////////////////////////////////////////////////////////////////////////

// timing is important in the decision engine. we use the same units as the
// system clock.
typedef std::chrono::system_clock::duration Duration;
typedef std::chrono::system_clock::time_point Timepoint;

// more convenience typedefs
typedef NodeID SRNID;
typedef controlchannel::Location Location;

// a tracker lets the decision engine keep track of something over a set window
// of time extending into the past. This is similar to the TrafficStat class of
// the statscenter.
template <typename T> class Tracker {
public:
  typedef T type;

  struct TrackedItem {
    T item;
    Timepoint t;
  };

  Tracker(Duration windowSize) : _windowSize(windowSize) {}
  Tracker(Tracker const &other)
      : _windowSize(other._windowSize), _items(other._items) {}

  void track(T const &item) {
    auto const now = std::chrono::system_clock::now();
    _items.push_back({item, now});
    _refresh(now);
  }

  size_t size() const { return _items.size(); }

  void apply(std::function<void(T const &)> f) const {
    for (auto const &ti : _items)
      f(ti.item);
  }

  std::vector<T> items() const {
    std::vector<T> o;
    o.reserve(size());
    for (auto const &ti : _items)
      o.push_back(ti.item);
    return o;
  }

  std::deque<TrackedItem> const &trackedItems() const { return _items; }

  boost::optional<T> newest() const {
    if (size() > 0) {
      return _items.back().item;
    } else {
      return boost::none;
    }
  }

  boost::optional<T> previous() const {
    if (size() > 1) {
      return _items[size() - 2].item;
    } else {
      return boost::none;
    }
  }

  void clear() { _items.clear(); }

private:
  Duration const _windowSize;

  std::deque<TrackedItem> _items;

  void _refresh(Timepoint t) {
    while (_items.size() > 0) {
      auto v = _items.front();
      auto diff = t - v.t;
      if (diff > _windowSize) {
        _items.pop_front();
      } else {
        break;
      }
    }
  }
};

// overlay map conversion
controlchannel::OverlapMap overlapMapFromLisp(cl_object obj);

///////////////////////////////////////////////////////////////////////////////
// Information about our SRNs
///////////////////////////////////////////////////////////////////////////////

struct MandateInfo {
  stats::IndividualMandate mandate;
  std::vector<SRNID> activeSRNs; // can be empty
  boost::optional<stats::FlowPerformance> performance;
  boost::optional<stats::FlowInfo> endpoints;
};

// the decisionengine also wants to track our SRNs during a match. we use a
// different data structure than for competitors because we have more
// fine-grained information about our own nodes
struct BAMSRNInfo {
  Tracker<Location> location;
  Tracker<std::vector<FlowUID>> mandatesOffered;
  Tracker<TransmitAssignment> txAssignment;
  Tracker<stats::DutyCycleInfo> dutyCycle;
  Tracker<psdsensing::PSDData> psd;

  BAMSRNInfo();
};

///////////////////////////////////////////////////////////////////////////////
// Information about peers/competitors
///////////////////////////////////////////////////////////////////////////////

// the decisionengine wants to track other networks during a match.
typedef boost::asio::ip::address_v4 NetworkID;
typedef uint32_t CompetitorSRNID;
typedef collab::NetworkType NetworkType;

// a band of continues frequencies. we assume that the members are absolute (not
// baseband) values.
struct FrequencyBand {
  double lower;
  double upper;
};

// otherwise known as spectrum voxel
struct FrequencyBandUsage {
  FrequencyBand band;
  CompetitorSRNID transmitter;
  std::vector<CompetitorSRNID> receivers;
  double txPowerdB;
  Timepoint start;
  boost::optional<Timepoint> end;
};

// an instantaneous idea of spectrum usage of a competitor's network is given as
// a vector of frequency bands.
typedef std::vector<FrequencyBandUsage> SpectrumUsage;

// some idea about a competitor's mandate performance. This copies all mandatory
// fields from the detailedperformance CIL message
struct CompetitorMandatePerformance {
  double scalarPerformance;
  std::vector<CompetitorSRNID> radioIDs;
  uint32_t holdPeriod;
  uint32_t achievedDuration;
};

struct CompetitorMandates {
  uint32_t mandateCount;
  Timepoint mpEnd;
  uint32_t mandatesAchieved;
  uint32_t totalScoreAchieved;
  uint32_t bonusThreshold;
  std::vector<CompetitorMandatePerformance> mandates;
};

// some information about the competitor's SRNs
struct CompetitorSRNInfo {
  CompetitorSRNID const id;
  Tracker<Location> location;

  CompetitorSRNInfo(SRNID id);
};

// if the incumbent is passive, there is some extra information. The DSRC
// incumbent on the other hand acts like a competitor (it has mandates). For the
// passive incumbent, we simply copy the contents of the incumbentnotify message
struct PassiveIncumbentMessage {
  enum Type { Unknown = 0, Report, Violation } type;
  int32_t incumbentID;
  Timepoint reportTime;
  double power;
  double threshold;
  int64_t centerFreq; // absolute (not baseband)
  int64_t bandwidth;
  bool thresholdExceeded;

  // help me i am tired
  bool overlaps(TransmitAssignment ass, int64_t cfreq) const;
  bool overlaps(FrequencyBandUsage uss, int64_t cfreq) const;
};

struct ActiveIncumbentMessage {
  enum Type { Unknown = 0, Report, Violation } type;
  int32_t incumbentID;
  Timepoint reportTime;
  double inr;
  double threshold;
  int64_t centerFreq; // absolute (not baseband)
  int64_t bandwidth;
  bool thresholdExceeded;
};

// all of the information we want to track about other networks during a match
struct NetworkInfo {
  NetworkType type;
  std::vector<CompetitorSRNInfo> SRNs;
  Tracker<CompetitorMandates> mandates;
  Tracker<SpectrumUsage> spectrumUsage;
  // this obv. only applies to passive and active incumbents.
  // for all other competitor networks, this tracker will be empty
  Tracker<PassiveIncumbentMessage> passiveIncumbentMessages;
  Tracker<ActiveIncumbentMessage> activeIncumbentMessages;

  NetworkInfo(NetworkType t);
  void addSRN(CompetitorSRNID id);
  CompetitorSRNInfo &findSRN(CompetitorSRNID id);
};

///////////////////////////////////////////////////////////////////////////////
// Decision Engine
///////////////////////////////////////////////////////////////////////////////

// one step of the decisionengine can be triggered by different events
enum Trigger { PeriodicStep = 0, EnvironmentUpdate, IMUpdate };

// the little engine that runs our decision process. her friends are the
// algorithms. she plays with them when she is triggered. some of her friends
// are deterministic, some stochastic. her brightest friends are learning.
class WITH_LISPY_METHODS(DecisionEngine) {
public:
  // options
  struct Options {
    NodeID const gatewayID;
    Duration const step_period;
    Duration const channel_alloc_delay;
    Duration const cil_broadcast_period;
    double const data_tx_gain;
    double const control_tx_gain;
    double const sample_rate;
    float const guard_band;
    std::string const max_wf;
    psdsensing::PSDSensing::HistParams const psd_hist_params;
  } const options;

  // tors
  DecisionEngine(zmq::context_t &ctx,
                 collab::CollabClient::ConnectionParams ccparams,
                 AbstractRadioController::sptr radioCtrl, Options const &opts);
  ~DecisionEngine();
  DecisionEngine(DecisionEngine const &other) = delete;

  // external API
  void start();
  void stop();
  bool running() const;

  // LISP env
  lisp::LispThread lisp;

  // lisp methods
  LISPY_METHOD(callUpdateTxAssignment, TxAssignmentUpdate,
               TxAssignmentUpdate::fromLisp);
  LISPY_METHOD(updateOverlapInfo, controlchannel::OverlapMap,
               overlapMapFromLisp);

  // shared_ptr convention
  typedef std::shared_ptr<DecisionEngine> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<DecisionEngine>(std::forward<Args>(args)...);
  }

protected:
  // internal state management
  std::atomic_bool _running;
  std::vector<NotificationCenter::SubToken> _st;

  // thread and work queue
  boost::asio::io_context _io;
  boost::asio::executor_work_guard<decltype(_io)::executor_type> _iow;
  std::thread _thread;

  // information providers
  AbstractRadioController::sptr _radioCtrl;
  collab::CollabClient::sptr _collabClient;
  controlchannel::CCData::sptr _ccData;

  // run the lisp code
  void _lispStep(Trigger trigger, Timepoint stepTime, uint64_t stepCount);

  // timing
  boost::asio::system_timer _periodicTriggerTimer;
  boost::asio::system_timer _cilBroadcastTimer;
  Duration const _t_step;
  Duration const _t_broadcast;
  uint64_t _stepCount;

  // track information about peers and my network
  std::map<NetworkID, NetworkInfo> _peers;
  std::map<SRNID, BAMSRNInfo> _srns;
  Tracker<std::vector<SRNID>> _myNetwork;

  // track information about mandates and environment updates
  Tracker<std::map<FlowUID, stats::IndividualMandate>> _offeredMandates;
  Tracker<c2api::EnvironmentManager::Environment> _environment;

  // track information about mandate porformance
  Tracker<std::map<FlowUID, MandateInfo>> _mandatePerformance;

  struct CILTransmitAssignmentReport {
    std::chrono::system_clock::time_point t_start;
    std::chrono::system_clock::time_point t_end;
    TransmitAssignment::Map tx_assignment;
  };
  Tracker<CILTransmitAssignmentReport> _cilReportedAssignment;

  // logic
  void _step(Trigger t);
  void _cilBroadcast(Trigger t);
  // XXX this is a kludge to synchronize channel allocation changes. refactor
  // once all decisions are done in lisp
  void _sendSpectrumUsage(Timepoint tstart, Timepoint tend,
                          TransmitAssignment::Map const &txAss);
  Trigger _currentTrigger;
  Timepoint _currentTriggerTime;
  bool _channel_updated;
  Timepoint _last_su_time;

  // helper functions
  NetworkInfo &_addPeer(NetworkID id, NetworkType type);
  void _trackSRN(SRNID id);
  uint32_t _updateTxAssignment(TransmitAssignment::Map txAss,
                               bool channel_updated);
  void _addSpectrumUsage(std::shared_ptr<sc2::cil::CilMessage> msg,
                         Timepoint tstart, Timepoint tend, bool measured,
                         TransmitAssignment::Map const &txAss);
  std::shared_ptr<sc2::cil::CilMessage> _makeLocationUpdate(Timepoint t);
  std::shared_ptr<sc2::cil::CilMessage> _makeDetailedPerformance(Timepoint t);
  bool _ensureCollab();

  NodeID const _myID;
  Tracker<Location> _myLocation; // fallback

  std::vector<uint8_t> _thresholdPSD(std::vector<float> const &raw_psd);

  /*

  // needed bandwidth calculation
  double _occupied_bw;
  std::vector<double> _bandwidthNeeded(void);

#warning restructure
  ///////////////////////////////////////////////////////////////////////////////
  // LEGACY HELP ME
  ///////////////////////////////////////////////////////////////////////////////
  psdsensing::FreqAlloc _freq_alloc;
  void _allocateChannels();
  void _handleSpectrumUsage(std::shared_ptr<sc2::cil::SpectrumUsage> msg,
                            collab::Metadata md);
  void _handleLocationUpdate(std::shared_ptr<sc2::cil::LocationUpdate> msg,
                             collab::Metadata md);
  void
  _handleDetailedPerformance(std::shared_ptr<sc2::cil::DetailedPerformance> msg,
                             collab::Metadata md);

  */
};

///////////////////////////////////////////////////////////////////////////////
// Conversions
///////////////////////////////////////////////////////////////////////////////

cl_object toLisp(Location const &l);
cl_object toLisp(Trigger const &t);
cl_object toLisp(Timepoint const &t);
cl_object toLisp(c2api::EnvironmentManager::Environment const &env);
cl_object toLisp(std::map<FlowUID, MandateInfo> const &mandatePerformance);
cl_object toLisp(PassiveIncumbentMessage const &msg);
cl_object toLisp(ActiveIncumbentMessage const &msg);
cl_object toLisp(psdsensing::PSDData const &psd);

cl_object threshPSDtoLisp(psdsensing::PSDData const &psd,
                          psdsensing::PSDSensing::HistParams const &params);
cl_object linkRatesToLisp(
    std::map<NodeID, std::map<NodeID, float>> const &offeredLinkRates);

// misc lisp

// a symbol in the :bam-radio package
inline cl_object BRSymbol(std::string const &sym) {
  return lisp::Symbol(sym, "bam-radio");
}

// populate the bandwidth table
void populateTables(collab::CollabClient::ConnectionParams ccparams);

} // namespace decisionengine
} // namespace bamradio

#endif // b4f3fb1d82addc3
