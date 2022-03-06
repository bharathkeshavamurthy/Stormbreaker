// DE.
//
// Copyright (c) 2018 Dennis Ogbe

#include "de.h"
#include "bandwidth.h"
#include "events.h"
#include "util.h"

#include <numeric>
#include <volk/volk.h>

namespace bamradio {
namespace decisionengine {

using namespace std::chrono_literals;
using namespace std::chrono;

// debug
#define BAM_DE_DEBUG 1
#ifdef BAM_DE_DEBUG
#include <sstream>
template <typename T> std::string timepoint2human(T t) {
  auto const tt = std::chrono::system_clock::to_time_t(t);
  std::ostringstream s;
  s << std::put_time(std::localtime(&tt), "[%H:%M:%S]") << " ("
    << duration_cast<nanoseconds>(t.time_since_epoch()).count() << ")";
  return s.str();
}
char const *trigger2human(Trigger t) {
  switch (t) {
  case Trigger::PeriodicStep:
    return "pstep";
  case Trigger::EnvironmentUpdate:
    return "env";
  case Trigger::IMUpdate:
    return "im";
  }
}
#endif

///////////////////////////////////////////////////////////////////////////////
// MISC
///////////////////////////////////////////////////////////////////////////////

DecisionEngine::~DecisionEngine() { stop(); }

void DecisionEngine::start() {
  if (!_running) {
    _running = true;
    // periodic decisionengine step
    _periodicTriggerTimer.expires_from_now(1s);
    _periodicTriggerTimer.async_wait(
        [this](auto &e) { this->_step(Trigger::PeriodicStep); });
    // periodic cil broadcast
    _cilBroadcastTimer.expires_from_now(3s);
    _cilBroadcastTimer.async_wait(
        [this](auto &e) { this->_cilBroadcast(Trigger::PeriodicStep); });
  }
}

void DecisionEngine::stop() {
  _running = false;
  _st.clear();
  _periodicTriggerTimer.cancel();
  _cilBroadcastTimer.cancel();
  _iow.reset();
  _io.stop();
  if (_thread.joinable()) {
    _thread.join();
  }
}

bool DecisionEngine::running() const { return _running; }

NetworkInfo &DecisionEngine::_addPeer(NetworkID id, NetworkType type) {
  if (_peers.find(id) == end(_peers)) {
    auto r = _peers.emplace(id, type);
    return r.first->second;
  }
  else {
    auto &p = _peers.at(id);
    p.type = type;
    return p;
  }
}

// initialize trackers for reasonable intervals, make longer or shorter if
// necessary
NetworkInfo::NetworkInfo(NetworkType t)
    : type(t), mandates(1min), spectrumUsage(1min),
      passiveIncumbentMessages(20min), activeIncumbentMessages(20min) {}
BAMSRNInfo::BAMSRNInfo()
    : location(5min), mandatesOffered(5min), txAssignment(5min),
      dutyCycle(1min), psd(1min) {}
CompetitorSRNInfo::CompetitorSRNInfo(SRNID id) : id(id), location(1min) {}

inline Timepoint cilTime2Timepoint(sc2::cil::TimeStamp const &ts) {
  using namespace std::chrono;
  nanoseconds const fracNS(ts.picoseconds() * 1000);
  seconds const fullSecSinceEpoch(ts.seconds());
  nanoseconds const NSSinceEpoch(fullSecSinceEpoch);
  return Timepoint(NSSinceEpoch + fracNS);
}

inline sc2::cil::TimeStamp *timepoint2cilTime(Timepoint p) {
  using namespace std::chrono;
  nanoseconds const NSSinceEpoch(p.time_since_epoch());
  seconds const fullSecSinceEpoch(duration_cast<seconds>(p.time_since_epoch()));
  nanoseconds const fracNS(NSSinceEpoch - nanoseconds(fullSecSinceEpoch));
  auto ts = new sc2::cil::TimeStamp();
  ts->set_seconds(fullSecSinceEpoch.count());
  ts->set_picoseconds(fracNS.count() * 1000);
  return ts;
}

void NetworkInfo::addSRN(CompetitorSRNID id) {
  if (std::find_if(begin(SRNs), end(SRNs),
                   [id](auto const &i) { return i.id == id; }) == end(SRNs)) {
    SRNs.emplace_back(id);
  }
}

CompetitorSRNInfo &NetworkInfo::findSRN(CompetitorSRNID id) {
  auto fit = std::find_if(begin(SRNs), end(SRNs),
                          [id](auto const &i) { return i.id == id; });
  if (fit == end(SRNs)) {
    SRNs.emplace_back(id);
    return SRNs.back();
  } else {
    return *fit;
  }
}

void DecisionEngine::_trackSRN(SRNID id) {
  if (_srns.find(id) == end(_srns)) {
    _srns.emplace(id, BAMSRNInfo());
    std::vector<SRNID> ids;
    ids.reserve(_srns.size());
    for (auto const &s : _srns) {
      auto const &i = s.first;
      ids.push_back(i);
    }
    _myNetwork.track(ids);
  }
}

uint32_t DecisionEngine::_updateTxAssignment(
    TransmitAssignment::Map txAss, bool channel_updated) {

  // compute the time at which the update is applied at the nodes
  auto const updateTime = [this] {
    if ((_currentTrigger == Trigger::EnvironmentUpdate) &&
        (c2api::env.current().scenario_rf_bandwidth !=
         c2api::env.previous()->scenario_rf_bandwidth)) {
      return _currentTriggerTime;
    } else {
      return _currentTriggerTime + options.channel_alloc_delay;
    }
  }();

  // end time is always the next periodic trigger time + the delay
  auto const endTime =
      _currentTrigger == Trigger::PeriodicStep
          ? _currentTriggerTime + options.step_period +
                options.channel_alloc_delay
          : _periodicTriggerTimer.expiry() + options.channel_alloc_delay;
  assert(endTime > updateTime);

  // change my internal state
  for (auto const &a : txAss) {
    auto const &id = a.first;
    auto ass = a.second;
    _trackSRN(id);
    _srns[id].txAssignment.track(ass);
  }

  // call CCData to notify other nodes in network
  auto const uid = _ccData->updateOFDMParams(txAss, updateTime);

  // broadcast the new allocation via CIL

  // if the actual position of channels changed, send a new message.
  if (channel_updated) {
    _channel_updated = true;
    _sendSpectrumUsage(updateTime, endTime, txAss);
  }

  return uid;
}

inline bool c_overlaps(double cl, double cu, double ml, double mu) {
  return !((cu < ml && cl < ml) || (cl > mu && cu > mu));
}

// FIXME do we need this?
bool PassiveIncumbentMessage::overlaps(TransmitAssignment ass,
                                       int64_t cfreq) const {
  double const cl = centerFreq - cfreq - bandwidth / 2.0;
  double const cu = centerFreq - cfreq + bandwidth / 2.0;
  auto const env = c2api::env.current();
  auto const ass_chan = ass.toRCChannel(env.scenario_rf_bandwidth);
  return c_overlaps(cl, cu, ass_chan.lower(), ass_chan.upper());
}

bool PassiveIncumbentMessage::overlaps(FrequencyBandUsage uss,
                                       int64_t cfreq) const {
  double const cl = centerFreq - cfreq - bandwidth / 2.0;
  double const cu = centerFreq - cfreq + bandwidth / 2.0;
  if (uss.band.lower > uss.band.upper) {
    return true; // ???
  }
  double const ml = uss.band.lower - cfreq;
  double const mu = uss.band.upper - cfreq;
  return c_overlaps(cl, cu, ml, mu);
}

//
// Lisp environment / FFI etc
//

// lisp environment initializer pointer
extern "C" {
extern void init_lib_BAM_RADIO__ALL_SYSTEMS(cl_object);
}
const lisp::Environment::Initializers lisp_init = {
    init_lib_BAM_RADIO__ALL_SYSTEMS};

// we are bound to this odd function signature when registering C functions to
// call from lisp code. for small, non-oop stuff like logging a string this is
// ok, but once we throw objects in the mix this becomes more complicated. there
// are some hacks at the bottom of lisp.h that enable this.

cl_object logLispString(cl_narg narg, ...) {
  va_list args;
  va_start(args, narg);
  cl_object cl_s = va_arg(args, cl_object);
  if (lisp::isString(cl_s)) {
    log::text(lisp::fromString(cl_s));
    return lisp::t;
  } else {
    return lisp::nil;
  }
}

// dump a lisp object to the data base -- we always post a StepEvent, if we need
// more fine-grained logging, we can add tables.
cl_object logStepInput(cl_narg narg, ...) {
  va_list args;
  va_start(args, narg);
  cl_object stepInput = va_arg(args, cl_object);
  if (lisp::isByteVector(stepInput)) {
    auto const data =
        std::make_shared<std::vector<uint8_t>>(lisp::fromByteVector(stepInput));
    NotificationCenter::shared.post(StepEvent, StepEventInfo{data});
    return lisp::t;
  } else {
    return lisp::nil;
  }
}

cl_object logStepOutput(cl_narg narg, ...) {
  va_list args;
  va_start(args, narg);
  cl_object stepOutput = va_arg(args, cl_object);
  if (lisp::isByteVector(stepOutput)) {
    auto const data =
        std::make_shared<std::vector<uint8_t>>(lisp::fromByteVector(stepOutput));
    NotificationCenter::shared.post(StepOutputEvent, StepOutputEventInfo{data});
    return lisp::t;
  } else {
    return lisp::nil;
  }
}

cl_object timeNow(cl_narg narg, ...) {
  // dirty, but the (now) call from local-time does not return nanoseconds...
  return toLisp(std::chrono::system_clock::now());
}

// a place to register all C FFI functions to the lisp runtime
void initlisp() {
  // enable string logging from lisp
  lisp::addFunction(BRSymbol("log-string"), logLispString);
  // enable lisp object logging from lisp
  lisp::addFunction(BRSymbol("log-decision-engine-input"), logStepInput);
  lisp::addFunction(BRSymbol("log-decision-engine-output"), logStepOutput);
  // new transmit assignment
  lisp::addFunction(BRSymbol("update-tx-assignment"),
                    DecisionEngine::callUpdateTxAssignment_LISPMETHOD);
  // overlap info
  lisp::addFunction(BRSymbol("update-overlap-info"),
                    DecisionEngine::updateOverlapInfo_LISPMETHOD);
  // current time
  lisp::addFunction(BRSymbol("get-time-now"), timeNow);
}

void populateTables(collab::CollabClient::ConnectionParams ccparams) {
  // bandwidth table
  [] {
    std::map<int, int> table;
    int k = 0;
    for (auto const &sc : bam::dsp::SubChannel::table()) {
      table[k++] = sc.bw();
    }
    lisp::Funcall(BRSymbol("set-bandwidths"), lisp::toLisp(table));
  }();
  // discrete channel table
  lisp::Funcall(BRSymbol("set-channelization-table"), lisp::toLisp(Channelization::table));
  // network id
  lisp::Funcall(BRSymbol("set-network-id"), lisp::Cons(lisp::toLisp(ccparams.client_ip.to_ulong()),
                                                       lisp::toLisp(ccparams.client_ip.to_string())));
  // set rng to known state
  lisp::Funcall(BRSymbol("init-random"));
}

// tors & external API

DecisionEngine::DecisionEngine(zmq::context_t &ctx,
                               collab::CollabClient::ConnectionParams ccparams,
                               AbstractRadioController::sptr radioCtrl,
                               Options const &opts)
    : options(opts), lisp(lisp_init,
                          [this, &ccparams] {
                            // set the pointer to myself
                            lisp::Funcall(BRSymbol("set-decision-engine-ptr"),
                                          lisp::CPointer(this));
                            // populate some data tables in lisp env
                            populateTables(ccparams);
                            // register C functions
                            initlisp();
                          }),
      _running(false), _iow(boost::asio::make_work_guard(_io)), _thread([this] {
        bamradio::set_thread_name("decisionEngine");
        _io.run();
      }),
      _radioCtrl(radioCtrl),
      _collabClient(collab::CollabClient::make(ctx, ccparams)),
      _ccData(radioCtrl->ccData()), _periodicTriggerTimer(_io),
      _cilBroadcastTimer(_io), _t_step(opts.step_period),
      _t_broadcast(opts.cil_broadcast_period), _stepCount(0),
      // long tracking period for mandates and environment, effectively tracks
      // over duration of a match
      _myNetwork(48h), _offeredMandates(48h), _environment(48h),
      _mandatePerformance(3min), _cilReportedAssignment(1min),
      _last_su_time(system_clock::now()),
      _myID(opts.gatewayID), _myLocation(1min) /*,
#warning restructure
      _freq_alloc(_io, _ccData, ccparams.client_ip.to_ulong(), 256),
      _occupied_bw(c2api::env.current().scenario_rf_bandwidth * 0.25) */
{

  ///////////////////////////////////////////////////////////////////////////////
  // TRACKING
  ///////////////////////////////////////////////////////////////////////////////

  // subscribe to environment and IM updates, track them
  _st.push_back(
      NotificationCenter::shared.subscribe<EnvironmentUpdateEventInfo>(
          EnvironmentUpdateEvent, _io, [this](auto ei) {
            _environment.track(ei);
            this->_step(Trigger::EnvironmentUpdate);
          }));

  _st.push_back(NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
      OutcomesUpdateEvent, _io, [this](auto ei) {
        _offeredMandates.track(stats::IndividualMandate::fromJSON(ei));
        log::text("Reiceived mandate update: " + ei.j.dump());
        this->_step(Trigger::IMUpdate);
      }));

  // subscribe to control channel receptions, track relevant information
  _st.push_back(NotificationCenter::shared.subscribe<bool>(
      controlchannel::CCData::NewCCDataNotification, _io, [this](auto ei) {
        auto srn_ids = _ccData->getAllSRNIDs();
        // add previously unseen SRNs to our state
        for (auto const &srn_id : srn_ids) {
          this->_trackSRN(srn_id);
        }
        // track duty cycle
        auto dcmap = _ccData->getDutyCycle();
        for (auto const &id : srn_ids) {
          auto it = dcmap.find(id);
          if (it != dcmap.end()) {
            _srns[id].dutyCycle.track(it->second);
          }
        }
        // track location
        auto locmap = _ccData->getLocationMap();
        for (auto const &srn_id : srn_ids) {
          if (locmap.find(srn_id) != end(locmap)) {
            _srns[srn_id].location.track(locmap[srn_id]);
          }
        }
        // track mandates offered to SRNs
        auto currentMandates = _offeredMandates.newest();
        if (currentMandates) {
          // for every flow in our offered mandates, check whether it is active
          // at any SRNs. if it is, see whether we have performance/endpoint
          // information. add any information obtained to the
          // _mandatePerformance tracker. then, update our _srns tracker with
          // the current set of active flows
          auto const activeFlows = _ccData->getActiveFlows();
          auto const flowPerformance = _ccData->getFlowPerformance();
          decltype(_mandatePerformance)::type newMP;
          for (auto const &m : *currentMandates) {
            auto const &flowUID = m.first;
            auto const &im = m.second;
            // find SRNs on which this flow is active
            std::vector<SRNID> activeSRNs;
            for (auto const &af : activeFlows) {
              auto const &sid = af.first;
              for (auto const &f : af.second) {
                auto const &fid = f.flow_uid;
                if (fid == flowUID) {
                  activeSRNs.push_back(sid);
                }
              }
            }
            // try to find performance and endpoint information for this flow
            auto const performance =
                [&flowPerformance,
                 &flowUID]() -> boost::optional<stats::FlowPerformance> {
              if (flowPerformance.find(flowUID) == end(flowPerformance)) {
                return boost::none;
              } else {
                return flowPerformance.at(flowUID);
              }
            }();
            auto const endpoints =
                [this, &flowUID]() -> boost::optional<stats::FlowInfo> {
              auto const fi = _ccData->getFlowInfo(flowUID);
              if (fi.available) {
                return fi;
              } else {
                return boost::none;
              }
            }();
            // track this information as MandateInfo
            newMP.emplace(flowUID, MandateInfo{.mandate = im,
                                               .activeSRNs = activeSRNs,
                                               .performance = performance,
                                               .endpoints = endpoints});
          }
          _mandatePerformance.track(newMP);
          // update the _srns tracker
          auto convertActiveFlows = [](auto const &v) {
            std::vector<FlowUID> o;
            o.reserve(v.size());
            for (auto const &e : v) {
              o.push_back(e.flow_uid);
            }
            return o;
          };
          for (auto const &af : activeFlows) {
            auto const &sid = af.first;
            if (_srns.find(sid) != _srns.end())
              _srns.at(sid).mandatesOffered.track(
                  convertActiveFlows(af.second));
          }
        }
      }));

  // track PSDs
  _st.push_back(
      NotificationCenter::shared.subscribe<
          std::pair<dll::Segment::sptr, std::shared_ptr<std::vector<uint8_t>>>>(
          std::hash<std::string>{}("New Rx PSD Segment"), _io,
          [this](auto data) {
            auto &seg = data.first;
            auto psd_seg =
                std::dynamic_pointer_cast<psdsensing::PSDSegment>(seg);
            if (!psd_seg) {
              return;
            }
            try {
              auto psd_data = psdsensing::PSDSensing::parsePSDSegment(
                  psd_seg->packetContentsBuffer());
              auto const srn_id = psd_data.src_srnid;
              this->_trackSRN(srn_id);
              _srns[srn_id].psd.track(psd_data);
              // log received psd to check if it's correct
              if (psd_data.psd) {
                auto now = std::chrono::system_clock::now().time_since_epoch();
                int64_t now_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
                NotificationCenter::shared.post(psdsensing::PSDRxEvent,
                    psdsensing::PSDRxEventInfo{ srn_id, *(psd_data.psd), now_ns });
              }
            } catch (...) {
              ;
            }
          }));
  // track my PSDs
  _st.push_back(
      NotificationCenter::shared.subscribe<psdsensing::PSDUpdateEventInfo>(
          psdsensing::PSDUpdateEvent, _io, [this](auto data) {
            psdsensing::PSDData d{
                .src_srnid = _myID,
                .time_ns = static_cast<uint64_t>(data.time_ns),
                .psd = std::make_shared<std::vector<float>>(data.psd)};
            this->_trackSRN(_myID);
            _srns[_myID].psd.track(d);
          }));

  // subscribe to peer CIL message reception, track peers

  // SpectrumUsage
  _collabClient->register_handler(
      [this](std::shared_ptr<sc2::cil::SpectrumUsage> msg, auto md) {
        _io.post([this, msg, md] {
          auto const networkID = boost::asio::ip::address_v4(md.sender_id);
          this->_addPeer(networkID, md.type);
          // pull spectrum voxels out of the message
          SpectrumUsage su;
          su.reserve(msg->voxels_size());
          for (auto const &vox : msg->voxels()) {
            // these are mandatory but you never know
            if ((!vox.has_spectrum_voxel()) || (!vox.has_transmitter_info()) ||
                vox.measured_data()) {
              continue;
            }
            FrequencyBand fb{.lower = vox.spectrum_voxel().freq_start(),
                             .upper = vox.spectrum_voxel().freq_end()};
            // throw this away if it does not make sense
            if (fb.lower >= fb.upper) {
              continue;
            }
            // otherwise add this band to the used bands
            CompetitorSRNID const txID = vox.transmitter_info().radio_id();
            double const txPowerdB = vox.transmitter_info().power_db().value();
            Timepoint const start =
                cilTime2Timepoint(vox.spectrum_voxel().time_start());
            auto tend = [&vox]() -> boost::optional<Timepoint> {
              if (vox.spectrum_voxel().has_time_end()) {
                return cilTime2Timepoint(vox.spectrum_voxel().time_end());
              } else {
                return boost::none;
              }
            }();
            std::vector<CompetitorSRNID> receivers;
            receivers.reserve(vox.receiver_info_size());
            for (auto const rx : vox.receiver_info()) {
              receivers.push_back(rx.radio_id());
            }
            su.push_back(FrequencyBandUsage{.band = fb,
                                            .transmitter = txID,
                                            .receivers = receivers,
                                            .txPowerdB = txPowerdB,
                                            .start = start,
                                            .end = tend});
            // add any SRN IDs to the set of tracked SRNs
            _peers.at(networkID).addSRN(txID);
            for (auto const rx : vox.receiver_info()) {
              CompetitorSRNID const rxID = rx.radio_id();
              _peers.at(networkID).addSRN(rxID);
            }
          }
          // if the spectrum usage changed, make sure to keep track of it
          if (su.size() > 0) {
            _peers.at(networkID).spectrumUsage.track(su);
          }
        });
/* #warning restructure
        this->_handleSpectrumUsage(msg, md);
*/
      });

  // LocationUpdate
  _collabClient->register_handler(
      [this](std::shared_ptr<sc2::cil::LocationUpdate> msg, auto md) {
        _io.post([this, msg, md] {
          auto const networkID = boost::asio::ip::address_v4(md.sender_id);
          this->_addPeer(networkID, md.type);
          for (auto const &lu : msg->locations()) {
            if (!lu.has_location()) {
              continue;
            }
            // pull location out of message and update state
            CompetitorSRNID const id = lu.radio_id();
            auto srnInfo = _peers.at(networkID).findSRN(id);
            srnInfo.location.track(
                Location{.latitude = lu.location().latitude(),
                         .longitude = lu.location().longitude(),
                         .elevation = lu.location().elevation()});
          }
        });
/* #warning restructure
        this->_handleLocationUpdate(msg, md);
*/
      });

  // DetailedPerformance
  _collabClient->register_handler(
      [this](std::shared_ptr<sc2::cil::DetailedPerformance> msg, auto md) {
        _io.post([this, msg, md] {
          auto const networkID = boost::asio::ip::address_v4(md.sender_id);
          this->_addPeer(networkID, md.type);
          auto const mandateCount = msg->mandate_count();
          auto const mpEnd = cilTime2Timepoint(msg->timestamp());
          auto const mandatesAchieved = msg->mandates_achieved();
          auto const totalScoreAchieved = msg->total_score_achieved();
          auto const bonusThreshold = msg->scoring_point_threshold();
          std::vector<CompetitorMandatePerformance> cmp;
          cmp.reserve(msg->mandates().size());
          for (auto const &mp : msg->mandates()) {
            std::vector<CompetitorSRNID> ids;
            ids.reserve(mp.radio_ids_size());
            for (auto const &id : mp.radio_ids()) {
              ids.push_back(id);
              // make sure tracked SRN state is consistent
              _peers.at(networkID).addSRN(id);
            }
            cmp.emplace_back(CompetitorMandatePerformance{
                .scalarPerformance = mp.scalar_performance(),
                .radioIDs = ids,
                .holdPeriod = mp.hold_period(),
                .achievedDuration = mp.achieved_duration()});
          }
          _peers.at(networkID).mandates.track(
              CompetitorMandates{.mandateCount = mandateCount,
                                 .mpEnd = mpEnd,
                                 .mandatesAchieved = mandatesAchieved,
                                 .totalScoreAchieved = totalScoreAchieved,
                                 .bonusThreshold = bonusThreshold,
                                 .mandates = cmp});
        });
/* #warning restructure
        this->_handleDetailedPerformance(msg, md);
*/
      });

  // IncumbentNotify
  _collabClient->register_handler(
      [this](std::shared_ptr<sc2::cil::IncumbentNotify> msg, auto md) {
        _io.post([this, msg, md] {
          auto const networkID = boost::asio::ip::address_v4(md.sender_id);
          this->_addPeer(networkID, md.type);
          if (msg->payload_case() ==
              sc2::cil::IncumbentNotify::PayloadCase::kData) {
            auto const msgType = [&msg] {
              switch (msg->data().msg_type()) {
              case sc2::cil::IncumbentPassiveInfo::UNKNOWN:
                return PassiveIncumbentMessage::Type::Unknown;
              case sc2::cil::IncumbentPassiveInfo::REPORT:
                return PassiveIncumbentMessage::Type::Report;
              case sc2::cil::IncumbentPassiveInfo::VIOLATION:
                return PassiveIncumbentMessage::Type::Violation;
              default:
                return PassiveIncumbentMessage::Type::Unknown;
              }
            }();
            _peers.at(networkID).passiveIncumbentMessages.track(
                PassiveIncumbentMessage{
                    .type = msgType,
                    .incumbentID = msg->data().incumbent_id(),
                    .reportTime = cilTime2Timepoint(msg->data().report_time()),
                    .power = msg->data().power(),
                    .threshold = msg->data().threshold(),
                    .centerFreq = msg->data().center_freq(),
                    .bandwidth = msg->data().bandwidth(),
                    .thresholdExceeded = msg->data().threshold_exceeded()});

          } else if (msg->payload_case() ==
                     sc2::cil::IncumbentNotify::PayloadCase::kDataActive) {
            auto const msgType = [&msg] {
              switch (msg->data_active().msg_type()) {
              case sc2::cil::IncumbentActiveInfo::UNKNOWN:
                return ActiveIncumbentMessage::Type::Unknown;
              case sc2::cil::IncumbentActiveInfo::REPORT:
                return ActiveIncumbentMessage::Type::Report;
              case sc2::cil::IncumbentActiveInfo::VIOLATION:
                return ActiveIncumbentMessage::Type::Violation;
              default:
                return ActiveIncumbentMessage::Type::Unknown;
              }
            }();
            _peers.at(networkID).activeIncumbentMessages.track(
                ActiveIncumbentMessage{
                    .type = msgType,
                    .incumbentID = msg->data_active().incumbent_id(),
                    .reportTime = cilTime2Timepoint(msg->data_active().report_time()),
                    .inr = msg->data_active().inr(),
                    .threshold = msg->data_active().threshold(),
                    .centerFreq = msg->data_active().center_freq(),
                    .bandwidth = msg->data_active().bandwidth(),
                    .thresholdExceeded = msg->data_active().threshold_exceeded()});
          } else {
            log::text("Received invalid payload in IncumbentNotify");
            return;
          }
        });
      });

  // as fallback mechanism for CIL broadcast, we track the gateway's location
  _st.push_back(NotificationCenter::shared.subscribe<gps::GPSEventInfo>(
      gps::GPSEvent, _io, [this](auto ei) {
        if (ei.type == gps::GPSEventType::READ_GOOD) {
          _myLocation.track(Location{
              .latitude = ei.lat, .longitude = ei.lon, .elevation = ei.alt});
        }
      }));

  // initialize the env tracker with the current environment
  auto env = c2api::env.current();
  _environment.track(env);

  // attempt to connect to CIL server
  if ((!_collabClient->connected()) && _collabClient->tryConnect()) {
    _collabClient->start();
  }

  // give the lisp thread a name
  lisp.run([] { bamradio::set_thread_name("bamradio-lisp"); });
}

// logic

///////////////////////////////////////////////////////////////////////////////
// THINKING
///////////////////////////////////////////////////////////////////////////////

// FIXME: I have a feeling this can be done better...

void DecisionEngine::_step(Trigger t) {
  auto restartTimer = [this, t] {
    if (t != Trigger::PeriodicStep) {
      return;
    }
    // we *could* run into the problem of the serialization of the decision
    // engine input taking so long that it throws off our timing. in that case,
    // we attempt to recover and simply... turn it off.
    auto next_call = _periodicTriggerTimer.expiry() + _t_step;
    if (next_call <= std::chrono::system_clock::now()) {
      lisp.run([] {
        lisp::Funcall(BRSymbol("disable-db-logging"));
      });
      auto const now = std::chrono::system_clock::now();
      while (next_call <= now) {
        next_call += _t_step;
      }
    }
    _periodicTriggerTimer.expires_at(next_call);
    _periodicTriggerTimer.async_wait(
        [this](auto &e) { this->_step(Trigger::PeriodicStep); });
  };

#ifdef BAM_DE_DEBUG
  auto ti1 = system_clock::now();
#endif

  _currentTrigger = t;
  _currentTriggerTime = t == Trigger::PeriodicStep
                            ? _periodicTriggerTimer.expiry()
                            : std::chrono::system_clock::now();
  _channel_updated = false;

#ifdef BAM_DE_DEBUG
  auto ti2 = system_clock::now();
#endif

  // enter into the LISP environment
  if (!lisp.run(
          [this, &t] { _lispStep(t, _currentTriggerTime, _stepCount++); })) {
    log::text("WARNING! Fatal error in Lisp environment. Restarting.");
  }

#ifdef BAM_DE_DEBUG
  auto to2 = system_clock::now() - ti2;
  log::text(boost::format("XXX lisp exec: %1%ms") %
            duration_cast<milliseconds>(to2).count());
#endif

  // transmit the channel allocation in case the channel was not updated
  if (!_channel_updated) {
    auto const tstart =
        _periodicTriggerTimer.expiry() + options.channel_alloc_delay;
    auto const tend = tstart + options.step_period;
    auto const txAss = [this] {
      TransmitAssignment::Map o;
      for (auto const &s : _srns) {
        auto const &id = s.first;
        auto const &info = s.second;
        auto const txAssignment = info.txAssignment.newest();
        if (txAssignment) {
          o[id] = *txAssignment;
        }
      }
      return o;
    }();
#ifdef BAM_DE_DEBUG
    log::text("XXX send spectrum usage.");
#endif
    _sendSpectrumUsage(tstart, tend, txAss);
  }

#ifdef BAM_DE_DEBUG
  auto to1 = system_clock::now() - ti1;
  log::text(boost::format("XXX step expiry: %1% now: %2% trigger: %3% time: "
                          "%4%ms channel updated? %5%") %
            timepoint2human(_periodicTriggerTimer.expiry()) %
            timepoint2human(ti1) % trigger2human(t) %
            duration_cast<milliseconds>(to1).count() %
            (_channel_updated ? "t" : "f"));
#endif

  // done thinking. call again if needed
  restartTimer();
}

void DecisionEngine::callUpdateTxAssignment(TxAssignmentUpdate update) {
  // help
  auto lmsg = [this] {
    auto lmsg = std::make_shared<BAMLogPb::ChannelAllocEventInfo>();
    lmsg->set_sample_rate(options.sample_rate);
    lmsg->set_guard_band(options.guard_band);
    lmsg->set_safety_margin(112e3);
    lmsg->set_ntries(1);

    auto srn_ids = _ccData->getAllSRNIDs();
    std::sort(begin(srn_ids), end(srn_ids));
    // N.B. The slot mapping remains here because we are to lazy to change the
    // log message and it doesn't matter
    std::map<uint8_t, uint8_t> srn2slot;
    uint8_t k = 0;
    for (auto const &srn_id : srn_ids)
      srn2slot[srn_id] = k++;

    for (auto const &rx_id : srn_ids) {
      lmsg->add_node_ids(rx_id);
      lmsg->add_tx_slots(srn2slot[rx_id]);
    }
    return lmsg;
  }();

  log::text("Updating Tx Assignments.");

  // call the actual function
  auto const channel_updated =
      update.channel_updated || update.bandwidth_updated;
  auto uid = _updateTxAssignment(update.assignment_map, channel_updated);

  // log
  lmsg->set_ofdm_params_update_id(uid);
  NotificationCenter::shared.post(decisionengine::ChannelAllocEvent,
                                  decisionengine::ChannelAllocEventInfo{lmsg});
}

void DecisionEngine::updateOverlapInfo(controlchannel::OverlapMap m) {
  log::text("Updating Overlaps.");
  _ccData->updateOverlapInfo(m);
}

///////////////////////////////////////////////////////////////////////////////
// BROADCASTING
///////////////////////////////////////////////////////////////////////////////

void DecisionEngine::_cilBroadcast(Trigger t) {
  auto restartTimer = [this, t] {
    if (t != Trigger::PeriodicStep) {
      return;
    }
    _cilBroadcastTimer.expires_at(_cilBroadcastTimer.expiry() + _t_broadcast);
    _cilBroadcastTimer.async_wait(
        [this](auto &e) { this->_cilBroadcast(Trigger::PeriodicStep); });
  };
  if (!_ensureCollab()) {
    restartTimer();
    return;
  }
#ifdef BAM_DE_DEBUG
  log::text(boost::format("XXX cilBroadcast expiry: %1% now: %2% ") %
            timepoint2human(_cilBroadcastTimer.expiry()) %
            timepoint2human(system_clock::now()));
#endif
  // broadcast messages
  auto const tstart = _cilBroadcastTimer.expiry();
  auto lu = _makeLocationUpdate(tstart);
  auto dp = _makeDetailedPerformance(tstart);
#ifdef BAM_DE_DEBUG
  if (!lu) {
    log::text("XXX lu == NULL");
  }
  if (!dp) {
    log::text("XXX dp == NULL");
  }
#endif
  for (auto msg : {lu, dp}) {
    if (msg) {
      _collabClient->broadcast(msg);
    }
  }

  // done. make sure that we are called again
  restartTimer();
}

bool DecisionEngine::_ensureCollab() {
  // attempt to connect to CIL server if not connected
  if (!_collabClient->connected()) {
    if (_collabClient->tryConnect()) {
      _collabClient->start();
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

void DecisionEngine::_sendSpectrumUsage(
    Timepoint tstart, Timepoint tend,
    TransmitAssignment::Map const &txAss) {
  // start collab client if not running
  if (!_ensureCollab()) {
    return;
  }

  // safeguard against transmitting in the same second -- this happens when, for
  // example, an environment update happens right after we sent out a
  // spectrum_usage message
  auto const dlast =
      duration_cast<seconds>(system_clock::now() - _last_su_time);

  if (dlast < 1s) {
    return;
#ifdef BAM_DE_DEBUG
    log::text("XXX blocking spectrum_usage transmission.");
#endif
  }

  // construct spectrum usage message
  auto su = std::make_shared<sc2::cil::CilMessage>();
  su->set_sender_network_id(_collabClient->client_ip().to_ulong());
  // future
  _addSpectrumUsage(su, tstart, tend, false, txAss);
  // present
  auto present_report = _cilReportedAssignment.newest();
  if (present_report) {
    _addSpectrumUsage(su, present_report->t_start, present_report->t_end, false,
                      present_report->tx_assignment);
  }
  // past
  auto prev_report = _cilReportedAssignment.previous();
  if (prev_report) {
    _addSpectrumUsage(su, prev_report->t_start, prev_report->t_end, true,
                      prev_report->tx_assignment);
  }

  // track new reported voxels
  _cilReportedAssignment.track(CILTransmitAssignmentReport{
      .t_start = tstart, .t_end = tend, .tx_assignment = txAss});

#ifdef BAM_DE_DEBUG
  if (!su) {
    log::text("XXX su == NULL");
  }
#endif
  if (su) {
    _last_su_time = system_clock::now();
    _collabClient->broadcast(su);
  }
}

void DecisionEngine::_addSpectrumUsage(
    std::shared_ptr<sc2::cil::CilMessage> msg, Timepoint tstart, Timepoint tend,
    bool measured, TransmitAssignment::Map const &txAss) {
  // need the center frequency of the current environment
  auto const env = c2api::env.current();
  auto const cfreq = (double)(env.scenario_center_frequency);

  auto su = msg->mutable_spectrum_usage();

  // iterate over all SRNs and set voxels accordingly
  for (auto const &s : txAss) {
    auto const &id = s.first;
    auto const txAssignment = s.second;
    auto const bw =
        bam::dsp::SubChannel::table()[txAssignment.bw_idx].bw();
 double const offset =
     Channelization::get(env.scenario_rf_bandwidth).center_offsets[txAssignment.chan_idx] + txAssignment.chan_ofst;
    auto const lower = cfreq + offset - bw / 2.0;
    auto const upper = cfreq + offset + bw / 2.0;
    auto const txPowerdB = options.data_tx_gain - txAssignment.atten;
    auto const duty_cycle = s.second.silent ? 0.0 : [&]() -> double {
      auto dcs = _srns[id].dutyCycle.trackedItems();
      // sort according to timestamp of the measurements
      std::sort(dcs.begin(), dcs.end(), [](auto const &a, auto const &b) {
        return a.item.t < b.item.t;
      });
      for (auto it = dcs.rbegin(); it != dcs.rend(); ++it) {
        if (it->item.t <= tend) {
          if (it != dcs.rbegin()) {
            // dc1.t <= tend < dc2.t, linear interpolation
            auto &dc1 = it->item, &dc2 = (it - 1)->item;
            auto lambda = double((tend - dc1.t).count()) /
                          double((dc2.t - dc1.t).count());
            return dc1.duty_cycle + lambda * (dc2.duty_cycle - dc1.duty_cycle);
          } else {
            // tend is in the future, return most recent measurement
            return it->item.duty_cycle;
          }
        }
      }
      return 0.8;
    }();
    // SpectrumVoxelUsage
    auto vu = su->add_voxels();
    vu->set_measured_data(measured);
    // SpectrumVoxel
    auto vox = vu->mutable_spectrum_voxel();
    vox->set_freq_start(lower);
    vox->set_freq_end(upper);
    vox->set_allocated_time_start(timepoint2cilTime(tstart));
    vox->set_allocated_time_end(timepoint2cilTime(tend));
    vox->mutable_duty_cycle()->set_value(duty_cycle);
    // TransmitterInfo
    auto txinfo = vu->mutable_transmitter_info();
    txinfo->set_radio_id(id);
    txinfo->mutable_power_db()->set_value(txPowerdB);
    // ReceiverInfo -- all our SRNs receive from all transmitters (in theory)
    for (auto const &ss : _srns) {
      auto ri = vu->add_receiver_info();
      ri->set_radio_id(ss.first);
    }
  }

  // add the control channel allocation to this message. WLOG just claim that
  // gateway is transmitting on it.
  auto const cchans = _radioCtrl->ctrlChannelAlloc();
  for (auto const &cchan : cchans) {
    auto const offset = cchan.offset;
    auto const bw = cchan.bandwidth;
    auto const lower = cfreq + offset - bw / 2.0;
    auto const upper = cfreq + offset + bw / 2.0;
    auto const txPowerdB = options.control_tx_gain;
    // SpectrumVoxelUsage
    auto vu = su->add_voxels();
    vu->set_measured_data(measured);
    // SpectrumVoxel
    auto vox = vu->mutable_spectrum_voxel();
    vox->set_freq_start(lower);
    vox->set_freq_end(upper);
    vox->set_allocated_time_start(timepoint2cilTime(tstart));
    vox->set_allocated_time_end(timepoint2cilTime(tend));
    // n.b. might be able to go even lower due to randomness in CC fsk
    // transmissions
    vox->mutable_duty_cycle()->set_value(0.2);
    // n.b. we claim this band for ever.
    auto txinfo = vu->mutable_transmitter_info();
    txinfo->set_radio_id(_myID);
    txinfo->mutable_power_db()->set_value(txPowerdB);
    // ReceiverInfo
    for (auto const &ss : _srns) {
      auto ri = vu->add_receiver_info();
      ri->set_radio_id(ss.first);
    }
    if (vu->receiver_info_size() == 0) {
      auto ri = vu->add_receiver_info();
      ri->set_radio_id(_myID);
    }
  }
}

std::shared_ptr<sc2::cil::CilMessage>
DecisionEngine::_makeLocationUpdate(Timepoint t) {
  // prepare the message object
  auto msg = std::make_shared<sc2::cil::CilMessage>();
  msg->set_sender_network_id(_collabClient->client_ip().to_ulong());
  auto lu = msg->mutable_location_update();

  // iterate through all SRNs and pull locations out
  for (auto const &s : _srns) {
    auto const &id = s.first;
    auto const &info = s.second;
    auto const location = info.location.newest();
    if (!location) {
      continue;
    }
    auto li = lu->add_locations();
    li->set_radio_id(id);
    li->set_allocated_timestamp(timepoint2cilTime(t));
    auto loc = li->mutable_location();
    loc->set_latitude(location->latitude);
    loc->set_longitude(location->longitude);
    loc->set_elevation(location->elevation);
  }

  // if we don't have the locations of our SRN's, maybe we have the location of
  // /this/ SRN
  if (lu->locations_size() == 0) {
    auto const myLoc = _myLocation.newest();
    if (!myLoc) {
      // nothing to report
      return nullptr;
    } else {
      auto li = lu->add_locations();
      li->set_radio_id(_myID);
      li->set_allocated_timestamp(timepoint2cilTime(t));
      auto loc = li->mutable_location();
      loc->set_latitude(myLoc->latitude);
      loc->set_longitude(myLoc->longitude);
      loc->set_elevation(myLoc->elevation);
    }
  }

  return msg;
}

std::shared_ptr<sc2::cil::CilMessage>
DecisionEngine::_makeDetailedPerformance(Timepoint t) {
  // Get the current environment
  auto env = c2api::env.current();
  // check whether we have a value for bonus_threshold. if not, don't
  // create this message
  if (env.timestamp == c2api::INIT_ENV_MAGIC) {
    return nullptr;
  }

  // prepare the message object
  auto msg = std::make_shared<sc2::cil::CilMessage>();
  msg->set_sender_network_id(_collabClient->client_ip().to_ulong());
  auto dp = msg->mutable_detailed_performance();
  dp->set_allocated_timestamp(timepoint2cilTime(t));

  // check the number of available mandates in my network
  auto const currentMandates = _mandatePerformance.newest();
  uint32_t nMandates = currentMandates ? currentMandates->size() : 0;
  uint32_t nMandatesAchieved = 0;
  // The total score achieved by my network
  uint32_t total_score_achieved = 0;
  // Get the scoring_point_threshold from the current environment
  auto bonus_threshold = (uint32_t)(env.bonus_threshold);
  dp->set_scoring_point_threshold(bonus_threshold);
  if (nMandates == 0) {
    dp->set_mandate_count(nMandates);
    dp->set_mandates_achieved(nMandatesAchieved);
    dp->set_total_score_achieved(total_score_achieved);
    return msg;
  }

  // go through all mandates and report on their performance if we have
  // performance information.
  for (auto const &mandate : *currentMandates) {
    auto const &fid = mandate.first;
    auto const &info = mandate.second;
    auto md = dp->add_mandates();
    md->set_flow_id(fid);
    md->set_hold_period(info.mandate.hold_period);
    md->set_point_value(info.mandate.point_value);
    if (info.performance) {
      md->set_scalar_performance(info.performance->scalar_performance);
      auto const ad = info.mandate.hold_period < info.performance->mps
                          ? info.performance->mps - info.mandate.hold_period
                          : 0;
      md->set_achieved_duration(ad);
      if (info.performance->scalar_performance >= 1.0) {
        ++nMandatesAchieved;
        total_score_achieved = total_score_achieved + info.mandate.point_value;
      }
    } else {
      md->set_scalar_performance(0.0);
      md->set_achieved_duration(0);
    }
    // add participating radio IDs
    for (auto const &srn : info.activeSRNs) {
      md->add_radio_ids(srn);
    }
  }

  // final inventory and we are done
  dp->set_mandate_count(nMandates);
  dp->set_mandates_achieved(nMandatesAchieved);
  dp->set_total_score_achieved(total_score_achieved);
  return msg;
}

///////////////////////////////////////////////////////////////////////////////
// LEGACY
///////////////////////////////////////////////////////////////////////////////

/*
void DecisionEngine::_handleSpectrumUsage(
    std::shared_ptr<sc2::cil::SpectrumUsage> msg, collab::Metadata md) {

  // pull current environment
  auto env = c2api::env.current();
  auto cfreq = (double)(env.scenario_center_frequency);

  // extract voxels and publish to notification center
  std::map<psdsensing::nodeid_t, std::vector<Channel>> tx_map;
  std::map<psdsensing::nodeid_t, double> tx_power;
  int sz = msg->voxels_size();
  if (sz > 0) {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t now_s =
        std::chrono::duration_cast<std::chrono::seconds>(now).count();
    for (int i = 0; i < sz; ++i) {
      auto vox = msg->voxels(i);
      if (vox.has_spectrum_voxel() && vox.has_transmitter_info()) {
        if (!vox.spectrum_voxel().has_time_end() ||
            (now_s - (int64_t)(vox.spectrum_voxel().time_end().seconds()) <=
                 6 &&
             (int64_t)(vox.spectrum_voxel().time_start().seconds() - now_s <=
                       60))) {
          auto const &txinfo = vox.transmitter_info();
          psdsensing::nodeid_t id{unsigned(md.sender_id),
                                  unsigned(txinfo.radio_id())};
          double left = vox.spectrum_voxel().freq_start();
          double right = vox.spectrum_voxel().freq_end();
          double bandwidth = right - left;
          double offset = left + bandwidth / 2.0 - cfreq;
          Channel ch(bandwidth, offset, options.sample_rate);
          tx_map[id].push_back(ch);
          tx_power[id] = txinfo.power_db().value();
        }
      }
    }
    NotificationCenter::shared.post(psdsensing::FreqAlloc::TxBandNotification,
                                    tx_map);
    NotificationCenter::shared.post(psdsensing::FreqAlloc::TxPowerNotification,
                                    tx_power);
  }
}

void DecisionEngine::_handleLocationUpdate(
    std::shared_ptr<sc2::cil::LocationUpdate> msg, collab::Metadata md) {

  std::map<psdsensing::nodeid_t, controlchannel::Location> their_locmap;
  int sz = msg->locations_size();
  if (sz > 0) {
    for (int i = 0; i < sz; ++i) {
      auto locinfo = msg->locations(i);
      if (locinfo.has_location()) {
        auto loc = locinfo.location();
        psdsensing::nodeid_t id{unsigned(md.sender_id),
                                unsigned(locinfo.radio_id())};
        controlchannel::Location location{loc.latitude(), loc.longitude(),
                                          loc.elevation()};
        their_locmap.insert(std::make_pair(id, location));
      }
    }
    NotificationCenter::shared.post(
        psdsensing::FreqAlloc::NodeLocationNotification, their_locmap);
  }
}

void DecisionEngine::_handleDetailedPerformance(
    std::shared_ptr<sc2::cil::DetailedPerformance> msg, collab::Metadata md) {
  uint32_t id = md.sender_id;
  int n = msg->mandates_achieved();
  // The total score achieved by our peers extracted from their
  // DetailedPerformanceMessage
  uint32_t total_score_achieved = msg->total_score_achieved();
  // The scoring point threshold value for each of our peers extracted from
  // their DetailedPerformanceMessage
  uint32_t scoring_point_threshold = msg->scoring_point_threshold();
  // The difference between the total_score_achieved and the
  // scoring_point_threshold
  int score_delta = total_score_achieved - scoring_point_threshold;
  NotificationCenter::shared.post(
      psdsensing::FreqAlloc::PerformanceNotification,
      std::make_tuple(id, n, score_delta));
}

std::vector<double> DecisionEngine::_bandwidthNeeded() {
  double scenario_bw = c2api::env.current().scenario_rf_bandwidth;
  // find our score and compare with other teams
  // TODO: this should be done in StatCenter
  int our_score = 0;
  auto const currentMandates = _mandatePerformance.newest();
  if (currentMandates) {
    for (auto const &mandate : *currentMandates) {
      auto const &info = mandate.second;
      if (info.performance) {
        if (info.performance->scalar_performance >= 1.0)
          our_score += info.mandate.point_value;
      }
    }
  }
  int bonus_threshold = c2api::env.current().bonus_threshold;
  // determine total bw
  // go through _peers to see their performance
  int team_above = 0, team_total = 1;  // count myself
  for (auto const &p : _peers) {
    auto const &info = p.second;
    if (info.type == NetworkType::Competitor) {
      ++team_total;
      auto const cm = info.mandates.newest();
      if (cm && cm->totalScoreAchieved - cm->bonusThreshold >= 0)
        ++team_above;
    }
  }
  if (_occupied_bw > scenario_bw * 0.6) {
    _occupied_bw = scenario_bw / team_total;
  } else {
    if (our_score < bonus_threshold && (double)team_above / team_total >= 0.5) {
      _occupied_bw *= 1.2;
    } else if (our_score - bonus_threshold > 5 &&
               (double)team_above / team_total <= 0.5) {
      _occupied_bw *= 0.9;
    } else {
      _occupied_bw *= 1.05;
    }
  }

  // tool function to normalize weights
  auto normalize = [](auto const &src, auto &dst) {
    double total = std::accumulate(
        src.begin(), src.end(), 0.0,
        [](auto const &value, auto const &p) { return value + p.second; });
    if (total != 0) {
      for (auto const &p : src)
        dst[p.first] = p.second / total;
    } else {
      auto const N = src.size();
      for (auto const &p : src)
        dst[p.first] = 1.0 / N;
    }
  };

  // distribute total_bw among our nodes
  // 1. load offered
  std::map<NodeID, double> load_ratio;
  std::map<NodeID, double> offered_load_tx;
  auto srn_ids = _ccData->getAllSRNIDs();
  for (auto const rx_id : srn_ids) {
    auto linkstate = _ccData->getLinkStateInfo(rx_id);
    // go through each rx link
    for (auto const &ls : linkstate) {
      auto const tx_id = ls.first;
      auto const load = _ccData->getOfferedDataRate(tx_id, rx_id);
      if (offered_load_tx.find(tx_id) == offered_load_tx.end())
        offered_load_tx[tx_id] = 0.0;
      else
        offered_load_tx[tx_id] += load;
    }
  }
  normalize(offered_load_tx, load_ratio);

  // 2. points offered and
  // 3. spectrum efficiency (points per Hz)
  std::map<NodeID, double> points_ratio;
  std::map<NodeID, double> eff_ratio;
  std::map<NodeID, double> offered_points_tx;
  std::map<NodeID, double> achieved_points_tx;
  if (currentMandates) {
    for (auto const &mandate : *currentMandates) {
      auto const &flowid = mandate.first;
      auto const &info = mandate.second;
      double pv = info.mandate.point_value;
      auto const flowinfo = _ccData->getFlowInfo(flowid);
      if (flowinfo.available) {
        auto const tx_id = flowinfo.src;
        if (offered_points_tx.find(tx_id) == offered_points_tx.end())
          offered_points_tx[tx_id] = 0.0;
        else
          offered_points_tx[tx_id] += pv;
        if (info.performance && info.performance->scalar_performance >= 1.0) {
          if (achieved_points_tx.find(tx_id) == achieved_points_tx.end())
            achieved_points_tx[tx_id] = 0.0;
          else
            achieved_points_tx[tx_id] += pv;
        }
      }
    }
    normalize(offered_points_tx, points_ratio);
    for (auto &p : achieved_points_tx) {
      if (_srns.find(p.first) != _srns.end()) {
        auto const currentTx = _srns.at(p.first).txAssignment.newest();
        if (currentTx) {
          double bw =
              bam::dsp::SubChannel::table()[currentTx->subchannelIdx].bw();
          p.second /= bw;
        }
      }
    }
    normalize(achieved_points_tx, eff_ratio);
  }

  // weight the 3 ratios
  std::vector<double> allocated_bw;
  allocated_bw.reserve(srn_ids.size());
  for (auto const id : srn_ids) {
    auto get_val = [](auto const &rmap, auto const &id, auto const &v) {
      return rmap.find(id) != rmap.end() ? rmap.at(id) : v;
    };
    double lr = get_val(load_ratio, id, 0.0);
    double pr = get_val(points_ratio, id, 0.0);
    double er = get_val(eff_ratio, id, 0.0);
    static double weight[] = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    double ratio = lr * weight[0] + pr * weight[1] + er * weight[2];
    allocated_bw.push_back(_occupied_bw * ratio);
  }
  return allocated_bw;
}

void DecisionEngine::_allocateChannels() {
  // constants
  auto srate = options.sample_rate;
  // minimum guard band between channels
  auto guard_band = options.guard_band;
  // safety margin to band edges (empirically determined)
  auto safety_margin = 112e3;
  // Are two frequencies within 1Hz of each other?
  auto close_1hz = [](auto const &a, auto const &b) {
    return std::abs(a - b) < 1.0;
  };
  // set up the logger output
  auto lmsg = std::make_shared<BAMLogPb::ChannelAllocEventInfo>();
  lmsg->set_sample_rate(srate);
  lmsg->set_guard_band(guard_band);
  lmsg->set_safety_margin(safety_margin);
  lmsg->set_ntries(0);
  lmsg->set_ofdm_params_update_id(0); // indicates unsuccessful attempt
  auto logThis = [&] {
    NotificationCenter::shared.post(
        decisionengine::ChannelAllocEvent,
        decisionengine::ChannelAllocEventInfo{lmsg});
  };

  //
  // get the target waveforms for the given offered loads
  //

  // FIXME bandwidth_needed(...) needs to be replaced
  auto bandwidth_needed = [](double offered_load, ofdm::MCS::Name mcs_id,
                             ofdm::SeqID::ID seq_id) -> double {
    // we compute the "raw" possible rate and throw our hands up and say that we
    // will (subject to overhead and block errors) be able to get wihtin 85% of
    // it
    static const double eff = 0.85;

    auto const mcs = ofdm::MCS::table[mcs_id];
    auto const sym_len = ofdm::SeqID::symLen(seq_id);
    auto const cp_len = ofdm::SeqID::cpLen(seq_id);
    auto const occ = ofdm::SeqID::occupiedCarriers(seq_id);

    // the bandwidth we'd need to service the offered load
    return (offered_load / eff) * (sym_len + cp_len) /
           (occ * mcs.bitsPerSymbol() * mcs.codeRate.k / mcs.codeRate.n);
  };

  auto which_waveform = [this](float target_bw) -> waveform::ID {
    // make sure to have the bandwidths cached
    static std::vector<float> bandwidths;
    if (bandwidths.size() == 0) {
      // i know you shouldn't iterate over enums but Trust Me I'm an Engineer™
      auto ns = static_cast<size_t>(waveform::ID::NUM_SYMBOLS);
      bandwidths.reserve(ns);
      for (size_t i = 0; i < ns; ++i) {
        bandwidths.push_back(
            get(static_cast<waveform::ID>(i)).bw(options.sample_rate));
      }
    }

    // get the bandwidth that is larger than or equal to the target
    auto bw =
        std::upper_bound(begin(bandwidths), end(bandwidths), target_bw,
                         [](auto const &a, auto const &b) { return a <= b; });
    if (bw == end(bandwidths)) {
      return static_cast<waveform::ID>(bandwidths.size() - 1);
    } else {
      return static_cast<waveform::ID>(std::distance(begin(bandwidths), bw));
    }
  };

  auto srn_ids = _ccData->getAllSRNIDs();
  std::sort(begin(srn_ids), end(srn_ids));
  std::map<uint8_t, uint8_t> srn2slot;
  uint8_t k = 0;
  for (auto const &srn_id : srn_ids)
    srn2slot[srn_id] = k++;
  auto nsrn = srn_ids.size();
  // we also want to save the *actual* target bandwidth in order to potentially
  // back off later

  // std::vector<std::pair<double, waveform::ID>> target(
  //     nsrn, {0.0, static_cast<waveform::ID>(0)});
  // // every SRN has a set of rx links. we need to take this data and translate it
  // // to a set of target bandwidths for the transmitters.
  // for (auto const rx_id : srn_ids) {
  //   auto linkstate = _ccData->getLinkStateInfo(rx_id);
  //   // go through each rx link and update the target bandwidth for this channel
  //   // if necessary
  //   for (auto const &ls : linkstate) {
  //     auto const tx_id = ls.first;
  //     auto const mcs = ls.second.mcs;
  //     auto const seq_id = ls.second.seqid;
  //     auto const offered_load = _ccData->getOfferedDataRate(tx_id, rx_id);
  //     auto const target_bw = bandwidth_needed(offered_load, mcs, seq_id);
  //     auto const target_waveform = which_waveform(target_bw);
  //     // if the target bandwidth is larger than the currenly saved target
  //     // bandwidth, update.
  //     if (target_bw > target[srn2slot[tx_id]].first) {
  //       std::pair<double, waveform::ID> t(target_bw, target_waveform);
  //       target[srn2slot[tx_id]] = t;
  //     }
  //   }
  // }

  auto needed_bw = _bandwidthNeeded();
  std::vector<std::pair<double, waveform::ID>> target;
  target.reserve(needed_bw.size());
  for (auto const &bw : needed_bw) {
    auto const target_waveform = which_waveform(bw);
    target.emplace_back(bw, target_waveform);
  }

  std::vector<waveform::ID> channel_waveform;
  for (auto &t : target) {
    // cap max bw
    if (options.max_wf != "") {
      const int max_waveform =
          static_cast<int>(waveform::stringNameToIndex(options.max_wf));
      if (static_cast<uint32_t>(t.second) > max_waveform)
        t.second = static_cast<waveform::ID>(max_waveform);
    }
    channel_waveform.push_back(t.second);
  }
  // add the targets to the log output (they are saved ordered by SLOT NUMBER)
  for (auto const &rx_id : srn_ids) {
    lmsg->add_node_ids(rx_id);
    lmsg->add_tx_slots(srn2slot[rx_id]);
  }
  for (auto const &t : target) {
    auto tp = lmsg->add_target();
    tp->set_bandwidth_needed(t.first);
    tp->set_waveform_id(static_cast<uint32_t>(t.second));
  }

  _freq_alloc.print_debug_info();

  std::vector<std::vector<float>> psd;
  psd.reserve(_srns.size());
  bool all_available = true;
  for (const auto &s : _srns) {
    auto psddata = s.second.psd.newest();
    if (psddata) {
      psd.push_back(*(psddata->psd));
    } else {
      all_available = false;
      break;
    }
  }
  if (!all_available)
    psd.clear();
  auto ctrl_alloc = _radioCtrl->ctrlChannelAlloc();
  auto fa = _freq_alloc.allocate_freq_sinr(channel_waveform, ctrl_alloc, psd);
  auto &chan_vec = fa.channels;
  if ((chan_vec.size() > 0) && (_srns.size() >= chan_vec.size())) {
    if (fa.is_new) {
      // broadcast the new channel assignment
      auto const slot2srn = [this] {
        std::vector<SRNID> o;
        o.reserve(_srns.size());
        for (auto const s : _srns) {
          o.push_back(s.first);
        }
        std::sort(begin(o), end(o));
        return o;
      }();

      std::map<SRNID, ofdm::TransmitAssignment> newAss;
      for (int i = 0; i < chan_vec.size(); ++i) {
        _trackSRN(slot2srn[i]);
        auto const offset = chan_vec[i].cfreq;
        auto const subchannelIdx = static_cast<uint32_t>(chan_vec[i].waveform);
        auto const oldAss = _srns[slot2srn[i]].txAssignment.newest();
        auto const attenuationdB = oldAss ? oldAss->attenuationdB : 0.0f;
        newAss.emplace(slot2srn[i],
                       ofdm::TransmitAssignment{.subchannelIdx = subchannelIdx,
                                                .offset = offset,
                                                .attenuationdB = attenuationdB,
                                                .silent = _silent});
      }

      auto uid = _updateTxAssignment(newAss, true);
      lmsg->set_ofdm_params_update_id(uid);
      logThis();
    }
  } else {
    // failed allocation
    logThis();
  }
}
*/

///////////////////////////////////////////////////////////////////////////////
// Conversions
///////////////////////////////////////////////////////////////////////////////

using lisp::Eval;
using lisp::Funcall;
using lisp::Keyword;
using lisp::List;
using lisp::Print;
using lisp::Quote;
using lisp::Symbol;
using lisp::Value;
using lisp::toLisp;

cl_object toLisp(controlchannel::Location const &l) {
  // clang-format off
  return Funcall(Symbol("make-instance"), BRSymbol("location"),
                 Keyword("latitude"), toLisp(l.latitude),
                 Keyword("longitude"), toLisp(l.longitude),
                 Keyword("elevation"), toLisp(l.elevation));
  // clang-format on
}

cl_object toLisp(Trigger const &t) {
  switch (t) {
  case Trigger::PeriodicStep:
    return BRSymbol("step");
  case Trigger::EnvironmentUpdate:
    return BRSymbol("env-update");
  case Trigger::IMUpdate:
    return BRSymbol("im-update");
  }
}

cl_object toLisp(c2api::EnvironmentManager::Environment const &env) {
  auto collab_network_type = [](auto const &cnt) {
    using NetworkType =
        c2api::EnvironmentManager::Environment::CollabNetworkType;
    switch (cnt) {
    case NetworkType::Internet:
      return BRSymbol("internet");
    case NetworkType::SATCOM:
      return BRSymbol("satcom");
    case NetworkType::HF:
      return BRSymbol("hf");
    case NetworkType::UNSPEC:
      return BRSymbol("unspec");
    }
  }(env.collab_network_type);
  return Funcall(Symbol("make-instance"), BRSymbol("environment"),
                 Keyword("collab-network-type"), collab_network_type,
                 Keyword("rf-bandwidth"), toLisp(env.scenario_rf_bandwidth),
                 Keyword("center-freq"), toLisp(env.scenario_center_frequency),
                 Keyword("bonus-threshold"), toLisp(env.bonus_threshold),
                 Keyword("stage-number"), toLisp(env.stage_number),
                 Keyword("has-incumbent"),
                 env.has_incumbent ? lisp::t : lisp::nil);
}

cl_object toLisp(Timepoint const &t) {
  using namespace std::chrono;
  nanoseconds const NSSinceEpoch(t.time_since_epoch());
  seconds const fullSecSinceEpoch(duration_cast<seconds>(t.time_since_epoch()));
  nanoseconds const fracNS(NSSinceEpoch - nanoseconds(fullSecSinceEpoch));
  return Funcall(BRSymbol("unix-to-timestamp"),
                 toLisp(fullSecSinceEpoch.count()), Keyword("nsec"),
                 toLisp(fracNS.count()));
}

cl_object toLisp(std::map<FlowUID, MandateInfo> const &mandatePerformance) {
  auto list = lisp::List();
  auto const make_instance = Symbol("make-instance");
  // go through all mandates and add to the output. if there is performance
  // and endpoint information available, add it
  for (auto const &m : mandatePerformance) {
    auto const &mi = m.second;
    // these exist for all mandates
    auto const id = toLisp(m.first);
    auto const pointValue = toLisp(mi.mandate.point_value);
    auto const holdPeriod = toLisp(mi.mandate.hold_period);
    // extra information might not exist, add it if it does.
    auto active = lisp::nil;
    auto tx = lisp::nil;
    auto rx = lisp::nil;
    auto perf = lisp::nil;
    if (mi.performance) {
      perf = Funcall(make_instance, BRSymbol("mandate-performance"),
                     Keyword("mps"), toLisp(mi.performance->mps),
                     Keyword("scalar-performance"), toLisp(mi.performance->scalar_performance));
      active = lisp::t;
    }
    if (mi.endpoints) {
      tx = toLisp(mi.endpoints->src);
      rx = toLisp(mi.endpoints->dst);
    }
    // add the mandate
    auto const lobj =
        Funcall(make_instance, BRSymbol("mandate"),
                Keyword("id"), id,
                Keyword("point-value"), pointValue,
                Keyword("hold-period"), holdPeriod,
                Keyword("active"), active,
                Keyword("tx"), tx,
                Keyword("rx"), rx,
                Keyword("performance"), perf);
    lisp::Push(lobj, list);
  }
  return list;
}

cl_object
linkRatesToLisp(std::map<NodeID, std::map<NodeID, float>> const &offeredLinkRates) {
  auto list = lisp::List();
  auto const make_instance = Symbol("make-instance");
  for (auto const &s : offeredLinkRates) {
    for (auto const &d : s.second) {
      auto const lobj = Funcall(make_instance, BRSymbol("offered-traffic-rate"),
                                Keyword("src"), toLisp(s.first),
                                Keyword("dst"), toLisp(d.first),
                                Keyword("bps"), toLisp(d.second));
      lisp::Push(lobj, list);
    }
  }
  return list;
}

cl_object toLisp(PassiveIncumbentMessage const &msg) {
  auto const kind = [&msg] {
    switch (msg.type) {
    case PassiveIncumbentMessage::Type::Unknown:
      return BRSymbol("unknown");
    case PassiveIncumbentMessage::Type::Report:
      return BRSymbol("report");
    case PassiveIncumbentMessage::Type::Violation:
      return BRSymbol("violation");
    }
  }();
  auto const env = c2api::env.current();
  // clang-format off
  return Funcall(Symbol("make-instance"), BRSymbol("passive-incumbent-message"),
                 Keyword("kind"), kind,
                 Keyword("incumbent-id"), toLisp(msg.incumbentID),
                 Keyword("report-time"), toLisp(msg.reportTime),
                 Keyword("power"), toLisp(msg.power),
                 Keyword("threshold"), toLisp(msg.threshold),
                 Keyword("offset"), toLisp(msg.centerFreq - env.scenario_center_frequency),
                 Keyword("bandwidth"), toLisp(msg.bandwidth),
                 Keyword("threshold-exceeded"), toLisp(msg.thresholdExceeded));
  // clang-format on
}

cl_object toLisp(ActiveIncumbentMessage const &msg) {
  auto const kind = [&msg] {
    switch (msg.type) {
    case ActiveIncumbentMessage::Type::Unknown:
      return BRSymbol("unknown");
    case ActiveIncumbentMessage::Type::Report:
      return BRSymbol("report");
    case ActiveIncumbentMessage::Type::Violation:
      return BRSymbol("violation");
    }
  }();
  auto const env = c2api::env.current();
  // clang-format off
  return Funcall(Symbol("make-instance"), BRSymbol("active-incumbent-message"),
                 Keyword("kind"), kind,
                 Keyword("incumbent-id"), toLisp(msg.incumbentID),
                 Keyword("report-time"), toLisp(msg.reportTime),
                 Keyword("inr"), toLisp(msg.inr),
                 Keyword("threshold"), toLisp(msg.threshold),
                 Keyword("offset"), toLisp(msg.centerFreq - env.scenario_center_frequency),
                 Keyword("bandwidth"), toLisp(msg.bandwidth),
                 Keyword("threshold-exceeded"), toLisp(msg.thresholdExceeded));
  // clang-format on
}

cl_object toLisp(psdsensing::PSDData const &psd) {
  if (!(psd.psd)) {
    return lisp::nil;
  }
  return lisp::toLisp(*(psd.psd));
}

// DEPRECATED
cl_object threshPSDtoLisp(psdsensing::PSDData const &psd,
                          psdsensing::PSDSensing::HistParams const &params) {
  if (!(psd.psd)) {
    return lisp::nil;
  }
  auto const psddB = [&] {
    std::vector<float> o((psd.psd)->size());
    volk_32f_log2_32f(o.data(), psd.psd->data(), o.size());
    // 10.0 * log10(2) = 3.010299956639812
    volk_32f_s32f_multiply_32f(o.data(), o.data(), 3.010299956639812, o.size());
    return o;
  }();
  auto const tpsd = psdsensing::PSDSensing::thresholdPSD(psddB, params);
  return lisp::util::toLispBitVector(tpsd);
}

controlchannel::OverlapMap overlapMapFromLisp(cl_object obj) {
  using namespace lisp;
  // assumes the map is a list if cons cells ((id . t/nil) (id . t/nil) ...)
  controlchannel::OverlapMap o;
  forEach(obj, [&o](auto const &elem) {
    auto const id = (NodeID)fromInt(Car(elem));
    auto const overlaps = (Cdr(elem) == t);
    o.emplace(id, overlaps);
  });
  return o;
}

// convert all data, call the LISP entry function
void DecisionEngine::_lispStep(Trigger trigger, Timepoint stepTime,
                               uint64_t stepCount) {
#ifdef BAM_DE_DEBUG
  auto ti = system_clock::now();
#endif

  // environment
  auto const env = c2api::env.current();

  // mandates
  auto const currentMandates = _mandatePerformance.newest();
  auto const mandateInfo =
      currentMandates ? toLisp(*currentMandates) : lisp::nil;

  // nodes
  auto const myNetwork = _myNetwork.newest();
  auto const nodeInfo = !myNetwork ? lisp::nil : [this](auto const &net) {
    auto list = lisp::List();
    auto const make_instance = Symbol("make-instance");
    auto const node = BRSymbol("internal-node");
    for (auto const &srnID : net) {
      // find this srn
      auto const srnInfo =
          std::find_if(cbegin(_srns), cend(_srns),
                       [&srnID](auto const &s) { return s.first == srnID; });
      if (srnInfo == cend(_srns)) {
        continue;
      }
      // pull out BAMSRNInfo data
      auto const now = system_clock::now();
      auto const id = toLisp(srnID);
      auto const tx_assign = [&srnInfo] {
        auto const txAss = srnInfo->second.txAssignment.newest();
        return txAss ? toLisp(*txAss) : lisp::nil;
      }();
      auto const duty_cycle = [&srnInfo] {
        auto const tdc = srnInfo->second.dutyCycle.newest();
        return tdc ? toLisp(tdc->duty_cycle) : lisp::nil;
      }();
      auto const loc = [&srnInfo] {
        auto const tloc = srnInfo->second.location.newest();
        return tloc ? toLisp(*tloc) : lisp::nil;
      }();
      auto const real_psd = [&srnInfo, &now] {
        auto const tpsd = srnInfo->second.psd.newest();
        return tpsd ? tpsd->time() > (now - 2s) ? toLisp(*tpsd) : lisp::nil : lisp::nil;
      }();
      auto const thresh_psd = [&srnInfo, this, &now] {
        auto const tpsd = srnInfo->second.psd.newest();
        return (tpsd ? (tpsd->time() > (now - 2s)
                            ? (tpsd->psd ? lisp::util::toLispBitVector(
                                               _thresholdPSD(*(tpsd->psd)))
                                         : lisp::nil)
                            : lisp::nil)
                     : lisp::nil);
      }();
      // make lisp object from the above
      auto const lobj = Funcall(make_instance, node,
                                Keyword("id"), id,
                                Keyword("location"), loc,
                                Keyword("tx-assignment"), tx_assign,
                                Keyword("est-duty-cycle"), duty_cycle,
                                Keyword("real-psd"), real_psd,
                                Keyword("thresh-psd"), thresh_psd);
      lisp::Push(lobj, list);
    }
    return list;
  }(*myNetwork);

  // collaborators and spectrum usage (can do this in one shot)
  auto collaboratorInfo = lisp::List();
  auto spectrumInfo = lisp::List();
  [this, &collaboratorInfo, &spectrumInfo] {

    auto const make_instance = Symbol("make-instance");
    auto const network = BRSymbol("network");
    auto const node = BRSymbol("node");
    auto const spectrum_user = BRSymbol("spectrum-user");
    auto const spectrum_usage = BRSymbol("spectrum-usage");
    auto const frequency_band = BRSymbol("frequency-band");

    for (auto const &p : _peers) {
      if (!(p.second.type == NetworkType::Competitor || p.second.type == NetworkType::Unknown)) {
        continue;
      }
      // add network info
      auto const &ip = p.first;
      auto const &ni = p.second;
      // pull score
      auto const mi = ni.mandates.newest();
      auto const score = mi ? toLisp(mi->totalScoreAchieved) : lisp::nil;
      auto const score_threshold = mi ? toLisp(mi->bonusThreshold) : lisp::nil;
      // pull locations
      auto nodes = lisp::List();
      for (auto const &srn : ni.SRNs) {
        auto const id = toLisp(srn.id);
        auto const loc = srn.location.newest()
                             ? toLisp(*(srn.location.newest()))
                             : lisp::nil;
        auto const n = Funcall(make_instance, node,
                               Keyword("id"), id,
                               Keyword("location"), loc);
        lisp::Push(n, nodes);
      }
      auto const net = Funcall(make_instance, network,
                               Keyword("id"), toLisp(ip.to_ulong()),
                               Keyword("ip"), toLisp(ip.to_string()),
                               Keyword("reported-score"), score,
                               Keyword("scoring-point-threshold"), score_threshold,
                               Keyword("nodes"), nodes);
      lisp::Push(net, collaboratorInfo);

      // add spectrum usage
      auto const su = ni.spectrumUsage.newest();
      if (su) {
        // get all bands they are declaring and add to spectrum info
        for (auto const &fbu : *su) {
          auto const start = toLisp((int64_t)(fbu.band.lower));
          auto const stop = toLisp((int64_t)(fbu.band.upper));
          auto user = List(Funcall(make_instance, spectrum_user,
                                   Keyword("kind"), BRSymbol("network"),
                                   Keyword("network-id"), toLisp(ip.to_ulong()),
                                   Keyword("tx-id"), toLisp(fbu.transmitter),
                                   Keyword("tx-pow-db"), toLisp(fbu.txPowerdB)));
          auto const si =
              Funcall(make_instance, spectrum_usage,
                      Keyword("band"), Funcall(make_instance, frequency_band,
                                               Keyword("start"), start,
                                               Keyword("stop"), stop),
                      Keyword("users"), user);
          lisp::Push(si, spectrumInfo);
        }
      }
    }
  }();

  auto const incumbentPassiveInfo = !env.has_incumbent ? lisp::nil : [this, &env] {
    // find passive incumbent
    auto const incumbent =
        std::find_if(begin(_peers), end(_peers), [](auto const &p) {
          return p.second.type == NetworkType::IncumbentPassive;
        });
    if (incumbent == end(_peers)) {
      return lisp::nil;
    }
    // find & convert the last message, if applicable
    auto const msg = incumbent->second.passiveIncumbentMessages.newest();
    auto const last_message = msg ? toLisp(*msg) : lisp::nil;
    // for now, take the center-freq and bandwidth from the env definition. note
    // that this is duplicated in the message
    // clang-format off
    return Funcall(
        Symbol("make-instance"), BRSymbol("passive-incumbent"),
        Keyword("offset"), toLisp(env.incumbent_protection.center_frequency - env.scenario_center_frequency),
        Keyword("bandwidth"), toLisp(env.incumbent_protection.rf_bandwidth),
        Keyword("last-message"), last_message);
    // clang-format on
  }();

  auto const incumbentActiveInfo = !env.has_incumbent ? lisp::nil : [this, &env] {
    // find active incumbent
    auto const incumbent =
        std::find_if(begin(_peers), end(_peers), [](auto const &p) {
          return p.second.type == NetworkType::IncumbentActive;
        });
    if (incumbent == end(_peers)) {
      return lisp::nil;
    }
    // find & convert the last message, if applicable
    auto const msg = incumbent->second.activeIncumbentMessages.newest();
    auto const last_message = msg ? toLisp(*msg) : lisp::nil;
    // for now, take the center-freq and bandwidth from the env definition. note
    // that this is duplicated in the message
    // clang-format off
    return Funcall(
        Symbol("make-instance"), BRSymbol("active-incumbent"),
        Keyword("offset"), toLisp(env.incumbent_protection.center_frequency - env.scenario_center_frequency),
        Keyword("bandwidth"), toLisp(env.incumbent_protection.rf_bandwidth),
        Keyword("last-message"), last_message);
    // clang-format on
  }();

  if ((incumbentPassiveInfo != lisp::nil) &&
      (incumbentActiveInfo != lisp::nil)) {
    log::text("Both active and passive incumbents exist in the match. Not sure "
              "what to do.");
  }

  auto const incumbentInfo =
      incumbentPassiveInfo
          ? incumbentPassiveInfo
          : (incumbentActiveInfo ? incumbentActiveInfo : lisp::nil);

  // offered traffic stats
  auto const offeredTrafficRates =
      !myNetwork ? lisp::nil : [this](auto &srnIDs) {
        std::map<NodeID, std::map<NodeID, float>> offeredLinkRates;
        for (auto const &tx : srnIDs) {
          std::map<NodeID, float> links;
          for (auto const &rx : srnIDs) {
            links[rx] = _ccData->getOfferedDataRate(tx, rx);
          }
          offeredLinkRates[tx] = links;
        }
        return linkRatesToLisp(offeredLinkRates);
      }(*myNetwork);

  // put it all together
  auto const data = Funcall(Symbol("make-instance"),
                            BRSymbol("decision-engine-input"),
                            // the trigger
                            Keyword("trigger"), toLisp(trigger),
                            // the step time
                            Keyword("time-stamp"), toLisp(stepTime),
                            // the count
                            Keyword("id"), toLisp(stepCount),
                            // the environment
                            Keyword("env"), toLisp(env),
                            // mandate information
                            Keyword("mandates"), mandateInfo,
                            // node information
                            Keyword("nodes"), nodeInfo,
                            // competitor information
                            Keyword("collaborators"), collaboratorInfo,
                            // spectrum declared using CIL
                            Keyword("declared-spectrum"), spectrumInfo,
                            // incumbent information
                            Keyword("incumbent"), incumbentInfo,
                            // offered traffic stats
                            Keyword("offered-traffic-rates"), offeredTrafficRates);

#ifdef BAM_DE_DEBUG
  auto to = system_clock::now() - ti;
  log::text(boost::format("XXX lisp prep time: %1%ms") %
            duration_cast<milliseconds>(to).count());
#endif

  // step
  Funcall(BRSymbol("decision-engine-step"), data);
}

//
// PSD thresholding -- copypasta from test/psd_thresh.cc as of <2019-09-02 Mon>
//

// don't use this anywhere else.
struct freq_band {
  int64_t lower;
  int64_t upper;
};

template <typename threshfun>
std::vector<uint8_t>
thresh_remove_known(std::vector<float> const &raw_data,
                    std::map<NodeID, freq_band> const &bands,
                    int64_t rf_bandwidth, threshfun tfun) {
  // convert offsett freq to bin index
  auto const ofst2bin = [](int64_t ofst, int64_t nbin) {
    auto const clip = [](auto num, auto lo, auto hi) {
      if (num < lo) {
        return lo;
      } else if (num > hi) {
        return hi;
      } else {
        return num;
      }
    };
    auto const in_range = [](auto num, auto lo, auto hi) {
      if (num < lo) {
        return false;
      } else if (num > hi) {
        return false;
      } else {
        return true;
      }
    };
    auto full = bam::dsp::sample_rate;
    auto half = full / 2;
    if (!in_range((double)ofst, -half, half)) {
      throw std::runtime_error("ofst not in range");
    }
    return (size_t)clip(floor((double)nbin * (((double)ofst / full) + 0.5)),
                        0.0, (double)nbin - 1);
  };

  // get all bins occupied by my transmisstions
  std::vector<size_t> occ_bins;
  for (auto const &b : bands) {
    // N.B overestimating a little to cover all bins
    auto const lower_bin = ofst2bin(b.second.lower, raw_data.size()) - 2;
    auto const upper_bin = ofst2bin(b.second.upper, raw_data.size()) + 2;
    for (auto i = lower_bin; i <= upper_bin; ++i) {
      occ_bins.push_back(i);
    }
  }

  // add the control channel bins as well
  // FIXME: this is hardcoded. undo.
  auto const cc_bw = 480e3;
  auto const cc_edge_offset = 380e3;
  for (auto const cfreq : {rf_bandwidth / 2 - cc_edge_offset,
                           -1 * (rf_bandwidth / 2 - cc_edge_offset)}) {
    auto const lower_bin = ofst2bin(cfreq - cc_bw / 2, raw_data.size()) - 2;
    auto const upper_bin = ofst2bin(cfreq + cc_bw / 2, raw_data.size()) + 2;
    for (auto i = lower_bin; i <= upper_bin; ++i) {
      occ_bins.push_back(i);
    }
  }

  // compute all bins NOT occupied by me, use only those in the thresholding
  std::sort(begin(occ_bins), end(occ_bins));
  std::vector<size_t> all_bins(raw_data.size());
  std::iota(begin(all_bins), end(all_bins), 0);
  std::vector<size_t> free_bins;
  std::set_difference(cbegin(all_bins), cend(all_bins), cbegin(occ_bins),
                      cend(occ_bins), std::back_inserter(free_bins));

  // prepare the data to send to the thresholder
  std::map<size_t, size_t> reverse_map;
  std::vector<float> in_vector;
  in_vector.reserve(free_bins.size());
  std::size_t k = 0;
  for (auto const &bin : free_bins) {
    in_vector.push_back(raw_data[bin]);
    reverse_map.emplace(k++, bin);
  }

  // run the thresholding algorithm and construct the output
  auto const thresh_data = tfun(in_vector);
  // set to two for known transmissions
  std::vector<uint8_t> out_vector(raw_data.size(), 2);
  for (size_t i = 0; i < thresh_data.size(); ++i) {
    out_vector[reverse_map[i]] = thresh_data[i];
  }

  return out_vector;
}

std::vector<float> psd_to_dB(std::vector<float> const &raw_psd) {
  std::vector<float> of(raw_psd.size());
  volk_32f_log2_32f(of.data(), raw_psd.data(), of.size());
  volk_32f_s32f_multiply_32f(of.data(), of.data(), 3.010299956639812,
                             of.size());
  return of;
}

std::vector<uint8_t> intv2uintv(std::vector<int8_t> v) {
  std::vector<uint8_t> o(v.size());
  size_t k = 0;
  std::generate(begin(o), end(o), [&] { return v[k++]; });
  return o;
}

// estimate the noise floor using libvolk. use this as input to the
// histogram-based thresholder. use empirically determined hist_params
std::vector<uint8_t>
est_noise_floor_hist_thresh(std::vector<float> const &raw_psd) {
  using hist_params = bamradio::psdsensing::PSDSensing::HistParams;
  float const spectralExclusionValue = 20.0f;
  float noise_floor_dB;
  auto const psddB = psd_to_dB(raw_psd);
  volk_32f_s32f_calc_spectral_noise_floor_32f(
      &noise_floor_dB, psddB.data(), spectralExclusionValue, raw_psd.size());
  hist_params hp{.bin_size = 0.1,
                 .empty_bin_thresh = 2,
                 .sn_gap_bins = 30,
                 .avg_len = 5,
                 .noise_floor = noise_floor_dB};
  auto const psdThresh =
      bamradio::psdsensing::PSDSensing::thresholdPSD(psddB, hp);
  return intv2uintv(psdThresh);
}

std::vector<uint8_t>
DecisionEngine::_thresholdPSD(std::vector<float> const &psd) {
  auto env = c2api::env.current();
  // get the correct freq_band map
  std::map<NodeID, freq_band> bands;
  auto const myNetwork = _myNetwork.newest();
  if (myNetwork) {
    for (auto const &srnID : *myNetwork) {
      auto const srnInfo =
          std::find_if(cbegin(_srns), cend(_srns),
                       [&srnID](auto const &s) { return s.first == srnID; });
      if (srnInfo == cend(_srns)) {
        continue;
      }
      auto const txAss = srnInfo->second.txAssignment.newest();
      if (txAss) {
        // check whether the environment changed from under us by comparing
        // timestamps of the tx assignment update and the environment. if it
        // did, we use the previous environment. since we always update the
        // transmit assignment in bulk. this is ok.
        auto const txAssTime =
            srnInfo->second.txAssignment.trackedItems().back().t;
        auto const envTime = Timepoint(Duration(env.timestamp));
        if (envTime > txAssTime) {
          env = *c2api::env.previous();
        }
        auto const channelization =
            Channelization::get(env.scenario_rf_bandwidth);
        auto const offset = channelization.center_offsets[txAss->chan_idx];
        int64_t const halfbw =
            bam::dsp::SubChannel::table()[txAss->bw_idx].bw() / 2;
        bands.emplace(srnID,
                      freq_band{.lower = offset - halfbw, .upper = offset + halfbw});
      }
    }
  }
  // return the thresholded PSD
  return thresh_remove_known(
      psd, bands, env.scenario_rf_bandwidth,
      [](auto const &data) { return est_noise_floor_hist_thresh(data); });
}

} // namespace decisionengine
} // namespace bamradio
