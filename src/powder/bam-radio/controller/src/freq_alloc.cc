#include "freq_alloc.h"
#include "events.h"
#include "c2api.h"
#include "bandwidth.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <numeric>
#include <limits>

namespace bamradio {
namespace psdsensing {

NotificationCenter::Name const FreqAlloc::NodeLocationNotification =
    std::hash<std::string>{}("Node Location");
NotificationCenter::Name const FreqAlloc::TxBandNotification =
    std::hash<std::string>{}("Tx Band");
NotificationCenter::Name const FreqAlloc::TxPowerNotification =
    std::hash<std::string>{}("Tx Power");
NotificationCenter::Name const FreqAlloc::PerformanceNotification =
    std::hash<std::string>{}("Detailed Performance");

FreqAlloc::FreqAlloc(boost::asio::io_service& bbc_ios,
                     controlchannel::CCData::sptr ccData,
                     unsigned int _our_network_id, int _num_bins)
    : _ios(bbc_ios),
      _ccData(ccData),
      d_our_network_id(_our_network_id),
      d_num_bins(_num_bins),
      d_half_bw(options::phy::bandwidth / 2.0),
      d_interference_threshold(options::psdsensing::interf_th),
      d_perf_state(PerformanceState::Worst),
      d_new_mandates(true)
{
  d_spectrum.emplace(FreqBand(-d_half_bw, d_half_bw), std::vector<nodenum_t>());

  update_location(d_their_locmap);
  d_total_nodes = d_our_nodes = d_our_locmap.size();

  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<nodeid_t, controlchannel::Location>>(
          NodeLocationNotification, _ios,
          [this](auto loc_map) { this->update_location(loc_map); }));
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<nodeid_t, std::vector<Channel>>>(
          TxBandNotification, _ios,
          [this](auto tx_map) { this->update_tx_list(tx_map); }));
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<nodeid_t, double>>(
          TxPowerNotification, _ios,
          [this](auto tx_power) { this->update_tx_power(tx_power); }));
  // mandated outcomes are used in channel allocation
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
          OutcomesUpdateEvent, _ios,
          [this](auto ei) { this->update_mandated_outcomes(ei.j); }));
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::tuple<networkid_t, int, int>>(
          PerformanceNotification, _ios,
          [this](auto p) { this->update_peer_performance(p); }));
}

void FreqAlloc::FreqBandList::insert(const FreqBand& ins) {
  auto it1 = fb_vec.begin();
  for (; it1 != fb_vec.end(); ++it1) {
    if (it1->rf >= ins.lf) {   // if *it not entirely on the left of ins
      if (it1->lf > ins.rf) {  // no overlap case
        fb_vec.insert(it1, ins);
        return;
      } else {  // partial overlap
        auto it2 = it1;
        for (; it2 != fb_vec.end(); ++it2)
          if (it2->lf > ins.rf)  // [it1, it2) are all partial overlapping
            break;
        it1->lf = std::min(it1->lf, ins.lf);
        it1->rf = std::max((it2 - 1)->rf, ins.rf);
        fb_vec.erase(it1 + 1, it2);
        return;
      }
    }
  }
  // ins is on the right of everything
  fb_vec.insert(it1, ins);
}

void FreqAlloc::FreqBandList::erase(const FreqBand& era) {
  if (fb_vec.empty()) return;
  auto it1 = fb_vec.begin();
  for (; it1 != fb_vec.end(); ++it1)
    if (it1->rf > era.lf)
      break;
  if (it1->lf >= era.rf)
    return;
  auto it2 = it1 + 1;
  for (; it2 != fb_vec.end(); ++it2)
    if (it2->lf >= era.rf)
      break;
  --it2;
  if (it1 == it2) {
    if (it1->lf < era.lf) {
      if (it1->rf > era.rf) {
        double t = it1->lf;
        it1->lf = era.rf;
        fb_vec.insert(it1, FreqBand(t, era.lf));
      }
      else
        it1->rf = era.lf;
    }
    else {
      if (it1->rf > era.rf)
        it1->lf = era.rf;
      else
        fb_vec.erase(it1);
    }
  }
  else {
    if (it1->lf < era.lf) {
      it1->rf = era.lf;
      if (it2->rf > era.rf) {
        it2->lf = era.rf;
        fb_vec.erase(it1 + 1, it2);
      }
      else
        fb_vec.erase(it1 + 1, it2 + 1);
    }
    else {
      if (it2->rf > era.rf) {
        it2->lf = era.rf;
        fb_vec.erase(it1, it2);
      }
      else
        fb_vec.erase(it1, it2 + 1);
    }
  }
}

bool FreqAlloc::FreqBandList::contain(const FreqBand& fb) const {
  for (const auto& mfb : fb_vec)
    if (mfb.lf <= fb.lf && fb.rf <= mfb.rf)
      return true;
  return false;
}

// d_spectrum is a map between FreqBand and nodes that are transmitting
// example:
// node 1 & 2: [-10,-8]; node 3: [2,3]; node 4: [2,5]; then
// FreqBand:            [-10,-8] [-8,2] [2,3] [3,5] [5,10]
// nodes (vector<int>): {1,2}    {}     {3,4} {4}   {}
void FreqAlloc::spectrum_insert(FreqBand ins, nodenum_t i) {
  if (d_spectrum.empty())
    d_spectrum.emplace(FreqBand(-d_half_bw, d_half_bw), std::vector<nodenum_t>());
  if (ins.rf <= -d_half_bw || ins.lf >= d_half_bw)
    return;
  if (ins.lf < -d_half_bw)
    ins.lf = -d_half_bw;
  if (ins.rf > d_half_bw)
    ins.rf = d_half_bw;
  auto it1 = d_spectrum.begin();
  for (; it1 != d_spectrum.end(); ++it1)
    if ((it1->first).rf > ins.lf)
      break;
  auto it2 = it1;
  for (; it2 != d_spectrum.end(); ++it2)
    if ((it2->first).lf >= ins.rf)
      break;
  // now [it1, it2) are overlapping ones
  // divide bands
  const FreqBand& fb1 = it1->first;
  --it2;
  const FreqBand& fb2 = it2->first;
  if (it1 == it2) {
    // ins within existing band
    if (fb1.lf < ins.lf) {
      d_spectrum.emplace(FreqBand(fb1.lf, ins.lf), it1->second);
    }
    if (fb1.rf > ins.rf) {
      d_spectrum.emplace(FreqBand(ins.rf, fb1.rf), it1->second);
    }
    auto vec = it1->second;
    vec.push_back(i);
    d_spectrum.erase(it1);
    d_spectrum.emplace(ins, vec);
  } else {
    auto nit1 = it1, nit2 = it2;
    if (fb1.lf < ins.lf) {
      d_spectrum.emplace(FreqBand(fb1.lf, ins.lf), it1->second);
      // nit1 = start of ins
      nit1 = d_spectrum.insert(
          it1, std::make_pair(FreqBand(ins.lf, fb1.rf), it1->second));
      d_spectrum.erase(it1);
    }
    if (fb2.rf > ins.rf) {
      d_spectrum.emplace(FreqBand(fb2.lf, ins.rf), it2->second);
      // nit2 = end of ins
      nit2 = d_spectrum.insert(
          it2, std::make_pair(FreqBand(ins.rf, fb2.rf), it2->second));
      d_spectrum.erase(it2);
    }
    for (auto it = nit1; it != nit2; ++it)
      (it->second).push_back(i);
  }
}

bool FreqAlloc::empty_spectrum(void) {
  for (auto const& spec : d_spectrum)
    if (!spec.second.empty())
      return false;
  return true;
}

std::map<nodeid_t, FreqAlloc::FreqBandList> FreqAlloc::get_available_fb(void) {
  std::map<nodeid_t, FreqBandList> available_fb;
  for (auto const& spec : d_spectrum) {
    std::vector<double> tx_signals(d_total_nodes, 0);
    // see who's transmitting
    for (auto const& node_num : (spec.second)) {
      auto findit = d_tx_power.find(d_node_ids[node_num]);
      if (findit != d_tx_power.end())
        tx_signals[node_num] = findit->second;
      else
        tx_signals[node_num] = 1;
    }
    auto interference = chan_M_multiply(tx_signals);
    for (int i = 0; i < d_our_nodes; ++i) {
      if (interference[i] <= d_interference_threshold) {
        available_fb[d_node_ids[i]].insert(spec.first);
      }
    }
  }
  return available_fb;
}

std::vector<Channel> FreqAlloc::allocate_freq(void) {
  auto available_fb = get_available_fb();
  auto allocated_fb = greedy_alloc(available_fb);
  std::vector<Channel> dst;
  for (const auto& fb : allocated_fb)
    dst.emplace_back(fb.width(), fb.offset(), options::phy::data::sample_rate);
  return dst;
}

std::vector<FreqAlloc::FreqBand> FreqAlloc::greedy_alloc(
    std::map<nodeid_t, FreqBandList> available_fb) {
  auto dst = d_current_fb;
  std::vector<int> need_alloc;
  for (int j = 0; j < d_our_nodes; ++j) {
    if (std::abs(d_current_fb[j].width() - d_req_bw[j]) > 0.1 ||
        !available_fb[d_node_ids[j]].contain(d_current_fb[j]))
      need_alloc.push_back(j);
  }
  for (auto j : need_alloc) {
    bool success = false;
    FreqBandList& fbl = available_fb[d_node_ids[j]];
    for (int i = 0; i < fbl.size(); ++i) {
      double f = fbl[i].lf + d_req_bw[j];
      if (fbl[i].rf >= f) {
        dst[j] = FreqBand(fbl[i].lf, f);
        for (auto& k : available_fb)
          k.second.erase(dst[j]);
        success = true;
        break;
      }
    }
    if (!success) {
      // cannot find available fb
      // do nothing... keep current band
    }
  }
  d_current_fb = dst;
  return dst;
}

// update location and d_node_ids
void FreqAlloc::update_location(const std::map<nodeid_t, controlchannel::Location>& others_map) {
  std::lock_guard<std::mutex> lock(d_mutex);
  // location map
  d_our_locmap = map_convert(_ccData->getLocationMap());
  auto srn_ids = _ccData->getAllSRNIDs();
  for (auto const& i : srn_ids) {
    auto id = nodeid_t{ d_our_network_id, i };
    if (d_our_locmap.find(id) == d_our_locmap.end())
      d_our_locmap[id] = controlchannel::Location{ 0, 0, 0 };
  }
  // since loc_map is incremental
  if (&others_map != &d_their_locmap) {
    for (auto const& loc : others_map) {
      d_their_locmap[loc.first] = loc.second;
    }
  }
  d_our_nodes = d_our_locmap.size();
  d_total_nodes = d_our_nodes + d_their_locmap.size();

  //node ids
  nodenum_t i = 0;
  for (auto const& loc : d_our_locmap) {
    d_node_ids[i++] = loc.first;
  }
  for (auto const& loc : d_their_locmap) {
    d_node_ids[i++] = loc.first;
  }
  auto all_map = d_their_locmap;
  all_map.insert(d_our_locmap.begin(), d_our_locmap.end());

  // use average location of other nodes in the network if location is not available
  std::map<networkid_t, controlchannel::Location> avg_loc;
  std::map<networkid_t, int> count;
  auto is_valid = [](auto const& loc) {
    // location is valid iff it is finite (!= inf or nan) && not all 0
    return std::isfinite(loc.latitude) &&
      std::isfinite(loc.longitude) &&
      std::isfinite(loc.elevation) &&
      !(std::abs(loc.latitude) < 1e-4 &&
      std::abs(loc.longitude) < 1e-4 &&
      std::abs(loc.elevation) < 1e-4);
  };
  for (auto const& loc : all_map) {
    networkid_t nid = loc.first.network_id;
    if (avg_loc.find(nid) == avg_loc.end()) {
      avg_loc[nid] = controlchannel::Location{ 0.0, 0.0, 0.0 };
      count[nid] = 0;
    }
    if (is_valid(loc.second)) {
      avg_loc[nid].latitude += loc.second.latitude;
      avg_loc[nid].longitude += loc.second.longitude;
      avg_loc[nid].elevation += loc.second.elevation;
      ++count[nid];
    }
  }
  for (auto& loc : avg_loc) {
    networkid_t nid = loc.first;
    if (count[nid] != 0) {
      avg_loc[nid].latitude /= count[nid];
      avg_loc[nid].longitude /= count[nid];
      avg_loc[nid].elevation /= count[nid];
    }
  }
  for (auto& loc : all_map) {
    if (!is_valid(loc.second)) {
      loc.second = avg_loc[loc.first.network_id];
    }
  }
  
  // channel matrix
  d_chan_M.resize(d_total_nodes, std::vector<double>(d_our_nodes, 0));
  for (auto& v : d_chan_M)
    v.resize(d_our_nodes, 0);

  for (int i = 0; i < d_our_nodes; ++i) {
    for (int j = 0; j < i; ++j)
      (d_chan_M[j])[i] = (d_chan_M[i])[j];
    (d_chan_M[i])[i] = 0;
    for (int j = i + 1; j < d_total_nodes; ++j) {
      double d = collab::distance(all_map[d_node_ids[i]], all_map[d_node_ids[j]]);
      if (d < 0.1) d = 0.1;
      (d_chan_M[j])[i] = std::pow(d, -options::psdsensing::path_loss_exponent);
    }
  }
}

void FreqAlloc::update_tx_list(const std::map<nodeid_t, std::vector<Channel>>& channels_map) {
  std::lock_guard<std::mutex> lock(d_mutex);
  if (channels_map.empty()) return;
  // clear entries in d_tx_channels where network_id matches
  auto nid = channels_map.begin()->first.network_id;
  for (auto& ch : d_tx_channels) {
    if (ch.first.network_id == nid)
      ch.second.clear();
  }
  for (auto const& cm : channels_map) {
    for (auto chan : cm.second) {
      // if bandwidth ~= 0, assume 100 kHz
      if (chan.bandwidth < 100e3)
        chan.bandwidth = 100e3;
      d_tx_channels[cm.first].insert(FreqBand(chan));
    }
  }
  if (!d_tx_channels.empty()) {
    d_spectrum.clear();
    for (auto const& chan : d_tx_channels) {
      auto it = std::find_if(d_node_ids.begin(), d_node_ids.end(),
          [&](const auto& p) { return p.second == chan.first; });
      if (it != d_node_ids.end()) {
        for (auto const& fb : chan.second.fb_vec)
          spectrum_insert(fb, it->first);
      }
    }
  }
}

void FreqAlloc::update_tx_power(const std::map<nodeid_t, double>& tx_power) {
  std::lock_guard<std::mutex> lock(d_mutex);
  for (auto const& p : tx_power) {
    double power_db = std::max(10.0, std::min(p.second, 30.0));
    d_tx_power[p.first] =
      std::pow(10.0, power_db / 10.0);
  }
}

void FreqAlloc::update_mandated_outcomes(const nlohmann::json& j) {
  std::lock_guard<std::mutex> lock(d_mutex);
  using json = nlohmann::json;
  for (auto goal : j) {
    flowuid_t uid = goal["flow_uid"];
    auto flowinfo = _ccData->getFlowInfo(uid);
    if (!flowinfo.available) continue;
    auto np = std::make_pair(
        nodeid_t{ d_our_network_id, flowinfo.src }, nodeid_t{ d_our_network_id, flowinfo.dst });
    auto req = goal["requirements"];
    if (req["max_latency_s"].is_number_float()) {
      d_flow_specs[np].has_latency = true;
      d_flow_specs[np].latency = req["max_latency_s"];
    }
    if (req["min_throughput_bps"].is_number_float()) {
      d_flow_specs[np].has_throughput = true;
      d_flow_specs[np].throughput = req["min_throughput_bps"];
    }
  }
  d_new_mandates = true;
}

void FreqAlloc::update_our_performance(void) {
  std::lock_guard<std::mutex> lock(d_mutex);
  auto perf_map = _ccData->getFlowPerformance();
  int n_th = 0, n_hp = 0;
  // The total score achieved by our network
  uint32_t total_score_achieved = 0;
  // Pull the current environment
  auto env = c2api::env.current();
  // Set the current scoring_point_threshold
  auto bonus_threshold = (uint32_t)(env.bonus_threshold);
  auto tock = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tock - d_alloc_tick);
  // assume 30s hold period with 2s grace period
  double thresh = std::max(0.1, std::min(1.0, (duration_ms.count()/1000.0 - 2) / 30));
  for (auto const& pm : perf_map) {
    auto& fp = pm.second;
    if (fp.scalar_performance >= thresh) {
      ++n_th;
      if (fp.scalar_performance >= 1.0) {
        ++n_hp;
        total_score_achieved += fp.point_value;
      }
    }
  }
  // The difference between the total_score_achieved and the scoring_point_threshold
  int score_delta = total_score_achieved - bonus_threshold;
  d_performance[d_our_network_id] = MandatedPerformance{ n_th, n_hp, score_delta };
  update_perf_state();
}

void FreqAlloc::update_peer_performance(const std::tuple<networkid_t, int, int>& p) {
  std::lock_guard<std::mutex> lock(d_mutex);
  d_performance[std::get<0>(p)] = MandatedPerformance{ 0, std::get<1>(p), std::get<2>(p) };
  update_perf_state();
}

void FreqAlloc::update_perf_state(void) {
  // get current state
  PerformanceState state;
  if (d_performance.empty()) return;
  // find min in ensemble excluding ourself and 0-performance
  auto it = std::min_element(d_performance.begin(), d_performance.end(),
      [&](auto const& a, auto const& b) {
        if (a.first == d_our_network_id)
          return false;
        if (b.first == d_our_network_id)
          return true;
        if (a.second.n_achieved == 0)
          return false;
        if (b.second.n_achieved == 0)
          return true;
        return a.second < b.second; });
  auto& worst = it->second;
  auto& us = d_performance[d_our_network_id];
  if (us.score_delta <= worst.score_delta) {
    // no potential to achieve better
    int n_better = std::count_if(d_performance.begin(), d_performance.end(),
        [&](auto const& p) {
          return us.n_passed_threshold < p.second.n_achieved; });
    if (n_better >= 0.5 * d_performance.size() || us.n_passed_threshold <= 0)
      state = PerformanceState::Worst;
    else
      state = PerformanceState::Average;
  }
  else if (us.score_delta > 0 && worst << us) {  // << means "way smaller"
    state = PerformanceState::Better;
  }
  else {
    state = PerformanceState::Average;
  }
  // state change
  if (state != d_perf_state) {
    d_perf_state = state;
    d_state_tick = std::chrono::high_resolution_clock::now();
  }
}

FreqAlloc::PerformanceState FreqAlloc::check_perf_state(void) {
  std::lock_guard<std::mutex> lock(d_mutex);
  // this only happens at the begining of the match
  if (d_performance.empty()) {
    return PerformanceState::Worst;
  }
  auto tock = std::chrono::high_resolution_clock::now();
  if (tock - d_state_tick < std::chrono::milliseconds(5000)) {
    return PerformanceState::JustChanged;
  }
  else if (tock - d_alloc_tick < std::chrono::milliseconds(5000) ||
      tock - d_env_tick < std::chrono::milliseconds(20000)) {
    return PerformanceState::JustReallocated;
  }
  else {
    return d_perf_state;
  }
}

std::vector<double> FreqAlloc::chan_M_multiply(const std::vector<double>& src, bool apply_mask) {
  std::vector<double> sum(d_our_nodes, 0);
  if (!apply_mask) {
    for (int i = 0; i < d_our_nodes; ++i)
      for (int j = 0; j < d_total_nodes; ++j)
        sum[i] += (d_chan_M[j])[i] * src[j];
  }
  else {
    for (int i = 0; i < d_our_nodes; ++i)
      for (int j = 0; j < d_total_nodes; ++j) {
        auto key = std::make_pair(d_node_ids[j], d_node_ids[i]);
        auto it = d_chan_mask.find(key);
        if (it != d_chan_mask.end()) {
          sum[i] += (d_chan_M[j])[i] * it->second * src[j];
        }
        else {
          d_chan_mask[key] = 1;
          sum[i] += (d_chan_M[j])[i] * src[j];
        }
      }
  }
  return sum;
}

void FreqAlloc::update_chan_mask(/*std::vector<double>*/) {
  for (auto const& spec : d_spectrum) {
    std::vector<double> tx_signals(d_total_nodes, 0);
    // see who's transmitting
    for (auto const& node_num : (spec.second)) {
      auto findit = d_tx_power.find(d_node_ids[node_num]);
      if (findit != d_tx_power.end())
        tx_signals[node_num] = findit->second;
      else
        tx_signals[node_num] = 1;
    }
    auto interference = chan_M_multiply(tx_signals, true);
    for (int i = 0; i < d_our_nodes; ++i) {
      // compare interference[i] and node_psd

    }
  }
}

std::vector<double> target_bandwidth(const std::vector<waveform::ID>& wf) {
  std::vector<double> bw(wf.size());
  for (int i = 0; i < wf.size(); ++i) {
    bw[i] = waveform::get(wf[i]).bw(options::phy::data::sample_rate) + 1.2 * options::phy::data::guard_band;
  }
  return bw;
}

std::vector<CFreqWaveform> FreqAlloc::reduce_bandwidth(const std::vector<waveform::ID>& suggest_wf) {
  auto channels = d_current_allocation;
  if (channels.empty()) return channels;
  // reduce bw of channel whose (actual bw - suggest bw) is largest
  auto it = [&]() {
    if (channels.size() == suggest_wf.size()) {
      std::vector<int> idx(channels.size());
      std::iota(idx.begin(), idx.end(), 0);
      auto maxidx = *std::max_element(idx.begin(), idx.end(),
          [&](auto& i, auto& j) {
            int diff1 = static_cast<int>(channels[i].waveform) - static_cast<int>(suggest_wf[i]);
            int diff2 = static_cast<int>(channels[j].waveform) - static_cast<int>(suggest_wf[j]);
            return diff1 < diff2;
          });
      return channels.begin() + maxidx;
    }
    else {
      return std::max_element(channels.begin(), channels.end(),
        [](auto const& a, auto const& b){ return static_cast<int>(a.waveform) < static_cast<int>(b.waveform); });
    }
  }();
  int n = static_cast<int>(it->waveform);
  int min_wf = static_cast<int>(waveform::ID::DFT_S_OFDM_128_500K);
  it->waveform = static_cast<waveform::ID>(n > min_wf ? n - 1 : n);
  return channels;
}

// return empty vector if channel allocation failed due to insufficient information
FreqAllocRet FreqAlloc::allocate_freq_sinr(
    const std::vector<waveform::ID>& suggest_wf, std::vector<Channel> const& ctrl_alloc,
    const std::vector<std::vector<float>>& psd) {
  // update location map and our performance
  update_location(d_their_locmap);
  update_our_performance();

  std::vector<CFreqWaveform> channels;
  if (suggest_wf.size() != d_our_nodes) {
    log::text("d_our_nodes = " + std::to_string(d_our_nodes) +
        " suggest_wf.size() = " + std::to_string(suggest_wf.size()) + "\n");
    return FreqAllocRet{ true, channels };
  }
  
  auto update_current_alloc = [&](auto const& alloc) {
    if (d_current_allocation == alloc) {
      return false;
    }
    else {
      d_current_allocation = alloc;
      d_alloc_tick = std::chrono::high_resolution_clock::now();
      return true;
    }
  };

  // available bw can change
  auto env = c2api::env.current();
  double hbw = (double)env.scenario_rf_bandwidth / 2;
  bool bw_changed = !equal_to(hbw, d_half_bw, 0.1);
  if (bw_changed) {  // half_bw changed
    // update d_spectrum
    d_half_bw = hbw;
    d_spectrum.clear();
    d_spectrum.emplace(FreqBand(-d_half_bw, d_half_bw), std::vector<nodenum_t>());
    if (!d_tx_channels.empty()) {
      for (auto const& chan : d_tx_channels) {
        auto it = std::find_if(d_node_ids.begin(), d_node_ids.end(),
          [&](const auto& p) { return p.second == chan.first; });
        if (it != d_node_ids.end()) {
          for (auto const& fb : chan.second.fb_vec)
            spectrum_insert(fb, it->first);
        }
      }
    }
  }
  // HACK for payline 5MHz bandwidth scenario
  if (equal_to(env.scenario_rf_bandwidth, 5e6, 100.0) &&
    d_our_nodes <= hardcoded_allocation_5MHz.size()) {
    auto clipped = hardcoded_allocation_5MHz;
    clipped.erase(clipped.begin() + d_our_nodes, clipped.end());
    bool r = update_current_alloc(clipped);
    return FreqAllocRet{ r, clipped };
  }
  // HACK for qual scenario (spectrum is empty)
  if (empty_spectrum() &&
      equal_to(env.scenario_rf_bandwidth, 10e6, 100.0) &&
      d_our_nodes <= hardcoded_allocation.size()) {
    auto clipped = hardcoded_allocation;
    clipped.erase(clipped.begin() + d_our_nodes, clipped.end());
    bool r = update_current_alloc(clipped);
    return FreqAllocRet{ r, clipped };
  }
  auto real_wf = suggest_wf;
  if (bw_changed || d_new_mandates) {
    // set initial bw after env update or new mandates
    auto initial_wf = [&]() {
      if (equal_to(env.scenario_rf_bandwidth, 5e6, 100.0))
        return waveform::stringNameToIndex(options::psdsensing::min_wf_base);
      if (equal_to(env.scenario_rf_bandwidth, 8e6, 100.0))
        return waveform::ID::DFT_S_OFDM_128_288K;
      if (equal_to(env.scenario_rf_bandwidth, 10e6, 100.0))
        return waveform::ID::DFT_S_OFDM_128_500K;
      if (equal_to(env.scenario_rf_bandwidth, 20e6, 100.0))
        return waveform::ID::DFT_S_OFDM_128_1M;
      if (equal_to(env.scenario_rf_bandwidth, 25e6, 100.0))
        return waveform::stringNameToIndex(options::psdsensing::initial_wf_pay);
      if (equal_to(env.scenario_rf_bandwidth, 40e6, 100.0))
        return waveform::ID::DFT_S_OFDM_128_2M;
      return waveform::ID::DFT_S_OFDM_128_500K;
    }();
    std::for_each(real_wf.begin(), real_wf.end(), [&](auto& wf) {
      if (static_cast<int>(wf) < static_cast<int>(initial_wf))
        wf = initial_wf;
    });
    if (bw_changed)
      d_env_tick = std::chrono::high_resolution_clock::now();
    else  // reset
      d_new_mandates = false;
  }
  else {  // half_bw didn't change, normal
    // check state
    auto state = check_perf_state();
    if (d_our_nodes == d_current_allocation.size() && state != PerformanceState::Worst) {
      if (state == PerformanceState::JustReallocated ||
        state == PerformanceState::JustChanged ||
        state == PerformanceState::Average) {
        return FreqAllocRet{ false, d_current_allocation };
      }
      else if (state == PerformanceState::Better &&
          !equal_to(env.scenario_rf_bandwidth, 5e6, 100.0)) {
        d_current_allocation = reduce_bandwidth(suggest_wf);
        d_alloc_tick = std::chrono::high_resolution_clock::now();
        return FreqAllocRet{ true, d_current_allocation };
      }
    }
  }
  // HACK: fixed bw
  if (options::psdsensing::fixed_wf != "") {
    std::fill(real_wf.begin(), real_wf.end(),
        waveform::stringNameToIndex(options::psdsensing::fixed_wf));
  }
  // now either # nodes or half_bw changed or we are the worst performing
  // reallocate using the suggested bw
  auto chan_bw = target_bandwidth(real_wf);
  std::vector<nodenum_t> unchanged_idx, relocate_idx;
  if (d_our_nodes != d_current_allocation.size() || bw_changed) {
    // reallocate everything
    relocate_idx.resize(d_our_nodes);
    std::iota(relocate_idx.begin(), relocate_idx.end(), 0);
  }
  else if (options::psdsensing::num_relocate_chans == d_our_nodes) {
    // shortcut, reallocate everything
    relocate_idx.resize(d_our_nodes);
    std::iota(relocate_idx.begin(), relocate_idx.end(), 0);
  }
  else {
    // reallocate the worst 2 channels (options::psdsensing::num_relocate_chans)
    // find two worst channels
    auto perf_map = _ccData->getFlowPerformance();
    std::map<NodeID, double> total_scalar;
    std::map<NodeID, int> num_flows;
    for (auto const& pm : perf_map) {
      auto flowinfo = _ccData->getFlowInfo(pm.first);
      if (!flowinfo.available) continue;
      auto src = flowinfo.src;
      if (total_scalar.find(src) == total_scalar.end()) {
        total_scalar[src] = 0.0;
        num_flows[src] = 0;
      }
      total_scalar[src] += pm.second.scalar_performance;
      ++num_flows[src];
    }
    for (auto& p : total_scalar) {
      p.second /= num_flows[p.first];
    }
    for (int i = 0; i < options::psdsensing::num_relocate_chans; ++i) {
      if (!total_scalar.empty()) {
        auto it = std::min_element(total_scalar.begin(), total_scalar.end(),
          [](auto& a, auto& b) { return a.second < b.second; });
        auto nodeid = it->first;
        auto it2 = std::find_if(d_node_ids.begin(), d_node_ids.end(),
          [&](auto& a) { return d_our_network_id == a.second.network_id && nodeid == a.second.radio_id; });
        if (it2 != d_node_ids.end()) {
          relocate_idx.push_back(it2->first);
        }
        total_scalar.erase(it);
      }
    }
    // in case not enough info is available to determine
    // which channels to relocate, randomly pick them
    while (relocate_idx.size() < options::psdsensing::num_relocate_chans) {
      int i = std::rand() % d_our_nodes;
      if (std::find(relocate_idx.begin(), relocate_idx.end(), i) == relocate_idx.end())
        relocate_idx.push_back(i);
    }
    for (int i = 0; i < d_our_nodes; ++i) {
      if (std::find(relocate_idx.begin(), relocate_idx.end(), i) == relocate_idx.end())
        unchanged_idx.push_back(i);
    }
  }
  // hack: make unchanged_idx have 0 bw so that only relocate_idx channels will be allocated
  for (auto i : unchanged_idx) {
    chan_bw[i] = 0.0;
  }

  // sort bw in decsending order, with gw node being the first
  // i.e. gw node has highest prioty in allocation, and others'
  // priorities are relative to their bw
  std::vector<int> idx(chan_bw.size());
  std::iota(idx.begin(), idx.end(), 0);
  unsigned int gw_id = _ccData->getGatewaySRNID();
  std::stable_sort(idx.begin(), idx.end(), [&](auto i, auto j) {
    if (d_node_ids[j].radio_id == gw_id)
      return false;
    if (d_node_ids[i].radio_id == gw_id)
      return true;
    return chan_bw[i] > chan_bw[j];
  });

  std::vector<int> chan_bw_bins;
  for (auto const& v : chan_bw)
    chan_bw_bins.push_back(bw_to_bins(v));
  // calculate Values
  // if psd is available, calculate values based on psd, otherwise by cil
  // TODO: hybrid between psd and cil
  if (psd.size() == d_our_nodes 
      && d_current_allocation.size() == d_our_nodes
      && options::psdsensing::psd_cil_input > 0.5) {
    update_node_values_psd(chan_bw_bins, psd);
  } else {
    update_node_values(chan_bw_bins);
  }

  // mask control channel and unchanged data channels
  std::vector<int> mask(d_num_bins, 1);
  for (auto const &cc : ctrl_alloc) {
    for (int i = freq_to_bin(cc.lower()); i <= freq_to_bin(cc.upper()); ++i) {
      mask[i] = 0;
    }
  }
  for (auto idx : unchanged_idx) {
    auto& ch = d_current_allocation[idx];
    for (int i = freq_to_bin(ch.lf()); i <= freq_to_bin(ch.rf()); ++i) {
      mask[i] = 0;
    }
  }
  // mask peers' channels if they're below Bonus Threshold and we're above BT
  mask_peer_channels(mask);

  std::vector<int> bw_bins_sorted(d_our_nodes);
  for (int i = 0; i < d_our_nodes; ++i)
    bw_bins_sorted[i] = chan_bw_bins[idx[i]];
  std::vector<std::vector<double>> values_sorted;
  for (int i = 0; i < d_our_nodes; ++i)
    values_sorted.push_back(d_node_values[d_node_ids[idx[i]]]);

  if (chan_bw_bins.size() == values_sorted.size() && values_sorted.size() > 0) {
    auto cfreq_val = assign_channels(mask, bw_bins_sorted, values_sorted);
    log::text("Total value = " + std::to_string(cfreq_val.total_value) + "\n");
    channels.resize(d_our_nodes);
    int cfsize = cfreq_val.center_freqs.size();
    for (int i = 0; i < cfsize; ++i) {
      if (bw_bins_sorted[i] > 0)
        channels[idx[i]] = CFreqWaveform{ bin_to_freq(cfreq_val.center_freqs[i],
          bw_bins_sorted[i] % 2), real_wf[idx[i]] };
      else
        channels[idx[i]] = d_current_allocation[idx[i]];
    }
    // assign_channels may return a vector whose size is smaller than
    // input chan_bw vector size in the case of insufficient bandwidth.
    // HACK: in such case, put dummy values in remaining channels so
    //       that radio controller would turn off this channel.
    for (int i = cfsize; i < d_our_nodes; ++i) {
      channels[idx[i]] = CFreqWaveform{ 2.718e28, real_wf[idx[i]] };
    }
  }
  d_current_allocation = channels;
  d_alloc_tick = std::chrono::high_resolution_clock::now();
  return FreqAllocRet{ true, channels };
}

void FreqAlloc::update_node_values(const std::vector<int>& chan_bw) {
  for (int i = 0; i < d_our_nodes; ++i) {
    if (d_node_values[d_node_ids[i]].size() != d_num_bins) {
      d_node_values[d_node_ids[i]].resize(d_num_bins);
      d_node_interf[d_node_ids[i]].resize(d_num_bins);
    }
  }
  for (auto const& spec : d_spectrum) {
    std::vector<double> tx_signals(d_total_nodes, 0);
    // calculate interference
    for (auto const& node_num : (spec.second)) {
      auto findit = d_tx_power.find(d_node_ids[node_num]);
      if (findit != d_tx_power.end())
        tx_signals[node_num] = findit->second;
      else
        tx_signals[node_num] = 30;  // ~15dB
      // TODO: use average tx power in a network
    }
    auto interference = chan_M_multiply(tx_signals);
    // calculate interf at all f in spec.first
    int lower = freq_to_bin(spec.first.lf);
    int upper = freq_to_bin(spec.first.rf);
    for (int i = 0; i < d_our_nodes; ++i) {
      for (int f = lower; f < upper; ++f) {
        d_node_interf[d_node_ids[i]][f] = interference[i];
      }
    }
  }
  for (int i = 0; i < d_our_nodes; ++i) {
    int bw = chan_bw[i];
    if (bw == 0) continue;
    // calculate link weights
    std::vector<double> weight(d_our_nodes, 0);
    bool has_weight = false;
    for (int j = 0; j < d_our_nodes; ++j) {
      if (j != i) {
        auto np = std::make_pair(d_node_ids[i], d_node_ids[j]);
        if (d_flow_specs.find(np) != d_flow_specs.end()) {
          auto& fs = d_flow_specs[np];
          if (fs.has_throughput) {
            weight[j] = fs.throughput;
            has_weight = true;
          }
        }
      }
    }
    double sum_thr = std::accumulate(weight.begin(), weight.end(), 0.0);
    if (has_weight && sum_thr != 0) {
      for (auto& w : weight)
        w /= sum_thr;
    }
    else {
      int n = (d_our_nodes == 1 ? 1 : d_our_nodes - 1);
      for (auto& w : weight)
        w = 1.0 / (double)n;
    }
    // calculate values
    for (int cf = 0; cf < d_num_bins; ++cf) {
      if (cf - bw / 2 < 0 || cf + bw / 2 >= d_num_bins)
        d_node_values[d_node_ids[i]][cf] = -1000; // avoid edges
      else {
        double val = 0;
        for (int j = 0; j < d_our_nodes; ++j) {
          if (j != i) {
            double avg_sinr = 0;
            for (int f = cf - bw / 2; f <= cf + bw / 2; ++f) {
              // can use snr estimate between i and j?
              double sinr = d_chan_M[i][j] / d_node_interf[d_node_ids[j]][f];
              avg_sinr += std::max(std::min(10 * std::log10(sinr), 20.0), -20.0);
            }
            avg_sinr /= bw;
            val += avg_sinr * weight[j]; // weighted sum
          }
        }
        d_node_values[d_node_ids[i]][cf] = val;
      }
    }
  }
}

void FreqAlloc::update_node_values_psd(const std::vector<int>& chan_bw,
    const std::vector<std::vector<float>>& psd) {
  for (int i = 0; i < d_our_nodes; ++i) {
    if (d_node_values[d_node_ids[i]].size() != d_num_bins) {
      d_node_values[d_node_ids[i]].resize(d_num_bins);
      d_node_interf[d_node_ids[i]].resize(d_num_bins);
    }
  }
  // find channel gain between our nodes
  // M[i][j] effectively = power at node j in i's tx band
  int vec_len = psd[0].size();
  auto M = std::vector<std::vector<double>>(d_our_nodes, std::vector<double>(d_our_nodes));
  for (int i = 0; i < d_our_nodes; ++i) {
    double cf = d_current_allocation[i].cfreq;
    double bw = waveform::get(d_current_allocation[i].waveform).bw(
        bam::dsp::sample_rate);
    int lf = freq_to_bin(cf - bw / 2, bam::dsp::sample_rate, vec_len);
    int rf = freq_to_bin(cf + bw / 2, bam::dsp::sample_rate, vec_len);
    for (int j = 0; j < d_our_nodes; ++j) {
      double s = std::accumulate(psd[j].begin() + lf, psd[j].begin() + rf + 1, 0.0);
      M[i][j] = s / (rf - lf + 1);
    }
  }
  // interference(j, f) = psd(j, f)
  for (int j = 0; j < d_our_nodes; ++j) {
    for (int f = 0; f < d_num_bins; ++f) {
      // psd does not have the same range as scenario bw
      double freq = bin_to_freq(f, false);
      int psd_idx = freq_to_bin(freq, bam::dsp::sample_rate, vec_len);
      d_node_interf[d_node_ids[j]][f] = psd[j][psd_idx];
    }
  }
  for (int i = 0; i < d_our_nodes; ++i) {
    int bw = chan_bw[i];
    if (bw == 0) continue;
    // calculate link weights
    std::vector<double> weight(d_our_nodes, 0);
    bool has_weight = false;
    for (int j = 0; j < d_our_nodes; ++j) {
      if (j != i) {
        auto np = std::make_pair(d_node_ids[i], d_node_ids[j]);
        if (d_flow_specs.find(np) != d_flow_specs.end()) {
          auto& fs = d_flow_specs[np];
          if (fs.has_throughput) {
            weight[j] = fs.throughput;
            has_weight = true;
          }
        }
      }
    }
    double sum_thr = std::accumulate(weight.begin(), weight.end(), 0.0);
    if (has_weight && sum_thr != 0) {
      for (auto& w : weight)
        w /= sum_thr;
    }
    else {
      int n = (d_our_nodes == 1 ? 1 : d_our_nodes - 1);
      for (auto& w : weight)
        w = 1.0 / (double)n;
    }
    // calculate values
    for (int cf = 0; cf < d_num_bins; ++cf) {
      if (cf - bw / 2 < 0 || cf + bw / 2 >= d_num_bins)
        d_node_values[d_node_ids[i]][cf] = -1000; // avoid edges
      else {
        double val = 0;
        for (int j = 0; j < d_our_nodes; ++j) {
          if (j != i) {
            double avg_sinr = 0;
            for (int f = cf - bw / 2; f <= cf + bw / 2; ++f) {
              double sinr = M[i][j] / d_node_interf[d_node_ids[j]][f];
              avg_sinr += std::min(10 * std::log10(sinr), 20.0);
            }
            avg_sinr /= bw;
            val += avg_sinr * weight[j]; // weighted sum
          }
        }
        d_node_values[d_node_ids[i]][cf] = val;
      }
    }
  }
}

void FreqAlloc::mask_peer_channels(std::vector<int>& mask) {
  if (d_performance[d_our_network_id].score_delta > 0) {  // TODO: greater by a certain margin?
    for (auto const& p : d_tx_channels) {
      if (d_performance[p.first.network_id].score_delta < 0) {
        for (auto const& fb : p.second.fb_vec) {
          int a = freq_to_bin(fb.lf), b = freq_to_bin(fb.rf);
          for (int i = a; i <= b; ++i)
            mask[i] = 0;
        }
      }
    }
  }
}

void FreqAlloc::print_debug_info(void) {
  std::lock_guard<std::mutex> lock(d_mutex);
  auto to_ip4 = [](unsigned n) {
    return std::to_string((n >> 24) & 0xFF) + "." +
           std::to_string((n >> 16) & 0xFF) + "." +
           std::to_string((n >> 8) & 0xFF) + "." +
           std::to_string(n & 0xFF);
  };
  auto perf_state_string = [](auto const& s) {
    switch (s) {
    case PerformanceState::Worst: return "Worst";
    case PerformanceState::Average: return "Average";
    case PerformanceState::Better: return "Better";
    case PerformanceState::JustReallocated: return "JustReallocated";
    case PerformanceState::JustChanged: return "JustChanged";
    }
    return "";
  };
  std::stringstream info;
  info << "    state: " << perf_state_string(d_perf_state);
  info << "    num_nodes: " << d_our_nodes << " " << d_total_nodes << std::endl;
  info << "    id => location; channels; power" << std::endl;
  int i = 0;
  for (auto const& p : d_our_locmap) {
    auto id = p.first;
    info << "    (" << to_ip4(id.network_id) << ", " << id.radio_id << ") => ";
    info << "{" << p.second.latitude << " " << p.second.longitude << " " << p.second.elevation << "}";
    if (d_our_nodes == d_current_allocation.size()) {
      auto const& ch = d_current_allocation[i++];
      double bw = waveform::get(ch.waveform).bw(options::phy::data::sample_rate) + options::phy::data::guard_band;
      info << "; [" << ch.cfreq - bw / 2 << "," << ch.cfreq + bw / 2 << "]";
    }
    info << std::endl;
  }
  for (auto const& p : d_their_locmap) {
    auto id = p.first;
    info << "    (" << to_ip4(id.network_id) << ", " << id.radio_id << ") => ";
    info << "{" << p.second.latitude << " " << p.second.longitude << " " << p.second.elevation << "};";
    for (auto const& fb : d_tx_channels[id].fb_vec)
      info << " [" << fb.lf << "," << fb.rf << "]";
    info << "; ";
    info << d_tx_power[id] << std::endl;
  }
  log::text(info.str());
}

// recursively assign center_freq based on bandwidth requirement and value for given node lists. 
// Once a node is assigned then remove it from node list
/* this function returns center freq assigned for all nodes
input:
mask_bands:
B: array of length num_node denoting bandwidth requirement for each node
V_original: 10*128 matrix, where V[i][f] denotes value of choosing center-freq f for node i

output:
vector of length num_node+1, where output[0] is total_value for the assignment, and output[i] denotes center freq for node i
*/
ChannelsValues assign_channels(std::vector<int> mask_bands, std::vector<int> B, std::vector<std::vector<double>> V){
  auto const inf = std::numeric_limits<double>::infinity();
  // helper function for 1D convolution
  auto conv = [](std::vector<int> const &f, std::vector<int> const &g) {
    int const nf = f.size();
    int const ng = g.size();
    int const n = nf + ng - 1;
    std::vector<int> out(n, 0);
    for (auto i(0); i < n; ++i) {
      int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
      int const jmx = (i <  nf - 1) ? i : nf - 1;
      for (auto j(jmn); j <= jmx; ++j) {
        out[i] += (f[j] * g[i - j]);
      }
    }
    return out;
  };
  //helper function: return a vector containing indices i where aux[i] < val
  auto find_idx = [](const std::vector<int>& aux, const int val) {
    std::vector<int> indices;
    for (int i = 0; i<aux.size(); ++i){
      if (aux[i] < val){
        indices.push_back(i);
      }
    }
    return indices;
  };

  ChannelsValues ret_val;
  // check at least one feasible assignment exists
  // construct a ones array with size of B[0]
  std::vector<int> ones(B[0], 1);
  // aux includes all available bandwidths. 
  std::vector<int> aux = conv(mask_bands, ones);

  // if no feasible assignments then we are supposed to recursively go back to the previous level.
  // if there are more than one feasible assignments then assign the first one available
  double total_value;
  // if we do not have any availale bands for current node then set total_val to -inf and go back to previous level of tree.
  if (*std::max_element(aux.begin(), aux.end())<B[0])
  {
    ret_val.total_value = -inf;
  }
  else{
    // we have feasible assignments. So take the first assignment available
    //crop first and last few values from convolution.
    int halfBl = B[0] / 2;
    int halfBr = std::floor((B[0] - 1) / 2.0);
    aux.erase(aux.end() - halfBl, aux.end());
    aux.erase(aux.begin(), aux.begin() + halfBr);
    //eliminate masked_bands
    auto pos = find_idx(mask_bands, 1);
    for (auto const& i : pos)
      V[0][i] = -inf;
    // eliminate gaps that are too narrow
    auto pos_narrow = find_idx(aux, B[0]);
    for (auto const& i : pos_narrow)
      V[0][i] = -inf;

    //sort choices from highest to lowest value, and store it in values vector
    auto values = V[0];
    int chan_size = values.size();
    // initializing index array idx
    std::vector<int> idx(chan_size);
    std::iota(idx.begin(), idx.end(), 0);
    // sort in decending order of values
    std::stable_sort(idx.begin(), idx.end(), [&](auto i, auto j){ return values[i] > values[j]; });
    // if we are at the last node, then we found a feasible assignment for all nodes just return 
    if (B.size() == 1){
      ret_val.total_value = values[idx[0]];
      // just take the first available choices
      ret_val.center_freqs.push_back(idx[0]);

    }
    // the case that we found solution for current node, but needs to proceed to next node.
    else{
      // evaluate first choice
      int choice1 = idx[0];
      auto mask_bands1 = mask_bands;
      //update mask_bands to eliminate the chosen band for current node
      for (int i = std::max(choice1 - halfBl,0); i<std::min(choice1 + halfBr + 1, chan_size); ++i)
        mask_bands1[i] = 0;

      // recursive part: pop current node from B and V
      B.erase(B.begin()); //B=B(2:end)
      V.erase(V.begin()); //V = V(2:end,:)
      auto freq_alloc1 = assign_channels(mask_bands1, B, V);
      auto total_value1 = freq_alloc1.total_value;
      total_value1 = total_value1 + values[idx[0]];

      // evaluate second choice
      int choice_i = 1;
      // find the second available center_freq . Q: Why are we not doing this for the first center_freq choice?
      while (abs(choice1 - idx[choice_i]) < halfBl)
        choice_i = choice_i + 1;
      int choice2 = idx[choice_i];
      auto mask_bands2 = mask_bands;
      for (int i = std::max(choice2 - halfBl,0); i<std::min(choice2 + halfBr + 1, chan_size); ++i)
        mask_bands2[i] = 0;
      auto freq_alloc2 = assign_channels(mask_bands2, B, V);
      auto total_value2 = freq_alloc2.total_value;
      total_value2 = total_value2 + values[idx[choice_i]];

      // third choice
      choice_i = choice_i + 1;
      // find the second available center_freq . Q: Why are we not doing this for the first center_freq choice?
      while (abs(choice1 - idx[choice_i]) < halfBl)
        choice_i = choice_i + 1;
      int choice3 = idx[choice_i];
      auto mask_bands3 = mask_bands;
      for (int i = std::max(choice3 - halfBl,0); i<std::min(choice3 + halfBr + 1, chan_size); ++i)
        mask_bands3[i] = 0;
      auto freq_alloc3 = assign_channels(mask_bands3, B, V);
      auto total_value3 = freq_alloc3.total_value;
      total_value3 = total_value3 + values[idx[choice_i]];

      {
        // return chosen solution
        // find the corresponding choice index for the max total_value and return corresponding choice and center_freq
        std::vector<double> total_val{ total_value1, total_value2, total_value3 };
        std::vector<std::vector<int>> choice_freqs{ freq_alloc1.center_freqs, freq_alloc2.center_freqs, freq_alloc3.center_freqs };
        std::vector<int> choices{ choice1, choice2, choice3 };
        auto it = std::max_element(total_val.begin(), total_val.end());
        int idx = std::distance(total_val.begin(), it);
        ret_val.total_value = total_val[idx];
        ret_val.center_freqs.push_back(choices[idx]);
        for (auto const& v : choice_freqs[idx])
          ret_val.center_freqs.push_back(v);
      }
    }

  } // end else: case that we have available bands
  return ret_val;
} // end center_freq assignment

}
}
