#ifndef INCLUDED_FREQ_ALLOC_H
#define INCLUDED_FREQ_ALLOC_H

#include "cc_data.h"
#include "collab.h"
#include "notify.h"
#include "options.h"
#include "statistics.h"
#include "radiocontroller_types.h"

#include <map>
#include <mutex>
#include <vector>
#include <chrono>
#include <cmath>

#include "json.hpp"
#include <boost/asio.hpp>

namespace bamradio {
namespace psdsensing {

auto equal_to = [](double a, double b, double th) {
  return std::abs(a - b) < th;
};
template<typename T>
T clip_range(const T val, const T min, const T max) {
  if (val < min) return min;
  if (max < val) return max;
  return val;
}

typedef uint32_t networkid_t;
struct nodeid_t {
  networkid_t network_id;
  unsigned int radio_id;
  bool operator<(const nodeid_t& i) const {
    return network_id < i.network_id
        || (network_id == i.network_id && radio_id < i.radio_id);
  }
  bool operator==(const nodeid_t& i) const {
    return network_id == i.network_id && radio_id == i.radio_id;
  }
};
typedef int nodenum_t;
typedef uint16_t flowuid_t;
struct FlowRequirement {
  bool has_latency = false, has_throughput = false;
  // in sec and bps
  double latency = 0, throughput = 0;
};
struct CFreqWaveform {
  double cfreq;
  waveform::ID waveform;
  bool operator==(const CFreqWaveform& c) const {
    return waveform == c.waveform && equal_to(cfreq, c.cfreq, 0.1);
  }
  double lf(void) const {
    double bw = waveform::get(waveform).bw(options::phy::data::sample_rate)
      + options::phy::data::guard_band;
    return cfreq - bw / 2;
  }
  double rf(void) const {
    double bw = waveform::get(waveform).bw(options::phy::data::sample_rate)
      + options::phy::data::guard_band;
    return cfreq + bw / 2;
  }
};
struct FreqAllocRet {
  bool is_new;
  std::vector<CFreqWaveform> channels;
};
struct MandatedPerformance {
  // # mandates that passed the perf threshold, but not yet the hold period
  int n_passed_threshold = 0;
  // # mandates that have been maintained over the hold period
  int n_achieved = 0;
  // Score Delta
  int score_delta = 0;
  bool operator<(const MandatedPerformance& other) const {
	return score_delta < other.score_delta;
  }
  bool operator<=(const MandatedPerformance& other) const {
	return score_delta <= other.score_delta;
  }
  bool operator<<(const MandatedPerformance& other) const {
    return score_delta <= other.score_delta - 5 && other.score_delta > 0;
  }
  bool too_low(void) const { return n_achieved <= 0; }
};

class FreqAlloc {
public:
  static NotificationCenter::Name const NodeLocationNotification;
  static NotificationCenter::Name const TxBandNotification;
  static NotificationCenter::Name const TxPowerNotification;
  static NotificationCenter::Name const PerformanceNotification;

  const std::vector<CFreqWaveform> hardcoded_allocation{
    { -3.6e6, waveform::ID::DFT_S_OFDM_128_715K },
    { -2.8e6, waveform::ID::DFT_S_OFDM_128_715K },
    { -2.0e6, waveform::ID::DFT_S_OFDM_128_715K },
    { -1.2e6, waveform::ID::DFT_S_OFDM_128_715K },
    { -0.4e6, waveform::ID::DFT_S_OFDM_128_715K },
    {  0.4e6, waveform::ID::DFT_S_OFDM_128_715K },
    {  1.2e6, waveform::ID::DFT_S_OFDM_128_715K },
    {  2.0e6, waveform::ID::DFT_S_OFDM_128_715K },
    {  2.8e6, waveform::ID::DFT_S_OFDM_128_715K },
    {  3.6e6, waveform::ID::DFT_S_OFDM_128_715K }
  };
  const std::vector<CFreqWaveform> hardcoded_allocation_5MHz{
    { -1.656e6, waveform::ID::DFT_S_OFDM_128_288K },
    { -1.288e6, waveform::ID::DFT_S_OFDM_128_288K },
    { -0.920e6, waveform::ID::DFT_S_OFDM_128_288K },
    { -0.552e6, waveform::ID::DFT_S_OFDM_128_288K },
    { -0.184e6, waveform::ID::DFT_S_OFDM_128_288K },
    {  0.184e6, waveform::ID::DFT_S_OFDM_128_288K },
    {  0.552e6, waveform::ID::DFT_S_OFDM_128_288K },
    {  0.920e6, waveform::ID::DFT_S_OFDM_128_288K },
    {  1.288e6, waveform::ID::DFT_S_OFDM_128_288K },
    {  1.656e6, waveform::ID::DFT_S_OFDM_128_288K }
  };

  class FreqBand {
  public:
    double lf, rf;
    FreqBand(double _lf = 0, double _rf = 0) : lf(_lf), rf(_rf) {}
    FreqBand(const Channel& chan) : lf(chan.lower()), rf(chan.upper()) {}
    double width(void) const { return (rf - lf); }
    double offset(void) const { return (lf + rf) / 2; }
    bool operator<(const FreqBand& fb) const {
      return (lf < fb.lf || (lf == fb.lf && rf < fb.rf));
    }
    FreqBand& operator=(const FreqBand& fb) {
      lf = fb.lf;
      rf = fb.rf;
      return *this;
    }
  };
  class FreqBandList {
  public:
    std::vector<FreqBand> fb_vec;
    void insert(const FreqBand& ins);
    void erase(const FreqBand& era);
    void clear(void) { fb_vec.clear(); }
    int size(void) const { return fb_vec.size(); }
    FreqBand& operator[](int i) { return fb_vec[i]; }
    bool contain(const FreqBand& fb) const;
  };

private:
  int d_total_nodes, d_our_nodes;
  int d_num_bins;
  networkid_t d_our_network_id;
  std::map<FreqBand, std::vector<nodenum_t>> d_spectrum;
  std::map<nodeid_t, double> d_tx_power;
  // channel gain matrix
  std::vector<std::vector<double>> d_chan_M;
  // channel mask. mask on pair.first to pair.second
  std::map<std::pair<nodeid_t, nodeid_t>, double> d_chan_mask;
  double d_half_bw;
  double d_interference_threshold;
  std::vector<double> d_req_bw;  // TODO: remove this
  std::vector<FreqBand> d_current_fb;  //TODO: remove this
  std::vector<CFreqWaveform> d_current_allocation;
  std::chrono::high_resolution_clock::time_point d_alloc_tick;
  std::chrono::high_resolution_clock::time_point d_env_tick;
  std::map<networkid_t, MandatedPerformance> d_performance;
  std::map<nodenum_t, nodeid_t> d_node_ids;
  std::map<nodeid_t, FreqBandList> d_tx_channels;
  std::map<nodeid_t, controlchannel::Location> d_our_locmap;
  std::map<nodeid_t, controlchannel::Location> d_their_locmap;

  std::map<std::pair<nodeid_t, nodeid_t>, FlowRequirement> d_flow_specs;
  bool d_new_mandates;
  std::map<nodeid_t, std::vector<double>> d_node_values;
  std::map<nodeid_t, std::vector<double>> d_node_interf;

  void spectrum_insert(FreqBand ins, nodenum_t i);
  bool empty_spectrum(void);
  std::vector<double> chan_M_multiply(const std::vector<double>& src, bool apply_mask = false);
  std::vector<FreqBand> greedy_alloc(std::map<nodeid_t, FreqBandList> available_fb);
  std::map<nodeid_t, FreqBandList> get_available_fb(void);
  void update_node_values(const std::vector<int>& chan_bw);
  void update_node_values_psd(const std::vector<int>& chan_bw,
    const std::vector<std::vector<float>>& psd);
  
  enum class PerformanceState {
    JustReallocated,
    JustChanged,
    Better,
    Average,
    Worst
  };
  std::chrono::high_resolution_clock::time_point d_state_tick;
  PerformanceState d_perf_state;
  void update_perf_state(void);
  PerformanceState check_perf_state(void);
  std::vector<CFreqWaveform> reduce_bandwidth(const std::vector<waveform::ID>& suggest_wf);
  void mask_peer_channels(std::vector<int>& mask);
  
  // helpers
  template<typename Id, typename Val>
  std::map<nodeid_t, Val> map_convert(const std::map<Id, Val>& src) {
    std::map<nodeid_t, Val> dst;
    for (auto const& item : src)
      dst.emplace(nodeid_t{d_our_network_id, unsigned(item.first)}, item.second);
    return dst;
  }
  int freq_to_bin(const double f) const {
    return clip_range((int)((f / (2 * d_half_bw) + 0.5) * d_num_bins), 0, d_num_bins - 1);
  }
  int freq_to_bin(const double f, const int vec_len) const {
    return clip_range((int)((f / (2 * d_half_bw) + 0.5) * vec_len), 0, vec_len - 1);
  }
  int freq_to_bin(const double f, const double bw, const int vec_len) const {
    return clip_range((int)((f / bw + 0.5) * vec_len), 0, vec_len - 1);
  }
  int bw_to_bins(const double bw) const {
    return (int)std::ceil(bw / (2 * d_half_bw) * d_num_bins);
  }
  double bin_to_freq(const int b, bool odd) const {
    // when bw_bins is odd, cf is the center of the bin
    if (odd) {
      return clip_range(((double)b + 0.5) / d_num_bins * 2 * d_half_bw - d_half_bw, -d_half_bw, d_half_bw);
    }
    // when bw_bins is even, cf is the left of the bin
    else {
      return clip_range((double)b / d_num_bins * 2 * d_half_bw - d_half_bw, -d_half_bw, d_half_bw);
    }
  }

  std::vector<NotificationCenter::SubToken> d_nc_tokens;
  // io_service
  boost::asio::io_service& _ios;
  boost::asio::io_service::work* _ios_work;
  std::thread _work_thread;  // Single thread
  std::mutex d_mutex;

  controlchannel::CCData::sptr _ccData;

public:
  FreqAlloc(
      boost::asio::io_service& bbc_ios,
      controlchannel::CCData::sptr ccData,
      unsigned int _our_network_id, int _num_bins);
  ~FreqAlloc() {}

  void update_chan_mask(void);
  void update_location(const std::map<nodeid_t, controlchannel::Location>& loc_map);
  void update_tx_list(const std::map<nodeid_t, std::vector<Channel>>& channels_map);
  void update_tx_power(const std::map<nodeid_t, double>& tx_power);
  void update_mandated_outcomes(const nlohmann::json& j);
  void update_our_performance(void);
  void update_peer_performance(const std::tuple<networkid_t, int, int>& p);

  std::vector<Channel> allocate_freq(void);
  FreqAllocRet allocate_freq_sinr(const std::vector<waveform::ID>& wf, std::vector<Channel> const& ctrl_alloc,
      const std::vector<std::vector<float>>& psd = {});
  void print_debug_info(void);
};

struct ChannelsValues{
  std::vector<int> center_freqs;
  double total_value = 0;
};
ChannelsValues assign_channels(std::vector<int> mask_bands, std::vector<int> B, std::vector<std::vector<double>> V);

}
}

#endif
