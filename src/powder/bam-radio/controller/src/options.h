// -*- c++ -*-
//  Copyright © 2016-2017 Stephen Larew

#ifndef _INCLUDED_OPTIONS_H
#define _INCLUDED_OPTIONS_H

#include <boost/program_options.hpp>
#include <log4cpp/Category.hh>
#include <string>
#include <vector>

namespace bamradio {

namespace options {
/// Option variables_map
/// e.g., if (vm[center_freq].as<int>() > 1000.0) { ...
extern boost::program_options::variables_map vm;

// global configuration options
extern unsigned int verbose;
extern std::string uid;
extern bool batch_mode;
extern std::string log_json_path;
extern std::string log_sqlite_path;
extern bool log_append;

// network-related options
namespace net {
extern std::string tun_iface_prefix;
extern std::string tun_ip4;
extern std::string tun_ip4_netmask;
extern size_t tun_mtu;
extern bool mock_traffic;
extern float route_timeout;
} // namespace net

namespace dll {
extern float arq_resend_timeout; // FIXME remove when adaptive resend timeout exists
extern float burst_max_inter_segment_arrival_duration;
extern int new_burst_inter_segment_arrival_ratio;
extern int sched_filept_rel_prio;
extern float sched_alpha;
extern float sched_beta;
extern int sched_frame_overhead;
extern float sched_goodput;
extern float sched_arq_goodput;
} // namespace dll

/// Common PHY options
namespace phy {
extern std::string uhd_args;
extern std::string fftw_wisdom;
extern size_t max_noutput_items;
extern double center_freq;
extern double bandwidth;
extern double lo_offset_tx;
extern double lo_offset_rx;
extern size_t max_n_nodes;
// adaptive mcs threshold
extern float mcs_thresh;
extern float mcs_alpha;
/// Data PHY options
// TEMP means these options are only temporary
// and should disappear in the future
namespace data {
extern double freq_offset; // TEMP
extern double tx_gain;     // TEMP
extern double rx_gain;     // TEMP
extern double sample_rate; // TEMP
extern size_t rx_frame_queue_size;
extern float sync_detection_threshold; // TEMP
extern std::vector<float> debug_channel_alloc;
extern float guard_band;
extern std::string initial_waveform;
extern std::string header_mcs_name;
extern std::string initial_payload_mcs_name;
extern std::string initial_payload_symbol_seq_name;
extern size_t header_demod_nthreads;
extern size_t payload_demod_nthreads;
extern size_t tx_nthreads;
extern bool mcs_adaptation;
extern float mcs_hyst_err_ub;
extern float mcs_hyst_err_lb;
extern size_t channelizer_fft_multiplier;
extern size_t channelizer_nbatch;
extern size_t sample_buffer_size;
extern bool multihop;
extern float variance_filter_window;
extern float variance_hist_min;
extern float variance_hist_max;
extern size_t variance_hist_nbins;
} // namespace data

// control PHY related options
namespace control {
extern double freq_offset; // Deprecated
extern double band_edge_offset;
extern double tx_gain;
extern double rx_gain;
extern double sample_rate;
extern double bandwidth;
extern double atten;
extern unsigned int num_fsk_points;
extern unsigned int rs_k;
extern unsigned int min_soft_decs;
extern unsigned int id;
extern double t_slot;
extern double max_delay;
extern double convergence_time;
extern float ccsegment_interval;
extern bool frequency_hopping;
} // namespace control
} // namespace phy

/// Collaboration options
namespace collab {
extern bool gateway;
extern std::string collab_server_ip;
extern int server_id;
extern int netid;
extern long server_port;
extern long client_port;
extern long peer_port;
extern std::string log_filename;
} // namespace collab

/// PSD sensing options
namespace psdsensing {
// hist sensing
extern float bin_size;
extern float empty_bin_items;
extern int sn_gap_bins;
extern int hist_avg_len;
extern float noise_floor_db;

extern long fft_len;
extern unsigned int mov_avg_len;
extern long reset_period;
extern float snr_threshold;
extern int holes_select_mode;
extern int contain_count_th;
extern double interf_th;
extern double path_loss_exponent;
extern std::string min_wf_base;
extern std::string initial_wf_pay;
extern std::string max_wf;
extern std::string fixed_wf;
extern int num_relocate_chans;
extern float psd_cil_input;
} // namespace psdsensing

namespace c2api {
extern int port;
extern std::string status_path;
extern std::string env_recovery_path;
extern std::string mo_recovery_path;
}

/// Handle command line arguments and initializes options
void init(
    int argc, const char *argv[],
    boost::program_options::options_description const *extra_ops = nullptr);

} // namespace options
} // namespace bamradio

#endif
