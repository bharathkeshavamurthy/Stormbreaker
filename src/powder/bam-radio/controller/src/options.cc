// -*- c++ -*-
//  Copyright © 2016-2017 Stephen Larew

#include "options.h"
#include "events.h"
#include "bandwidth.h"
#include "c2api.h"
#include <cstdlib>
#include <fstream>
#include <gnuradio/prefs.h>
#include <iostream>
#include <uhd/utils/msg.hpp>
#include <boost/format.hpp>

namespace po = boost::program_options;

namespace bamradio {

using std::string;

namespace options {
namespace detail {
string config_file;
string write_config_template;
string colosseum_config_file;
}
/// Option variables_map
/// e.g., if (vm[center_freq].as<int>() > 1000.0) { ...
boost::program_options::variables_map vm;

// global configuration options
unsigned int verbose;
std::string uid;
bool batch_mode;
std::string log_json_path;
std::string log_sqlite_path;
bool log_append;

// network-related options
namespace net {
std::string tun_iface_prefix;
std::string tun_ip4;
std::string tun_ip4_netmask;
size_t tun_mtu;
bool mock_traffic;
float route_timeout;
}

namespace dll {
float arq_resend_timeout;
float burst_max_inter_segment_arrival_duration;
int new_burst_inter_segment_arrival_ratio;
int sched_filept_rel_prio;
float sched_beta, sched_alpha;
int sched_frame_overhead;
float sched_goodput, sched_arq_goodput;
} // namespace dll

namespace phy {
std::string uhd_args;
std::string fftw_wisdom;
size_t max_noutput_items;
double center_freq;
double bandwidth;
double lo_offset_tx;
double lo_offset_rx;
size_t max_n_nodes;
float mcs_thresh;
float mcs_alpha;
double snr_upper;
float winsize;

namespace data {
// added by diyu to set mcs threshold
double freq_offset;
double tx_gain;
double rx_gain;
double sample_rate;
size_t rx_frame_queue_size;
float sync_detection_threshold;
std::vector<float> debug_channel_alloc;
float guard_band;
std::string initial_waveform;
std::string header_mcs_name;
std::string initial_payload_mcs_name;
std::string initial_payload_symbol_seq_name;
size_t header_demod_nthreads;
size_t payload_demod_nthreads;
size_t tx_nthreads;
bool mcs_adaptation;
float mcs_hyst_err_ub;
float mcs_hyst_err_lb;
size_t channelizer_fft_multiplier;
size_t channelizer_nbatch;
size_t sample_buffer_size;
bool multihop;
float variance_filter_window;
float variance_hist_min;
float variance_hist_max;
size_t variance_hist_nbins;
} // namespace data

namespace control {
double freq_offset;
double tx_gain;
double rx_gain;
double sample_rate;
double bandwidth;
double atten;
unsigned int num_fsk_points;
unsigned int rs_k;
unsigned int min_soft_decs;
unsigned int id;
double t_slot;
double max_delay;
double convergence_time;
float ccsegment_interval;
bool frequency_hopping;
double band_edge_offset;
}
}

namespace collab {
bool gateway;
std::string collab_server_ip;
int server_id;
int netid;
long server_port;
long client_port;
long peer_port;
std::string log_filename;
}

namespace psdsensing {
long fft_len;
unsigned int mov_avg_len;
long reset_period;
float bin_size;
int sn_gap_bins;
float empty_bin_items;
int hist_avg_len;
float noise_floor_db;
float snr_threshold;
int holes_select_mode;
int contain_count_th;
double interf_th;
double path_loss_exponent;
string min_wf_base;
string initial_wf_pay;
string max_wf;
string fixed_wf;
int num_relocate_chans;
float psd_cil_input;
}

namespace c2api {
int port;
string status_path;
string env_recovery_path;
string mo_recovery_path;
}

}

static void my_uhd_msg_handler(uhd::msg::type_t type, const string &msg);
static void null_uhd_msg_handler(uhd::msg::type_t type, const string &msg);

void options::init(int argc, const char *argv[],
                   po::options_description const *extra_ops) {
  po::options_description op_phy("Common PHY options");
  op_phy.add_options()(
      // Option
      "phy.args,args", po::value<string>(&options::phy::uhd_args)
                           ->required()
                           ->value_name("key/value comma separated list"),
      "UHD device arguments")(
      // Option
      "phy.max-noutput-items",
      po::value<size_t>(&options::phy::max_noutput_items)
          ->value_name("count")
          ->required(),
      "GR top_block max_noutput_items")(
      // Option
      "phy.fftw_wisdom", po::value<string>(&options::phy::fftw_wisdom)
                             ->required()
                             ->value_name("filename"),
      "FFTW wisdom")(
      "phy.adaptive-mcs-alpha", po::value<float>(&options::phy::mcs_alpha)
                             ->required()
                             ->default_value(1.0)
                             ->value_name("threshold"),


      "FFTW wisdom")(
      // Option
      "phy.max_n_nodes",
      po::value<size_t>(&options::phy::max_n_nodes)
          ->required()
          ->default_value(10),
      "GR top_block max_noutput_items")(
      // Option
      "RF.center_frequency", po::value<double>(&options::phy::center_freq)
                                 ->required()
                                 ->value_name("Hz"),
      "RF center frequency")(
      // Option
      "RF.rf_bandwidth",
      po::value<double>(&options::phy::bandwidth)->required()->value_name("Hz"),
      "RF bandwidth")(
      // Option
      "RF.LO-offset-tx", po::value<double>(&options::phy::lo_offset_tx)
                             ->required()
                             ->value_name("Hz")
                             ->default_value(-42.0e6), // NI recommendation
      "LO offset (Tx)")(
      // Option
      "RF.LO-offset-rx", po::value<double>(&options::phy::lo_offset_rx)
                             ->required()
                             ->value_name("Hz")
                             ->default_value(42.0e6), // NI recommendation
      "LO offset (Rx)");

  po::options_description op_phy_data("Data PHY options");
  op_phy_data.add_options()(
      // Option
      "phy_data.freq-offset",
      po::value<double>(&options::phy::data::freq_offset)
          ->value_name("Hz")
          ->required(),
      "TX frequency offset")(
      // Option
      "phy_data.tx-gain",
      po::value<double>(&options::phy::data::tx_gain)
          ->value_name("dB")
          ->required()
          ->default_value(20), // NI recommendation @1GHz
      "TX gain")(
      // Option
      "phy_data.rx-gain",
      po::value<double>(&options::phy::data::rx_gain)
          ->value_name("dB")
          ->required()
          ->default_value(7), // NI recommendation @1GHz
      "RX gain")(
      // Option
      "phy_data.sample-rate",
      po::value<double>(&options::phy::data::sample_rate)
          ->value_name("Hz")
          ->required()
          ->default_value(bam::dsp::sample_rate), // FIXME remove this parameter.
      "sample rate")(
      // Option
      "phy_data.sync-threshold",
      po::value<float>(&options::phy::data::sync_detection_threshold)
          ->required(),
      "Sync metric threshold [0,1]")(
      // Option
      "phy_data.rx-frame-queue-size",
      po::value<size_t>(&options::phy::data::rx_frame_queue_size)
          ->value_name("count")
          ->required(),
      "Maximum size of RX frame queue")(
      // Option
      "phy_data.debug-channel-alloc",
      po::value<std::vector<float>>(&options::phy::data::debug_channel_alloc)
          ->multitoken(),
      "initial channel allocation for debug")(
      // Option
      "phy_data.initial-waveform",
      po::value<std::string>(&options::phy::data::initial_waveform)->required(),
      "Initial waveform")(
      // Option
      "phy_data.guard-band",
      po::value<float>(&options::phy::data::guard_band)
          ->required()
          ->default_value(80e3),
      "Guard band")(
      // Option
      "phy_data.header-mcs",
      po::value<decltype(phy::data::header_mcs_name)>(
          &options::phy::data::header_mcs_name)
          ->required()
          ->default_value("QPSK_R12_N648"),
      "name of header MCS")(
      // Option
      "phy_data.initial-payload-mcs",
      po::value<decltype(phy::data::initial_payload_mcs_name)>(
          &options::phy::data::initial_payload_mcs_name)
          ->required()
          ->default_value("QPSK_R12_N1944"),
      "name of initial payload MCS")(
      // Option
      "phy_data.initial-payload-symbol-seq",
      po::value<decltype(phy::data::initial_payload_symbol_seq_name)>(
          &options::phy::data::initial_payload_symbol_seq_name)
          ->required()
          ->default_value("ZIG_128_12_108_12_QPSK"),
      "name of initial payload symbol sequence")(
      // Option
      "phy_data.mcs_adaptation",
      po::value<bool>(
          &options::phy::data::mcs_adaptation)
          ->required()
          ->default_value(true),
      "enable MCS adaptation")(
      // Option
      "phy_data.mcs-hyst-err-ub",
      po::value<float>(&options::phy::data::mcs_hyst_err_ub)
          ->required()
          ->default_value(0.2),
      "MCS error rate hysteresis upper bound")(
      // Option
      "phy_data.mcs-hyst-err-lb",
      po::value<float>(&options::phy::data::mcs_hyst_err_lb)
          ->required()
          ->default_value(0.05),
      "MCS error rate hysteresis lower bound")(
      // Option mcs choice. false for old algorithm, true for new (Andrew's) algo
      "phy_data.variance-filter-window",
      po::value<decltype(options::phy::data::variance_filter_window)>(
          &options::phy::data::variance_filter_window)
          ->required()
          ->default_value(5.0),
      "filter length [seconds] for receive noise variance meadian filter")(
      // Option
      "phy_data.variance-hist-min",
      po::value<decltype(options::phy::data::variance_hist_min)>(
          &options::phy::data::variance_hist_min)
          ->value_name("dB")
          ->required()
          ->default_value(0.0),
      "minimum variance for noise variance histogram")(
      // Option
      "phy_data.variance-hist-max",
      po::value<decltype(options::phy::data::variance_hist_max)>(
          &options::phy::data::variance_hist_max)
          ->value_name("dB")
          ->required()
          ->default_value(40.0),
      "maximum variance for noise variance histogram")(
      // Option
      "phy_data.variance-hist-nbins",
      po::value<decltype(options::phy::data::variance_hist_nbins)>(
          &options::phy::data::variance_hist_nbins)
          ->required()
          ->default_value(80),
      "number of bins of noise variance histogram")(
      // Option
      "phy_data.multihop",
      po::value<bool>(
          &options::phy::data::multihop)
          ->required()
          ->default_value(true),
      "enable multihop routing")(
      // Option
      "phy_data.channelizer-fft-multiplier",
      po::value<decltype(options::phy::data::channelizer_fft_multiplier)>(
          &options::phy::data::channelizer_fft_multiplier)
          ->required()
          ->default_value(16),
      "FFT size multiplier for Rx OS fast convolution processing")(
      // Option
      "phy_data.channelizer-nbatch",
      po::value<decltype(options::phy::data::channelizer_nbatch)>(
          &options::phy::data::channelizer_nbatch)
          ->required()
          ->default_value(100),
      "number of batches for Rx OS fast convolution processing")(
      // Option
      "phy_data.sample-buffer-size",
      po::value<decltype(options::phy::data::sample_buffer_size)>(
          &options::phy::data::sample_buffer_size)
          ->required()
          ->default_value(1e7),
      "size of PHY complex sample buffers")(
      // Option
      "phy_data.header-demod-nthreads",
      po::value<decltype(phy::data::header_demod_nthreads)>(
          &options::phy::data::header_demod_nthreads)
          ->required()
          ->default_value(10),
      "number of header demodulation threads")(
      // Option
      "phy_data.payload-demod-nthreads",
      po::value<decltype(phy::data::payload_demod_nthreads)>(
          &options::phy::data::payload_demod_nthreads)
          ->required()
          ->default_value(20),
      "number of payload demodulation threads")(
      // Option
      "phy_data.tx-nthreads",
      po::value<decltype(phy::data::tx_nthreads)>(
          &options::phy::data::tx_nthreads)
          ->required()
          ->default_value(2),
      "number of transmitter threads");

  po::options_description op_phy_control("Control PHY options");
  op_phy_control.add_options()(
      // Option
      "phy_control.freq-offset",
      po::value<double>(&options::phy::control::freq_offset)  // DEPRECATED
          ->value_name("Hz")
          ->required()
          ->default_value(0),
      "TX frequency offset")(
      // Option
      "phy_control.band-edge-offset",
      po::value<decltype(options::phy::control::band_edge_offset)>(&options::phy::control::band_edge_offset)
          ->value_name("Hz")
          ->required()
          ->default_value(380e3),
      "Offset from band edges")(
      // Option
      "phy_control.tx-gain", po::value<double>(&options::phy::control::tx_gain)
                                 ->value_name("dB")
                                 ->required()
                                 ->default_value(20),
      "TX gain")(
      // Option
      "phy_control.rx-gain", po::value<double>(&options::phy::control::rx_gain)
                                 ->value_name("dB")
                                 ->required()
                                 ->default_value(7),
      "RX gain")(
      // Option
      "phy_control.sample-rate",
      po::value<double>(&options::phy::control::sample_rate)
          ->value_name("Hz")
          ->required()
          ->default_value(480e3),
      "sample rate")(
      // Option
      "phy_control.bandwidth",
      po::value<double>(&options::phy::control::bandwidth)
          ->value_name("Hz")
          ->required()
          ->default_value(480e3),
      "bandwidth")(
      // Option
      "phy_control.num_fsk_points",
      po::value<unsigned int>(&options::phy::control::num_fsk_points)
          ->required(),
      "Number of FSK points")(
      // Option
      "phy_control.rs_k", po::value<unsigned int>(&options::phy::control::rs_k)
                              ->value_name("bytes")
                              ->required(),
      "RS code block size")(
      // Option
      "phy_control.min_soft_decs",
      po::value<unsigned int>(&options::phy::control::min_soft_decs)
          ->value_name("val")
          ->required(),
      "min_soft_decs")(
      // Option
      "phy_control.frequency_hopping",
      po::value<bool>(
          &options::phy::control::frequency_hopping)
          ->required()
          ->default_value(true),
      "enable frequency hopping")(
      // Option
      "phy_control.id",
      po::value<unsigned int>(&options::phy::control::id)->required(),
      "Node ID (1, 2, 3, ...)")(
      // Option
      "phy_control.t_slot", po::value<double>(&options::phy::control::t_slot)
                                ->value_name("seconds")
                                ->required(),
      "Tx slot length")(
      // Option
      "phy_control.atten",
      po::value<double>(&options::phy::control::atten)
          ->value_name("val")
          ->required()
          ->notifier([](const double &v) {
            if (v <= 0.0) {
              throw po::validation_error(
                  po::validation_error::invalid_option_value, "atten", "atten");
            }
          }),
      "attenuation scalar for complex baseband samples (not in dB)")(
      // Option
      "phy_control.max_delay",
      po::value<double>(&options::phy::control::max_delay)
          ->value_name("seconds")
          ->required()
          ->default_value(0.3),
      "Maximum allowed latency in control channel Tx flowgraph")(
      // Option
      "phy_control.ccsegment_interval",
      po::value<float>(&options::phy::control::ccsegment_interval)
          ->value_name("seconds")
          ->required()
          ->default_value(0.1),
      "Duration of bootstrap process")(
      // Option
      "phy_control.convergence_time",
      po::value<double>(&options::phy::control::convergence_time)
          ->value_name("seconds")
          ->required()
          ->default_value(15),
      "Duration of bootstrap process");

  po::options_description op_cl("Command line options");
  op_cl.add_options()(
      // Option
      "help,h", "print help message")(
      // Option
      "version", "print version information")(
      // Option
      "config",
      po::value<string>(&options::detail::config_file)->value_name("filename"),
      "configuration file")(
      // Option
      "colosseum_config",
      po::value<string>(&options::detail::colosseum_config_file)
          ->value_name("filename"),
      "colosseum_config.ini file")(
      // Option
      "write-config-template",
      po::value<string>(&options::detail::write_config_template)
          ->value_name("filename")
          ->implicit_value("stdout"),
      "write a template config file to stdout");

  po::options_description op_basic("Basic options");
  op_basic.add_options()(
      // Option
      "global.verbose,verbose,v",
      po::value<unsigned int>(&options::verbose)
          ->value_name("level")
          ->implicit_value(1)
          ->default_value(1)
          ->notifier([](unsigned int level) {
            if (level != 0) {
              // Redirect UHD logging to our handler.
              uhd::msg::register_handler(&my_uhd_msg_handler);
            } else {
              // we should prob just get rid of this option
              uhd::msg::register_handler(&my_uhd_msg_handler);
            }
          }),
      "enable verbose output (level=0 to quiet stdout)")(
      // Option
      "global.grprefs,grprefs",
      po::value<string>()
          ->value_name("filename")
          ->notifier([](const string &filename) {
            gr::prefs::singleton()->add_config_file(filename);
          }),
      "additional GR preferences")(
      // Option
      "global.batch,batch",
      po::bool_switch(&options::batch_mode)->default_value(false),
      "true if in batch mode")(
      // Option
      "global.uid", po::value<string>(&options::uid)->required(),
      "user name to drop privileges")(
      // Option
      "global.log_json_path",
      po::value<string>(&options::log_json_path)->value_name("filename"),
      "statistics JSON log file name")(
      // Option
      "global.log_sqlite_path",
      po::value<string>(&options::log_sqlite_path)->value_name("filename"),
      "statistics SQLite log file name")(
      // Option
      "global.log_append",
      po::bool_switch(&options::log_append)->default_value(true));

  po::options_description op_net("Network options");
  op_net.add_options()(
      // Option
      "net.tun-iface-prefix", po::value<string>(&options::net::tun_iface_prefix)
                                  ->required()
                                  ->default_value("tun"),
      "prefix for name of TUN interface")(
      // Option
      "net.tun-ip4", po::value<string>(&options::net::tun_ip4)->required(),
      "IPv4 address on TUN interface")(
      // Option
      "net.tun-ip4-netmask", po::value<string>(&options::net::tun_ip4_netmask)
                                 ->required()
                                 ->default_value("255.255.255.0"),
      "netmask of IPv4 address on TUN interface")(
      // Option
      "net.tun-mtu", po::value<size_t>(&options::net::tun_mtu)
                         ->required()
                         ->default_value(1500),
      "MTU of TUN interface")(
      // Option
      "net.route-timeout", po::value<float>(&options::net::route_timeout)
                         ->required()
                         ->default_value(20),
      "Routing timeout")(
      // Option
      "net.mock-traffic", po::bool_switch(&options::net::mock_traffic)
                              ->required()
                              ->default_value(false),
      "send mock IP traffic ");

  po::options_description op_dll("DLL options");
  op_dll.add_options()(
      // Option
      "dll.arq-resend-timeout",
      po::value<float>(&options::dll::arq_resend_timeout)
          ->required()
          ->value_name("seconds")
          ->default_value(0.350f),
      "duration to wait before resending segments")(
      // Option
      "dll.burst-max-inter-segment-arrival-duration",
      po::value<float>(&options::dll::burst_max_inter_segment_arrival_duration)
          ->required()
          ->value_name("seconds")
          ->default_value(0.5),
      "maximum duration between arrivals of segments from same burst")(
      // Option
      "dll.new-burst-inter-segment-arrival-ratio",
      po::value<int>(&options::dll::new_burst_inter_segment_arrival_ratio)
          ->required()
          ->default_value(750),
      "ratio of current to previous arrival intervals for new bursts")(
      // Option
      "dll.sched-filept-rel-prio",
      po::value<int>(&options::dll::sched_filept_rel_prio)
          ->required()
          ->value_name("{-1(demote files),0,1(promote files)}")
          ->default_value(0)
          ->notifier([](const int &v) {
            if (v < -1 || v > 1) {
              throw po::validation_error(
                  po::validation_error::invalid_option_value,
                  "sched_filept_rel_prio", "sched_filept_rel_prio");
            }
          }),
      "relative priority of files to streams when scheduling");

  po::options_description op_collab("Collaboration options");
  op_collab.add_options()(
      // Option
      "collaboration.gateway", po::bool_switch(&options::collab::gateway)
                                   ->required()
                                   ->default_value(false),
      "true if the node is a gateway")(
      // Option
      "COLLABORATION.collab_server_ip",
      po::value<string>(&options::collab::collab_server_ip)
          ->required()
          ->default_value("127.0.0.1"),
      "IP address of collaboration server")(
      // Option
      "collaboration.server_id",
      po::value<int>(&options::collab::server_id)->required()->default_value(2),
      "Collaboration server id")(
      // Option
      "collaboration.netid",
      po::value<int>(&options::collab::netid)->required()->default_value(0),
      "Collaboration subnet")(
      // Option
      "COLLABORATION.collab_server_port",
      po::value<long>(&options::collab::server_port)
          ->required()
          ->default_value(0),
      "Collaboration server port")(
      // Option
      "COLLABORATION.collab_client_port",
      po::value<long>(&options::collab::client_port)
          ->required()
          ->default_value(0),
      "Collaboration client port")(
      // Option
      "COLLABORATION.collab_peer_port",
      po::value<long>(&options::collab::peer_port)
          ->required()
          ->default_value(0),
      "Collaboration peer port")(
      // Option
      "collaboration.log_filename",
      po::value<string>(&options::collab::log_filename)
          ->required()
          ->default_value("collab.log"),
      "Collaboration client log file name");

  po::options_description op_psd_sensing("PSD Sensing options");
  op_psd_sensing.add_options()(
      // Option
      "psd_sensing.fft_len",
      po::value<long>(&options::psdsensing::fft_len)->required(), "FFT length")(
      // Option
      "psd_sensing.mov_avg_len",
      po::value<unsigned int>(&options::psdsensing::mov_avg_len)->required(),
      "Moving average length")(
      // Option
      "psd_sensing.reset_period",
      po::value<long>(&options::psdsensing::reset_period)->required(),
      "Moving average reset period")(
      // Option
      "psd_sensing.bin_size",
      po::value<decltype(options::psdsensing::bin_size)>(&options::psdsensing::bin_size)->required(),
      "Histogram bin size")(
      // Option
      "psd_sensing.sn_gap_bins",
      po::value<decltype(options::psdsensing::sn_gap_bins)>(&options::psdsensing::sn_gap_bins)->required(),
      "Histogram signal-noise gap bins")(
      // Option
      "psd_sensing.empty_bin_items",
      po::value<decltype(options::psdsensing::empty_bin_items)>(&options::psdsensing::empty_bin_items)
          ->required(),
      "Histogram empty bin items")(
      // Option
      "psd_sensing.hist_avg_len",
      po::value<decltype(options::psdsensing::hist_avg_len)>(&options::psdsensing::hist_avg_len)->required(),
      "Histogram averaging length")(
      // Option
      "psd_sensing.noise_floor_db",
      po::value<decltype(options::psdsensing::noise_floor_db)>(&options::psdsensing::noise_floor_db)
          ->required()
          ->value_name("dB"),
      "Noise floor")(
      // Option
      "psd_sensing.snr_threshold",
      po::value<float>(&options::psdsensing::snr_threshold)
          ->required()
          ->value_name("dB"),
      "SNR threshold")(
      // Option
      "psd_sensing.holes_select_mode",
      po::value<int>(&options::psdsensing::holes_select_mode)->required(),
      "Freq Holes Selection Mode")(
      // Option
      "psd_sensing.contain_count_th",
      po::value<int>(&options::psdsensing::contain_count_th)->required(),
      "FreqAlloc contain_count threshold")(
      // Option
      "psd_sensing.interf_th",
      po::value<double>(&options::psdsensing::interf_th)->required(),
      "FreqAlloc interference threshold")(
      // Option
      "psd_sensing.path_loss_exponent",
      po::value<double>(&options::psdsensing::path_loss_exponent)->required()
          ->default_value(3.0),
          "Path loss exponent")(
      // Option
      "psd_sensing.min_wf_base",
      po::value<string>(&options::psdsensing::min_wf_base)->required()
          ->default_value("DFT_S_OFDM_128_500K"),
          "Min waveform in Baseline")(
      // Option
      "psd_sensing.initial_wf_pay",
      po::value<string>(&options::psdsensing::initial_wf_pay)->required()
          ->default_value("DFT_S_OFDM_128_288K"),
          "Initial waveform in Payline")(
      // Option
      "psd_sensing.max_wf",
      po::value<string>(&options::psdsensing::max_wf)->required()
          ->default_value("DFT_S_OFDM_128_2M"),
          "Max waveform from DE")(
      // Option
      "psd_sensing.fixed_wf",
      po::value<string>(&options::psdsensing::fixed_wf)->required()
          ->default_value(""),
          "Fixed waveform for everything")(
      // Option
      "psd_sensing.num_relocate_chans",
      po::value<int>(&options::psdsensing::num_relocate_chans)->required()
          ->default_value(10),
          "Num chans to be relocated each time")(
      // Option
      // 0.0 means relying on CIL, 1.0 means relying on PSD
      "psd_sensing.psd_cil_input",
      po::value<float>(&options::psdsensing::psd_cil_input)->required()
          ->default_value(1.0),
          "Preference of using PSD or CIL as input to interference estimation");

  po::options_description op_c2api("C2API options");
  op_c2api.add_options()(
      // Option
      "c2api.port",
      po::value<decltype(options::c2api::port)>(&options::c2api::port)
          ->required()
          ->default_value(9999),
      "C2API in socket port")(
      // Option
      "c2api.status-path",
      po::value<decltype(options::c2api::status_path)>(
          &options::c2api::status_path)
          ->required()
          ->default_value("/root/radio_api/status.txt"),
      "C2API status path")(
      // Option
      "c2api.env-recovery-path",
      po::value<decltype(options::c2api::env_recovery_path)>(
          &options::c2api::env_recovery_path)
          ->required()
          ->default_value("/root/radio_api/environment.json"),
      "C2API environment update path")(
      // Option
      "c2api.mo-recovery-path",
      po::value<decltype(options::c2api::mo_recovery_path)>(
          &options::c2api::mo_recovery_path)
          ->required()
          ->default_value("/root/radio_api/mandated_outcomes.json"),
      "C2API mandated_outcomes update path");

  po::options_description op_sched("Scheduler options");
  op_c2api.add_options()(
      // Option
      "sched.alpha",
      po::value<float>(&options::dll::sched_alpha)
          ->required()
          ->default_value(0.75),
      "Scheduler alpha")(
      // Option
      "sched.beta",
      po::value<float>(&options::dll::sched_beta)
          ->required()
          ->default_value(0.96),
      "Scheduler beta")(
      // Option
      "sched.frame-overhead",
      po::value<int>(&options::dll::sched_frame_overhead)
          ->required()
          ->default_value(5),
      "Num OFDM symbols per frame")(
      // Option
      "sched.goodput",
      po::value<float>(&options::dll::sched_goodput)
          ->required()
          ->default_value(0.98f),
      "Scheduler goodput scalar")(
      // Option
      "sched.arq-goodput",
      po::value<float>(&options::dll::sched_arq_goodput)
          ->required()
          ->default_value(0.8f),
      "Scheduler arq goodput scalar");

  po::options_description op_final_cl(80, 40);
  op_final_cl.add(op_phy_data)
      .add(op_phy_control)
      .add(op_phy)
      .add(op_basic)
      .add(op_net)
      .add(op_dll)
      .add(op_collab)
      .add(op_psd_sensing)
      .add(op_c2api)
      .add(op_sched)
      .add(op_cl);
  if (extra_ops != nullptr) {
      op_final_cl.add(*extra_ops);
  }

  po::options_description op_final_config;
  op_final_config.add(op_phy_data)
      .add(op_phy_control)
      .add(op_phy)
      .add(op_basic)
      .add(op_net)
      .add(op_dll)
      .add(op_collab)
      .add(op_psd_sensing)
      .add(op_sched)
      .add(op_c2api);
  po::options_description op_final_colosseum_config;
  op_final_colosseum_config.add(op_phy).add(op_collab);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(op_final_cl)
                  .style(po::command_line_style::unix_style)
                  .run(),
              vm);

    // colosseum_config.ini
    if (vm.count("colosseum_config") > 0) {
      // Try to open the config file and parse it.
      std::ifstream conf(vm["colosseum_config"].as<string>());
      if (conf.is_open()) {
        // Config file exists and is opened for reading. Parse it.
        po::store(po::parse_config_file(conf, op_final_colosseum_config), vm);
        conf.close();
      } else {
        std::cout << "Failed to open colloseum config file "
                  << vm["colosseum_config"].as<string>() << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    if (vm.count("config") > 0) {
      // Try to open the config file and parse it.
      std::ifstream conf(vm["config"].as<string>());
      if (conf.is_open()) {
        // Config file exists and is opened for reading. Parse it.
        po::store(po::parse_config_file(conf, op_final_config), vm);
        conf.close();
      } else {
        std::cout << "Failed to open config file " << vm["config"].as<string>()
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (vm.count("version")) {
    std::cout << "💩" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  if (vm.count("help")) {
    std::cout << op_final_cl << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  if (vm.count("write-config-template") > 0) {
    auto filename = vm["write-config-template"].as<string>();

    std::ostream *stream = &std::cout;

    std::unique_ptr<std::ofstream> ofs;

    if (filename != "stdout") {
      ofs = std::make_unique<std::ofstream>(filename, std::ios_base::out |
                                                          std::ios_base::trunc);
      if (ofs->is_open()) {
        stream = ofs.get();
      }
    }

    for (const auto &op : op_final_config.options()) {
      const auto &ln = op->long_name();
      *stream << ln << "=";
      boost::any v;
      if (op->semantic()->apply_default(v)) {
        if (v.type() == typeid(string)) {
          *stream << boost::any_cast<string>(v);
        } else if (v.type() == typeid(size_t)) {
          *stream << boost::any_cast<size_t>(v);
        } else if (v.type() == typeid(float)) {
          *stream << boost::any_cast<float>(v);
        } else if (v.type() == typeid(double)) {
          *stream << boost::any_cast<double>(v);
        } else if (v.type() == typeid(bool)) {
          *stream << boost::any_cast<bool>(v);
        } else if (v.type() == typeid(unsigned int)) {
          *stream << boost::any_cast<unsigned int>(v);
        } else if (v.type() == typeid(int)) {
          *stream << boost::any_cast<int>(v);
        } else if (v.type() == typeid(unsigned short)) {
          *stream << boost::any_cast<unsigned short>(v);
        } else if (v.type() == typeid(short)) {
          *stream << boost::any_cast<short>(v);
        } else {
          *stream << " #default exists";
        }
      }
      *stream << std::endl;
    }
    std::exit(EXIT_SUCCESS);
  }

  try {
    po::notify(vm);
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // FIXME: Get rid of these option parameters!
  options::phy::data::sample_rate = bam::dsp::sample_rate;
  options::phy::control::sample_rate = bam::dsp::control_sample_rate;
  options::phy::control::bandwidth = bam::dsp::control_sample_rate;
}

static void my_uhd_msg_handler(uhd::msg::type_t type, const string &msg) {
  using namespace uhdfeedback;
  NotificationCenter::shared.post(UHDMsgEvent, UHDMsgEventInfo{type, msg});
}

static void null_uhd_msg_handler(uhd::msg::type_t, const string &) {
  // be quiet
  ;
}

} // namespace bamradio
