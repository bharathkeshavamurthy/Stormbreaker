// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#include "radiocontroller.h"
#include "c2api.h"
#include "common.h"
#include "events.h"
#include "options.h"
#include "util.h"

#include <iostream>
#include <vector>

namespace de = bamradio::decisionengine;

namespace bamradio {

std::string time_spec_to_string(::uhd::time_spec_t t) {
  return (boost::format("{%d %.12f}") % t.get_full_secs() % t.get_frac_secs())
      .str();
}

usrp_monitor::usrp_monitor(std::string const &name,
                           ::uhd::usrp::multi_usrp::sptr dev)
    : _running(true), _name(name) {
  auto const usrp_dev = dev->get_device();
  _thread = std::thread([=] {
    bamradio::set_thread_name("usrp_monitor");
    while (_running) {
      ::uhd::async_metadata_t md;
      while (!usrp_dev->recv_async_msg(md))
        if (!_running)
          break;
      int64_t const time =
          md.has_time_spec
              ? uhd_time_spec_to_duration<std::chrono::nanoseconds>(
                    md.time_spec)
                    .count()
              : 0;
      NotificationCenter::shared.post(
          uhdfeedback::UHDAsyncEvent,
          uhdfeedback::UHDAsyncEventInfo{md.channel, time, md.event_code});
    }
  });
}

usrp_monitor::~usrp_monitor() {
  _running = false;
  _thread.join();
}

RadioController::RadioController(boost::asio::io_service &net_ios)
    : _srnid(options::phy::control::id),
      // io_service
      _ios_work(new boost::asio::io_service::work(_ios)), _work_thread([this] {
        bamradio::set_thread_name("radiocontroller_work");
        _ios.run();
      }),
      _n_rx_chains(options::phy::max_n_nodes - 1),
      _buf_sz(options::phy::data::sample_buffer_size),
      _channelizer_ios_work(boost::asio::make_work_guard(_channelizer_ios)),
      _channelizer_thread([this] {
        bamradio::set_thread_name("channelizer_synch");
        _channelizer_ios.run();
      }),
      _channelizer(_n_rx_chains,
                   options::phy::data::channelizer_fft_multiplier *
                       (bam::dsp::SubChannel::table()[0].taps.size() - 1),
                   options::phy::data::channelizer_nbatch,
                   bam::dsp::SubChannel::table()),
      _timer_chan_update(_ios),
      // max_in_samps needs to be the same as channelizer, see FIXME above
      _synch(
          [] {
            using namespace bamradio::ofdm;
            auto hparams =
                DFTSOFDMFrame(
                    0, 0, {},
                    MCS::stringNameToIndex(options::phy::data::header_mcs_name),
                    MCS::stringNameToIndex(
                        options::phy::data::initial_payload_mcs_name),
                    SeqID::stringNameToIndex(
                        options::phy::data::initial_payload_symbol_seq_name),
                    0, 0)
                    .headerParams(false, 20);
            auto sym_len = hparams.symbols[0].second->symbol_length;
            return static_cast<size_t>(sym_len / 2);
          }(),
          _n_rx_chains, _channelizer.getOutStride(),
          _channelizer.getStreams()) {
  log::text("RadioController [Making USRP devs]...");
  // Create radio sources and sinks for each mboard slot
  // slot A: OFDM
  // slot B: FSK
  //
  // FIXME: For now, we pull the UHD device out of the GNU radio block. If any1
  // wants to clean this up that's fine.
  auto stream_args = uhd::stream_args_t("fc32", "sc16");
  stream_args.args = uhd::device_addr_t("underflow_policy=next_burst");

  auto dev_args = boost::format("%1%,master_clock_rate=%2%") %
                  options::phy::uhd_args % bam::dsp::master_clock_rate;
  for (size_t cn : {0, 1}) {
    stream_args.channels = {cn};
    _usrpSrc[cn] = gr::uhd::usrp_source::make(
        uhd::device_addr_t(dev_args.str()), stream_args);
    _usrpSnk[cn] = gr::uhd::usrp_sink::make(uhd::device_addr_t(dev_args.str()),
                                            stream_args);
  }
  log::text("RadioController [Done].");

  // tune the radios
  log::text("RadioController [Tuning USRPs]...");
  auto const init_env = c2api::env.current();
  _center_frequency = init_env.scenario_center_frequency;
  auto rf_bandwidth = (double)init_env.scenario_rf_bandwidth;

  // slot A
  _usrpSnk[0]->set_gain(options::phy::data::tx_gain);
  _usrpSnk[0]->set_samp_rate(options::phy::data::sample_rate);
  ::uhd::tune_request_t tr0_tx(_center_frequency, options::phy::lo_offset_tx);
  tr0_tx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSnk[0]->set_center_freq(tr0_tx);
  _usrpSrc[0]->set_gain(options::phy::data::rx_gain);
  _usrpSrc[0]->set_samp_rate(options::phy::data::sample_rate);
  ::uhd::tune_request_t tr0_rx(_center_frequency, options::phy::lo_offset_rx);
  tr0_rx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSrc[0]->set_center_freq(tr0_rx);

  // slot B
  auto ctrl_offset = rf_bandwidth / 2 - options::phy::control::band_edge_offset;
  _usrpSnk[1]->set_gain(options::phy::control::tx_gain);
  _usrpSnk[1]->set_samp_rate(bam::dsp::sample_rate);
  ::uhd::tune_request_t tr1_tx(_center_frequency, options::phy::lo_offset_tx);
  tr1_tx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSnk[1]->set_center_freq(tr1_tx);
  _usrpSrc[1]->set_gain(options::phy::control::rx_gain);
  _usrpSrc[1]->set_samp_rate(options::phy::control::sample_rate);
  ::uhd::tune_request_t tr1_rx(_center_frequency + ctrl_offset,
                               options::phy::lo_offset_rx);
  tr1_rx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSrc[1]->set_center_freq(tr1_rx);

  // grab NTP time and set on radio
  _usrpSnk[0]->set_time_now(system_clock_now());
  //_usrpSnk[1]->set_time_now(system_clock_now());
  log::text("RadioController [Done].");

  // create the tx streamer (OFDM)
  log::text("RadioController [Creating Streamers]...");
  stream_args.channels = {0};
  _txStreamer = _usrpSnk[0]->get_device()->get_tx_stream(stream_args);
  uhd::tx_metadata_t beg;
  beg.start_of_burst = true;
  beg.end_of_burst = false;
  beg.has_time_spec = false;
  _txStreamer->send("", 0, beg);
  beg.start_of_burst = false;
  beg.end_of_burst = true;
  _txStreamer->send("", 0, beg);

  // create rx streamer (OFDM)
  _rxStreamer = _usrpSrc[0]->get_device()->get_rx_stream(stream_args);

  // create the tx streamer (FSK)
  stream_args.channels = {1};
  _txStreamer_fsk = _usrpSnk[1]->get_device()->get_tx_stream(stream_args);
  uhd::tx_metadata_t beg_fsk;
  beg_fsk.start_of_burst = true;
  beg_fsk.end_of_burst = false;
  beg_fsk.has_time_spec = false;
  _txStreamer_fsk->send("", 0, beg_fsk);
  beg_fsk.start_of_burst = false;
  beg_fsk.end_of_burst = true;
  _txStreamer_fsk->send("", 0, beg_fsk);
  log::text("RadioController [Done].");

  // usrp monitors
  log::text("RadioController [Starting Monitor]....");
  _monitor = std::make_shared<usrp_monitor>("mon", _usrpSrc[0]->get_device());
  log::text("RadioController [Done].");

  //
  // Create the OFDM link
  //
  log::text("RadioController [Making top_block]...");
  _tb = gr::make_top_block("BAM! Radio");
  log::text("RadioController [Done].");

  log::text("RadioController [Setting intial offsets]...");
  auto initial_waveform_id = bam::dsp::SubChannel::stringNameToIndex(
      options::phy::data::initial_waveform);
  auto initial_waveform = bam::dsp::SubChannel::table()[initial_waveform_id];

  assert(options::phy::max_n_nodes * initial_waveform.bw() <=
         options::phy::data::sample_rate);

  // set channel allocation to demuxer
  std::vector<float> initial_offsets(_n_rx_chains, 0.0f);
  _channelizer_ios.post([
    initial_offsets, initial_id = static_cast<size_t>(initial_waveform_id), this
  ] {
    for (size_t i = 0; i < initial_offsets.size(); ++i) {
      auto const offset_normalized = initial_offsets[i] / bam::dsp::sample_rate;
      _channelizer.setOffsetFreq(i, offset_normalized);
      _channelizer.setBandwidth(i, initial_id);
    }
  });

  // Create buffer: channelizer -> ringbuf_src
  // Make ring buffers for channelizer outputs
  for (size_t i = 0; i < _n_rx_chains; ++i) {
    _chanout.push_back(ofdm::ChannelOutputBuffer::make(_buf_sz));
  }
  log::text("RadioController [Done].");

  // create the dll and connect appropriately
  log::text("RadioController [Creating DLL]...");
  _ofdmDll = std::make_shared<ofdm::DataLinkLayer>(
      "ofdm0", net_ios, ofdm::phy_tx::make(options::phy::data::tx_nthreads),
      options::phy::data::rx_frame_queue_size, options::net::tun_mtu,
      options::phy::control::id);
  _ofdmDll->tx()->connect(_txStreamer);
  // connect the channel output ring buffers to PHY receiver
  _ofdmDll->rx()->connect(_chanout, [&] {
    std::vector<int> os(_n_rx_chains);
    for (size_t i = 0; i < os.size(); ++i)
      os[i] = initial_waveform.os;
    return os;
  }());
  log::text("RadioController [Done].");

  //
  // Create the FSK control channel link
  //

  // CC data
  log::text("RadioController [Creating control channel]...");
  _ccData = std::make_shared<controlchannel::CCData>(
      options::phy::control::id, options::phy::control::ccsegment_interval,
      options::collab::gateway);
  // control channel controller
  _cccontroller = controlchannel::CCController::make(
      _ccData, _ofdmDll,
      [this](auto const &samples, auto nsamp, auto time_spec) -> bool {
        // TODO: check whether the time spec is realizable
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst = false;
        md.has_time_spec = true;
        md.time_spec = time_spec;
        auto const mspp = _txStreamer_fsk->get_max_num_samps();
        auto nsent = 0;
        while (nsent < nsamp) {
          nsent += _txStreamer_fsk->send(
              samples.data() + nsent, std::min(mspp, nsamp - nsent), md, 1.0);
          md.start_of_burst = false;
          md.has_time_spec = false;
        }
        md.end_of_burst = true;
        _txStreamer_fsk->send("", 0, md, 1.0);
        // FIXME
        return true;
      });
  // connect receive chain to GNU Radio top_block
  _tb->connect(_usrpSrc[1], 0, _cccontroller->rx(), 0);
  log::text("RadioController [Done].");

  //
  // PSD Sensing
  //
  log::text("RadioController [Creating PSD]...");
  _psd = std::make_shared<psdsensing::PSDSensing>(options::phy::control::id,
                                                  _ccData, _ofdmDll);
  log::text("RadioController [Done].");

  // subscribe to channel allocation updates
  log::text("RadioController [Subscribing to notifications]...");
  _subTokens.push_back(
      NotificationCenter::shared
          .subscribe<controlchannel::CCData::OFDMChannelUpdateInfo>(
              controlchannel::CCData::OFDMChannelBandNotification, _ios,
              [this](auto v) {
                // Warn if the channel update is "late"
                if (v.t_effective < std::chrono::system_clock::now()) {
                  log::text("RadioController: OFDMChannelUpdate Late");
                }
                // schedule channel update
                _timer_chan_update.expires_at(v.t_effective);
                _timer_chan_update.async_wait([this, v](auto &e) {
                  // check whether this update was made before the current
                  // environment's timestamp. if yes, ignore.
                  using namespace std::chrono; // FIXME save timepoint in c2api
                  auto const env_timestamp = system_clock::time_point(
                      system_clock::duration(c2api::env.current().timestamp));
                  // FYI
                  log::text((boost::format(
                                 "RadioController: OFDMChannelUpdate update: "
                                 "t_last: %1% env_timestamp: %2%") %
                             v.t_last_update.time_since_epoch().count() %
                             env_timestamp.time_since_epoch().count())
                                .str());
                  this->_changeChannels(v.channels);
                });
              }));

  // MCS Adaptation
  if (options::phy::data::mcs_adaptation) {
    _subTokens.push_back(NotificationCenter::shared.subscribe<ofdm::MCSRequest>(
        ofdm::AdaptiveMCSController::MCSRequestNotification,
        _adap_mcs.io_service(), [this](auto mcs_request) {
          if (mcs_request.src_srnid == options::phy::control::id) {
            _ofdmDll->setMCSOFDM(mcs_request.dst_srnid, mcs_request.mcs,
                                 mcs_request.seqid);
          }
        }));
  }

  // subscribe to re-tuning notification / environment updates
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<EnvironmentUpdateEventInfo>(
          EnvironmentUpdateEvent, _ios, [this](auto env) {
            if (env.scenario_center_frequency != _center_frequency) {
              // need to re-tune radios
              _center_frequency = env.scenario_center_frequency;

              ::uhd::tune_request_t tr0_tx(_center_frequency,
                                           options::phy::lo_offset_tx);
              tr0_tx.args = uhd::device_addr_t("mode_n=integer");
              _usrpSnk[0]->set_center_freq(tr0_tx);

              ::uhd::tune_request_t tr0_rx(_center_frequency,
                                           options::phy::lo_offset_rx);
              tr0_rx.args = uhd::device_addr_t("mode_n=integer");
              _usrpSrc[0]->set_center_freq(tr0_rx);

              ::uhd::tune_request_t tr1_tx(_center_frequency,
                                           options::phy::lo_offset_tx);
              tr1_tx.args = uhd::device_addr_t("mode_n=integer");
              _usrpSnk[1]->set_center_freq(tr1_tx);
            }
            // in addition, we always want to update the control channel center
            // frequencies.
            if (_cccontroller) {
              _cccontroller->updateRFBandwidth(env.scenario_rf_bandwidth);
              _cccontroller->forceRetune();
            }
            // if the RF bandwidth changes, we shut the transmitter down until
            // we get another assignment.
            auto const prev_env = c2api::env.previous();
            if (!prev_env) {
              _ofdmDll->setTxChannel(boost::none);
            } else {
              if (env.scenario_rf_bandwidth !=
                  prev_env->scenario_rf_bandwidth) {
                _ofdmDll->setTxChannel(boost::none);
              }
            }
          }));

  // FSK frequency hopping
  _subTokens.push_back(NotificationCenter::shared.subscribe<float>(
      controlchannel::CCTuneNotification, _ios, [this](auto offset) {
        auto const env = c2api::env.current();
        auto const cfreq = static_cast<double>(env.scenario_center_frequency);
        // FIXME: we should not change the LO frequency in this function
        ::uhd::tune_request_t tr1_rx(cfreq + offset,
                                     options::phy::lo_offset_rx);
        tr1_rx.args = uhd::device_addr_t("mode_n=integer");
        _usrpSrc[1]->set_center_freq(tr1_rx);
      }));

  // subscribe to individual mandate updates
  _subTokens.push_back(
      NotificationCenter::shared.subscribe<OutcomesUpdateEventInfo>(
          OutcomesUpdateEvent, _ios, [this](auto ei) {
            this->_ofdmDll->addIndividualMandates(
                IndividualMandate::fromJSON(ei.j));
          }));
  log::text("RadioController [Done].");
}

RadioController::~RadioController() {
  delete _ios_work;
  _work_thread.join();
}

void RadioController::_channelizer_synch_work() {
  static std::vector<size_t> nsamp(_n_rx_chains);
  if (_channelizer_running) {
    if (_channelizer.load(_rx_rb.get())) {
      _channelizer.execute();
      _channelizer.getNumOutput(nsamp);
      _synch.execute(nsamp, _channelizer.getOutPtr());
      _synch.write_output(nsamp, _channelizer.getOutPtr(), _chanout);
    }
    _channelizer_ios.post([this] { _channelizer_synch_work(); });
  }
}

void RadioController::start() {
  //
  // Start the top block
  //
  _tb->start(options::phy::max_noutput_items);

  // start the DLL
  _ofdmDll->start();

  // Launch the receiving thread
  //
  _rx_rb = PinnedComplexRingBuffer::make(_buf_sz);

  _reading = true;
  _reader = std::thread([this] {
    bamradio::set_thread_name("radiocontroller_rx");
    size_t const npsd = options::psdsensing::fft_len;
    uhd::rx_metadata_t md;
    while (_reading) {
      // Get raw samples from UHD
      size_t nproduced =
          _rxStreamer->recv(_rx_rb->write_ptr(), _rx_rb->space_avail(), md);

      // send the first npsd samples to the psd chain if it has room in its
      // queue.
      if (nproduced >= npsd) {
        auto ptr = std::make_shared<std::vector<fcomplex>>(npsd);
        std::memcpy(ptr->data(), _rx_rb->write_ptr(), npsd * sizeof(fcomplex));
        NotificationCenter::shared.post(psdsensing::RawPSDNotification, ptr);
      }
      // tell the buffer we are done
      _rx_rb->produce(nproduced);
    }
  });
  _rxStreamer->issue_stream_cmd(
      uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);

  //
  // run the channelizer/synch loop
  //
  _channelizer_running = true;
  _channelizer_ios.post([this] { _channelizer_synch_work(); });

  //
  // run the control channel controller
  //
  auto const env = c2api::env.current();
  _cccontroller->updateRFBandwidth(env.scenario_rf_bandwidth);
  _cccontroller->start();

  //
  // run PSD sensing
  //
  if (!options::collab::gateway) {
    // Report PSD to gateway
    _psd->start();
  }
}

void RadioController::stop() {
  _tb->stop();
  _rxStreamer->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
  _reading = false;
  _channelizer_running = false;
  _reader.join();
  _channelizer_ios_work.reset();
  _channelizer_thread.join();
}

void RadioController::_changeChannels(de::TransmitAssignment::Map channels) {

  auto const env = c2api::env.current();

  // perform one last sanity check in case something went out of sync
  auto const do_change = [&channels, &env] {
    // we want to ensure that all chan_idx are within the range of the current
    // environment's channelization and make sure that all bw-idx are less than
    // the current channel's max bandwidth.
    auto const channelization =
        decisionengine::Channelization::get(env.scenario_rf_bandwidth);
    auto const max_chan_idx = channelization.center_offsets.size() - 1;
    auto const max_bw_idx = channelization.max_bw_idx;
    bool o = true;
    for (auto const &ch : channels) {
      if (ch.second.chan_idx > max_chan_idx) {
        log::text(
            boost::format("WARNING (radiocontroller): SRN ID %1%: chan_idx %2% "
                          "> max_chan_idx %3%. Not switching channels.") %
            ch.first % ch.second.chan_idx % max_chan_idx);
        o = false;
      }
      if (ch.second.bw_idx > max_bw_idx) {
        log::text(
            boost::format("WARNING (radiocontroller): SRN ID %1%: bw_idx %2% > "
                          "max_bw_idx %3%. Not switching channels.") %
            ch.first % ch.second.bw_idx % max_bw_idx);
        o = false;
      }
    }
    return o;
  }();

  if (!do_change) {
    // if anything went wrong, we went out of synch and we'll turn off
    // transmission
    log::text(
        "WARNING (radiocontroller): one or more chan_idx or bw_idx are out "
        "of sync. Disabling Tx for now.");
    _ofdmDll->setTxChannel(boost::none);
    return;
  }

  // else we are good to go and need to set Rx and Tx parameters

  // Rx: we only need waveform ID and offset
  _channelizer_ios.post([channels, env, this] {
    size_t rx_chan_idx = 0;
    for (auto const &channel : channels) {
      if (channel.first == _srnid) {
        // we don't need a self channel
        continue;
      }
      auto rcchannel = channel.second.toRCChannel(env.scenario_rf_bandwidth);
      auto offset = rcchannel.offset;
      auto const offset_normalized = offset / bam::dsp::sample_rate;
      auto const subchannel_type_idx = channel.second.bw_idx; // ?
      _channelizer.setOffsetFreq(rx_chan_idx, offset_normalized);
      _channelizer.setBandwidth(rx_chan_idx, subchannel_type_idx);
      _ofdmDll->rx()->setOversampleRate(rx_chan_idx, rcchannel.interp_factor);
      rx_chan_idx++;
    }
    _ofdmDll->rx()->flushFilters();
  });

  // Tx: only set when I can find myself in the map. If I am silent, unset.
  if (channels.find(_srnid) != channels.end()) {
    auto const txChan = channels.at(_srnid);
    if (txChan.silent) {
      _ofdmDll->setTxChannel(boost::none);
    } else {
      _ofdmDll->setTxChannel(txChan.toRCChannel(env.scenario_rf_bandwidth));
    }
  }
}

std::vector<Channel> RadioController::ctrlChannelAlloc() {
  auto const cc_bw = options::phy::control::bandwidth;
  std::vector<Channel> out;
  out.reserve(2);
  auto const freq_table = _cccontroller->getFreqTable();
  for (auto const &f : freq_table) {
    out.push_back({cc_bw, f, options::phy::control::sample_rate});
  }
  return out;
}

} // namespace bamradio
