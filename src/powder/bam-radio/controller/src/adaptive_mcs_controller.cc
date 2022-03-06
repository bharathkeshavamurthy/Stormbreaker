// -*- c++ -*-
// Copyright (c) 2017 Tomohiro Arakawa

#include "adaptive_mcs_controller.h"
#include "cc_data.h"
#include "events.h"
#include "options.h"
#include "statistics.h"
#include <cmath>
#include <tuple>

namespace bamradio {
namespace ofdm {

NotificationCenter::Name const AdaptiveMCSController::MCSRequestNotification =
    std::hash<std::string>{}("MCS Request");

AdaptiveMCSController::AdaptiveMCSController()
    : _ios_work(new boost::asio::io_service::work(_ios)), _work_thread([this] {
        bamradio::set_thread_name("adaptive_mcs_controller");
        _ios.run();
      }) {
  // Subscribe to error rate variance notification
  _nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<uint8_t, double>>(
          stats::BLERNotification, _ios, [this](auto err_map) {
            for (auto const &x : err_map) {
              this->_err_map[x.first] = x.second;
            }
          }));
  _nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::map<uint8_t, double>>(
          stats::NoiseVarNotification, _ios, [this](auto noise_var_map) {
            for (auto const &x : noise_var_map) {
              this->_noise_var_map[x.first] = x.second;
              this->updateMCS(x.first);
            }
          }));
  _nc_tokens.push_back(
      NotificationCenter::shared
          .subscribe<controlchannel::CCData::OFDMChannelOverlapInfo>(
              controlchannel::CCData::OFDMChannelOverlapNotification, _ios,
              [this](auto overlap_info) {
                for (auto const &ol : overlap_info.overlap_map) {
                  if (this->_overlap_map[ol.first] != ol.second) {
                    this->_overlap_map[ol.first] = ol.second;
                    if (ol.second) {
                      // FIXME: set link mcs to QPSK 1/2
                      MCSRequest mcs_req;
                      mcs_req.dst_srnid = options::phy::control::id;
                      mcs_req.src_srnid = ol.first;
                      mcs_req.mcs = MCS::QPSK_R12_N1944;
                      mcs_req.seqid = SeqID::ID::ZIG_128_12_108_12_QPSK;
                      // Send request
                      this->requestMCS(mcs_req);
                    }
                  }
                }
              }));
}

AdaptiveMCSController::~AdaptiveMCSController() {
  delete _ios_work;
  _work_thread.join();
}

void AdaptiveMCSController::updateMCS(uint8_t srnid) {
  // get new MCS and SeqID
  auto new_mcs =
      _adap_mcs_map[srnid].getNextMCS(_noise_var_map[srnid], _err_map[srnid]);
  // need update?
  if (_mcs[srnid] != std::get<0>(new_mcs) && _overlap_map[srnid] == false) {
    _mcs[srnid] = std::get<0>(new_mcs);
    MCSRequest mcs_req;
    mcs_req.dst_srnid = options::phy::control::id;
    mcs_req.src_srnid = srnid;
    mcs_req.mcs = std::get<0>(new_mcs);
    mcs_req.seqid = std::get<1>(new_mcs);
    // Send request
    requestMCS(mcs_req);
  }
}

void AdaptiveMCSController::requestMCS(MCSRequest req) {
  NotificationCenter::shared.post(MCSRequestNotification, req);
  // log decision
  NotificationCenter::shared.post(
      ofdm::MCSDecisionEvent,
      ofdm::MCSDecisionEventInfo{
          .txNodeID = req.src_srnid,
          .noiseVar = static_cast<float>(_noise_var_map[req.src_srnid]),
          .errorRate = static_cast<float>(_err_map[req.src_srnid]),
          .overlapped = _overlap_map[req.src_srnid],
          .payloadMCS = req.mcs,
          .payloadSymSeqID = req.seqid});
}

} // namespace ofdm
} // namespace bamradio
