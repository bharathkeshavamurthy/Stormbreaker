// -*- c++ -*-
// Copyright (c) 2017 Tomohiro Arakawa

#ifndef INCLUDED_AMC_H_
#define INCLUDED_AMC_H_

#include "mcs.h"
#include "mcsv1.h"
#include "notify.h"
#include "ofdm.h"
#include "util.h"
#include "dll_types.h"
#include <boost/asio.hpp>
#include <map>
#include <thread>

namespace bamradio {
namespace ofdm {

struct MCSRequest {
  uint8_t src_srnid;
  uint8_t dst_srnid;
  MCS::Name mcs;
  SeqID::ID seqid;
};

class AdaptiveMCSController {
public:
  AdaptiveMCSController();
  ~AdaptiveMCSController();
  boost::asio::io_service &io_service() { return _ios; };
  static NotificationCenter::Name const MCSRequestNotification;

private:
  void updateMCS(uint8_t srnid);
  void requestMCS(MCSRequest req);
  std::map<uint8_t, double> _err_map;
  std::map<uint8_t, double> _noise_var_map;
  std::map<NodeID, bool> _overlap_map;
  std::map<uint8_t, MCSAdaptAlg> _adap_mcs_map;
  std::map<uint8_t, MCS::Name> _mcs;
  std::vector<NotificationCenter::SubToken>
      _nc_tokens; /// Notification center tokens
  // io_service
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  std::thread _work_thread; // Single thread
};
} // namespace ofdm
} // namespace bamradio
#endif
