// PSD Sensing
// Copyright (c) 2018 Tomohiro Arakawa

#ifndef PSD_H
#define PSD_H

#include "cc_data.h"
#include "dll.h"
#include "dll_types.h"
#include "events.h"
#include "notify.h"
#include "segment.h"
#include "fcomplex.h"
#include "bamfftw.h"

#include <chrono>
#include <vector>

namespace bamradio {
namespace psdsensing {

extern NotificationCenter::Name const RawPSDNotification;

struct PSDData {
  NodeID src_srnid = UnspecifiedNodeID;
  uint64_t time_ns = 0;
  std::shared_ptr<const std::vector<float>> psd;

  std::chrono::system_clock::time_point time() const;
};

class PSDSensing {
public:
  typedef std::shared_ptr<PSDSensing> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<PSDSensing>(std::forward<Args>(args)...);
  }

  PSDSensing(NodeID srnID, controlchannel::CCData::sptr ccdata,
             bamradio::AbstractDataLinkLayer::sptr dll);
  ~PSDSensing();

  // threshold PSD based on histogram
  struct HistParams {
    float bin_size;
    float empty_bin_thresh;
    int sn_gap_bins;
    int avg_len;
    float noise_floor;
  };
  static std::vector<int8_t> thresholdPSD(const std::vector<float>& psd,
      const HistParams& params);

  /// Parse PSD segment
  static PSDData parsePSDSegment(boost::asio::const_buffer data);

  // API
  void start();

private:
  typedef std::chrono::system_clock::duration Duration;

  void _processRawPSD(std::vector<fcomplex>& raw_vec);

  void _sendOFDM();
  void _receiveOFDM(boost::asio::const_buffer data);

  NodeID _srnID;

  // OFDM DLL/PHY
  bamradio::AbstractDataLinkLayer::sptr _ofdm_dll;
  Duration const _t_ofdm;

  // PSD
  std::map<NodeID, PSDData> _psd_data;

  // Averaging of PSD
  fft::CPUFFT::sptr _fft;
  const std::vector<float> _fft_window;
  std::deque<std::vector<float>> _psd_history;
  std::vector<float> _current_avg;

  // Gateway info
  controlchannel::CCData::sptr _ccdata;
  NodeID _gw_srn;

  // subscription
  std::vector<NotificationCenter::SubToken> d_nc_tokens;

  // io_service / timers
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  std::thread _work_thread;
  boost::asio::system_timer _timer_ofdm; // OFDM Segment generation
};

} // namespace psdsensing
} // namespace bamradio

#endif
