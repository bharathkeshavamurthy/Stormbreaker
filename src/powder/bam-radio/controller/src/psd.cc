#include "psd.h"
#include "psd.pb.h"

#include <volk/volk.h>
#include <algorithm>
#include <gnuradio/fft/window.h>

namespace bamradio {
namespace psdsensing {

using namespace boost::asio;
using namespace std::literals::chrono_literals;

NotificationCenter::Name const RawPSDNotification =
  NotificationCenter::makeName("Raw PSD Notification");

PSDSensing::PSDSensing(NodeID srnID, controlchannel::CCData::sptr ccdata,
                       bamradio::AbstractDataLinkLayer::sptr dll)
    : _srnID(srnID), _ccdata(ccdata), _ofdm_dll(dll), _t_ofdm(1s),
      _ios_work(new boost::asio::io_service::work(_ios)), _work_thread([this] {
        bamradio::set_thread_name("psd_sensing_work");
        _ios.run();
      }),
      _timer_ofdm(_ios), _gw_srn(UnspecifiedNodeID),
      _current_avg(options::psdsensing::fft_len, 0),
      _fft(fft::CPUFFT::make({ (size_t)options::psdsensing::fft_len }, true, 1,
          options::phy::fftw_wisdom)),
      _fft_window(gr::fft::window::blackman_harris(options::psdsensing::fft_len)) {
  // Subscribe to PSDUpdateEventInfo
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<PSDUpdateEventInfo>(
          PSDUpdateEvent, _ios, [this](auto ei) {
            _psd_data[_srnID] =
                PSDData{_srnID, static_cast<uint64_t>(ei.time_ns),
                        std::make_shared<std::vector<float>>(ei.psd)};
          }));
  // Subscribe to OFDM PSD segment notification
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<
          std::pair<dll::Segment::sptr, std::shared_ptr<std::vector<uint8_t>>>>(
          std::hash<std::string>{}("New Rx PSD Segment"), _ios,
          [this](auto data) {
            auto psd_seg =
                std::dynamic_pointer_cast<psdsensing::PSDSegment>(data.first);
            this->_receiveOFDM(psd_seg->packetContentsBuffer());
          }));
  // Subscribe to PSD raw data
  d_nc_tokens.push_back(
      NotificationCenter::shared.subscribe<std::shared_ptr<std::vector<fcomplex>>>(
          RawPSDNotification, _ios, [this](auto raw_vec) {
            this->_processRawPSD(*raw_vec);
          }));
}

void PSDSensing::start() {
  // Start OFDM timer
  auto now = std::chrono::system_clock::now();
  _timer_ofdm.expires_at(now + _t_ofdm);
  _timer_ofdm.async_wait([this](auto &e) { this->_sendOFDM(); });
}

PSDSensing::~PSDSensing() {
  d_nc_tokens.clear();
  _timer_ofdm.cancel();
  if (_ios_work) {
    delete _ios_work;
  }
  _ios.stop();
  if (_work_thread.joinable()) {
    _work_thread.join();
  }
}

void PSDSensing::_processRawPSD(std::vector<fcomplex>& raw_vec) {
  // blackman harris window
  size_t fftlen = raw_vec.size();
  volk_32fc_32f_multiply_32fc(raw_vec.data(), raw_vec.data(), _fft_window.data(), fftlen);
  
  // fft
  std::vector<fcomplex> fft_out(fftlen);
  _fft->execute(fftlen, raw_vec.data(), fft_out.data());
  
  // fft shift
  std::rotate(fft_out.begin(), fft_out.begin() + fftlen / 2, fft_out.end());
  
  // mag square
  std::vector<float> mag_sqr(fftlen);
  volk_32fc_magnitude_squared_32f(mag_sqr.data(), fft_out.data(), fftlen);
  
  // moving avg
  auto const& reset_period = options::psdsensing::reset_period;
  auto const& avg_len = options::psdsensing::mov_avg_len;
  static int counter = 0;
  _psd_history.push_back(std::move(mag_sqr));
  if (reset_period == 0 || (++counter %= reset_period) == 0) {
    // calibrate the avg once every reset_period
    if (_psd_history.size() > avg_len)
      _psd_history.pop_front();
    std::fill(_current_avg.begin(), _current_avg.end(), 0);
    for (auto const& v : _psd_history) {
      volk_32f_x2_add_32f(_current_avg.data(), _current_avg.data(), v.data(), fftlen);
    }
    volk_32f_s32f_multiply_32f(_current_avg.data(), _current_avg.data(), 1.0f/_psd_history.size(), fftlen);
  }
  else {
    // efficiently calculate the avg by adding appropriate delta
    std::vector<float> delta(fftlen);
    if (_psd_history.size() > avg_len) {
      //delta = (back-front)/size
      volk_32f_x2_subtract_32f(delta.data(), _psd_history.back().data(), _psd_history.front().data(), fftlen);
      _psd_history.pop_front();
    }
    else {
      //delta = (back-avg)/size
      volk_32f_x2_subtract_32f(delta.data(), _psd_history.back().data(), _current_avg.data(), fftlen);
    }
    volk_32f_s32f_multiply_32f(delta.data(), delta.data(), 1.0f/_psd_history.size(), fftlen);
    volk_32f_x2_add_32f(_current_avg.data(), _current_avg.data(), delta.data(), fftlen);
  }

  // log the averaged psd
  auto now = std::chrono::system_clock::now().time_since_epoch();
  int64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  NotificationCenter::shared.post(PSDUpdateEvent,
      PSDUpdateEventInfo{ _current_avg, now_ns });
}

std::vector<int8_t> PSDSensing::thresholdPSD(const std::vector<float>& psd,
    const HistParams& params) {
  size_t fftlen = psd.size();
  if (fftlen == 0) return {};
  // compute histogram (averaged)
  auto mm = std::minmax_element(psd.begin(), psd.end());
  const float min = *(mm.first), max = *(mm.second);
  if ((min < -200.0) || (max > 200.0)) {
    log::text(boost::format("WARNING: PSDSensing: min = %1% max = %2%") % min % max, __FILE__, __LINE__);
  }
  int nbins = int((max - min) / params.bin_size + 1);
  std::vector<float> hist(nbins, 0);
  const int half = (params.avg_len - 1) / 2;
  for (auto const& v : psd) {
    int idx = int((v - min) / params.bin_size);
    int jmin = std::max(idx - half, 0),
        jmax = std::min(idx + half, nbins - 1);
    for (int j = jmin; j <= jmax; ++j)
      ++hist[j];
  }
  volk_32f_s32f_multiply_32f(hist.data(), hist.data(), 1.0f / params.avg_len, nbins);
  // find lowest peak in histogram, thresh = right edge of peak
  bool reached_peak = false;
  int counter = 0;
  size_t k = 0;
  for (; k < nbins; ++k) {
    if (hist[k] <= params.empty_bin_thresh) {
      if (reached_peak) {
        ++counter;
        if (counter >= params.sn_gap_bins)
          break;
      }
    }
    else {
      reached_peak = true;
      counter = 0;
    }
  }
  float thresh = min + k * params.bin_size;
  if (thresh < params.noise_floor)
    thresh = params.noise_floor;
  // threshold the psd vector
  std::vector<int8_t> out(fftlen);
  for (size_t i = 0; i < fftlen; ++i)
    out[i] = psd[i] >= thresh;
  return out;
}

void PSDSensing::_sendOFDM() {
  if (_gw_srn == UnspecifiedNodeID) {
    // Need gateway SRN ID
    _gw_srn = _ccdata->getGatewaySRNID();
  }

  // Find my PSD data to send
  auto psddata_itr = _psd_data.find(this->_srnID);

  if (psddata_itr != _psd_data.end() && _gw_srn != UnspecifiedNodeID &&
      _ofdm_dll->running()) {
    // Construct a packet
    PSDPb::PSDMsg pbmsg;
    pbmsg.set_src_srnid(_srnID);
    pbmsg.set_timestamp(psddata_itr->second.time_ns);
    char const *p =
        reinterpret_cast<char const *>(psddata_itr->second.psd->data());
    pbmsg.set_psd(
        std::string(p, p + sizeof(float) * psddata_itr->second.psd->size()));

    // First two bytes are used for storing the payload (protobuf) data size
    // in bytes. The following byte contains the destination node ID (i.e.
    // gateway node ID).
    uint16_t payload_nbytes = pbmsg.ByteSize() + sizeof(NodeID);
    auto bs = std::make_shared<std::vector<uint8_t>>(
        sizeof(payload_nbytes) + sizeof(NodeID) + payload_nbytes);

    // Copy
    std::memcpy(bs->data(), &payload_nbytes, sizeof(payload_nbytes));
    std::memcpy(bs->data() + sizeof(payload_nbytes), &_gw_srn, sizeof(_gw_srn));
    pbmsg.SerializeWithCachedSizesToArray(bs->data() + sizeof(payload_nbytes) +
                                          sizeof(_gw_srn));

    // Construct a segment
    auto const seg = std::make_shared<PSDSegment>(
        _gw_srn, buffer(*bs), std::chrono::system_clock::now());
    try {
      _ofdm_dll->send(seg, bs);
    } catch (std::runtime_error e) {
      log::text("Failed sending PSD segment on OFDM DLL.");
    }
  }

  // Wait
  if (_ios_work) {
    _timer_ofdm.expires_at(_timer_ofdm.expiry() + _t_ofdm);
    _timer_ofdm.async_wait([this](auto &e) { this->_sendOFDM(); });
  }
}

PSDData PSDSensing::parsePSDSegment(boost::asio::const_buffer data) {
  auto const src = buffer_cast<uint8_t const *>(data);
  // Get nbytes
  uint16_t payload_nbytes = 0;
  std::memcpy(&payload_nbytes, src, sizeof(payload_nbytes));

  // Read the packet
  PSDPb::PSDMsg pbmsg;
  bool success = pbmsg.ParseFromArray(
      src + sizeof(payload_nbytes) + sizeof(NodeID), payload_nbytes - 1);
  if (!success) {
    log::text("Failed to deserialize PSD data.");
    throw;
  }
  auto psd_data_str = pbmsg.psd();
  assert(psd_data_str.size() % sizeof(float) == 0);
  auto psd_vec =
      std::make_shared<std::vector<float>>(psd_data_str.size() / sizeof(float));
  std::memcpy(psd_vec->data(), psd_data_str.data(), psd_data_str.size());
  return PSDData{static_cast<NodeID>(pbmsg.src_srnid()), pbmsg.timestamp(),
                 psd_vec};
}

void PSDSensing::_receiveOFDM(const_buffer data) {
  auto psd_data = parsePSDSegment(data);
  _psd_data[psd_data.src_srnid] = psd_data;
}

std::chrono::system_clock::time_point PSDData::time() const {
  using namespace std::chrono;
  return system_clock::time_point(nanoseconds(time_ns));
}
} // namespace psdsensing
} // namespace bamradio
