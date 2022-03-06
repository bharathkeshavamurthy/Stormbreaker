#include "discrete_channels.h"
#include "bandwidth.h"
#include "c2api.h"
#include "events.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace bamradio {
namespace decisionengine {

std::vector<uint8_t> Channelization::initial_assignment() const {
  std::mt19937 rng(center_offsets.size());
  std::vector<uint8_t> o(center_offsets.size());
  std::iota(begin(o), end(o), 0);
  std::shuffle(begin(o), end(o), rng);
  return o;
}

// generated my util/alloc2.m -- TODO change the script to actually generate
// this snippet
const std::map<int64_t, Channelization> Channelization::table = {
    {5000000,
     {1,
      {-1656000, -1288000, -920000, -552000, -184000, 184000, 552000, 920000,
       1288000, 1656000}}},
    {8000000,
     {1,
      {-3105882, -2717647, -2329412, -1941176, -1552941, -1164706, -776471,
       -388235, 0, 388235, 776471, 1164706, 1552941, 1941176, 2329412, 2717647,
       3105882}}},
    {10000000,
     {3,
      {-3870000, -3010000, -2150000, -1290000, -430000, 430000, 1290000,
       2150000, 3010000, 3870000}}},
    {20000000, {3, {-8895652, -8086957, -7278261, -6469565, -5660870, -4852174,
                    -4043478, -3234783, -2426087, -1617391, -808696,  0,
                    808696,   1617391,  2426087,  3234783,  4043478,  4852174,
                    5660870,  6469565,  7278261,  8086957,  8895652}}},
    {25000000,
     {4, {-11263636, -10190909, -9118182, -8045455, -6972727, -5900000,
          -4827273,  -3754545,  -2681818, -1609091, -536364,  536364,
          1609091,   2681818,   3754545,  4827273,  5900000,  6972727,
          8045455,   9118182,   10190909, 11263636}}},
    {40000000,
     {4, {-18778378, -17735135, -16691892, -15648649, -14605405, -13562162,
          -12518919, -11475676, -10432432, -9389189,  -8345946,  -7302703,
          -6259459,  -5216216,  -4172973,  -3129730,  -2086486,  -1043243,
          0,         1043243,   2086486,   3129730,   4172973,   5216216,
          6259459,   7302703,   8345946,   9389189,   10432432,  11475676,
          12518919,  13562162,  14605405,  15648649,  16691892,  17735135,
          18778378}}}};

Channelization const &Channelization::get(int64_t rf_bandwidth) {
  return table.at(rf_bandwidth);
}

// making this a simple cons might not be the right thing to do here but oh
// well...
cl_object toLisp(Channelization c) {
  return lisp::Cons(lisp::toLisp(c.max_bw_idx), lisp::toLisp(c.center_offsets));
}

double Channelization::get_current_max_bandwidth() {
  auto const rfb = c2api::env.current().scenario_rf_bandwidth;
  auto const mbwi = table.at(rfb).max_bw_idx;
  return bam::dsp::SubChannel::table()[mbwi].bw();
}

bamradio::Channel TransmitAssignment::toRCChannel(int64_t rf_bandwith) const {
  double const bandwidth = bam::dsp::SubChannel::table()[bw_idx].bw();
  double const offset =
      Channelization::get(rf_bandwith).center_offsets[chan_idx] + chan_ofst;
  // FIXME? better constructor?
  Channel chan(bandwidth, offset, bam::dsp::sample_rate);
  chan.sample_gain = std::pow(10, -1.0 * atten / 20);
  return chan;
}

CCDataPb::TransmitAssignmentMsg TransmitAssignment::toProto() const {
  CCDataPb::TransmitAssignmentMsg msg;
  msg.set_bw_idx(bw_idx);
  msg.set_chan_idx(chan_idx);
  msg.set_chan_ofst(chan_ofst);
  msg.set_atten(std::max(atten, 0.0f));
  msg.set_silent(silent);
  return msg;
}

CCDataPb::ChannelInfo *
TransmitAssignment::toLegacyChannelInfo(int64_t rf_bandwidth) const {
  auto msg = new CCDataPb::ChannelInfo;
  double const offset =
      Channelization::get(rf_bandwidth).center_offsets[chan_idx] + chan_ofst;
  msg->set_center_offset_hz(offset);
  msg->set_waveform_id(bw_idx);
  msg->set_attenuation_db(atten);
  msg->set_silent(silent);
  msg->set_channel_id(chan_idx);
  return msg;
}

TransmitAssignment
TransmitAssignment::fromProto(CCDataPb::TransmitAssignmentMsg const &msg) {
  return {.bw_idx = (uint8_t)(msg.bw_idx()),
          .chan_idx = (uint8_t)(msg.chan_idx()),
          .chan_ofst = msg.chan_ofst(),
          .atten = msg.atten(),
          .silent = msg.silent()};
}

TransmitAssignment TransmitAssignment::fromLisp(cl_object obj) {
  using lisp::Funcall;
  using lisp::Symbol;
  auto BRSymbol = [](auto const &s) { return Symbol(s, "bam-radio"); };
  auto access = [&BRSymbol, &obj](auto const &field, auto const &convert) {
    return convert(Funcall(BRSymbol(field), obj));
  };
  return {.bw_idx = (uint8_t)access("bw-idx", lisp::fromInt),
          .chan_idx = (uint8_t)access("chan-idx", lisp::fromInt),
          .chan_ofst = (int32_t)access("chan-ofst", lisp::fromInt),
          .atten = access("atten", lisp::fromFloat),
          .silent = access("silent", [](auto b) { return b == lisp::t; })};
}

cl_object toLisp(TransmitAssignment t) {
  using lisp::Funcall;
  using lisp::Keyword;
  using lisp::Symbol;
  auto BRSymbol = [](auto const &s) { return Symbol(s, "bam-radio"); };
  // clang-format off
  return Funcall(Symbol("make-instance"), BRSymbol("transmit-assignment"),
                 Keyword("bw-idx"), lisp::toLisp(t.bw_idx),
                 Keyword("chan-idx"), lisp::toLisp(t.chan_idx),
                 Keyword("chan-ofst"), lisp::toLisp(t.chan_ofst),
                 Keyword("atten"), lisp::toLisp(t.atten),
                 Keyword("silent"), t.silent ? lisp::t : lisp::nil);
  // clang-format on
}

TxAssignmentUpdate TxAssignmentUpdate::fromLisp(cl_object tau) {
  using namespace lisp;
  auto BRSymbol = [](auto const &s) { return Symbol(s, "bam-radio"); };
  // convert the lisp map first to a cons list and then to C++ (pull requests
  // welcome...)
  auto assList = Funcall(BRSymbol("get-consed-assignment-map"), tau);
  TransmitAssignment::Map am;
  forEach(assList, [&am](auto const &obj) {
    auto const id = (NodeID)fromInt(Car(obj));
    auto const assignment = TransmitAssignment::fromLisp(Cdr(obj));
    am.emplace(id, assignment);
  });
  // get the bools
  auto chnlUp = Funcall(BRSymbol("channel-updated?"), tau);
  auto bwUp = Funcall(BRSymbol("bandwidth-updated?"), tau);
  auto attenUp = Funcall(BRSymbol("atten-updated?"), tau);
  // return fresh object
  return {.assignment_map = am,
          .channel_updated = (chnlUp == lisp::t),
          .bandwidth_updated = (bwUp == lisp::t),
          .atten_updated = (attenUp == lisp::t)};
}

} // namespace decisionengine
} // namespace bamradio
