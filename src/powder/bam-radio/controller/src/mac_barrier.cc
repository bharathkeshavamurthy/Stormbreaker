// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#include "mac_barrier.h"
#include "util.h"

#include <boost/format.hpp>
#include <functional>
#include <iostream>

#include <uhd/types/time_spec.hpp>

#include <gnuradio/io_signature.h>

namespace bamradio {
namespace ofdm {

pmt::pmt_t const mac_barrier::delay_port_id = pmt::intern("delay_port");
pmt::pmt_t const mac_barrier::channel_use_port_id =
    pmt::intern("chan_use_port");

mac_barrier::sptr mac_barrier::make(gr::uhd::usrp_sink::sptr usrp_sink) {
  return gnuradio::get_initial_sptr(new mac_barrier(usrp_sink));
}

mac_barrier::mac_barrier(gr::uhd::usrp_sink::sptr usrp_sink)
    : gr::sync_block("mac_barrier",
                     gr::io_signature::make(1, 1, sizeof(gr_complex)),
                     gr::io_signature::make(1, 1, sizeof(gr_complex))),
      uhd_async_msgs(1000), _usrp_sink(usrp_sink) {
  // set_min_noutput_items(2000); // 2000 samples * 4 b/sample = 800 bytes <
  // 8000 MTU
  message_port_register_out(delay_port_id);
  message_port_register_in(channel_use_port_id);
  set_msg_handler(channel_use_port_id,
                  boost::bind(&mac_barrier::handle_chan_use_msg, this, _1));
}

mac_barrier::~mac_barrier() {}

bool mac_barrier::start() {
  _last_time = _usrp_sink->get_time_now(); // + ::uhd::time_spec_t(0.01);
  _last_sob = nitems_read(0);
  _samp_rate = _usrp_sink->get_samp_rate();
  return true;
}

gr::uhd::usrp_sink::sptr mac_barrier::usrp_sink() const { return _usrp_sink; }

void mac_barrier::handle_chan_use_msg(pmt::pmt_t msg) {
  //auto start = pmt::tuple_ref(msg, 0);
  //auto duration = pmt_to_time_spec(pmt::tuple_ref(msg, 1));
  // TODO: block transmission in (start,duration window)
}

int mac_barrier::work(int N, gr_vector_const_void_star &input_items,
                      gr_vector_void_star &output_items) {
  int n = 0;

  static auto const uhd_tx_time_tag_key = pmt::intern("tx_time");
  static auto const moab_time_tag_key = pmt::intern("moab_time");
  static auto const tx_sob_tag_key = pmt::intern("tx_sob");
  static auto const tx_eob_tag_key = pmt::intern("tx_eob");

  std::vector<gr::tag_t> tags;
  get_tags_in_window(tags, 0, 0, N, uhd_tx_time_tag_key);
  std::sort(tags.begin(), tags.end(), gr::tag_t::offset_compare);

  for (auto const &t : tags) {
    auto const tx_time = pmt_to_time_spec(t.value);

    // Advance past end of previous packet based on sample offset and samp_rate
    auto const earliest_time =
        _last_time + ::uhd::time_spec_t(0, t.offset - _last_sob, _samp_rate);

    // ensure requested tx_time is after this_time
    if (tx_time < earliest_time) {
      abort();
    }

    _last_sob = t.offset;
    _last_time = tx_time;
  }

  get_tags_in_window(tags, 0, 0, N, moab_time_tag_key);
  std::sort(tags.begin(), tags.end(), gr::tag_t::offset_compare);

  for (auto const &t : tags) {
    auto const now = time_spec_to_pmt(::uhd::time_spec_t::get_system_time());
    auto const msg = pmt::make_tuple(alias_pmt(), pmt::tuple_ref(t.value, 0),
                                     pmt::tuple_ref(t.value, 1), now);
    message_port_pub(delay_port_id, msg);
  }

  memcpy(output_items[0], input_items[0], N * sizeof(gr_complex));
  n = N;

  return n;
}
}
}
