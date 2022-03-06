/* -*- c++ -*- */
/*
 * Copyright 2017 $<+YOU OR YOUR COMPANY+>.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/filter/fft_filter.h>
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>
#include <algorithm>
#include <boost/format.hpp>
#include <iostream>
#include "mac_ctrl.h"
#include "util.h"

#define __CTRL_GEN__DEBUG__ 1

namespace gr {
namespace bamfsk {

mac_ctrl::sptr mac_ctrl::make(std::vector<float> const &filter,
                              std::string const &length_tag_name) {
  return gnuradio::get_initial_sptr(new mac_ctrl(filter, length_tag_name));
}

/*
 * The private constructor
 */
mac_ctrl::mac_ctrl(std::vector<float> const &filter,
                   std::string const &length_tag_name)
    : gr::tagged_stream_block(
          "mac_ctrl", gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(1, 1, sizeof(gr_complex)), length_tag_name),
      _length_tag_key(pmt::intern(length_tag_name)),
      _filter(1, filter) {
  pmt::pmt_t const input_port_id = pmt::intern("cctx_msg");
  message_port_register_in(input_port_id);
  set_msg_handler(input_port_id, boost::bind(&mac_ctrl::msg_handler, this, _1));
  set_tag_propagation_policy(TPP_DONT);
}

/*
 * Our virtual destructor.
 */
mac_ctrl::~mac_ctrl() {}

int mac_ctrl::work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items) {
  auto out = (gr_complex *)output_items[0];
  auto in = (const gr_complex *)input_items[0];

  // copy over the output samples
   memcpy(out, in, noutput_items * sizeof(gr_complex));
  //_filter.filter(ninput_items[0], in, out);

  // pop packet info
  if (d_packet_info.empty())
    throw "no cc pkt info";
  auto packet_info = d_packet_info.front();
  d_packet_info.pop();

  auto tx_time_pmt =
      pmt::dict_ref(packet_info, pmt::intern("time"), pmt::PMT_NIL);
  // add sob and eob tags in appropriate locations
  add_item_tag(0, nitems_read(0), _tx_time_key, tx_time_pmt);
  add_item_tag(0, nitems_read(0), _tx_sob_key, pmt::PMT_T);
  add_item_tag(0, nitems_read(0) + ninput_items[0] - 1, _tx_eob_key,
               pmt::PMT_T);

  // Tell runtime system how many output items we produced.
  return ninput_items[0];
}

int mac_ctrl::calculate_output_stream_length(
    const gr_vector_int &ninput_items) {
  return ninput_items[0];
}

void mac_ctrl::msg_handler(pmt::pmt_t msg) { d_packet_info.push(msg); }

} /* namespace bamfsk */
} /* namespace gr */
