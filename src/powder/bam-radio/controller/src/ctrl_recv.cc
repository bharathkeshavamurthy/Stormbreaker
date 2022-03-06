/* -*- c++ -*- */
/*
 * Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>.
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

#include "ctrl_recv.h"
#include <boost/asio.hpp>
#include <gnuradio/io_signature.h>
#include <iostream>
#include <pmt/pmt.h>
#include <string.h>

namespace gr {
namespace bamfsk {

using namespace bamradio::controlchannel;

ctrl_recv::sptr ctrl_recv::make(std::string const &length_tag_key,
                                CCData::sptr cc_data) {
  return gnuradio::get_initial_sptr(new ctrl_recv(length_tag_key, cc_data));
}

ctrl_recv::ctrl_recv(std::string const &length_tag_key, CCData::sptr cc_data)
    : gr::tagged_stream_block("ctrl_recv",
                              gr::io_signature::make(1, 1, sizeof(char)),
                              gr::io_signature::make(0, 0, 0), length_tag_key),
      d_cc_data(cc_data) {}

ctrl_recv::~ctrl_recv() {}

void ctrl_recv::set_cc_data(bamradio::controlchannel::CCData::sptr ccd) {
  std::unique_lock<decltype(_m)> l(_m);
  d_cc_data = ccd;
}

int ctrl_recv::work(int noutput_items, gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items) {
  std::unique_lock<decltype(_m)> l(_m);
  long packet_length = ninput_items[0];
  char *in = (char *)input_items[0];
  if (d_cc_data) {
    d_cc_data->deserializeShortMsg(
        boost::asio::const_buffer(in, packet_length));
    return packet_length;
  } else {
    return 0;
  }
}

} /* namespace bamfsk */
} /* namespace gr */
