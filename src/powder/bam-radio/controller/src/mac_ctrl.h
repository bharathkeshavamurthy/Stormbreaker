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

#ifndef INCLUDED_BAMFSK_MAC_CTRL_H
#define INCLUDED_BAMFSK_MAC_CTRL_H

#include "bbqueue.h"

#include <gnuradio/filter/fft_filter.h>
#include <gnuradio/tagged_stream_block.h>
#include <gnuradio/uhd/usrp_sink.h>
#include <pmt/pmt.h>

#include <uhd/types/metadata.hpp>
#include <uhd/types/time_spec.hpp>

#include <queue>

namespace gr {
namespace bamfsk {

class mac_ctrl : virtual public gr::tagged_stream_block {
 public:
  typedef boost::shared_ptr<mac_ctrl> sptr;
  static sptr make(std::vector<float> const &filter,
                   std::string const &length_tag_name);

  ~mac_ctrl();

  int work(int noutput_items, gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
  int calculate_output_stream_length(const gr_vector_int &ninput_items);

 private:
  // tags
  pmt::pmt_t _length_tag_key;
  pmt::pmt_t _tx_sob_key = pmt::intern("tx_sob");
  pmt::pmt_t _tx_eob_key = pmt::intern("tx_eob");
  pmt::pmt_t _tx_time_key = pmt::intern("tx_time");

  // constructor
  mac_ctrl(std::vector<float> const &filter,
           std::string const &length_tag_name);

  gr::filter::kernel::fft_filter_ccf _filter;

  void msg_handler(pmt::pmt_t msg);
  std::queue<pmt::pmt_t> d_packet_info;
};

}  // namespace bamfsk
}  // namespace gr

#endif /* INCLUDED_BAMFSK_MAC_CTRL_H */
