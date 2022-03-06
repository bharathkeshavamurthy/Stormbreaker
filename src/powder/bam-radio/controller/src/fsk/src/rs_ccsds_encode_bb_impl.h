/* -*- c++ -*- */
/*
 * Copyright 2013 Andrew Balmo
 *
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

#ifndef INCLUDED_BAMFSK_RS_CCSDS_ENCODE_BB_IMPL_H
#define INCLUDED_BAMFSK_RS_CCSDS_ENCODE_BB_IMPL_H

#include <bamfsk/rs_ccsds_encode_bb.h>

namespace gr {
namespace bamfsk {

class rs_ccsds_encode_bb_impl : public rs_ccsds_encode_bb {
private:
  std::string d_len_tag_key;
  int d_k;

  pmt::pmt_t d_tx_time;
  pmt::pmt_t d_tx_sob;
  pmt::pmt_t d_tx_eob;

protected:
  int calculate_output_stream_length(const gr_vector_int &ninput_items);

public:
  rs_ccsds_encode_bb_impl(const int k, const std::string &len_tag_key);
  ~rs_ccsds_encode_bb_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_RS_CCSDS_ENCODE_BB_IMPL_H */
