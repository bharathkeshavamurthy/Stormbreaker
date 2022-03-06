/* -*- c++ -*- */
/*
 * Copyright 2013 Andrew Marcum <acmarcum@prudue.edu>
 *				  Andrew Balmos <abalmos@purdue.edu>
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

#ifndef INCLUDED_BAMFSK_INSERTPREAMBLE_BB_IMPL_H
#define INCLUDED_BAMFSK_INSERTPREAMBLE_BB_IMPL_H

#include <bamfsk/insertPreamble_bb.h>
#include <vector>

namespace gr {
namespace bamfsk {

class insertPreamble_bb_impl : public insertPreamble_bb {
private:
  std::vector<unsigned char> d_preamble;

  pmt::pmt_t d_tx_time;
  pmt::pmt_t d_tx_sob;
  pmt::pmt_t d_tx_eob;

protected:
  int calculate_output_stream_length(const gr_vector_int &ninput_items);

public:
  insertPreamble_bb_impl(std::vector<unsigned char> preamble,
                         const std::string &lengthtagname);
  ~insertPreamble_bb_impl();

  int work(int noutput_items, gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_INSERTPREAMBLE_BB_IMPL_H */
