/* -*- c++ -*- */
/*
 * Copyright 2017 Dennis Ogbe.
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

#ifndef INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_IMPL_H
#define INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_IMPL_H

#include <bamfsk/tsb_chunks_to_symbols_bf.h>

namespace gr {
namespace bamfsk {

class tsb_chunks_to_symbols_bf_impl : public tsb_chunks_to_symbols_bf {
private:
  int _D;
  std::vector<float> _symbol_table;

  pmt::pmt_t _tx_time_key = pmt::intern("tx_time");
  pmt::pmt_t _tx_sob_key = pmt::intern("tx_sob");
  pmt::pmt_t _tx_eob_key = pmt::intern("tx_eob");

protected:
  int calculate_output_stream_length(const gr_vector_int &ninput_items);

public:
  tsb_chunks_to_symbols_bf_impl(const std::vector<float> &symbol_table,
                                const int D,
                                const std::string &length_tag_name);
  ~tsb_chunks_to_symbols_bf_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_IMPL_H */
