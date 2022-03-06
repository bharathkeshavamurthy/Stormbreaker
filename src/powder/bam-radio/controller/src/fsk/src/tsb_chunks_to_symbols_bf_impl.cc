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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tsb_chunks_to_symbols_bf_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace bamfsk {

tsb_chunks_to_symbols_bf::sptr
tsb_chunks_to_symbols_bf::make(const std::vector<float> &symbol_table,
                               const int D,
                               const std::string &length_tag_name) {
  return gnuradio::get_initial_sptr(
      new tsb_chunks_to_symbols_bf_impl(symbol_table, D, length_tag_name));
}

/*
 * The private constructor
 */
tsb_chunks_to_symbols_bf_impl::tsb_chunks_to_symbols_bf_impl(
    const std::vector<float> &symbol_table, const int D,
    const std::string &length_tag_name)
    : gr::tagged_stream_block("tsb_chunks_to_symbols_bf",
                              gr::io_signature::make(1, 1, sizeof(char)),
                              gr::io_signature::make(1, 1, sizeof(float)),
                              length_tag_name),
      _D(D), _symbol_table(symbol_table) {
  set_tag_propagation_policy(TPP_DONT);
  set_relative_rate((double)_D);
}

/*
 * Our virtual destructor.
 */
tsb_chunks_to_symbols_bf_impl::~tsb_chunks_to_symbols_bf_impl() {}

int tsb_chunks_to_symbols_bf_impl::calculate_output_stream_length(
    const gr_vector_int &ninput_items) {
  int noutput_items = _D * ninput_items[0];
  return noutput_items;
}

int tsb_chunks_to_symbols_bf_impl::work(int noutput_items,
                                        gr_vector_int &ninput_items,
                                        gr_vector_const_void_star &input_items,
                                        gr_vector_void_star &output_items) {
  auto *in = (const char *)input_items[0];
  auto *out = (float *)output_items[0];

  // copy the symbols
  for (size_t i = 0; i < ninput_items[0]; ++i) {
    assert(((size_t)in[i] * _D + _D) <= _symbol_table.size());
    memcpy(out, &_symbol_table[(size_t)in[i] * _D], _D * sizeof(float));
    out += _D;
  }

  // propagate tags correctly
  std::vector<tag_t> tags;
  get_tags_in_range(tags, 0, nitems_read(0), nitems_read(0) + ninput_items[0]);
  for (auto const &tag : tags) {
    if (pmt::eqv(tag.key, _tx_sob_key) || pmt::eqv(tag.key, _tx_time_key)) {
      // SOB and tx_time need to be at the start of the PDU
      add_item_tag(0, nitems_written(0), tag.key, tag.value);
    } else if (pmt::eqv(tag.key, _tx_eob_key)) {
      // EOB needs to be on the very last item of the PDU
      add_item_tag(0, nitems_written(0) + _D * ninput_items[0] - 1, tag.key,
                   tag.value);
    } else {
      // everything else can stay where it is.
      add_item_tag(0, nitems_written(0) + (tag.offset - nitems_read(0)) * _D,
                   tag.key, tag.value);
    }
  }

  // Tell runtime system how many output items we produced.
  return _D * ninput_items[0];
}

} /* namespace bamfsk */
} /* namespace gr */
