/* -*- c++ -*- */
/*
 * Copyright 2013 Andrew Marcum <acmarcum@purdue.edu>
 *				  Andrew Balmos	<abalmos@purdue.edu>
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

#include "insertPreamble_bb_impl.h"
#include <gnuradio/io_signature.h>
#include <vector>

namespace gr {
namespace bamfsk {

insertPreamble_bb::sptr
insertPreamble_bb::make(std::vector<unsigned char> preamble,
                        const std::string &lengthtagname) {
  return gnuradio::get_initial_sptr(
      new insertPreamble_bb_impl(preamble, lengthtagname));
}

/*
 * The private constructor
 */
insertPreamble_bb_impl::insertPreamble_bb_impl(
    std::vector<unsigned char> preamble, const std::string &lengthtagname)
    : gr::tagged_stream_block(
          "insertPreamble_bb",
          gr::io_signature::make(1, 1, sizeof(unsigned char)),
          gr::io_signature::make(1, 1, sizeof(unsigned char)), lengthtagname),
      d_preamble(preamble) {
  set_tag_propagation_policy(TPP_DONT);

  // Tags of instrest
  d_tx_time = pmt::string_to_symbol("tx_time");
  d_tx_sob = pmt::string_to_symbol("tx_sob");
  d_tx_eob = pmt::string_to_symbol("tx_eob");
}

/*
 * Our virtual destructor.
 */
insertPreamble_bb_impl::~insertPreamble_bb_impl() {}

int insertPreamble_bb_impl::calculate_output_stream_length(
    const gr_vector_int &ninput_items) {
  return ninput_items[0] + d_preamble.size();
}

int insertPreamble_bb_impl::work(int noutput_items, gr_vector_int &ninput_items,
                                 gr_vector_const_void_star &input_items,
                                 gr_vector_void_star &output_items) {
  const unsigned char *in = (const unsigned char *)input_items[0];
  unsigned char *out = (unsigned char *)output_items[0];

  memcpy(out, &d_preamble[0], d_preamble.size() * sizeof(unsigned char));
  memcpy(out + d_preamble.size(), in, ninput_items[0] * sizeof(unsigned char));

  // Try to pass along any tags
  std::vector<tag_t> tags;
  get_tags_in_range(tags, 0, nitems_read(0), nitems_read(0) + ninput_items[0]);

  BOOST_FOREACH (const tag_t tag, tags) {
    if (pmt::equal(tag.key, d_tx_time) || pmt::equal(tag.key, d_tx_sob)) {
      add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0), tag.key,
                   tag.value);
    } else if (pmt::equal(tag.key, d_tx_eob)) {
      add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0) +
                          d_preamble.size(),
                   tag.key, tag.value);
    }
  }

  return ninput_items[0] + d_preamble.size();
}

} /* namespace bamfsk */
} /* namespace gr */
