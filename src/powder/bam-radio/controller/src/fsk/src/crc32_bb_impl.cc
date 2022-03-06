/* -*- c++ -*- */
/*
 * Copyright 2014 Purdue University
 *			Andrew Balmos <abalmos@purdue.edu>
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

#include "crc32_bb_impl.h"
#include <gnuradio/io_signature.h>

#define STATISTICS_PORT pmt::mp("statistics")

namespace gr {
namespace bamfsk {

crc32_bb::sptr crc32_bb::make(const bool check,
                              const std::string &lengthtagname) {
  return gnuradio::get_initial_sptr(new crc32_bb_impl(check, lengthtagname));
}

crc32_bb_impl::crc32_bb_impl(const bool check, const std::string &lengthtagname)
    : gr::tagged_stream_block(
          "crc32_bb", gr::io_signature::make(1, 1, sizeof(char)),
          gr::io_signature::make(1, 1, sizeof(char)), lengthtagname),
      d_check(check), d_total(0), d_pass(0), d_fail(0) {
  message_port_register_out(STATISTICS_PORT);
  set_tag_propagation_policy(TPP_DONT);

  // Tags of instrest
  d_tx_time = pmt::string_to_symbol("tx_time");
  d_tx_sob = pmt::string_to_symbol("tx_sob");
  d_tx_eob = pmt::string_to_symbol("tx_eob");
  d_preamble_time = pmt::string_to_symbol("preamble_time");
}

crc32_bb_impl::~crc32_bb_impl() {}

int crc32_bb_impl::calculate_output_stream_length(
    const gr_vector_int &ninput_items) {
  if (d_check) {
    return ninput_items[0] - 4;
  } else {
    return ninput_items[0] + 4;
  }
}

int crc32_bb_impl::work(int noutput_items, gr_vector_int &ninput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items) {
  const unsigned char *in = (const unsigned char *)input_items[0];
  unsigned char *out = (unsigned char *)output_items[0];
  int output_size = ninput_items[0] + (d_check ? -4 : 4);
  unsigned int crc;

  // Clean up the CRC object for the coming calculation
  d_crc_impl.reset();

  // Try to pass along any tags
  std::vector<tag_t> tags;
  get_tags_in_range(tags, 0, nitems_read(0),
                    nitems_read(0) + (int)ninput_items[0]);

  // Check CRC
  if (d_check) {
    d_crc_impl.process_bytes(in, output_size);
    crc = d_crc_impl();

    d_total++;
    if (crc != *(unsigned int *)(in + output_size)) {
      d_fail++;

      // for(unsigned int i = 0; i < output_size; i++) {
      //	std::cout << static_cast<unsigned>(in[i]) << " ";
      //}
      // std::cout << std::endl << std::endl;
      output_size = 0;

    } else {
      d_pass++;
      memcpy(out, in, output_size);
    }

    // Generate the statistics info
    // idx 0: total looked at
    // idx 1: total that passed
    // idx 2: total that failed
    message_port_pub(STATISTICS_PORT, pmt::make_tuple(pmt::from_long(d_total),
                                                      pmt::from_long(d_pass),
                                                      pmt::from_long(d_fail)));

    // Add CRC
  } else {
    d_crc_impl.process_bytes(in, ninput_items[0]);
    crc = d_crc_impl();

    memcpy(out, in, ninput_items[0]);
    memcpy(out + ninput_items[0], &crc, 4);
  }

  BOOST_FOREACH (const tag_t tag, tags) {
    if (pmt::equal(tag.key, d_tx_time) || pmt::equal(tag.key, d_tx_sob) ||
        // pmt::equal(tag.key, d_tx_eob) ||
        pmt::equal(tag.key, d_preamble_time)) {
      add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0), tag.key,
                   tag.value);
    } else if (pmt::equal(tag.key, d_tx_eob)) {
      add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0) + 4,
                   tag.key, tag.value);
    }
  }

  return output_size;
}

} /* namespace bamfsk */
} /* namespace gr */
