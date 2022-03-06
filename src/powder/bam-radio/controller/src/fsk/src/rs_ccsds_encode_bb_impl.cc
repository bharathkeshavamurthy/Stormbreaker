/* -*- c++ -*- */
/*
 * Copyright 2013 Andrew Balmos
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

#include "rs_ccsds_encode_bb_impl.h"
#include <cmath>
#include <gnuradio/io_signature.h>

#include "rs_ccsds.h"

namespace gr {
namespace bamfsk {

rs_ccsds_encode_bb::sptr
rs_ccsds_encode_bb::make(const int k, const std::string &len_tag_key) {
  return gnuradio::get_initial_sptr(
      new rs_ccsds_encode_bb_impl(k, len_tag_key));
}

/*
 * The private constructor
 */
rs_ccsds_encode_bb_impl::rs_ccsds_encode_bb_impl(const int k,
                                                 const std::string &len_tag_key)
    : gr::tagged_stream_block(
          "rs_ccsds_encode_bb",
          gr::io_signature::make(1, 1, sizeof(unsigned char)),
          gr::io_signature::make(1, 1, sizeof(unsigned char)), len_tag_key),
      d_k(k), d_len_tag_key(len_tag_key) {
  assert(d_k <= rs::ccsds::K);

  // Tags of instrest
  d_tx_time = pmt::string_to_symbol("tx_time");
  d_tx_sob = pmt::string_to_symbol("tx_sob");
  d_tx_eob = pmt::string_to_symbol("tx_eob");
}

/*
 * Our virtual destructor.
 */
rs_ccsds_encode_bb_impl::~rs_ccsds_encode_bb_impl() {}

int rs_ccsds_encode_bb_impl::calculate_output_stream_length(
    const gr_vector_int &ninput_items) {
  int out_len = 0;
  if (d_k > 0) {
    set_tag_propagation_policy(TPP_DONT);
    out_len = ninput_items[0] +
              std::ceil(((double)ninput_items[0]) / d_k) * rs::ccsds::PARITY;
  } else {
    set_tag_propagation_policy(TPP_ONE_TO_ONE);
    out_len = ninput_items[0];
  }

  return out_len;
}

int rs_ccsds_encode_bb_impl::work(int noutput_items,
                                  gr_vector_int &ninput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items) {
  const unsigned char *in = (const unsigned char *)input_items[0];
  unsigned char *out = (unsigned char *)output_items[0];
  int out_len = 0;

  if (d_k > 0) {
    unsigned char data[rs::ccsds::K];

    int num_rs_blocks = std::ceil(((double)ninput_items[0]) / d_k);

    for (int i = 0; i < num_rs_blocks; i++) {

      // Determine the what "k" is for this block.
      // The last "block" maybe <= d_k
      int kp = std::min(ninput_items[0] - i * d_k, d_k);

      // Prepare the next block data
      memset(data, 0, rs::ccsds::K);
      memcpy(data + (rs::ccsds::K - kp), &in[i * d_k], sizeof(char) * kp);

      int out_idx = (d_k + rs::ccsds::PARITY) * i;

      // Copy the data to out
      memcpy(&out[out_idx], &in[i * d_k], sizeof(char) * kp);

      // Do the RS CCSDS encoding, store the codewords directly
      // into out
      rs::ccsds::encode(data, &out[out_idx + kp]);

      // Try to pass along any tags
      std::vector<tag_t> tags;
      get_tags_in_range(tags, 0, nitems_read(0) + i * d_k,
                        nitems_read(0) + i * d_k + kp);

      BOOST_FOREACH (const tag_t tag, tags) {
        if (pmt::equal(tag.key, d_tx_time) || pmt::equal(tag.key, d_tx_sob)) {
          add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0),
                       tag.key, tag.value);
        } else if (pmt::equal(tag.key, d_tx_eob)) {
          add_item_tag(0, nitems_written(0) + tag.offset - nitems_read(0) +
                              rs::ccsds::PARITY,
                       tag.key, tag.value);
        }
      }
    }

    out_len = ninput_items[0] + num_rs_blocks * rs::ccsds::PARITY;
  } else {
    memcpy(out, in, sizeof(char) * ninput_items[0]);
    out_len = ninput_items[0];
  }

  return out_len;
}

} /* namespace bamfsk */
} /* namespace gr */
