/* -*- c++ -*- */
/*
 * Copyright 2017 (c) Tomohiro Arakawa <tarakawa@purdue.edu>.
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

#include <gnuradio/io_signature.h>
#include "access_code_detector_impl.h"

namespace gr {
namespace bamfsk {

access_code_detector::sptr
access_code_detector::make(pmt::pmt_t const &len_tag_key,
                           std::vector<uint8_t> const preamble,
                           unsigned int const payload_nbits,
                           unsigned int const max_diff_nbits)
{
  return gnuradio::get_initial_sptr
         (new access_code_detector_impl(len_tag_key, preamble, 
                                        payload_nbits, max_diff_nbits));
}

/*
 * The private constructor
 */
access_code_detector_impl::access_code_detector_impl(pmt::pmt_t const
    &len_tag_key, std::vector<uint8_t> const preamble,
    unsigned int const payload_nbits, unsigned int const max_diff_nbits)
  : gr::block("access_code_detector",
              gr::io_signature::make(1, 1, sizeof(uint8_t)),
              gr::io_signature::make(1, 1, sizeof(uint8_t))),
    d_len_tag_key(len_tag_key),
    d_payload_nbits(payload_nbits),
    d_max_diff_nbits(max_diff_nbits)
{
  BOOST_FOREACH(uint8_t byte, preamble) {
    for(int i = 0; i < 8; i++) {
      d_preamble_bits.push_back(0x01 & (byte >> i));
    }
  }
  set_tag_propagation_policy(TPP_DONT);
  set_output_multiple(d_payload_nbits);
}

/*
 * Our virtual destructor.
 */
access_code_detector_impl::~access_code_detector_impl()
{
}

void
access_code_detector_impl::forecast (int noutput_items,
                                     gr_vector_int &ninput_items_required)
{
  ninput_items_required[0] = (noutput_items / d_payload_nbits) *
                             (d_payload_nbits + d_preamble_bits.size());
}

int
access_code_detector_impl::general_work (int noutput_items,
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items,
    gr_vector_void_star &output_items)
{
  const uint8_t *in = (const uint8_t *) input_items[0];
  uint8_t *out = (uint8_t *) output_items[0];

  size_t const in_nitems = ninput_items[0];
  size_t const out_nitems = noutput_items;
  size_t inbuf_pos = 0;
  size_t outbuf_pos = 0;

  while (inbuf_pos + d_preamble_bits.size() + d_payload_nbits <= in_nitems &&
         outbuf_pos + d_payload_nbits <= out_nitems) {
    size_t matched_nbits = 0;
    for (size_t i=0; i<d_preamble_bits.size(); ++i)
      if (in[inbuf_pos+i] == d_preamble_bits[i])
        ++matched_nbits;
    if (matched_nbits >= d_preamble_bits.size()-d_max_diff_nbits) {
      // copy payload, drop header
      memcpy(out + outbuf_pos, in + d_preamble_bits.size() + inbuf_pos,
             d_payload_nbits);
      // add tag
      tag_t len_tag;
      len_tag.offset = nitems_written(0) + outbuf_pos;
      len_tag.key = d_len_tag_key;
      len_tag.value = pmt::from_long(d_payload_nbits);
      add_item_tag(0, len_tag);
      //housekeeping
      outbuf_pos += d_payload_nbits;
    }
    ++inbuf_pos;
  }

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each (inbuf_pos);

  // Tell runtime system how many output items we produced.
  return outbuf_pos;
}

} /* namespace bamfsk */
} /* namespace gr */

