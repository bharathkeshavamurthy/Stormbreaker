/* -*- c++ -*- */
/* 
 * Copyright 2014 <+YOU OR YOUR COMPANY+>.
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

#include <algorithm>
#include <gnuradio/io_signature.h>
#include "drop_header_bb_impl.h"

namespace gr {
  namespace bamfsk {

    drop_header_bb::sptr
    drop_header_bb::make(const unsigned int header_len, const std::string& len_tag_key)
    {
      return gnuradio::get_initial_sptr
        (new drop_header_bb_impl(header_len, len_tag_key));
    }

    /*
     * The private constructor
     */
    drop_header_bb_impl::drop_header_bb_impl(const unsigned int header_len, 
			const std::string& len_tag_key)
      : gr::tagged_stream_block("drop_header_bb",
              gr::io_signature::make(1, 1, sizeof(unsigned char)),
              gr::io_signature::make(1, 1, sizeof(unsigned char)), len_tag_key),
	  d_header_len(header_len)
    {
		set_tag_propagation_policy(TPP_DONT);
	}

    /*
     * Our virtual destructor.
     */
    drop_header_bb_impl::~drop_header_bb_impl()
    {
    }

    int
    drop_header_bb_impl::calculate_output_stream_length(
			const gr_vector_int &ninput_items)
    {
      return ninput_items[0] - d_header_len;
    }

    int
    drop_header_bb_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
        const unsigned char *in = (const unsigned char *) input_items[0];
        unsigned char *out = (unsigned char *) output_items[0];

		int cnt = ninput_items[0] - d_header_len;	

		memcpy(out, &in[d_header_len], cnt);

		// Try to pass along any tags
		std::vector<tag_t> tags;
		get_tags_in_range(tags, 0,
				nitems_read(0),
				nitems_read(0) + ninput_items[0]);

		BOOST_FOREACH(const tag_t tag, tags) {
			uint64_t  idx = nitems_written(0);
                        uint64_t offset = tag.offset - nitems_read(0);
                        if (offset > d_header_len) {
                            idx += (offset - d_header_len);
                        }
			add_item_tag(0, idx, tag.key, tag.value);
		}

		return cnt;
    }

  }
} /* namespace gr */

