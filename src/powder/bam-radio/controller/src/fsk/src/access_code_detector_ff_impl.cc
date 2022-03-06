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

#include <vector>
#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include "access_code_detector_ff_impl.h"

namespace gr {
  namespace bamfsk {

    access_code_detector_ff::sptr
    access_code_detector_ff::make(const int payload_len, 
			const std::vector<int>pre, 
			const std::string& len_tag_key,
			const double upstream_dec_factor,
            const bool use_fft)
    {
      return gnuradio::get_initial_sptr
        (new access_code_detector_ff_impl(payload_len, pre, 
			len_tag_key, upstream_dec_factor, use_fft));
    }

    /*
     * The private constructor
     */
    access_code_detector_ff_impl::access_code_detector_ff_impl(
			const int payload_len, 
			const std::vector<int> pre, 
			const std::string& len_tag_key,
			const double upstream_dec_factor,
            const bool use_fft)
      : gr::sync_block("access_code_detector_ff",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float))),
        d_use_fft(use_fft),
	    d_block_len(),
		d_frame_len(),
		d_payload_len(payload_len),
		d_upstream_dec_factor(upstream_dec_factor),
		d_pre_bits(),
		d_samp_rate(-1),
		d_time_fracs(-1),
		d_time_seconds(0),
		d_time_offset(0)
    {

		d_pre_bits.reserve(pre.size()*8);
		BOOST_FOREACH(unsigned int byte, pre) {
			for(int i = 0; i < 8; i++) {
				d_pre_bits.push_back(0x01&(byte >> i));
			}
		}

		d_len_tag.key = pmt::string_to_symbol(len_tag_key);
		d_len_tag.value = pmt::from_long(d_payload_len + d_pre_bits.size());
		d_len_tag.srcid = pmt::string_to_symbol(name());

		// Save from calculating time and time again
		d_frame_len = d_pre_bits.size() + d_payload_len;
		d_block_len = d_frame_len + d_pre_bits.size();

		// Make sure we always see one header
		set_output_multiple(d_block_len);

		// Prepare the FFT if using FFT correlation
		if(use_fft) {

			// Time reverse to build convolution filter
			d_pre_bits_rev.resize(d_pre_bits.size());
			std::reverse_copy(d_pre_bits.begin(), d_pre_bits.end(),
					d_pre_bits_rev.begin());

			// Find the next biggest power of 2
			d_fft_size = (int) pow(2.0, ceil(log(double(d_block_len))/log(2.0)));

			// Prepare fft objects
			d_fft_fwd = new fft::fft_real_fwd(d_fft_size, 1);
			d_fft_rev = new fft::fft_real_rev(d_fft_size, 1);

			// Copy preamble filter into fft object
			memset(d_fft_fwd->get_inbuf(), 0,
				d_fft_fwd->inbuf_length()*sizeof(float));
			memcpy(d_fft_fwd->get_inbuf(), &d_pre_bits_rev[0],
				d_pre_bits_rev.size()*sizeof(float));

			// Compute preamble filter fft 
			d_fft_fwd->execute();

			// Make storage for the fft of the preamble filter
			d_fft_preamble = fft::malloc_complex(d_fft_fwd->outbuf_length());

			// Store result
			memcpy(d_fft_preamble, d_fft_fwd->get_outbuf(),
				d_fft_fwd->outbuf_length()*sizeof(gr_complex));
		}
    }

    /*
     * Our virtual destructor.
     */
    access_code_detector_ff_impl::~access_code_detector_ff_impl()
    {
		// Clean up fft objects
		delete d_fft_fwd;
		delete d_fft_rev;

		// Clean up preamble filter transform storage
		fft::free(d_fft_preamble);
    }

    int access_code_detector_ff_impl::work(int noutput_items,
			  gr_vector_const_void_star &input_items,
			  gr_vector_void_star &output_items)
    {
        const float *in = (const float *) input_items[0];
        float *out = (float *) output_items[0];

		memcpy(out, in, noutput_items*sizeof(float));

		for(int z = 0; z < noutput_items/d_block_len; z++) {
			unsigned int ind = 0;

			// fft Method
			if(d_use_fft) {

				// Copy data into fft
				memcpy(d_fft_fwd->get_inbuf(), 
						&in[z*d_block_len],
						d_block_len*sizeof(float));

				// Compute data fft
				d_fft_fwd->execute();

				// Multiply data fft and preamble filter fft to do
				// preamble filter convolution
				volk_32fc_x2_multiply_32fc(d_fft_rev->get_inbuf(),
						d_fft_fwd->get_outbuf(), d_fft_preamble,
						d_fft_fwd->outbuf_length());

				// Compute the time domain convolution
				d_fft_rev->execute();

				// Find the magnitude of the convolution
				float *conv = d_fft_rev->get_outbuf();
				std::transform(conv, conv+d_fft_size, conv, fabsf);

				// Find the index of the maximum value
				volk_32f_index_max_32u(&ind, conv, d_fft_rev->outbuf_length());

				// Account for the length of the preamble
				ind -= d_pre_bits.size() - 1;

				// If ind is negative here we did not find anything anyways...
				if((int)ind < 0) {
					ind = 0;
                }
                
			// Correlation Method
			} else {
			
                float max = 0;
                int corrLen = d_block_len - d_pre_bits.size() + 1;
                for(int i = 0; i < corrLen; i++) {

                    float dot_val = 0;
                    volk_32f_x2_dot_prod_32f(&dot_val,
                            &in[i+z*d_block_len],
                            &d_pre_bits[0],
                            d_pre_bits.size());

                    if(fabsf(dot_val) > max) {
                        max = fabsf(dot_val);
                        ind = i;
                    }
                }
                if(max!=25) continue;
            }

			// Look for any rx_time tags from the USRP
			std::vector<tag_t> tags;
			get_tags_in_range(tags, 0,
					nitems_read(0),
					nitems_read(0)+noutput_items,
					pmt::string_to_symbol("rx_time"));

			// Extract time value associated with rx_time tags
			// Is this simply recording the last received "rx_time" tag?
			BOOST_FOREACH(const tag_t &tag, tags){
				d_time_offset = tag.offset;
				const pmt::pmt_t &value = tag.value;
				d_time_seconds = pmt::to_uint64(pmt::tuple_ref(value, 0));
				d_time_fracs = pmt::to_double(pmt::tuple_ref(value, 1));
			}

			// Look for any rx_rate tags from the USRP
			get_tags_in_range(tags, 0,
					nitems_read(0),
					nitems_read(0)+noutput_items,
					pmt::string_to_symbol("rx_rate"));

			// Extract rate value associated with rx_rate tags
			BOOST_FOREACH(const tag_t &tag, tags){
				d_samp_rate = pmt::to_double(tag.value);
				d_samp_rate /= d_upstream_dec_factor;
			}

			// Determine USRP clock when this preamble was found
			d_len_tag.offset = nitems_written(0)+ind;
			add_item_tag(0, d_len_tag);
			add_time_tag(ind);

			if(ind + d_frame_len < d_block_len) {
				d_len_tag.offset = nitems_written(0) + ind + d_frame_len;
				add_item_tag(0, d_len_tag);
				add_time_tag(ind + d_frame_len);
			}

			if(ind > d_frame_len) {
				d_len_tag.offset = nitems_written(0) + ind - d_frame_len;
				add_item_tag(0, d_len_tag);
				add_time_tag(ind - d_frame_len);
			}
		}

        return noutput_items;
    }

	void access_code_detector_ff_impl::add_time_tag(unsigned int ind)
	{
		double tmp = (nitems_read(0)+ind-d_time_offset)/d_samp_rate
			+ d_time_fracs;
		uint64_t cur_seconds = d_time_seconds + std::floor(tmp);
		double cur_fracs = tmp - std::floor(tmp);

		// Tag this preamble with the time
		add_item_tag(0, 
				nitems_written(0)+ind,
				pmt::string_to_symbol("preamble_time"),
				pmt::make_tuple(
					pmt::from_uint64(cur_seconds),
					pmt::from_double(cur_fracs)
				),
				pmt::string_to_symbol(name()));
	}
 }
} /* namespace gr */

