/* -*- c++ -*- */
/* 
 * Copyright 2013 Joon Young Kim <kim415@purdue.edu>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>
#include <gnuradio/io_signature.h>
#include "stream_to_tagged_stream_impl.h"

namespace gr {
  namespace bamfsk {

	stream_to_tagged_stream::sptr
		stream_to_tagged_stream::make(const std::string& len_tag_key)
	{
		return gnuradio::get_initial_sptr
			(new stream_to_tagged_stream_impl(len_tag_key));
	}

	stream_to_tagged_stream_impl::stream_to_tagged_stream_impl(
			const std::string& len_tag_key)
		: gr::block("stream_to_tagged_stream",
			gr::io_signature::make(1, 1, sizeof(unsigned char)),
			gr::io_signature::make(1, 1, sizeof(unsigned char))),
		d_needed(0),
		d_len_tag_key(pmt::intern(len_tag_key))
	{
		set_tag_propagation_policy(TPP_DONT);
		
		d_len_tag.key = d_len_tag_key;
        d_len_tag.srcid = pmt::string_to_symbol(name());

		d_preamble_time_tag = pmt::string_to_symbol("preamble_time");
	}

	stream_to_tagged_stream_impl::~stream_to_tagged_stream_impl()
	{
	}

    void stream_to_tagged_stream_impl::forecast(int noutput_items, 
			gr_vector_int &ninput_items_required)
    {
        if(d_needed > 0) {
            ninput_items_required[0] = d_needed;
        } else {
            ninput_items_required[0] = 1024;
        }
    }

	int stream_to_tagged_stream_impl::general_work(int noutput_items,
			gr_vector_int &ninput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
	{
		const unsigned char *in = (const unsigned char *) input_items[0];
		unsigned char *out = (unsigned char *) output_items[0];

		//----- Start ACM's Code -----
		std::vector<tag_t> tags;
		get_tags_in_range(tags, 0, nitems_read(0),
				nitems_read(0) + ninput_items[0],
				d_len_tag_key);

        //std::cout << "nitems_read(0) = " << nitems_read(0) << std::endl;
        //std::cout << "nitems_read(0) + ninput_items[0] = " << nitems_read(0) + ninput_items[0] << std::endl;
        //std::cout << "nitems_wrriten(0) = " << nitems_written(0) << std::endl;

        // Sort tags into sane order
        std::sort(tags.begin(), tags.end(), &tag_t::offset_compare);

        if(d_needed > 0) {
            // Copy output 
            memcpy(out, in, sizeof(unsigned char)*d_needed);

            // Find all current tags, so we can propagate them 
            get_tags_in_range(tags, 0, nitems_read(0),
                    nitems_read(0) + d_needed);
            std::sort(tags.begin(), tags.end(), &tag_t::offset_compare);

            // Propagate tags on to the same sample they came from
            BOOST_FOREACH(const tag_t tag, tags) {
                //std::cout << "key = " << tag.key << std::endl;
                add_item_tag(0,
                    nitems_written(0) + tag.offset - nitems_read(0),
                    tag.key,
                    tag.value);
            }

            //std::cout << std::endl;

            //std::cout << "d_needed > 0 : d_needed = " << d_needed << std::endl;

            // Produce 
            produce(0, d_needed);
            consume(0, d_needed);
            
            // No more samples needed
            d_needed = 0;

        } else {


            //std::cout << "size = " << tags.size() << std::endl;
            if(!tags.empty()) {
                // Samples needed to copy in next work call
                d_needed = pmt::to_long(tags[0].value); 

                // Make sure we have enough samples in the next call
                // to produce this tag stream
                set_min_noutput_items(d_needed);

                //std::cout << "d_needed " << d_needed << std::endl;

                // Produce nothing and consume up to the tag
                consume(0, tags[0].offset - nitems_read(0));
            } else {

                //std::cout << "No tags " << ninput_items[0] << std::endl;
                // No tags found, consume everything
                consume(0, ninput_items[0]);
            }

            // This case never produces anything
            produce(0, 0);
        }   

        /*
		//----- Start ACM's Code -----
		std::vector<tag_t> tags;
		get_tags_in_range(tags, 0, nitems_read(0),
				nitems_read(0) + ninput_items[0],
				d_len_tag_key);

		
		//If in state where data cannot be output
		if (d_needed == 0){
            //If no tags are found
			if(tags.empty()){
				consume_each(ninput_items[0]);
			}
			//If at least one tag has been found
			else{
				//Consume up to item associated with the first tag
				consume_each(tags[0].offset - nitems_read(0));
				set_output_multiple(pmt::to_long(tags[0].value));
				d_needed = 1;
			}
                        produce(0,0);
		}
		//If in state where data can be output
		else{
			int length = pmt::to_long(tags[0].value);

                        // Add the tagged_stream length tag
                        d_len_tag.offset = nitems_written(0);
                        d_len_tag.value = tags[0].value;
                        add_item_tag(0, d_len_tag);
		
			std::vector<tag_t> old_tags;
			get_tags_in_range(old_tags, 0,
					nitems_read(0),
					nitems_read(0) + length);

			BOOST_FOREACH(const tag_t &tag, old_tags) {
				if(pmt::equal(tag.key, d_preamble_time_tag)) {
					add_item_tag(0,
						nitems_written(0) + tag.offset - nitems_read(0),
						tag.key,
						tag.value);
				}
			}


			memcpy(out, in, length*sizeof(unsigned char));
			produce(0, length); 

			//If more than 1 tag found in input stream
			if (tags.size() > 1) {
                for (std::vector<tag_t>::iterator it = tags.begin() + 1; it != tags.end(); ++it) {
                    if (it->offset > tags[0].offset) {
				        consume_each(tags[1].offset - nitems_read(0)); 
                        break;

                    }
                }
			}
			//Only one tag found
			else{
				consume_each(ninput_items[0]);
				d_needed = 0;
			}
		}
        */

		return WORK_CALLED_PRODUCE;
	}

  }
} /* namespace gr */

