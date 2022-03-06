/* -*- c++ -*- */
/* 
 * Copyright 2013 <+YOU OR YOUR COMPANY+>.
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


#ifndef INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_H
#define INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_H

#include <bamfsk/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace bamfsk {

    class BAMFSK_API stream_to_tagged_stream : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<stream_to_tagged_stream> sptr;

      static sptr make(const std::string& len_tag_key);
    };

  }
} // namespace gr

#endif /* INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_H */

