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

#ifndef INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_H
#define INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_H

#include <bamfsk/api.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
namespace bamfsk {

/*!
 * \brief <+description of block+>
 * \ingroup bamfsk
 *
 */
class BAMFSK_API tsb_chunks_to_symbols_bf
    : virtual public gr::tagged_stream_block {
public:
  typedef boost::shared_ptr<tsb_chunks_to_symbols_bf> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of
   * bamfsk::tsb_chunks_to_symbols_bf.
   *
   * To avoid accidental use of raw pointers, bamfsk::tsb_chunks_to_symbols_bf's
   * constructor is in a private implementation
   * class. bamfsk::tsb_chunks_to_symbols_bf::make is the public interface for
   * creating new instances.
   */
  static sptr make(const std::vector<float> &symbol_table, const int D,
                   const std::string &length_tag_name);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_TSB_CHUNKS_TO_SYMBOLS_BF_H */
