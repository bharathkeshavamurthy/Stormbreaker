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

#ifndef INCLUDED_BAMFSK_CRC32_BB_H
#define INCLUDED_BAMFSK_CRC32_BB_H

#include <bamfsk/api.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
namespace bamfsk {

/*!
 * \brief <+description of block+>
 * \ingroup pu
 *
 */
class BAMFSK_API crc32_bb : virtual public gr::tagged_stream_block {
public:
  typedef boost::shared_ptr<crc32_bb> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of pu::crc32_bb.
   *
   * To avoid accidental use of raw pointers, pu::crc32_bb's
   * constructor is in a private implementation
   * class. pu::crc32_bb::make is the public interface for
   * creating new instances.
   */
  static sptr make(const bool check, const std::string &lengthtagname);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_CRC32_BB_H */
