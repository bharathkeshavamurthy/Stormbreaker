/* -*- c++ -*- */
/* 
 * Copyright 2017 $<+YOU OR YOUR COMPANY+>.
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


#ifndef _QA_TSB_CHUNKS_TO_SYMBOLS_BF_H_
#define _QA_TSB_CHUNKS_TO_SYMBOLS_BF_H_

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>

namespace gr {
  namespace bamfsk {

    class qa_tsb_chunks_to_symbols_bf : public CppUnit::TestCase
    {
    public:
      CPPUNIT_TEST_SUITE(qa_tsb_chunks_to_symbols_bf);
      CPPUNIT_TEST(t1);
      CPPUNIT_TEST_SUITE_END();

    private:
      void t1();
    };

  } /* namespace bamfsk */
} /* namespace gr */

#endif /* _QA_TSB_CHUNKS_TO_SYMBOLS_BF_H_ */

