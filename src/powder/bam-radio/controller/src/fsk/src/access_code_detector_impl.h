/* -*- c++ -*- */
/*
 * Copyright 2017 (c) Tomohiro Arakawa <tarakawa@purdue.edu>.
 *
 */

#ifndef INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_IMPL_H
#define INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_IMPL_H

#include <bamfsk/access_code_detector.h>
#include <vector>

namespace gr {
namespace bamfsk {

class access_code_detector_impl : public access_code_detector
{
private:
  pmt::pmt_t const d_len_tag_key;
  std::vector<uint8_t> d_preamble_bits;
  unsigned int const d_payload_nbits;
  unsigned int const d_max_diff_nbits;

public:
  access_code_detector_impl(pmt::pmt_t const &len_tag_key,
                            std::vector<uint8_t> const preamble,
                            unsigned int const payload_nbits,
                            unsigned int const max_diff_nbits);
  ~access_code_detector_impl();

  // Where all the action really happens
  void forecast (int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items,
                   gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);

};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_IMPL_H */

