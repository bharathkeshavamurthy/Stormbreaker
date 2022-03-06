/* -*- c++ -*- */

#ifndef INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_H
#define INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_H

#include <bamfsk/api.h>
#include <gnuradio/block.h>

namespace gr {
namespace bamfsk {

/*!
 * \brief <+description of block+>
 * \ingroup bamfsk
 *
 */
class BAMFSK_API access_code_detector : virtual public gr::block
{
public:
  typedef boost::shared_ptr<access_code_detector> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of bamfsk::access_code_detector.
   *
   * To avoid accidental use of raw pointers, bamfsk::access_code_detector's
   * constructor is in a private implementation
   * class. bamfsk::access_code_detector::make is the public interface for
   * creating new instances.
   */
  static sptr make(pmt::pmt_t const &len_tag_key,
                   std::vector<uint8_t> const preamble, unsigned int const payload_nbits,
                   unsigned int const max_diff_nbits);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_ACCESS_CODE_DETECTOR_H */

