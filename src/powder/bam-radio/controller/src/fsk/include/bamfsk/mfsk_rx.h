/* -*- c++ -*- */

#ifndef INCLUDED_BAMFSK_MFSK_RX_H
#define INCLUDED_BAMFSK_MFSK_RX_H

#include <bamfsk/api.h>
#include <gnuradio/hier_block2.h>

namespace gr {
  namespace bamfsk {

    /*!
     * \brief <+description of block+>
     * \ingroup bamfsk
     *
     */
    class BAMFSK_API mfsk_rx : virtual public gr::hier_block2
    {
     public:
      typedef boost::shared_ptr<mfsk_rx> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of bamfsk::mfsk_rx.
       *
       * To avoid accidental use of raw pointers, bamfsk::mfsk_rx's
       * constructor is in a private implementation
       * class. bamfsk::mfsk_rx::make is the public interface for
       * creating new instances.
       */
      static sptr make(double sample_rate, std::vector<float> pulse, int num_fsk_points, int rs_k, std::vector<uint8_t> preamble, int min_soft_decs, int payload_len);
    };

  } // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_MFSK_RX_H */

