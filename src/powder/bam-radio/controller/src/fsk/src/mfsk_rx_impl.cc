/* -*- c++ -*- */
/*
 * MFSK Receiver
 * Tomohiro Arakawa <tarakawa@purdue.edu>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "mfsk_rx_impl.h"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <gnuradio/filter/fft_filter_ccc.h>
#include <gnuradio/blocks/complex_to_mag.h>
#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/blocks/float_to_char.h>
#include <gnuradio/blocks/repack_bits_bb.h>
#include <gnuradio/blocks/tag_debug.h>
#include <gnuradio/digital/crc32_bb.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/expj.h>

// deprecated GNU radio blocks... will be out in newer versions
#if __has_include(<gnuradio/blocks/argmax_fs.h>)
#include <gnuradio/blocks/argmax_fs.h>
#else
#include <gnuradio/blocks/argmax.h>
#endif

#if __has_include(<gnuradio/digital/chunks_to_symbols_sf.h>)
#include <gnuradio/digital/chunks_to_symbols_sf.h>
#else
#include <gnuradio/digital/chunks_to_symbols.h>
#endif


#include <bamfsk/crc32_bb.h>
#include <bamfsk/access_code_detector.h>
#include <bamfsk/ndaSync_ff.h>
#include <bamfsk/stream_to_tagged_stream.h>
#include <bamfsk/drop_header_bb.h>
#include <bamfsk/rs_ccsds_decode_bb.h>

#include <gnuradio/blocks/tag_debug.h>

namespace gr {
    namespace bamfsk {

        mfsk_rx::sptr
            mfsk_rx::make(double sample_rate, std::vector<float> pulse, int num_fsk_points, int rs_k, std::vector<uint8_t> preamble, int min_soft_decs, int payload_len)
            {
                return gnuradio::get_initial_sptr
                    (new mfsk_rx_impl(sample_rate, pulse, num_fsk_points, rs_k, preamble, min_soft_decs, payload_len));
            }


        /*
         * The private constructor
         */
        mfsk_rx_impl::mfsk_rx_impl(double sample_rate, std::vector<float> pulse, int num_fsk_points, int rs_k, std::vector<uint8_t> preamble, int min_soft_decs, int payload_len)
            : gr::hier_block2("mfsk_rx",
                    gr::io_signature::make(1, 1, sizeof(gr_complex)),
                    gr::io_signature::make(1, 1, sizeof(uint8_t)))
        {

            //Generate filter
            std::vector<float> freq_dev_vec(num_fsk_points);
            size_t k = 0;
            double start = sample_rate * (-1.0 / 2.0 + 2.0 / (2.0 * pulse.size() + 1));
            double stop = sample_rate * (1.0 / 2.0 - 2.0 / (2.0 * pulse.size() + 1));
            double step = (stop - start) / (freq_dev_vec.size() - 1);
            std::generate(freq_dev_vec.begin(), freq_dev_vec.end(), [&k, start, step] { return start + (step * k++); });
            auto bits_per_sym = (int)std::log2(num_fsk_points);
            size_t len_pulse = pulse.size();

            std::vector<filter::fft_filter_ccc::sptr> filter_blocks;
            std::vector<blocks::complex_to_mag::sptr> c_to_mag_blocks;
            for(int i=0; i<num_fsk_points; ++i){
                std::vector<gr_complex> tone(len_pulse, 1.0);
                double f = start + step * i;
                for(int j=0; j<len_pulse; ++j){
                    double t = j / sample_rate;
                    tone[j] *= gr_expj(2*M_PI*f*t);
                }
                filter_blocks.push_back(filter::fft_filter_ccc::make(1,tone));
            }

            //compute magnitude
            for(int i=0; i<num_fsk_points; ++i){
                c_to_mag_blocks.push_back(blocks::complex_to_mag::make(1));
            }

            //NDA Sync
            auto nda_sync = ndaSync_ff::make(len_pulse, min_soft_decs);

            //find abs max output
            auto amax = blocks::argmax_fs::make(1);

            //null sink (throw away 1st output of amax)
            auto nsink = blocks::null_sink::make(sizeof(short));

            //C2S
            std::vector<float> c2s_table;
            for(int i=0; i<num_fsk_points; ++i){
                for (int j=0; j<bits_per_sym; ++j){
                    uint8_t val = i;
                    c2s_table.push_back(1&(val>>j));
                }
            }
            auto c2s = digital::chunks_to_symbols_sf::make(c2s_table, bits_per_sym);

            //access code detector
            //double dec_factor = ((double)len_pulse)/bits_per_sym;
            //std::vector<int> preamble_int(preamble.begin(), preamble.end());
            int phy_payload_len = (payload_len + 4 + 32) * 8;
            //auto access_code = access_code_detector_ff::make(phy_payload_len, preamble_int, "length", dec_factor, false);
            auto access_code = access_code_detector::make(pmt::intern("length"), preamble, phy_payload_len, 1);

            //float to byte
            auto f2b = blocks::float_to_char::make(1, 1.0);

            //add tag
            //auto s2ts = stream_to_tagged_stream::make("length");

            //repack
            auto repack = blocks::repack_bits_bb::make(1, 8, "length", false);

            //drop the header
            //auto drop_header = drop_header_bb::make(preamble.size(), "length");

			//auto tag_dbg = blocks::tag_debug::make(sizeof(uint8_t), "tagdebug", "length");

            //RS
            auto rs_ccsds = rs_ccsds_decode_bb::make(rs_k, "length");

            //CRC32
            //auto crc_check = crc32_bb::make(true, "length");
			auto crc_check = digital::crc32_bb::make(true, "length");

            //make flowgraph
            for(int i=0; i<num_fsk_points; ++i){
                connect(self(), 0, filter_blocks[i], 0);
                connect(filter_blocks[i], 0, c_to_mag_blocks[i], 0);
                connect(c_to_mag_blocks[i], 0, nda_sync, i);
                connect(nda_sync, i, amax, i);
            }
            connect(amax, 0, nsink, 0);
            connect(amax, 1, c2s, 0);
			connect(c2s, 0, f2b, 0);
            connect(f2b, 0, access_code, 0);
            //connect(access_code, 0, s2ts, 0);
            //connect(f2b, 0, s2ts, 0);
            connect(access_code, 0, repack, 0);
            //connect(repack, 0, drop_header, 0);
            connect(repack, 0, rs_ccsds, 0);
			//connect(repack, 0, tag_dbg, 0);
            connect(rs_ccsds, 0, crc_check, 0);
            connect(crc_check, 0, self(), 0);
        }

        /*
         * Our virtual destructor.
         */
        mfsk_rx_impl::~mfsk_rx_impl()
        {
        }


    } /* namespace bamfsk */
} /* namespace gr */
