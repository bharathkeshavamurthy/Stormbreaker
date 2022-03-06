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

#include <stdio.h>                                                               
#include <stdlib.h>                                                              
#include <cmath>                                                                 
#include <vector>                                                                
#include <algorithm>                                                             
#include <volk/volk.h> 
#include <gnuradio/io_signature.h>
#include "ndaSync_ff_impl.h"

namespace gr {
  namespace bamfsk {

    ndaSync_ff::sptr
    ndaSync_ff::make(int pulseLen, int minSoftDecs)
    {
      return gnuradio::get_initial_sptr
        (new ndaSync_ff_impl(pulseLen, minSoftDecs));
    }

    /*
     * The private constructor
     */
    ndaSync_ff_impl::ndaSync_ff_impl(int pulseLen, int minSoftDecs)
      : gr::sync_decimator("ndaSync_ff",
              gr::io_signature::make(1, gr::io_signature::IO_INFINITE,
				  sizeof(float)),
              gr::io_signature::make(1, gr::io_signature::IO_INFINITE, 
				  sizeof(float)), pulseLen),
		d_pulseLen(pulseLen),
		d_minSoftDecs(minSoftDecs)
    {
		set_min_noutput_items(d_minSoftDecs);
	}

    /*
     * Our virtual destructor.
     */
    ndaSync_ff_impl::~ndaSync_ff_impl()
    {
    }

    int ndaSync_ff_impl::work(int noutput_items,
			  gr_vector_const_void_star &input_items,
			  gr_vector_void_star &output_items)
    {
		unsigned int ninput_items = noutput_items*d_pulseLen;  
		//printf("ninput_items = %d\n",ninput_items);
		unsigned int index = 0;
		unsigned int bestStart = 0;
		float maxEnergy = 0;
		//compute argmax() for each input stream                                 
		for (unsigned int z = 0; z < input_items.size(); z++){   
			float tempMax = 0;
			unsigned int indMax = 0;
			const float* input = (const float*)input_items[z];
			/*for (unsigned int n = 0; n < ninput_items; n++){  			
				if (input[n] > tempMax){
					tempMax = input[n];
					indMax = n;
				}	
			}*/
			volk_32f_index_max_32u(&indMax,input,ninput_items);
			index = indMax%d_pulseLen;
			float energySum = 0;
			float energyTemp = 0;
			std::vector<float> symbols(noutput_items);
			
			for (unsigned int l = 0; l < input_items.size(); l++){
				const float* input = (const float*)input_items[l];
				//std::vector<float> symbols(noutput_items);
				symbols.clear();
				for (unsigned int k = index; k < ninput_items; k+=d_pulseLen){
					//energySum += std::pow(input[k],2);
					symbols.push_back(input[k]);
				}
				volk_32f_x2_dot_prod_32f(&energyTemp,&symbols[0],&symbols[0],
						noutput_items);
				//volk_32f_s32f_power_32f(&energyTemp,&symbols[0],2,noutput_items);
				//volk_32f_x2_add_32f(&energySum,&Pow,&energySum,1);
				energySum += energyTemp;
			}
			//printf("energySum =%f\n",energySum);
			if (z == 0){
				maxEnergy = energySum;
				bestStart = index;
			}
			else{ //z > 0
				if (energySum > maxEnergy){
					maxEnergy = energySum;
					bestStart = index;
				}
			}
		}
		//printf("bestStart %d\n",bestStart);

		//Populate output vector
		for (int z = 0; z < input_items.size(); z++){
			const float* input = (const float*)input_items[z];
			float* output = (float*)output_items[z];
			int n = 0;
			for (int y = bestStart; y < ninput_items; y+=d_pulseLen){
				//printf("y = %u\n",y);
				output[n] = input[y];
				n++;
			}
		}

        // Tell runtime system how many output items we produced.
        return noutput_items;
    }

  } /* namespace feedback */
} /* namespace gr */

