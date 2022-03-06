#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: File Metadata Vector Sink
# Generated: Wed Jun  7 23:56:21 2017
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import gr, blocks
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser


class file_metadata_vector_sink(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "File Metadata Vector Sink")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 200000

        ##################################################
        # Blocks
        ##################################################
        self.blocks_vector_source_x_0 = blocks.vector_source_c(10*[0,1,2,3,4,5,6,7,8,9], True, 10, [])
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*10, 10010000)
        self.blocks_file_meta_sink_0 = blocks.file_meta_sink(gr.sizeof_gr_complex*10, 'C:\\Users\\Zyglabs\\Dropbox\\2017_05_Summer_Research\\EARS\\Refs\\GnuRadio\\Trials\\Trial4_file_metadata_vector_sink.out', samp_rate, 1, blocks.GR_FILE_FLOAT, True, 1000000, "", True)
        self.blocks_file_meta_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_meta_sink_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_head_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate


def main(top_block_cls=file_metadata_vector_sink, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == '__main__':
    main()
