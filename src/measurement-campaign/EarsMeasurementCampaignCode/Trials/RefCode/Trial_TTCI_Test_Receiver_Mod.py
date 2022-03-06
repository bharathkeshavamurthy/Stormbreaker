#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Thu Jun 15 11:48:26 2017
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from datetime import datetime
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio import wxgui
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.wxgui import fftsink2
from gnuradio.wxgui import forms
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import delayed_start
import gpsdo
import time
import wx

class top_block(grc_wxgui.top_block_gui):

    def __init__(self):

        grc_wxgui.top_block_gui.__init__(self, title="Top Block")
        _icon_path = "C:\Program Files\GNURadio-3.7\share\icons\hicolor\scalable/apps\gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.prefix = prefix = "C:\\Users\\Zyglabs\\Dropbox\\2017_05_Summer_Research\\EARS\\MeasurementCampaign\\RefCode\\"
        self.time_now = time_now = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        self.samp_rate = samp_rate = 5.2e6
        self.rx_gain = rx_gain = 76
        self.recfile = recfile = prefix + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".dat"
        self.logfile = logfile = prefix + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + "_log.txt"
        self.center_freq = center_freq = 99.9e6

        ##################################################
        # Blocks
        ##################################################
        _center_freq_sizer = wx.BoxSizer(wx.VERTICAL)
        self._center_freq_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_center_freq_sizer,
        	value=self.center_freq,
        	callback=self.set_center_freq,
        	label='center_freq',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._center_freq_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_center_freq_sizer,
        	value=self.center_freq,
        	callback=self.set_center_freq,
        	minimum=89e6,
        	maximum=108e6,
        	num_steps=190,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_center_freq_sizer)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=center_freq,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=samp_rate,
        	fft_size=1024,
        	fft_rate=2,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot',
        	peak_hold=False,
        )
        self.Add(self.wxgui_fftsink2_0.win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("", "master_clock_rate=52e6, --fifo=self.logfile")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_time_source('gpsdo', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_gain(rx_gain, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, recfile, False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.wxgui_fftsink2_0, 0))

        # ZYG
        # delayed_start.delayed_start(self.uhd_usrp_source_0, delay=2, pps=False, base_time=None, offset=0, gpsdo=True)
        # delayed_start.main(self.uhd_usrp_source_0)
        gpsdo.gpsdo(self.uhd_usrp_source_0)
        gpsdo.main(self.uhd_usrp_source_0, self.rx_gain, self.logfile)

    def get_prefix(self):
        return self.prefix

    def set_prefix(self, prefix):
        self.prefix = prefix
        self.set_recfile(self.prefix + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".dat")
        self.set_logfile(self.prefix + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + "_log.txt")

    def get_time_now(self):
        return self.time_now

    def set_time_now(self, time_now):
        self.time_now = time_now

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_fftsink2_0.set_sample_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 0)


    def get_recfile(self):
        return self.recfile

    def set_recfile(self, recfile):
        self.recfile = recfile
        self.blocks_file_sink_0.open(self.recfile)

    def get_logfile(self):
        return self.logfile

    def set_logfile(self, logfile):
        self.logfile = logfile

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self._center_freq_slider.set_value(self.center_freq)
        self._center_freq_text_box.set_value(self.center_freq)
        self.wxgui_fftsink2_0.set_baseband_freq(self.center_freq)
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
