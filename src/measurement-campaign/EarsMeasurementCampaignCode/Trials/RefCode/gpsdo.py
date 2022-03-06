#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gpsdo.py
#  
#  Copyright 2014 Balint Seeber <balint256@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import threading
import sys, socket, traceback, time, datetime
from optparse import OptionParser

from gnuradio import uhd

#def main():
class gpsdo():
	def __init__(self, usrp):


		#parser = OptionParser(usage="%prog: [options]")

		# FIXME: Add TCP server to avoid mess with pipes
		#parser.add_option("-p", "--port", type="int", default=12345, help="port [default=%default]")
		#parser.add_option("-a", "--args", type="string", default="", help="UHD device args [default=%default]")
		#parser.add_option("-f", "--fifo", type="string", default="gps_data.txt", help="UHD device args [default=%default]")
		#(options, args) = parser.parse_args()

		#f = None
		#f = "gps_data.txt"


		#print "Fifo option ", options.fifo
		#if options.fifo != "":
		#	try:
		#		f = open(options.fifo, 'w')
		#		print "Opened"
		#	except Exception, e:
		#		print "Failed to open FIFO:", options.fifo, e

		#usrp = uhd.usrp_source(
		#	device_addr=options.args,
		#	stream_args=uhd.stream_args(
		#		cpu_format="fc32",
		#		channels=range(1),
		#	),
		#)
		#self.src.set_subdev_spec(self.spec)
		#self.src.set_clock_source(ref, 0)
		#self.src.set_time_source(pps, 0)
		#self.src.set_samp_rate(requested_sample_rate)
		#self.src.set_center_freq(uhd.tune_request(freq, lo_offset), 0)
		#self.src.set_gain(selected_gain_proxy, 0)

		mboard_sensor_names = usrp.get_mboard_sensor_names()

		print mboard_sensor_names
		#for name in mboard_sensor_names:
			#sensor_value = usrp.get_mboard_sensor(name)
			#print name, "=", sensor_value
			#print sensor_value.name, "=", sensor_value.value
			#print sensor_value.to_pp_strin	def print_logfile(logfile):
		return

def log(usrp, chann_gain, filename):
		#nmea_sensors = ["gps_gpgga", "gps_gprmc", "gps_time", "gps_locked"]
		f = open(filename, 'w')
		print filename

		mboard_sensor_names = usrp.get_mboard_sensor_names()
		# Serious hack job here, just for making this work
		nmea_sensors = ["gps_gpgga", "gps_time", "gps_locked", "ref_locked"]

		try:
			while True:
				for name in nmea_sensors:
					if name not in mboard_sensor_names:
						print "Sensor '%s' not available" % (name)
						continue
					sensor_value = usrp.get_mboard_sensor(name)
					value = sensor_value.value.strip()
					if value == "":
						continue
					##print value
					## CRA Added the below code
					if name == "gps_gpgga":
						display_string = "GPS Position: " + value
						#print "GPS Position is: ", value
					elif name == "gps_gprmc":
						display_string = "GPS Position: " + value
					elif name == "gps_time":
						formatted_current_time = datetime.datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
						display_string = "Current Time: " + formatted_current_time
						#print "Current Time is: ", value
					elif name == "gps_locked":
						display_string = "GPS Locked:   " + value
					elif name == "ref_locked":
						display_string = "Rcvr Gain:    " + str(chann_gain)
					else:
						print "There was an error somewhere"
					#print display_string
					## End CRA Code

					if f is not None:
						try:
							f.write(display_string + "\r\n")
							f.flush()
						except IOError, e:
							if e.errno == 32:	# Broken pipe
								f.close()
								print "FIFO consumer broke pipe. Re-opening..."
								f = open(filename, 'w')	# Will block
								break
				time.sleep(1)
		except KeyboardInterrupt:
			print "Stopping..."

		except Exception, e:
			print "Unhandled exception:", e

		if f is not None:
			f.flush()
			f.close()

def main(usrp,chann_gain,filename):
	t = threading.Thread(target=log, name="gpsdo_thread", args=(usrp, chann_gain, filename))
	t.start()
	return

if __name__ == '__main__':
	main()
