# from lib import Test2_baseband_rx_file_sink_Mod
# Doesn't work. Python won't reload the gnu radio script for each execution.

import serial, time, os

cmdsToSend = ['FL500000\r\n']*5
strsToSend = ['Hello ...', 'My name is Python ...', 'What is your name?', '...', 'Bye~']

port = "COM5"
baud = 9600
ser5 = serial.Serial(port, baud, timeout=1)

for idx, str in enumerate(cmdsToSend):
    ser5.write(str.encode('ascii'))
    time.sleep(5) # In second.

    # Test2_baseband_rx_file_sink_Mod.main()
    # execfile('./ModifiedPyFiles/Test2_baseband_rx_file_sink_Mod.py')
    os.system('py -2.7 ./ModifiedPyFiles/Test2_baseband_rx_file_sink_Mod.py')