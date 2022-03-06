import serial

port = "COM5"
baud = 19200

try:
    ser = serial.Serial(port, baud, timeout=1)
    ser.isOpen() # try to open port, if possible print message and proceed with 'while True:'
    print ("port is opened!")

except IOError: # if port is already opened, close it and open it again and print message
    ser.close()
    ser.open()
    print ("port was already open, was closed and opened again!")