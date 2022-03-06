# Open a serial port even if it's currently occupied.
#
# Yaguang Zhang, Purdue University, 2017-06-13
import serial

def openPort (p="COM5", b=9600, t=1):
    try:
        ser = serial.Serial(port=p, baudrate=b, timeout=t)
        # Try to open port, if possible print message.
        ser.isOpen()
        print ("Port " + p + " is successfully opened!")
        return ser
    except IOError:
        # If port is already opened, close it and open it again, and print
        # message.
        try:
            ser.close()
            ser.open()
            print ("Port " + p + " was already open... Successfully closed & reopened!")
            return ser
        except Exception as e:
            print ("Failed in opening Port " + p + "!")
            print(e)
            raise