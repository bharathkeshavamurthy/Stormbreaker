# Open ports
import serial
ser12 = serial.Serial('COM12',9600)
ser14 = serial.Serial('COM14',9600)

# Close ports
ser12.close()
ser14.close()

# Cheatsheet
# FL
# AC
# SSDONE
# SP0
# FP0