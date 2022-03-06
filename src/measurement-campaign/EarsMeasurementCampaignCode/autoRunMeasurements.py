# AUTORUNMEASUREMENTS.PY
#
# Load the commands for controlling servos for the antenna, move the antenna
# accordingly, and carry out a signal measurement whenever there is a "Q".
#
# Yaguang Zhang, Purdue University, 2017-06-13

import serial, time, os, sys
from lib import safelyOpenSerial, stripNewlines

'''
Custom variables.
'''
BAUD = 9600
PORT_X = 'COM13'
PORT_Z = 'COM14'
CMDS_FILE_NAME = 'Measurements_Cmds_LargeScale_Slower.txt' #'Measurements_Cmds_MIMO.txt' #'Measurements_Cmds_LargeScale_Slower.txt'
MAX_MOVING_TIME = 300 # In second.

'''
The Script.
'''
FLAG_DEBUG_X_Z_POS_SYS = False

print('Summary for autoRunMeasuremnts.py')
print('    BAUD: '+str(BAUD))
print('    PORT_X: '+PORT_X)
print('    PORT_Z: '+PORT_Z)
print('    MAX_MOVING_TIME: '+str(MAX_MOVING_TIME))
print('')

print('Summary for USRP')
sys.stdout.flush()
# Make sure GNURadio is available.
os.system('uhd_find_devices')
print('')

countdown = 5
print("Please terminate the operation (ctrl + c) if USRP isn't there!")
while countdown > 0:
    print('    Start measurements in ' + str(countdown) + 's...')
    time.sleep(1)
    countdown -= 1
print('')

# Timing the measurements.
tStart = time.time()

# Read in the commands.
print('Loading commands from .txt file: ' + CMDS_FILE_NAME +'...')
with open(os.path.join('Cmds', CMDS_FILE_NAME)) as f:
    cmds = f.readlines()
# Remove \n and \r (if there is any)
cmds = stripNewlines.stripStrArray(cmds)
print('Commands loaded!')
print('')

print('Opening RS-232 ports...')
# Open the RS-232 ports.
serX = safelyOpenSerial.openPort(PORT_X, BAUD, 1)
serZ = safelyOpenSerial.openPort(PORT_Z, BAUD, 1)
print('All ports opened!')
print('\r\n')

print('Starting auto-measurements...')
# Send the commands for the servos and run a signal measurement whenever there
# is a "Q". Note that a servo command will start with the name for the axis
# (i.e. X and Z) to which that command should be sent.
totalNumCmds = len(cmds)
for idx, cmd in enumerate(cmds):
    if cmd[0]=='X':
        serX.write((cmd[1:]+'\r\n').encode('ascii'))
        print(' - ('+str(idx)+'/'+str(totalNumCmds)+') Sent to X: ' + cmd[1:])
    elif cmd[0]=='Z':
        serZ.write((cmd[1:]+'\r\n').encode('ascii'))
        print(' + ('+str(idx)+'/'+str(totalNumCmds)+') Sent to Z: ' + cmd[1:])
    elif cmd=='Q':
        # Make sure motor for axis X is done before the measurement.
        print('------------------------------------')
        print('Checking whether X is done moving...')
        serX.write('SSDONE\r\n'.encode('ascii'))
        try:
            respX = stripNewlines.stripStr(serX.readline())
            print('    Initial read trial received: "' + respX + '"' )

            numReadsX = 1;
            while('DONE' not in respX and numReadsX<MAX_MOVING_TIME):
                respX = stripNewlines.stripStr(serX.readline())
                numReadsX += 1
                print('    #'+str(numReadsX)+' read trial received: "' \
                    + respX + '"' )

            print('X done moving!')
            print('------------------------------------')
        except Exception as e:
            print('Error: Not able to read from serX!')
            print(e)

        # Make sure motor for axis Z is done before the measurement.
        print('++++++++++++++++++++++++++++++++++++')
        print('Checking whether Z is done moving...')
        serZ.write('SSDONE\r\n'.encode('ascii'))
        try:
            respZ = stripNewlines.stripStr(serZ.readline())
            print('    Initial read trial received: "' + respZ + '"' )

            numReadsZ = 1;
            while('DONE' not in respZ and numReadsZ<MAX_MOVING_TIME):
                respZ = stripNewlines.stripStr(serZ.readline())
                numReadsZ += 1
                print('    #'+str(numReadsZ)+' read trial received: "' \
                    + respZ + '"' )

            print('Z done moving!')
            print('++++++++++++++++++++++++++++++++++++')
        except Exception as e:
            print('Error: Not able to read from serZ!')
            print(e)

        if (stripNewlines.stripStr(respX) =='DONE' and \
            stripNewlines.stripStr(respZ) =='DONE'):
            print('==============================')
            print('Initiating new measurement...')
            print('')
            print('    Time Stamp: ' + str(int(time.time())))
            print('')
            sys.stdout.flush()
            if (not FLAG_DEBUG_X_Z_POS_SYS):
                os.system('py -2.7 ./measureSignal.py')
            print('')
            print('Measurement done!')
            print('==============================')
        else:
            if (stripNewlines.stripStr(respX) !='DONE'):
                print('')
                print('Error: servo for axis X timed out!')
                print('    Servo for axis X is not ready as expected...')
                print('    Expecting "DONE" but received: ""' + respX + '"')
                print('')
            if (stripNewlines.stripStr(respZ) !='DONE'):
                print('')
                print('Error: servo for axis Z timed out!')
                print('    Servo for axis Z is not ready as expected...')
                print('    Expecting "DONE" but received: ""' + respX + '"')
                print('')

            serX.write('FP0\r\n'.encode('ascii'))
            serZ.write('FP0\r\n'.encode('ascii'))
            print('Homing commands sent to both servos.')
            print('Measurements terminated...')
            break
    else:
        print('')
        print('('+str(idx)+'/'+str(totalNumCmds)+') Ignored unknown command: ' + cmd)
        print('')

# Timing the measurements: show results.
tUsed = time.time() - tStart
print('')
print('Total time used for the measurements: ' + "{:.2f}".format(tUsed) + 's')
print('                                     =' + "{:.2f}".format(tUsed/60) + 'min')
# EOF