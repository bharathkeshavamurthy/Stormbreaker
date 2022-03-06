@ECHO OFF
(
    ECHO ---------------------------------------------------------------------------------------
    ECHO Initializing GPSDO... (Please make sure this script is run using GNURadio command line)
    ECHO ---------------------------------------------------------------------------------------
)
ECHO
ECHO +++ Start in 3 seconds...
timeout 3
ECHO
uhd_usrp_probe
ECHO
ECHO +++ Waiting for the cold start to complete for GPSDO...
timeout 50
ECHO
ECHO +++ Manual check for the GPSDO status:
py -2.7 verifyGpsdoLocked.py