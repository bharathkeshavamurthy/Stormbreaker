# Ears Measurement Campaign Code

Code for automating the measurement campaign of Project EARS and post-processing the data collected.

## Measurement Setup

A Universal Software Radio Peripheral (USRP) B200 was utilized to record the signal. It also regularly sampled the RX’s location with the help of an on-board GPS disciplined oscillator (TCXO version).

More details can be found at [28-GHz Channel Measurements and Modeling for Suburban Environments](https://docs.lib.purdue.edu/ecetr/483/).

## Signal Recording

For controlling the USRP to carry out signal recordings, one will need:

  - GNU Radio Companion
  - Python

**Note:** The Python commands need to be run in via *GNURadio Command Prompt* to properly configure the USRP via GNU Radio, if the operating system is Windows.

### "+" Pattern Measurements for a Site

To run the "+" pattern measurements (3-second signal recording + 1 GPS sample for each RX antenna location):

```
python autoRunMeasurements.py
```

This will take care of the motors for moving the RX antenna (controlled by one of the command files under `Cmds`), firing up and stopping the USRP signal recording, and sample the GPS location once for each antenna location.

**Note:** All the out put files will be saved under "measureSignalOutput"; One may want to manually move and organize the data files collected regularly.

### Measurement for One Antenna Location

To manually carry out the measurement for only one antenna location (3-second signal recording + 1 GPS sample):

```
python measureSignal.py
```

### Continuous Signal Recording with GPS Samples

To run a continuous signal recording with regular GPS samples (1 sample per second):

```
python measureSignalCont.py
```

### To Inspect a GNU Radio `.out` File

One can load `measureSignalReplay.grc` into GNU Radio Companion and set the file source to a `.out` file to replay the signal recorded.

## Post Processing

Matlab is used for post-processing the data we collected in a measurement campaign on campus of the United States Naval Academy in 2017. All the codes for that are (and will be) stored under `PostProcssing`.

These codes are still under development, but they are organized by their functions and the proper order of running them, with a decent amount of comments added for each script and function. One should be able to process their data with only a few tweaks using these scripts if the data are collected using our Python code and if the data are organized properly. Some illustrations for the way we organized the data can be found under `measureSignalOutput/examples`.

Please contact us if you need more information.

## Contact

* **Yaguang Zhang** | *Purdue University* | Email: ygzhang@purdue.edu

## License

This project is licensed under the MIT License - please see the [LICENSE](LICENSE) file for details.