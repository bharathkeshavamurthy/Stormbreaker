# YAM3S

The code base for Yet Another Mobile Millimeter-wave Measurement System (YAM3S). YAM3S combines real-time kinematic (RTK) GPS, inertial measurement units (IMUs) with 2-axis rotators to automatically align the transmitter (TX) and receiver (RX) antennas. It is designed for a mobile RX with the TX installed at a fixed location.

## Checklist

Assuming that `settings.json` is filled out correctly, then for each measurement
recording, the steps one needs to carry out are listed below.

1. Start the remote mySQL server.
2. Manually adjust the TX and RX servos so that the antennas are aligned with
   enough moving/rotating room for the rotators to cover the area of interest.
3. Measure or find on a map the TX and RX GPS locations and update the
   `settings.json` file accordingly.
4. If desired, assign a string label to the measurement recording attempt to
   `measurement_attempt_label_prefix` in `settings.json`, which will be combined with the date and time when the measurement starts for labeling that recording activity.
5. On the RX side, initiated the CKT-GPS (Ref: <https://learn.sparkfun.com/tutorials/gps-rtk2-hookup-guide>)
   1. connect to the CKT-GPS board via USB
   2. open u-center and connect to the correct COM port
   3. adjust the baud rate of UART2 to 115200 (for streaming RTCM correction data via Bluetooth Mate)
   4. start correction data streaming on the phone using the NTRIP app
6. On both the TX and the RX sides,
   1. calibrate the IMU module according to <https://cdn.sparkfun.com/assets/c/6/f/4/9/Sensor-Calibration-Procedure-v1.1.pdf>
   2. start (i) Arduino, and (ii) the Python controller on the connect computer.

## Contact

* **Yaguang Zhang** | *Purdue University* | Email: ygzhang@purdue.edu

## License

This project is licensed under the MIT License.
