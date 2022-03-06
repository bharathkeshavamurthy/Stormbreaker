/*
 Auto align the RX antenna via the pan and tilt kit controlled by Arduino via
 PWM signals.

 The program is controlled by character commands via the serial port to the
 Arduino:

  - r/R for starting auto alignment procedure after a successful initialization;
    a 'r' (indicating this script is running on the RX side) or 't' (indicating
    this script is running on the TX side) will be sent back from the Arduino as
    an acknowledgement.

  - s/S for stopping the program.

 We will collect sensor information and send it to the controller via the serial
 port. Debug information are prefixed by "#", while IMU readings are organized
 as a byte stream prefixed by "+":

  - upTime
  - quatReal, quatI, quatJ, quatK
  - quatRadianAccuracy
  - magX, magY, magZ
  - magAccuracy

 and GPS data are organized as a byte stream prefixed by "@":

  - Lat

 We used SparkFun Blackboard and Arduino IDE v1.8.1 to run this sketch. Required
 Arduino libraries include:

  - SparkFun BNO080 Cortex Based IMU
  - SparkFun Ublox

 Yaguang Zhang, Purdue University, 2020/01/28
*/

#include <Servo.h>

#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"
#include "SparkFun_Ublox_Arduino_Library.h"

// For enabling debugging over the serial port.
#define DEBUG false

// The PWM range for the continuous servos.
#define MIN_PWM 1000
#define MID_PWM 1500
#define MAX_PWM 2000

// For converting data to bytes for serial communication.
#define FLOAT_SIZE_IN_BYTE         4
#define UNSIGNED_LONG_SIZE_IN_BYTE 4
#define          LONG_SIZE_IN_BYTE 4
#define UNSIGNED_INT_SIZE_IN_BYTE  2
#define          INT_SIZE_IN_BYTE  2
#define BYTE_SIZE_IN_BYTE          1

// X axis - Tilt (changing elevation); Z axis - Pan (changing azimuth).
int wpmPinX = 9, wpmPinZ = 10;
Servo servoX, servoZ;

// Time to wait for sensors in millisecond.
int timeToWaitForSensorsInMs = 100;

// We use relatively low sample rates to limit the computation resource
// consumption of the Arduino board and the controller PC. However, note that if
// the GPS sensor is set to update too slowly, it may block the code for too long.
int imuPeriodInMs = 100; // IMU data update period in millisecond.
int gpsPeriodInMs = 100; // GPS data update period in millisecond.

// Communication parameters.
long serialBoundRate = 115200;
long i2cClockInHz    = 400000;

// For receiving command via the serial port from the controller.
char programCommand = 0;

// For orientation data from the VR IMU.
BNO080 vrImu;

// RTK GPS.
SFE_UBLOX_GPS rtkGps;

// For adjusting rotator servos.
String newPwmStr = "";
int newPwmValue = MID_PWM;

// For limiting effective time for each servo adjustment.
int maxServoAdjustmentFreqInHz = 10;
volatile unsigned long lastUpTimeInMsForServoAdjustment;
int minTimeInMsToWaitForServoAdjustment = 1000/maxServoAdjustmentFreqInHz;

void setup() {
  // Serial COM.
  Serial.begin(serialBoundRate);
  //Wait for the controller to open terminal.
  while (!Serial);

  // I2C.
  Wire.begin();
  Wire.setClock(i2cClockInHz);

  // IMU. Wait until the sensors are available.
  while (vrImu.begin() == false) {
    // Serial.println(F(
    //   "#Error: VR IMU (BNO080) not detected at default I2C address!"));
    // Serial.print(F("#    We will try again after "));
    // Serial.print(timeToWaitForSensorsInMs);
    // Serial.println(F(" ms..."));
    delay(timeToWaitForSensorsInMs);
  }
  vrImu.enableRotationVector(imuPeriodInMs);
  vrImu.enableMagnetometer(imuPeriodInMs);

  // RTK GPS.
  while (rtkGps.begin() == false) {
    // Serial.println(F(
    //   "#Error: RTK GPS (u-blox) not detected at default I2C address!"));
    // Serial.print(F("#    We will try again after "));
    // Serial.print(timeToWaitForSensorsInMs);
    // Serial.println(F(" ms..."));
    delay(timeToWaitForSensorsInMs);
  }
  // Set the I2C port to output UBX only (turn off NMEA noise).
  rtkGps.setI2COutput(COM_TYPE_UBX);
  rtkGps.setNavigationFrequency((int) (1000.0/gpsPeriodInMs));
  // Configure the GPS to update navigation reports automatically.
  rtkGps.setAutoPVT(true);
  rtkGps.saveConfiguration();

  // Motors.
  servoX.attach(wpmPinX);
  servoZ.attach(wpmPinZ);

  servoX.writeMicroseconds(MID_PWM);
  servoZ.writeMicroseconds(MID_PWM);
  // The servos were successfully initialized.
  lastUpTimeInMsForServoAdjustment = millis();

  // Limit the effective time of the lastest servo adjustment using a timer
  // interrupt. This is crucial because reading GPS data may block the code for
  // hundreds of milliseconds, which may cause too much rotation of the servo.
  // Timer0 is already used for millis() - we'll just interrupt somewhere in the
  // middle and call the "Compare A" function below
  OCR0A = 0xAF; // 127 is roughly the center of 0 to 255.
  TIMSK0 |= _BV(OCIE0A);

  // If there is no errors in the Arduino initialization, this will be the first
  // message to the controller.
  Serial.println(F("#Initialization succeeded! "
                   "Waiting for controllor's run command (r)..."));
  while(true) {
    if (Serial.available() > 0) {
      programCommand = toLowerCase(Serial.read());
      if (programCommand == 'r') {
        // Send 'r' to indicate this is a RX.
        Serial.println(F("r"));
        Serial.println(F("#Auto antenna alignment procedure started!"));
        break;
      }
    }
  }
}

// Interrupt is called once a millisecond, to check whether it is necessary to
// stop motors.
SIGNAL(TIMER0_COMPA_vect)
{
  // Stop servos if the last adjustment has lasted long enough time.
  if (lastUpTimeInMsForServoAdjustment>0) {
    if (millis()-lastUpTimeInMsForServoAdjustment
      >=minTimeInMsToWaitForServoAdjustment) {
        servoX.writeMicroseconds(MID_PWM);
        servoZ.writeMicroseconds(MID_PWM);
        // Set timer to 0 to avoid redundant servo checks.
        lastUpTimeInMsForServoAdjustment = 0;
    }
  }
}

void loop() {
  // React to the command from the serial port with the highest priority.
  while (Serial.available() > 0) {
    programCommand = toLowerCase(Serial.read());

    switch (programCommand) {
      case 's':
        printCommand(programCommand);
        servoX.writeMicroseconds(MID_PWM);
        servoZ.writeMicroseconds(MID_PWM);
        Serial.println(F("#Freezing..."));
        while(true);
        break;
      case 'x':
        newPwmStr = Serial.readStringUntil('\n');
        newPwmValue = newPwmStr.toInt();

        servoX.writeMicroseconds(newPwmValue);
        lastUpTimeInMsForServoAdjustment = millis();

        // Debug info.
        if (DEBUG) {
          Serial.print(F("#Command received: "));
          Serial.print(programCommand);
          Serial.println(newPwmStr);
          Serial.print(F("#Setting PWM for X-axis servo to "));
          Serial.println(newPwmValue);
        }
        break;
      case 'z':
        newPwmStr = Serial.readStringUntil('\n');
        newPwmValue = newPwmStr.toInt();

        servoZ.writeMicroseconds(newPwmValue);
        lastUpTimeInMsForServoAdjustment = millis();

        // Debug info.
        if (DEBUG) {
          Serial.print(F("#Command received: "));
          Serial.print(programCommand);
          Serial.println(newPwmStr);
          Serial.print(F("#Setting PWM for Z-axis servo to "));
          Serial.println(newPwmValue);
        }
        break;
      default:
        Serial.print(F("#Unkown command received: "));
        Serial.println(programCommand);
        break;
    }
  }

  // Read the GPS data. Reference for data types:
  //   - long <=> int32_t
  //   - int  <=> int16_t
  //   - byte <=> int8_t
  if (rtkGps.getPVT() == true) {
    // Arduino up time.
    unsigned long upTimeInMs = millis();

    // Fetch high-precision GPS data.
    unsigned long timeOfWeekInMs = rtkGps.getTimeOfWeek();

    // Naming convention: [a]In[U]Xe[N] equals to [a]In[U] times 10^N, where [a]
    // is the variable name and [U] is the unit name.
    long latInDegXe7 = rtkGps.getHighResLatitude();
    long lonInDegXe7 = rtkGps.getHighResLongitude();
    long altInMmMeanSeaLevel = rtkGps.getMeanSeaLevel();
    long altInMmEllipsoid = rtkGps.getElipsoid();

    unsigned long horAccuracyInMXe4 = rtkGps.getHorizontalAccuracy();
    unsigned long verAccuracyInMXe4 = rtkGps.getVerticalAccuracy();

    // Extra information.
    byte satsInView = rtkGps.getSIV();
    byte fixType    = rtkGps.getFixType();

    unsigned int year = rtkGps.getYear();
    byte month        = rtkGps.getMonth();
    byte day          = rtkGps.getDay();
    byte hour   = rtkGps.getHour();
    byte minute = rtkGps.getMinute();
    byte second = rtkGps.getSecond();
    unsigned int millisecond = rtkGps.getMillisecond();
    // This value includes millisecond and can be negative.
    long nanosecond          = rtkGps.getNanosecond();

    long speedInMPerSX3  = rtkGps.getGroundSpeed();
    long headingInDegXe5 = rtkGps.getHeading();
    // Positional dilution of precision.
    unsigned int PDODXe2 = rtkGps.getPDOP();

    // Debug info.
    if (DEBUG) {
      Serial.print(F("#Up time: "));
      Serial.print(upTimeInMs);
      Serial.println(F(" ms"));

      Serial.print(F("#GPS Time of the week: "));
      Serial.print(timeOfWeekInMs);
      Serial.println(F(" ms"));

      Serial.print(F("#GPS (latInDegXe7, lonInDegXe7): ("));
      Serial.print(latInDegXe7);
      Serial.print(F(","));
      Serial.print(lonInDegXe7);
      Serial.println(F(")"));

      Serial.print(F("#GPS (altInMmMeanSeaLevel, altInMmEllipsoid): ("));
      Serial.print(altInMmMeanSeaLevel);
      Serial.print(F(","));
      Serial.print(altInMmEllipsoid);
      Serial.println(F(")"));

      Serial.print(F("#GPS (horAccuracyInMXe4, verAccuracyInMXe4): ("));
      Serial.print(horAccuracyInMXe4);
      Serial.print(F(","));
      Serial.print(verAccuracyInMXe4);
      Serial.println(F(")"));

      Serial.print(F("#GPS satellites in view: "));
      Serial.println(satsInView);

      Serial.print(F("#GPS fixed type: "));
      Serial.println(fixType);
      Serial.println(F("#    0: no fix"));
      Serial.println(F("#    1: dead reckoning only"));
      Serial.println(F("#    2: 2D-fix"));
      Serial.println(F("#    3: 3D-fix"));
      Serial.println(F("#    4: GNSS + dead reckoning combined"));
      Serial.println(F("#    5: time only fix"));

      Serial.print(F("#Human-readable GPS time: "));
      Serial.print(year);
      Serial.print(F("-"));
      Serial.print(month);
      Serial.print(F("-"));
      Serial.print(day);
      Serial.print(F(" "));
      Serial.print(hour);
      Serial.print(F(":"));
      Serial.print(minute);
      Serial.print(F(":"));
      Serial.print(second);
      Serial.print(F("."));
      //Pretty print leading zeros.
      if (millisecond < 100) Serial.print(F("0"));
      if (millisecond < 10)  Serial.print(F("0"));
      Serial.println(millisecond);

      Serial.print(F("#GPS nano seconds: "));
      Serial.println(nanosecond);

      Serial.print(F("#GPS speed: "));
      Serial.print(speedInMPerSX3);
      Serial.println(F(" (mm/s)"));

      Serial.print(F("#GPS heading: "));
      Serial.print(headingInDegXe5);
      Serial.println(F(" (degrees * 10^-5)"));

      Serial.print(F("#GPS positional dilution of precision: "));
      Serial.println(PDODXe2/100.0, 2);
    }

    // Send data over serial.
    Serial.print(F("@"));
    sendUnsignedLong(upTimeInMs);
    sendUnsignedLong(timeOfWeekInMs);
    sendLong(latInDegXe7);
    sendLong(lonInDegXe7);
    sendLong(altInMmMeanSeaLevel);
    sendLong(altInMmEllipsoid);
    sendUnsignedLong(horAccuracyInMXe4);
    sendUnsignedLong(verAccuracyInMXe4);
    sendByte(satsInView);
    sendByte(fixType);
    sendUnsignedInt(year);
    sendByte(month);
    sendByte(day);
    sendByte(hour);
    sendByte(minute);
    sendByte(second);
    sendUnsignedInt(millisecond);
    sendLong(nanosecond);
    sendLong(speedInMPerSX3);
    sendLong(headingInDegXe5);
    sendUnsignedInt(PDODXe2);

    // End of package.
    Serial.println();
  }
  // Read the IMU data.
  else if (vrImu.dataAvailable() == true) {
    // Arduino up time.
    unsigned long upTimeInMs = millis();

    // Fetch IMU data.
    float quatReal = vrImu.getQuatReal();
    float quatI = vrImu.getQuatI();
    float quatJ = vrImu.getQuatJ();
    float quatK = vrImu.getQuatK();
    float quatRadianAccuracy = vrImu.getQuatRadianAccuracy();

    float magX = vrImu.getMagX();
    float magY = vrImu.getMagY();
    float magZ = vrImu.getMagZ();
    byte magAccuracy = vrImu.getMagAccuracy();

    // Debug info.
    if (DEBUG) {
      Serial.print(F("#Up time: "));
      Serial.print(upTimeInMs);
      Serial.println(F(" ms"));

      Serial.print(F("#Rotation vector: "));
      Serial.print(quatReal, 2);
      Serial.print(F(","));
      Serial.print(quatI, 2);
      Serial.print(F(","));
      Serial.print(quatJ, 2);
      Serial.print(F(","));
      Serial.print(quatK, 2);
      Serial.print(F(","));
      Serial.print(quatRadianAccuracy, 2);

      Serial.println();

      Serial.print(F("#Magnetometer: "));
      Serial.print(magX, 2);
      Serial.print(F(","));
      Serial.print(magY, 2);
      Serial.print(F(","));
      Serial.print(magZ, 2);
      Serial.print(F(","));
      printAccuracyLevel(magAccuracy);

      Serial.println();
    }

    // Send data over serial.
    Serial.print(F("+"));
    sendUnsignedLong(upTimeInMs);
    sendFloat(quatReal);
    sendFloat(quatI);
    sendFloat(quatJ);
    sendFloat(quatK);
    sendFloat(quatRadianAccuracy);
    sendFloat(magX);
    sendFloat(magY);
    sendFloat(magZ);
    printAccuracyLevel(magAccuracy);

    // End of package.
    Serial.println();
  }
}

void printCommand (char command) {
  Serial.print(F("#Command received: "));
  Serial.println(command);
}

// Given an accuracy number, print to serial what it means.
void printAccuracyLevel(byte accuracyNumber)
{
  if(accuracyNumber == 0) Serial.print(F("U"));       // Unreliable
  else if(accuracyNumber == 1) Serial.print(F("L"));  // Low
  else if(accuracyNumber == 2) Serial.print(F("M"));  // Medium
  else if(accuracyNumber == 3) Serial.print(F("H"));  // High
}

void sendFloat (float arg)
{
  // Get access to the float as a byte-array and write the data to the serial.
  Serial.write((byte *) &arg, FLOAT_SIZE_IN_BYTE);
}

void sendUnsignedLong (unsigned long arg)
{
  // Get access to the unsigned long as a byte-array and write the data to the
  // serial.
  Serial.write((byte *) &arg, UNSIGNED_LONG_SIZE_IN_BYTE);
}

void sendLong (long arg)
{
  // Get access to the long as a byte-array and write the data to the serial.
  Serial.write((byte *) &arg, LONG_SIZE_IN_BYTE);
}

void sendUnsignedInt (unsigned int arg)
{
  // Get access to the unsigned long as a byte-array and write the data to the
  // serial.
  Serial.write((byte *) &arg, UNSIGNED_INT_SIZE_IN_BYTE);
}

void sendUnsignedInt (int arg)
{
  // Get access to the unsigned long as a byte-array and write the data to the
  // serial.
  Serial.write((byte *) &arg, INT_SIZE_IN_BYTE);
}

void sendByte (byte arg)
{
  // Get access to the byte as a byte-array and write the data to the serial.
  Serial.write((byte *) &arg, BYTE_SIZE_IN_BYTE);
}