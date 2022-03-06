/*
 * Eitri: The metal worker in Project Odin.
 * An Arduino sketch to perform the following tasks: 
 * 1. Publish the GPS readings received from the SparkFun UBlox ZED-F9P RTK-GPS module;
 * 2. Receive the required angles of rotation (yaw and pitch) from the XXRealm Python Controller over Bluetooth/USB; and
 * 3. Yaw and Pitch rotations using the SparkFun Blackboard Rev C, IMU BNO080, and ServoCity continuous servos -- along with IMU logging.
 * 
 * Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
 * Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
 * Copyright (c) 2021. All Rights Reserved.
*/

/* Pre-processor Directives */
#include <Wire.h>
#include <Servo.h>
#include "SparkFun_Ublox_Arduino_Library.h"
#include "SparkFun_BNO080_Arduino_Library.h"

/* Rotation Control Logic Definitions */
#define STEADY_STATE 125                  // Avoid "nan" yaw and pitch values by forcing steady-state reads...
#define READ_EQUILIBRIUM 25               // The number of samples to be read before we decide to use the IMU report
#define YAW_OFFSET 0.5                    // The precision of Z-axis rotations evaluated during pre-trial manual test-runs
#define PITCH_OFFSET 0.5                  // The precision of Z-axis rotations evaluated during pre-trial manual test-runs
#define PRECISION 2.5                     // If the absolute difference between the current angle (yaw/pitch) and mandated angle (yaw/pitch) is less than 2.5 degrees, don't try to meet it. Stay where you are. 
#define YAW_RESTRICTION 360.0             // A constraint (in degrees) on the maximum allowable yaw rotation in order to prevent the wires from getting tangled-up
#define PITCH_RESTRICTION 90.0            // A constraint (in degrees) on the maximum allowable pitch rotation in order to prevent the wires from getting tangled-up

/* The PWM ON pulse durations for controlling the continuous servos */
#define MAX_PWM 2000                      // Forward Motion (PWM ON pulse duration)
#define MID_PWM 1500                      // Stop Motion (PWM ON pulse duration)
#define MIN_PWM 1000                      // Backward Motion (PWM ON pulse duration)

/* IMU Instance Creation
 * Servo control pin definitions | Servo (X and Z) instance creation
 * X = Yaw movements | Z = Pitch & Roll movements | For pitch movements, place the object being rotated perpendicular to the pan plate -- in the X-Z plane | Manual height adjustments
*/
BNO080 bno080;
Servo servoX, servoZ;
int pinX = 9, pinZ = 10;

/* UBlox ZED-F9P Instance Creation */
SFE_UBLOX_GPS ublox;

/* Core Rotational Variables */
bool motorsInitiated = false;                                                       // A flag to indicate if the motors have been initiated or not...
float yaw = 0.0, prevYaw = 0.0, pitch = 0.0, prevPitch = 0.0;                       // The estimated yaw angle and its previous time-step counterpart | The estimated pitch angle and its previous time-step counterpart
float requiredYawAngle = 0.0, currentYawAngle = 0.0;                                // The core yaw axis variables facilitating motor control
float requiredPitchAngle = 0.0, currentPitchAngle = 0.0;                            // The core pitch axis variables facilitating motor control
float yawReconfigRequirement = 0.0, pitchReconfigRequirement = 0.0;                 // The reconfigured mandates in terms of the amount of angle that needs to be traversed -- either in the anti-clockwise direction or in the clockwise direction
float aclkYawRotation = 0.0, clkYawRotation = 0.0;                                  // The movement variables that encapsulate the amount of yaw rotation in the anti-clockwise & clockwise directions, respectively
float aclkPitchRotation = 0.0, clkPitchRotation = 0.0;                              // The movement variables that encapsulate the amount of pitch rotation in the anti-clockwise & clockwise directions, respectively
bool proceedToPitch = false, mandateAchieved = true, imuCalibrationDone = false;    // A flag to give the go-ahead to move on to pitch changes | A flag to indicate that the overall rotational mandate from the XXRealm Python Controller has been met | A flag to indicate IMU calibration status
unsigned int sampleCounter = 0, steadyStateCounter = 0;                             // A counter for ensuring that the IMU values are read post-equilibrium | A counter to force steady state reads

/* Core GPS Variables */
long prevTime = 0;                                                                  // A previous time member in order to limit the amount of traffic coming in from the UBlox ZED-F9P RTK-GPS module
bool gpsConnected = true;                                                           // A flag to indicate whether the ZED-F9P RTK-GPS module is connected to the uC-IMU setup
int32_t latitudeM = 0, longitudeM, ellipsoidM = 0, mslM = 0;                        // The main component of the latitude from HPPOSLLH as an int32_t in degrees * 10^-7 | The main component of the longitude from HPPOSLLH as an int32_t in degrees * 10^-7 | The main component of the height w.r.t an ellipsoid model as an int32_t in mm | The main component of the height above sea level as an int32_t in mm
int8_t latitudeHp = 0, longitudeHp = 0, ellipsoidHp = 0, mslHp = 0;                 // The high resolution component of latitude from HPPOSLLH as an int8_t in degrees * 10^-9 | The high resolution component of longitude from HPPOSLLH as an int8_t in degrees * 10^-9 | The high resolution component of the height w.r.t an ellipsoid model as an int8_t in mm * 10^-1 | The high resolution component of the height above sea level as an int8_t in mm * 10^-1
double d_lat = 0.0, d_lon = 0.0;                                                    // The assembled latitude component | The assembled longitude component
float f_ellipsoid = 0.0, f_msl = 0.0;                                               // The assembled height w.r.t an "ellipsoid model" component | The assembled height w.r.t a "mean sea level model" component
uint32_t h_accuracy = 0, v_accuracy = 0;                                            // The raw horizontal accuracy component | The raw vertical accuracy component
float horizontal_accuracy = 0.0, vertical_accuracy = 0.0;                           // The assembled horizontal accuracy component | The assembled vertical accuracy component

/* The setup routine */
void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("[INFO] || Project Odin || Eitri || Arduino sketch execution: GPS Publish | Angle Subscribe | Yaw and Pitch Rotations");

  if (sizeof(double) < 8) {                                                         // Check that this platform supports 64-bit (8 byte) double
    Serial.println("[WARN] || Project Odin || Eitri || The platform does not support 64-bit double | The latitude and longitude will be inaccurate.");
  }
  
  Wire.begin();                                                                     // Begin the Wire instance to start reading the I2C interface...
  
  if (!bno080.begin()) {                                                            // Check for the IMU at the I2C interface
    Serial.println("[ERROR] || Project Odin || Eitri || BNO080 IMU not connected");
    return;
  }
  Wire.setClock(400000);                                                            // I2C Data Rate = 400kHz
  bno080.enableRotationVector(100);                                                 // IMU Refresh at 100ms

  if (ublox.begin(Wire) == false) {                                                 // Check to see if the ZED-F9P RTK-GPS module is connected -- if not, default to the pre-defined values
    Serial.println("[WARN] || Project Odin || Eitri || ZED-F9P RTK-GPS module not connected | Post pre-defined defaults to the XXRealm Python Controller");
    gpsConnected = false;
  }
  ublox.setI2COutput(COM_TYPE_UBX);                                                 // Set the I2C port to output UBX only (turn off NMEA noise)

  servoX.attach(pinX);                                                              // Attach to servoX | Reset to zero motion | Yaw Servo
  servoX.writeMicroseconds(MID_PWM);
 
  servoZ.attach(pinZ);                                                              // Attach to servoZ | Reset to zero motion | Pitch Servo
  servoZ.writeMicroseconds(MID_PWM);
}

/* The loop routine */
void loop() {
  /* Calibration save (manual interrupt based save to Flash): DCD save to the Flash outside the general auto-calibration & auto-recording done by the IMU every 300s */
  if ( Serial.available() && (imuCalibrationDone == false) ) {
    if (Serial.read() == 'S') {
      bno080.saveCalibration();                                                             // Save the current DCD to the Flash
      bno080.requestCalibrationStatus();
      for (int i = 0; i < 100; i++) {
        if (i > 99) {
          Serial.println("[WARN] || Project Odin || Eitri || Flash Save Failed | Retry");   // Calibration Failed
          break;
        } else {
          if (bno080.dataAvailable() && bno080.calibrationComplete()) {                     // Calibration successful
            Serial.println("[INFO] || Project Odin || Eitri || Calibration & Flash Save Successful");
            delay(1000);
            imuCalibrationDone = true;
            break;
          }
          delay(1);
        }
      }
    }
  }

  /* Query module only every second | The ZED-F9P RTK-GPS module only responds when a new position is available
   *  SparkFun UBlox GPS Library API calls:
   *    getHighResLatitude: Returns the latitude from HPPOSLLH as an int32_t in degrees * 10^-7;
   *    getHighResLatitudeHp: Returns the high resolution component of latitude from HPPOSLLH as an int8_t in degrees * 10^-9;
   *    getHighResLongitude: Returns the longitude from HPPOSLLH as an int32_t in degrees * 10^-7;
   *    getHighResLongitudeHp: Returns the high resolution component of longitude from HPPOSLLH as an int8_t in degrees * 10^-9;
   *    getElipsoid: Returns the height above ellipsoid as an int32_t in mm;
   *    getElipsoidHp: Returns the high resolution component of the height above ellipsoid as an int8_t in mm * 10^-1;
   *    getMeanSeaLevel: Returns the height above mean sea level as an int32_t in mm;
   *    getMeanSeaLevelHp: Returns the high resolution component of the height above mean sea level as an int8_t in mm * 10^-1; and
   *    getHorizontalAccuracy: Returns the horizontal accuracy estimate from HPPOSLLH as an uint32_t in mm * 10^-1. These are a few relevant calls in this GPS API.
   *    
   *    TODO (v21.06): CHANGE DEFAULT VALUES HERE IN CASE THIS uC-BASED CONTROLLER DOES NOT HAVE A GPS MODULE
  */
  if ( Serial.available() && ( (millis() - prevTime) > 1000 ) ) {
    prevTime = millis();
    
    Serial.print("$");
    Serial.print(gpsConnected ? ublox.getGnssFixOk() : 1);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getSIV() : 25);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getFixType() : 3);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getCarrierSolutionType() : 1);
    Serial.print(",");
    
    latitudeM = ublox.getHighResLatitude();
    Serial.print(gpsConnected ? latitudeM : 404288533);
    Serial.print(",");
    latitudeHp = ublox.getHighResLatitudeHp();
    Serial.print(gpsConnected ? latitudeHp : -18);
    Serial.print(",");
    d_lat = ( (double)latitudeM ) / 10000000.0;                       // Convert latitude from (degrees * 10^-7) to degrees
    d_lat += ( (double)latitudeHp ) / 1000000000.0;                   // Add the high resolution component (degrees * 10^-9)
    if (gpsConnected) {
      Serial.print(d_lat, 9);
    } else {
      Serial.print(40.428855895);
    }
    Serial.print(",");
    
    longitudeM = ublox.getHighResLongitude();
    Serial.print(gpsConnected ? longitudeM : -868999516);
    Serial.print(",");
    longitudeHp = ublox.getHighResLongitudeHp();
    Serial.print(gpsConnected ? longitudeHp : 18);
    Serial.print(",");
    d_lon = ( (double)longitudeM ) / 10000000.0;                      // Convert longitude from degrees * 10^-7 to degrees
    d_lon += ( (double)longitudeHp ) / 1000000000.0;                  // Add the high resolution component (degrees * 10^-9)
    if (gpsConnected) {
      Serial.print(d_lon, 9);
    } else {
      Serial.print(-86.899948120);
    }
    Serial.print(",");
    
    ellipsoidM = ublox.getElipsoid();
    Serial.print(gpsConnected ? ellipsoidM : 124489);
    Serial.print(",");
    ellipsoidHp = ublox.getElipsoidHp();
    Serial.print(gpsConnected ? ellipsoidHp : -4);
    Serial.print(",");
    f_ellipsoid = (ellipsoidM * 10) + ellipsoidHp;                   // Calculate the height above ellipsoid in mm * 10^-1
    f_ellipsoid = f_ellipsoid / 10000.0;                             // Convert to m: Convert from mm * 10^-1 to m
    if (gpsConnected) {
      Serial.print(f_ellipsoid, 4);
    } else {
      Serial.print(124.4886);
    }
    Serial.print(",");
    
    mslM = ublox.getMeanSeaLevel();
    Serial.print(gpsConnected ? mslM : 158030.06);
    Serial.print(",");
    mslHp = ublox.getMeanSeaLevelHp();
    Serial.print(gpsConnected ? mslHp : 0);
    Serial.print(",");
    f_msl = (mslM * 10) + mslHp;                                    // Calculate the height above mean sea level in mm * 10^-1
    f_msl = f_msl / 10000.0;                                        // Convert to m: Convert from mm * 10^-1 to m
    if (gpsConnected) {
      Serial.print(f_msl, 4);
    } else {
      Serial.print(158.0301);
    }
    Serial.print(",");
    
    Serial.print(gpsConnected ? ublox.getGroundSpeed() : 16805864);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getHeading() : 1660);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getHorizontalAccEst() : 2725);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getVerticalAccEst() : 204);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getSpeedAccEst() : 2187329);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getHeadingAccEst() : -1);
    Serial.print(",");
    
    Serial.print(gpsConnected ? ublox.getNedNorthVel() : 20);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getNedEastVel() : -7);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getNedDownVel() : 118);
    Serial.print(",");
    
    Serial.print(gpsConnected ? ublox.getPDOP() : 0);
    Serial.print(",");
    
    Serial.print(gpsConnected ? ublox.getMagAcc() : 0);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getMagDec() : 135);
    Serial.print(",");
    
    Serial.print(gpsConnected ? ublox.getGeometricDOP() : 118);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getPositionDOP() : 65);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getTimeDOP() : 55);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getHorizontalDOP() : 105);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getVerticalDOP() : 42);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getNorthingDOP() : 36);
    Serial.print(",");
    Serial.print(gpsConnected ? ublox.getEastingDOP() : 36);
    Serial.print(",");
    
    h_accuracy = ublox.getHorizontalAccuracy();
    horizontal_accuracy = h_accuracy / 10000.0;
    if (gpsConnected) {
      Serial.print(horizontal_accuracy, 4);
    } else {
      Serial.print(1.6578);
    }
    Serial.print(",");
    v_accuracy = ublox.getVerticalAccuracy();
    vertical_accuracy = v_accuracy / 10000.0;
    if (gpsConnected) {
      Serial.print(vertical_accuracy, 4);
    } else {
      Serial.print(2.7278);
    }
    Serial.print("\n");
  }
  
  /* 
   * Data extraction and logging:
   * 1. Read the Quaternion Complex Vector values [i, j, k, real] and Normalize them;
   * 2. Parse Input Report from BNO080 --> Bosch Sensor Hub Transport Protocol (SHTP) over I2C --> Parse raw quaternion values if the channel is CHANNEL_GYRO in the SHTP Header OR if the SHTP Payload is referenced by SENSOR_*_ROTATION_VECTOR; and
   * 3. Use the BNO080 Library for this | Modify the C++ code if needed with the Q-values | https://github.com/fm4dd/pi-bno080 | https://github.com/sparkfun/SparkFun_BNO080_Arduino_Library/blob/master/src/SparkFun_BNO080_Arduino_Library.cpp.
  */
  if (bno080.dataAvailable()) {
    float quatI = bno080.getQuatI();
    float quatJ = bno080.getQuatJ();
    float quatK = bno080.getQuatK();
    float quatReal = bno080.getQuatReal();
    float norm = sqrt( (quatI * quatI) + (quatJ * quatJ) + (quatK * quatK) + (quatReal * quatReal) );
    quatI /= norm;
    quatJ /= norm;
    quatK /= norm;
    quatReal /= norm;
    
    yaw = atan2( (2.0 * ( (quatReal * quatK) + (quatI * quatJ) ) ), ( 1.0 - ( 2.0 * ( (quatJ * quatJ) + (quatK * quatK) ) ) ) ) * (180 / PI);      // Determine the Yaw Angle (in degrees)
    
    float temp = 2.0 * ( (quatReal * quatJ) - (quatK * quatI) );                                                                                   // Sine Inverse Range Restrictions [-1.0, 1.0]
    temp = (temp > 1.0) ? 1.0 : temp;
    temp = (temp < -1.0) ? -1.0 : temp;
    pitch = asin(temp) * (180 / PI);                                                                                                               // Determine the Pitch Angle (in degrees)

    sampleCounter += 1;                                                                                                                            // Update the sample counter
    steadyStateCounter += 1;                                                                                                                       // Update the steady state counter
  }
  
  if (steadyStateCounter < STEADY_STATE) {
    return;
  }
  
  /* Motor Control */
  if ( Serial.available() && mandateAchieved && (sampleCounter >= READ_EQUILIBRIUM) ) {
    String message = Serial.readString();
    if ( message.startsWith("^") && message.endsWith("\n") && (message.indexOf("#") != -1) ) {         // Custom Odin IMUTrace format: Messages begin with '^'
      String yawAngleString = message.substring(1, message.indexOf("#"));                              // ^YAW_ANGLE#PITCH_ANGLE<LF>
      String pitchAngleString = message.substring(message.indexOf("#") + 1, message.length() - 1);     // ^YAW_ANGLE#PITCH_ANGLE<LF>
      if ( (yawAngleString == "") || (pitchAngleString == "") ) {
        return;
      }
      requiredYawAngle = yawAngleString.toFloat();
      requiredPitchAngle = pitchAngleString.toFloat();
      if ( (requiredYawAngle < 0.0) || (requiredYawAngle >= YAW_RESTRICTION) || ( requiredPitchAngle <= (-1 * PITCH_RESTRICTION) ) || (requiredPitchAngle >= PITCH_RESTRICTION) ) {
        Serial.println("[ERROR] || Project Odin || Eitri || Invalid NMEA message received from the XXRealm Python Controller over Serial: Either the structure or the data is invalid/out-of-bounds.");
        return;
      }
    }
    sampleCounter = 0;                                                 // Start fresh with new samples in order to achieve READ_EQUILIBRIUM
    mandateAchieved = false;                                           // Get ready to start working in order to meet the provided mandate
    prevYaw = yaw;                                                     // Raw, unfiltered previous yaw store
    prevPitch = pitch;                                                 // Raw, unfiltered previous pitch store
  }

  /* Yaw Rotation Control */
  if ( (mandateAchieved == false) && (proceedToPitch == false) && (sampleCounter >= READ_EQUILIBRIUM) ) {
    currentYawAngle = (yaw < 0.0) ? abs(360.0 - yaw) : yaw;
    
    if (motorsInitiated == false) {
      if ( (abs(requiredYawAngle - currentYawAngle) <= PRECISION) ) {                                                    // The platform has almost (already) satisfied the provided Yaw Mandate
         Serial.println("[INFO] || Project Odin || Eitri || The platform has already satisfied the provided Yaw Mandate.");
         proceedToPitch = true;
         sampleCounter = 0;
         return;
      }
      Serial.println("[INFO] || Project Odin || Eitri || Yaw Rotation Mandate Received | Achieving Mandate...");
      if (currentYawAngle <= requiredYawAngle) {                                                                         // Variant-1
        aclkYawRotation = requiredYawAngle - currentYawAngle;
        clkYawRotation = 360.0 - (requiredYawAngle - currentYawAngle);
      } else {                                                                                                           // Variant-2
        clkYawRotation = currentYawAngle - requiredYawAngle;
        aclkYawRotation = 360.0 - (currentYawAngle - requiredYawAngle);
      }
      yawReconfigRequirement = (aclkYawRotation <= clkYawRotation) ? aclkYawRotation : clkYawRotation;
      servoX.writeMicroseconds( (aclkYawRotation <= clkYawRotation) ? MAX_PWM : MIN_PWM );                               // Use the optimal rotation direction (a simple comparison-based decision heuristic)
      motorsInitiated = true;
    }

    if ( abs(yaw - prevYaw) >= yawReconfigRequirement ) {                                                                // Stop, once the mandate has been met -- and, move on to pitch adjustments...
      servoX.writeMicroseconds(MID_PWM);
      proceedToPitch = true;
      motorsInitiated = false;
    }

    sampleCounter = 0; 
  }

  /* Pitch Rotation Control */
  if ( (mandateAchieved == false) && (proceedToPitch == true) && (sampleCounter >= READ_EQUILIBRIUM) ) {
    currentPitchAngle = pitch;

    if (motorsInitiated == false) {
      if ( abs(requiredPitchAngle - currentPitchAngle) <= PRECISION ) {                                                  // The platform has almost (already) satisfied the provided Pitch Mandate
         Serial.println("[INFO] || Project Odin || Eitri || The platform has already satisfied the provided Pitch Mandate.");
         proceedToPitch = false;
         mandateAchieved = true;
         sampleCounter = 0;
         return;
      }
      Serial.println("[INFO] || Project Odin || Eitri || Pitch Rotation Mandate Received | Achieving Mandate...");
      pitchReconfigRequirement = abs(currentPitchAngle - requiredPitchAngle);
      servoZ.writeMicroseconds( (currentPitchAngle < requiredPitchAngle) ? MAX_PWM  : MIN_PWM );                        // Use the optimal rotation direction (a simple comparison-based decision heuristic)
      motorsInitiated = true;
    }
    
    Serial.print("[INFO] || Project Odin || Eitri || Pitch Rotation Mandate Received | Achieving Mandate...");

    if ( abs(pitch - prevPitch) >=  pitchReconfigRequirement ) {                                                        // Stop, once the mandate has been met -- and, move on to the next mandate (if one exists).
      servoZ.writeMicroseconds(MID_PWM);
      proceedToPitch = false;
      motorsInitiated = false;
      mandateAchieved = true;
    }

    sampleCounter = 0;
  }
}
