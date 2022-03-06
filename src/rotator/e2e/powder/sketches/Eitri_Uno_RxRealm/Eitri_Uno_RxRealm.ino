/*
 * Eitri: The metal worker in Project Odin.
 * An Arduino sketch to perform the following tasks: 
 * 1. Publish the GPS readings received from the SparkFun UBlox ZED-F9P RTK-GPS module;
 * 2. Receive the required angles of rotation (yaw and pitch) from the XXRealm Python Controller over Bluetooth/USB; and
 * 3. Yaw and Pitch rotations using the SparkFun Blackboard Rev C, IMU BNO080, and ServoCity continuous servos -- along with IMU logging.
 * 
 * Eitri_Rx: With GPS
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
#define STEADY_STATE 10                    // Avoid "nan" yaw and pitch values by forcing steady-state reads...
#define READ_EQUILIBRIUM 0                 // The number of samples to be read before we decide to use the IMU report
#define YAW_RESTRICTION 350.0              // A constraint (in degrees) on the maximum allowable yaw rotation in order to prevent the wires from getting tangled-up
#define PITCH_RESTRICTION 8.00             // A constraint (in degrees) on the maximum allowable pitch rotation in order to prevent the wires from getting tangled-up

/* Servo PWM Control Values */
#define MAX_PWM 2000                       // Forward Motion (PWM ON pulse duration)
#define MID_PWM 1500                       // Stop Motion (PWM ON pulse duration)
#define MIN_PWM 1000                       // Backward Motion (PWM ON pulse duration)

/* IMU Instance Creation
 * Servo control pin definitions | Servo (X and Z) instance creation
 * X = Yaw movements | Z = Pitch & Roll movements | For pitch movements, place the object being rotated perpendicular to the pan plate -- in the X-Z plane | Manual height adjustments
*/
BNO080 bno080;
int pinX = 9, pinZ = 10;
Servo servoX, servoZ;

/* UBlox ZED-F9P Instance Creation */
SFE_UBLOX_GPS ublox;

/* Core Rotational Variables */
float yaw = 0.0, prevYaw = 0.0, pitch = 0.0, prevPitch = 0.0;                       // The estimated yaw angle and its previous time-step counterpart | The estimated pitch angle and its previous time-step counterpart
float tmp_yaw = 720.0, requiredYawAngle = 0.0, currentYawAngle = 0.0;               // The core yaw axis variables facilitating motor control
float tmp_pitch = 720.0, requiredPitchAngle = 0.0, currentPitchAngle = 0.0;         // The core pitch axis variables facilitating motor control
float yawReconfigRequirement = 0.0, pitchReconfigRequirement = 0.0;                 // The reconfigured mandates in terms of the amount of angle that needs to be traversed -- either in the anti-clockwise direction or in the clockwise direction
float aclkYawRotation = 0.0, clkYawRotation = 0.0;                                  // The movement variables that encapsulate the amount of yaw rotation in the anti-clockwise & clockwise directions, respectively
float aclkPitchRotation = 0.0, clkPitchRotation = 0.0;                              // The movement variables that encapsulate the amount of pitch rotation in the anti-clockwise & clockwise directions, respectively
boolean proceedToPitch = false, motorsInitiated = false, mandateAchieved = true;    // A flag to give the go-ahead to move on to pitch changes | A flag to indicate if the motors have been initiated or not | A flag to indicate that the overall correction mandate from the XXRealm Python Controller has been met
unsigned int sampleCounter = 0, steadyStateCounter = 0;                             // A counter for ensuring that the IMU values are read post-equilibrium | A counter to force steady state reads

/* Core GPS Variable */
long prevTime = 0;                                                                  // A previous time member in order to limit the amount of traffic coming in from the UBlox ZED-F9P RTK-GPS module

/* Core Mandate Extraction Variables */
const byte numChars = 32;                                                           // The length of the mandate (correction string from the XXRealm Python Controller) character array
char receivedChars[numChars];                                                       // The mandate (correction string from the XXRealm Python Controller) character array
boolean newData = false;                                                            // A flag to indicate that a new mandate has been received

/* The setup routine */
void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  Wire.begin();                                                                     // Begin the Wire instance to start reading the I2C interface...
  
  if (!bno080.begin()) {                                                            // Check for the IMU at the I2C interface
    return;
  }
  Wire.setClock(400000);                                                            // I2C Data Rate = 400kHz
  bno080.enableRotationVector(20);                                                  // IMU Refresh at 20ms

  if (ublox.begin(Wire) == false) {                                                 // Check to see if the ZED-F9P RTK-GPS module is connected -- if not, default to the pre-defined values
    return;
  }
  ublox.setI2COutput(COM_TYPE_UBX);                                                 // Set the I2C port to output UBX only (turn off NMEA noise)
  
  servoX.attach(pinX);                                                              // Attach to servoX | Reset to zero motion | Yaw Servo
  servoX.writeMicroseconds(MID_PWM);
  servoZ.attach(pinZ);                                                              // Attach to servoZ | Reset to zero motion | Pitch Servo
  servoZ.writeMicroseconds(MID_PWM);
}

/* The loop routine */
void loop() {
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
   *    NOTE (v21.06): Handling the high-precision component assembly in the XXRealm Python Controller...
  */
  if ( (mandateAchieved == true) && ( (millis() - prevTime) > 5000 ) ) {
    prevTime = millis();
    
    double d_lat, d_lon;
    float f_ellipsoid, f_msl;
      
    bool is_gnss_fix_ok = ublox.getGnssFixOk();
    uint8_t siv = ublox.getSIV();
    uint8_t fix_type = ublox.getFixType();
    uint8_t carrier_solution_type = ublox.getCarrierSolutionType();
    
    int32_t latitude = ublox.getHighResLatitude();
    int8_t latitudeHp = ublox.getHighResLatitudeHp();
    d_lat = ((double)latitude) / 10000000.0;
    d_lat += ((double)latitudeHp) / 1000000000.0;
    
    int32_t longitude = ublox.getHighResLongitude();
    int8_t longitudeHp = ublox.getHighResLongitudeHp();
    d_lon = ((double)longitude) / 10000000.0;
    d_lon += ((double)longitudeHp) / 1000000000.0;
    
    int32_t ellipsoid = ublox.getElipsoid();
    int8_t ellipsoidHp = ublox.getElipsoidHp();
    f_ellipsoid = (ellipsoid * 10) + ellipsoidHp;
    f_ellipsoid = f_ellipsoid / 10000.0;
    
    int32_t msl = ublox.getMeanSeaLevel();
    int8_t mslHp = ublox.getMeanSeaLevelHp();
    f_msl = (msl * 10) + mslHp;
    f_msl = f_msl / 10000.0;

    Serial.print("$");
    Serial.print(is_gnss_fix_ok);
    Serial.print(",");
    Serial.print(siv);
    Serial.print(",");
    Serial.print(fix_type);
    Serial.print(",");
    Serial.print(carrier_solution_type);
    Serial.print(",");
    
    Serial.print(latitude);
    Serial.print(",");
    Serial.print(latitudeHp);
    Serial.print(",");
    Serial.print(d_lat, 9);
    Serial.print(",");
    
    Serial.print(longitude);
    Serial.print(",");
    Serial.print(longitudeHp);
    Serial.print(",");
    Serial.print(d_lon);
    Serial.print(",");
    
    Serial.print(ellipsoid);
    Serial.print(",");
    Serial.print(ellipsoidHp);
    Serial.print(",");
    Serial.print(f_ellipsoid);
    Serial.print(",");
    
    Serial.print(msl);
    Serial.print(",");
    Serial.print(mslHp);
    Serial.print(",");
    Serial.print(f_msl);
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
    yaw = atan2( ( 2.0 * ( (quatReal * quatK) + (quatI * quatJ) ) ), ( 1.0 - ( 2.0 * ( (quatJ * quatJ) + (quatK * quatK) ) ) ) ) * (180 / PI);     // Determine the Yaw Angle (in degrees)
    pitch = atan2( ( 2.0 * ( (quatReal * quatI) + (quatJ * quatK) ) ), ( 1.0 - ( 2.0 * ( (quatI * quatI) + (quatJ * quatJ) ) ) ) ) * (180 / PI);   // Determine the Pitch Angle (in degrees)
    sampleCounter += 1;                                                                                                                            // Update the sample counter
    steadyStateCounter += 1;                                                                                                                       // Update the steady state counter
  }
  
  if (steadyStateCounter < STEADY_STATE) {
    return;
  }

  /* Mandate Extraction */
  recvWithStartEndMarkers();
  if (newData == true) {
    tmp_yaw = atof(strtok(receivedChars, "#"));
    tmp_pitch = atof(strtok(NULL, "#"));
    newData = false;
  }

  /* Mandate Execution Dispatch */
  if ( (mandateAchieved == true) && (tmp_yaw != 720.0) && (tmp_pitch != 720.0) && ( (tmp_yaw != requiredYawAngle) || (tmp_pitch != requiredPitchAngle) ) ) {
    requiredYawAngle = tmp_yaw;
    requiredPitchAngle = tmp_pitch;
    sampleCounter = 0;                                                                                                    // Start fresh with new samples in order to achieve READ_EQUILIBRIUM
    prevYaw = yaw;                                                                                                        // Raw, unfiltered previous yaw store
    prevPitch = pitch;                                                                                                    // Raw, unfiltered previous pitch store
    mandateAchieved = false;
  }
  
  /* Yaw Rotation Control */
  if ( (mandateAchieved == false) && (proceedToPitch == false) && (sampleCounter >= READ_EQUILIBRIUM) ) {
    currentYawAngle = (yaw < 0.0) ? abs(360.0 - yaw) : yaw;
    if (motorsInitiated == false) {
      if (currentYawAngle <= requiredYawAngle) {                                                                         // Variant-1
        aclkYawRotation = requiredYawAngle - currentYawAngle;
        clkYawRotation = 360 - (requiredYawAngle - currentYawAngle);
      } else {                                                                                                           // Variant-2
        clkYawRotation = currentYawAngle - requiredYawAngle;
        aclkYawRotation = 360 - (currentYawAngle - requiredYawAngle);
      }
      yawReconfigRequirement = (aclkYawRotation <= clkYawRotation) ? aclkYawRotation : clkYawRotation;
      servoX.writeMicroseconds( (aclkYawRotation <= clkYawRotation) ? MAX_PWM : MIN_PWM );                               // Use the optimal rotation direction (a simple comparison-based decision heuristic)
      motorsInitiated = true;
    } else {
      if ( abs(yaw - prevYaw) >= yawReconfigRequirement ) {                                                              // Stop, once the mandate has been met -- and, move on to pitch adjustments...
        servoX.writeMicroseconds(MID_PWM);
        proceedToPitch = true;
        motorsInitiated = false;
      }
    }
    sampleCounter = 0; 
  }
  
  /* Pitch Rotation Control */
  if ( (mandateAchieved == false) && (proceedToPitch == true) && (sampleCounter >= READ_EQUILIBRIUM) ) {
    currentPitchAngle = pitch;
    if (motorsInitiated == false) {
      pitchReconfigRequirement = abs(currentPitchAngle - requiredPitchAngle);
      servoZ.writeMicroseconds( (currentPitchAngle <= requiredPitchAngle) ? MAX_PWM  : MIN_PWM );                       // Use the optimal rotation direction (a simple comparison-based decision heuristic)
      motorsInitiated = true;
    } else {
      if ( abs(pitch - prevPitch) >=  pitchReconfigRequirement ) {                                                      // Stop, once the mandate has been met -- and, move on to the next mandate (if one exists).
        servoZ.writeMicroseconds(MID_PWM);
        proceedToPitch = false;
        motorsInitiated = false;
        mandateAchieved = true;
      }
    }
    sampleCounter = 0;
  } 
}

/* Mandate (Correction String from XXRealm Python Controller) Read Routine */
void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;
    while ( (Serial.available() > 0) && (newData == false) ) {
        rc = Serial.read();
        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0';
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }
        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}
