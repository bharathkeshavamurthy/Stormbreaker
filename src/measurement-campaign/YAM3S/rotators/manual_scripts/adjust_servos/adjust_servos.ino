/*
 Manually adjust the servos.

 Yaguang Zhang, Purdue University, 2020/01/31
*/

#include <Servo.h>

// The PWM range for the continuous servos.
#define MAX_PWM 2000
#define MID_PWM 1500
#define MIN_PWM 1000

// X axis - Tilt (changing elevation); Z axis - Pan (changing azimuth).
int wpmPinX = 9, wpmPinZ = 10;
Servo servoX, servoZ;

// We will set the PWM signals for the specified time periods and then return
// them to MID_PWM.
int wpmToSetX = MIN_PWM, wpmEffTimeMsX = 1000,
    wpmToSetZ = MIN_PWM, wpmEffTimeMsZ = 1000;

void setup() {
  servoX.attach(wpmPinX);
  servoZ.attach(wpmPinZ);

  servoX.writeMicroseconds(wpmToSetX);
  delay(wpmEffTimeMsX);
  servoX.writeMicroseconds(MID_PWM);

  servoZ.writeMicroseconds(wpmToSetZ);
  delay(wpmEffTimeMsZ);
  servoZ.writeMicroseconds(MID_PWM);
}

void loop() {
  // Nothing to do here.
}
