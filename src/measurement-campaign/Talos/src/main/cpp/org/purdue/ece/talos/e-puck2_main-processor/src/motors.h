#ifndef MOTORS_H
#define MOTORS_H

#include <stdint.h>
#include <hal.h>

#define MOTOR_SPEED_LIMIT 1100 // [step/s]

extern struct stepper_motor_s right_motor;
extern struct stepper_motor_s left_motor;

 /**
 * @brief   Sets the speed of the left motor
 * 
 * @param speed     speed desired in step/s
 */
void left_motor_set_speed(int speed);

 /**
 * @brief   Sets the speed of the right motor
 * 
 * @param speed     speed desired in step/s
 */
void right_motor_set_speed(int speed);

 /**
 * @brief   Reads the position counter of the left motor
 * 
 * @return          position counter of the left motor in step
 */
int32_t left_motor_get_pos(void);

 /**
 * @brief   Reads the position counter of the right motor
 * 
 * @return          position counter of the right motor in step
 */
int32_t right_motor_get_pos(void);

/**
 * @brief 	sets the position counter of the left motor to the given value
 * 
 * @param counter_value 	value to store in the position counter of the motor in step
 */
void left_motor_set_pos(int32_t counter_value);

/**
 * @brief 	sets the position counter of the right motor to the given value
 * 
 * @param counter_value value to store in the position counter of the motor in step
 */
void right_motor_set_pos(int32_t counter_value);

 /**
 * @brief   Initializes the control of the motors.
 */
void motors_init(void);

/**
* @brief	Get the last desired speed set for the left motor
*
* @return	speed desired in step/s
*/
int left_motor_get_desired_speed(void);

/**
* @brief	Get the last desired speed set for the right motor
*
* @return	speed desired in step/s
*/
int right_motor_get_desired_speed(void);

/**
* @brief   Sets the speed of the chosen motor (low level).
*
* @param m         pointer to the motor. See stepper_motor_s
* @param speed     speed desired in step/s
*/
void motor_set_speed(struct stepper_motor_s *m, int speed);

#endif /* MOTOR_H */
