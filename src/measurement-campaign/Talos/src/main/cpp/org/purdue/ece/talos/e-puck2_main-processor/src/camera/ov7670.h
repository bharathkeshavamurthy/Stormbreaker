#ifndef OV7670_H
#define OV7670_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "hal.h"
#include "camera.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OV7670_MAX_WIDTH 640
#define OV7670_MAX_HEIGHT 480

typedef enum {
	OV7670_FORMAT_RGB565 = 0x01,
	OV7670_FORMAT_GREYSCALE = 0x02	// Actually this format is not supported, but in case it is requested a conversion will be performed.
} ov7670_format_t;

struct ov7670_configuration {
	uint16_t 		width;
	uint16_t 		height;
	ov7670_format_t curr_format;
	subsampling_t 	curr_subsampling_x;
	subsampling_t 	curr_subsampling_y;
};

/**
 * @brief       Initializes the clock generation for the po6030
 */
void ov7670_start(void);

 /**
 * @brief   Configures some parameters of the camera
 * 
 * @param fmt           format of the image. See format_t
 * @param imgsize       size of the image. See image_size_t
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end or if wrong imgsize
 *
 */
int8_t ov7670_config(ov7670_format_t fmt, image_size_t imgsize);

 /**
 * @brief   Configures advanced setting of the camera
 * 
 * @param fmt           format of the image. See format_t
 * @param x1            x coordinate of the upper left corner of the zone to capture from the sensor
 * @param y1            y coordinate of the upper left corner of the zone to capture from the sensor
 * @param width         width of the image to capture
 * @param height        height of the image to capture
 * subsampling_x        subsampling in the x axis. See subsampling_t
 * subsampling_y        subsampling in the y axis. See subsampling_t
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_advanced_config(  ov7670_format_t fmt, unsigned int x1, unsigned int y1,
                                unsigned int width, unsigned int height,
								subsampling_t subsampling_x, subsampling_t subsampling_y);

 /**
 * @brief   Sets the brigthness of the camera
 * 
 * @param value         Brightness. [7] = sign (positive if 0) and [6:0] the magnitude. => from -127 to 127
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 * @retval others       see in the implementation for details
 *
 */
int8_t ov7670_set_brightness(uint8_t value);

 /**
 * @brief   Sets the contrast of the camera
 * 
 * @param value         Contrast (0..255)
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_contrast(uint8_t value);

 /**
 * @brief   Sets mirroring for both vertical and horizontal orientations.
 * 
 * @param vertical      1 to enable vertical mirroring. 0 otherwise
 * @param horizontal    1 to enable horizontal mirroring. 0 otherwise
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_mirror(uint8_t vertical, uint8_t horizontal);

 /**
 * @brief   Enables/disables auto white balance.
 * 
 * @param awb      1 to enable auto white balance. 0 otherwise
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_awb(uint8_t awb);

 /**
 * @brief   Sets the white balance for the red, green and blue gains. 
 *          These values are considered only when auto white balance is disabled, so this function also disables auto white balance.
 *          The resulting gain is the value divided by 64 (max resulting gain = 4).
 *          
 * @param r             red gain. Default is 128 (2x).
 * @param g             green gain. Default is 0.
 * @param b             blue gain. Default is 128 (2x).
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_rgb_gain(uint8_t r, uint8_t g, uint8_t b);

 /**
 * @brief   Enables/disables auto exposure.
 * 
 * @param ae            1 to enable auto exposure. 0 otherwise
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_ae(uint8_t ae);

 /**
 * @brief   Sets integration time, aka the exposure.
 *          Total integration time is: integral * line time.
 * 
 * @param integral      unit is line time.
 *
 * @return              The operation status.
 * @retval MSG_OK       if the function succeeded.
 * @retval MSG_TIMEOUT  if a timeout occurred before operation end
 *
 */
int8_t ov7670_set_exposure(uint16_t integral);

 /**
 * @brief   Returns the current image size in bytes.
 *
 * @return              The image size in bytes
 *
 */
uint32_t ov7670_get_image_size(void);

/**
* @brief	Check whether the ov7670 camera is connected.
*
* @return	1 if the camera is present, 0 otherwise.
*
*/
uint8_t ov7670_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif
