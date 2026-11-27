# -*- coding: utf-8 -*-
# A robot routine. RobotInterface attaches this function as a method, so `self` is the robot
# and it is called as ROBOT.move_toward_colour_detected(...)
import time, math

from robot_routines import DRIVE_SPEED

def move_toward_colour_detected(self, colour="red", timelimit=5, mode='turning'):
    """
    Instructs the robot to continuously track and move towards a specific color blob.
    Calculates the angle of the blob relative to the camera center and adjusts trajectory.

    Args:
        colour (str): The color name from your YAML file to track. Defaults to "red".
        timelimit (int/float): Maximum time in seconds to execute the command before giving up. Defaults to 5.
        mode (str): Movement style.
                    - 'turning' will turn the robot's chassis to face the object while driving forward.
                    - 'drifting' will use mecanum wheels to slide diagonally toward it.

    Returns:
        dict: The final detection data snapshot and timestamp.
    """
    self.routine = "move_toward_colour_detected"
    data = {
        'routine': self.routine,
        'starttime': time.time()
    }

    # Give visual feedback that tracking has started
    self.set_led_colour(colour)
    self.CAMERA.add_detection_task("detect_colour")
    self.CAMERA.add_detection_colour(colour)
    endtime = time.time() + timelimit

    # How close (in pixels from the bottom of the frame) the object must get before stopping
    distance = 145

    while (time.time() < endtime) and (self.routine == "move_toward_colour_detected"):
        if not self.show_camera_window(): # Acts as our loop throttler
            break

        data.update(self.CAMERA.get_detection_data())

        colour_data = data.get('detect_colour', {}).get(colour, {})
        if colour_data.get('found'):
            # Extract the bounding box details
            center, size, angle = colour_data['rect']
            x, y = center

            # Calculate offsets from the exact center of the 640x480 frame
            deltaY = 480 - y
            deltaX = 320 - x

            # Angle to the object: 0 = straight ahead, positive = to the left
            angle_to_object = math.atan2(deltaX, deltaY)

            # If we are not close enough yet, keep moving
            if deltaY > distance:
                if mode == 'drifting':
                    self.set_velocity(linear_x=DRIVE_SPEED * math.cos(angle_to_object),
                                      linear_y=DRIVE_SPEED * math.sin(angle_to_object))
                elif mode == 'turning':
                    turn = angle_to_object
                    # Deadzone to prevent micro-stuttering if it's already basically centered
                    if abs(turn) <= 0.05:
                        turn = 0
                    self.set_velocity(linear_x=DRIVE_SPEED, angular_z=turn)
            else:
                # We have arrived at the object!
                break
        else:
            self.stop() # lost sight of it, wait for it to reappear

    self.stop_routine()
    self.CAMERA.end_detection()
    self.set_led_colour()
    data['endtime'] = time.time()
    return data
