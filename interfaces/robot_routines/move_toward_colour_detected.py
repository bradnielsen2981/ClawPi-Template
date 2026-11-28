# -*- coding: utf-8 -*-
# A robot routine, called as ROBOT.move_toward_colour_detected(...). `self` is the robot.
import time, math

from .routine_tools import pause_for_behaviour

def move_toward_colour_detected(self, colour="red", timelimit=5, mode='turning'):
    """
    Drives toward a colour seen by the camera until it is close, steering to keep it in front.

    Args:
        colour (str): The color name from your YAML file to track. Defaults to "red".
        timelimit (int/float): Maximum time in seconds to execute the command before giving up. Defaults to 5.
        mode (str): Movement style.
                    - 'turning' will turn the robot's chassis to face the object while driving forward.
                    - 'drifting' will use mecanum wheels to slide diagonally toward it.

    Returns:
        dict: The final detection data snapshot and timestamp.
    """
    self.set_routine("move_toward_colour_detected")
    data = {'routine': "move_toward_colour_detected", 'starttime': time.time()}
    detectors_before = self.CAMERA.get_detectors() # put back when finished, so other detectors aren't turned off
    colours_before = self.CAMERA.get_detection_colours()

    try:
        # Give visual feedback that tracking has started
        self.set_led_colour(colour)
        self.CAMERA.add_detector("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        endtime = time.time() + timelimit

        # How close (in pixels from the bottom of the frame) the object must get before stopping
        distance = 145

        while time.time() < endtime:
            if self.routine != "move_toward_colour_detected": # interrupted by a behaviour, or stopped
                paused = pause_for_behaviour(self, "move_toward_colour_detected")
                if paused is None:
                    break # stopped, so end the routine
                endtime += paused # the pause doesn't count toward the time limit
                continue

            data.update(self.CAMERA.wait_for_detection()) # waits for the next camera frame

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
                        self.set_velocity(linear_x=self.DRIVE_SPEED * math.cos(angle_to_object),
                                          linear_y=self.DRIVE_SPEED * math.sin(angle_to_object))
                    elif mode == 'turning':
                        turn = angle_to_object
                        # Deadzone to prevent micro-stuttering if it's already basically centered
                        if abs(turn) <= 0.05:
                            turn = 0
                        self.set_velocity(linear_x=self.DRIVE_SPEED, angular_z=turn)
                else:
                    # We have arrived at the object!
                    break
            else:
                self.stop() # lost sight of it, wait for it to reappear

        if not self.is_shutdown:
            self.set_led_colour() # LEDs off

    except Exception as e:
        self.emergency_shutdown(f"move_toward_colour_detected crashed: {e}")
        raise
    finally:
        # Always runs - finished, stopped or crashed
        self.CAMERA.set_detectors(detectors_before)
        self.CAMERA.set_detection_colours(colours_before)
        self.stop_routine()

    data['endtime'] = time.time()
    return data
