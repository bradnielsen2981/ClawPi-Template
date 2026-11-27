# -*- coding: utf-8 -*-
# A robot routine. RobotInterface attaches this function as a method, so `self` is the robot
# and it is called as ROBOT.move_direction_until_detection(...)
import time

from robot_routines import DRIVE_SPEED, TURN_SPEED, pause_for_behaviour

def move_direction_until_detection(self, movetype='forward', distanceto=250, detection_types=None,
                                   detection_colours=None, timelimit=5, confirmlevel=1, target_class='person'):
    """
    Moves the robot in a continuous pattern until a specific sensor or visual target is triggered.
    Highly versatile function for autonomous exploration.

    Args:
        movetype (str): How to move ('forward', 'turnright', 'turnleft', 'circleright', 'circleleft', 'slideright', 'slideleft').
        distanceto (int): Threshold distance for Lidar (cm) or Line (pixels from the bottom of the frame). Defaults to 250.
        detection_types (list): What to look for (['lidar', 'line', 'colour', 'object', 'all']). 'model' still works for 'object'.
        detection_colours (list): Which specific colors to look for if 'colour' is active. Defaults to ['red'].
        timelimit (int): Max time to spend searching before giving up. Defaults to 5.
        confirmlevel (int): How many distinct detection types must be triggered simultaneously to succeed. Defaults to 1.
        target_class (str): Which YOLO26 object class to trigger on (e.g., 'person', 'cup').

    Returns:
        dict: The final detection data snapshot and timestamp.
    """
    if detection_types is None:
        detection_types = ['colour']
    if detection_colours is None:
        detection_colours = ['red']
    detection_types = ['object' if t == 'model' else t for t in detection_types] # 'model' is the old name

    self.routine = "move_direction_until_detection"
    data = {'routine': self.routine, 'movetype': movetype }
    detections = [] # Keeps track of which sensors have tripped

    # 1. Setup the camera tasks based on user arguments
    if 'all' in detection_types:
        self.CAMERA.detect_all()
    else:
        if 'line' in detection_types:
            self.CAMERA.add_detection_task("detect_line")
        if 'colour' in detection_types:
            self.CAMERA.add_detection_task('detect_colour')
            self.CAMERA.set_detection_colours(detection_colours)
        if 'object' in detection_types:
            self.CAMERA.load_detection_model()
            self.CAMERA.add_detection_task("detect_object")

    data['starttime'] = time.time()
    endtime = time.time() + timelimit

    # 2. Choose the continuous movement pattern: (forward m/s, left m/s, anticlockwise rad/s)
    velocities = {
        "forward": (DRIVE_SPEED, 0, 0),
        "turnright": (0, 0, -TURN_SPEED),
        "turnleft": (0, 0, TURN_SPEED),
        "circleright": (DRIVE_SPEED, 0, -TURN_SPEED),
        "circleleft": (DRIVE_SPEED, 0, TURN_SPEED),
        "slideright": (0, -DRIVE_SPEED, 0),
        "slideleft": (0, DRIVE_SPEED, 0),
    }
    velocity = velocities.get(movetype, (0, 0, 0))

    # 3. Enter the active scanning loop
    while time.time() < endtime:
        if self.routine != "move_direction_until_detection": # interrupted by a behaviour, or stopped
            paused = pause_for_behaviour(self, "move_direction_until_detection")
            if paused is None:
                break # stopped (e.g. by stop_routine), so end the routine
            endtime += paused # the pause doesn't count toward the time limit
            continue

        self.set_velocity(*velocity) # resent every loop so the robot keeps moving
        num_detections_processed = 0
        if not self.show_camera_window():
            break

        data.update(self.CAMERA.get_detection_data())

        # --- LIDAR CHECK ---
        if 'lidar' in detection_types or 'all' in detection_types:
            lidar_distance = self.get_distance_ahead() # metres, inf if nothing in range
            lidar_cm = None if lidar_distance is None else lidar_distance * 100
            data['detect_lidar'] = { 'distance': lidar_cm }
            self.CAMERA.output_message += f" Lidar: {lidar_cm}"

            if lidar_cm is not None and lidar_cm < distanceto:
                print("Lidar detected!")
                if 'lidar' not in detections:
                    detections.append('lidar')
                    if len(detections) == confirmlevel:
                        break # Success! Break the while loop

            num_detections_processed += 1
            if num_detections_processed == len(detection_types):
                continue

        # --- LINE CHECK ---
        if 'line' in detection_types or 'all' in detection_types:
            # A crossing line is one running across the robot's path
            crossing = data.get('detect_line', {}).get('crossing')
            if crossing and (480 - crossing['y'] < distanceto):
                print("Line detected!")
                if 'line' not in detections:
                    detections.append('line')
                    if len(detections) == confirmlevel:
                        break

            num_detections_processed += 1
            if num_detections_processed == len(detection_types):
                continue

        # --- COLOUR CHECK ---
        if 'colour' in detection_types or 'all' in detection_types:
            for col in data.get('detect_colour', {}).keys():
                col_data = data['detect_colour'][col]
                if col_data.get('found'):
                    print(f"{col} detected!")
                    if col not in detections:
                        detections.append(col)
                        if len(detections) == confirmlevel:
                            break

            if len(detections) == confirmlevel:
                break

            num_detections_processed += 1
            if num_detections_processed == len(detection_types):
                continue

        # --- NEURAL NETWORK CHECK ---
        if 'object' in detection_types or 'all' in detection_types:
            object_data = data.get('detect_object', {})
            if object_data.get('found'):
                for target in object_data['targets']:
                    if target['class'] == target_class:
                        if 'object' not in detections:
                            print(f"{target_class} detected!")
                            detections.append('object')
                            if len(detections) == confirmlevel:
                                break

                if len(detections) == confirmlevel:
                    break

    self.stop_routine()
    self.CAMERA.end_detection()
    data['endtime'] = time.time()
    return data
