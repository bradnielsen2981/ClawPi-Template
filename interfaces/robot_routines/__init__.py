# -*- coding: utf-8 -*-
# Robot routines: each file holds one routine. A routine is a loop that uses the robot's interfaces (camera, lidar,
# line follower, wheels...) to do one thing, e.g. drive toward a colour. Tasks (in robot.py) combine routines.
#
# Robot in robot.py inherits from Routines, so every routine is called as self.name(...) / ROBOT.name(...).
# To add a routine: copy one of the files, import it below, add it to the Routines class and to ROUTINES.

from .routine_tools import PAUSETIME, pause_for_behaviour
from .move_toward_colour_detected import move_toward_colour_detected
from .move_direction_until_detection import move_direction_until_detection
from .face_tracking import face_tracking


class Routines:
    # The routines' names: used by ROBOT.start_routine(name) and the voice agent
    ROUTINES = ['move_toward_colour_detected', 'move_direction_until_detection', 'face_tracking']

    move_toward_colour_detected = move_toward_colour_detected
    move_direction_until_detection = move_direction_until_detection
    face_tracking = face_tracking
