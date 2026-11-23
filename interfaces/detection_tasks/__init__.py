# -*- coding: utf-8 -*-
# The detection tasks the CameraInterface can run, one file each.
# To add a new one: write a Detector subclass in its own file (see base.py) and add it to DETECTOR_CLASSES.
# The order here is the order they run and are drawn in.

from .base import Detector, KERNEL, largest_contour, keep_or_expire, json_safe
from .detect_line import LineDetector
from .detect_colour import ColourDetector
from .detect_object import ObjectDetector

# detect_depth (detect_depth.py) is switched off: the ClawPi has no depth camera by default. If one is fitted,
# add "from .detect_depth import DepthDetector" above, put DepthDetector in this list, and create the camera
# with CameraInterface(use_depth=True).
DETECTOR_CLASSES = [LineDetector, ColourDetector, ObjectDetector]
DETECTION_TASKS = [cls.name for cls in DETECTOR_CLASSES] # 'detect_line', 'detect_colour', 'detect_object'
