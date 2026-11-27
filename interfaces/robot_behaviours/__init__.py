# -*- coding: utf-8 -*-
# Robot behaviours: code that keeps running in the background on its own thread, on top of the robot's task and
# routines. None are on by default: turn one on with ROBOT.add_behaviour('name', ...) (see base.py).
# To add a new one: write a Behaviour subclass in its own file and add it to BEHAVIOUR_CLASSES.

from .base import Behaviour
from .stay_within_quadrant import StayWithinQuadrant

BEHAVIOUR_CLASSES = [StayWithinQuadrant]
BEHAVIOURS = {cls.name: cls for cls in BEHAVIOUR_CLASSES} # 'stay_within_quadrant' -> StayWithinQuadrant
