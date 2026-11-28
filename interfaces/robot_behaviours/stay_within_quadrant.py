# -*- coding: utf-8 -*-
# A robot behaviour: keeps the robot inside the quadrant (the area marked out by lines) it was placed in.
#
#   ROBOT.add_behaviour('stay_within_quadrant', degrees=180)   # turn around at a line
#   ROBOT.add_behaviour('stay_within_quadrant', degrees=-90)   # quarter turn counterclockwise at a line
#
# While the robot is doing something (a task, a routine, or a person driving it), as soon as the line follower under
# the front of the robot sees a boundary line it takes over the wheels, sets ROBOT.routine to 'stay_within_quadrant'
# (with ROBOT.interrupt_routine), reverses a little and turns, then hands the wheels back. Stop cuts the turn short.
# The running routine sees ROBOT.routine has changed and decides what to do: the routines in robot_routines/
# pause for PAUSETIME and then carry on (see pause_for_behaviour), but a routine could end instead.
#
# Which lines count is set on the line follower: ROBOT.LINE_FOLLOWER.set_mode('light_line') for light lines on
# a dark floor, 'dark_line' (the default) for dark lines on a light floor.

from .base import Behaviour


class StayWithinQuadrant(Behaviour):
    name = 'stay_within_quadrant'
    rate_hz = 50

    def setup(self, degrees=180, reverse_distance=0.1, reverse_speed=0.15):
        """
        Args:
            degrees (int/float): How far to turn after reversing off a line: usually 90 or 180.
                                 Positive = clockwise, negative = counterclockwise. Defaults to 180.
            reverse_distance (float): How far to back away from the line, in metres. Defaults to 0.1.
            reverse_speed (float): Speed to reverse at, in m/s. Defaults to 0.15.
        """
        if not degrees:
            raise ValueError("degrees can't be 0")
        self.degrees = degrees
        self.reverse_distance = abs(reverse_distance)
        self.reverse_speed = abs(reverse_speed)
        self.boundaries_hit = 0
        self._warned = False

    def check(self):
        robot = self.robot
        if not robot.is_active:
            return # nothing is driving the robot (e.g. Stop was pressed), so there is nothing to correct
        sensors = robot.LINE_FOLLOWER.read()
        if sensors is None:
            if not self._warned:
                self.logger.warning(f"{self.name} can't work: the line follower can't be read")
                self._warned = True
            return
        self._warned = False
        if any(sensors):
            self.avoid_boundary(sensors)

    def avoid_boundary(self, sensors):
        """Stops everything, backs off the line and turns."""
        robot = self.robot
        self.boundaries_hit += 1
        self.logger.info(f"Boundary line seen {sensors}: reversing and turning {self.degrees} degrees")
        with robot.take_movement_control(): # nothing else can drive the wheels until this is done
            stops = robot.stop_count # if this changes, Stop was pressed (or the robot is shutting down)
            robot.stop() # halt straight away
            robot.release_hold() # end any held dashboard button
            previous = robot.interrupt_routine(self.name) # tells the running routine a behaviour has interrupted it
            robot.move_backward(self.reverse_distance / self.reverse_speed, speed=self.reverse_speed)
            if robot.stop_count == stops and not self.stopping: # don't start the turn if Stop was pressed
                try:
                    robot.rotate_degrees(-self.degrees) # rotate_degrees counts anticlockwise as positive
                except RuntimeError as e: # no odometry: turn for the time it should take instead
                    self.logger.warning(f"rotate_degrees failed ({e}), turning by time instead")
                    seconds = abs(self.degrees) / 57.3 / robot.default_turn_speed # 57.3 degrees per radian
                    robot.rotate(seconds, 'clockwise' if self.degrees > 0 else 'anticlockwise')
        # If no routine was running (e.g. a person was driving), there is nothing to hand back to,
        # so put the routine state back
        if previous in ("Ready", "Shutdown"):
            robot.resume_routine(previous, by=self.name)
