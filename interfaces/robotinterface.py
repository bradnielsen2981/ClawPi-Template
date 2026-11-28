# -*- coding: utf-8 -*-
# container for more advanced functionality
# use Ctrl + K, Ctrl + 0 to minimise all coding blocks in VSCODE
#
# RobotInterface IS a ClawPi (it inherits the ROS movement, servo, LED and sensor functions) and HAS the other
# interfaces (CAMERA, LIDAR, LINE_FOLLOWER, SOUND, MAPPING). It also keeps track of what the robot is doing:
#
#   self.task      the job the robot is doing, e.g. 'automated_search' (tasks are in robot.py), or None
#   self.routine   the routine running right now, e.g. 'face_tracking' (routines are in robot_routines/),
#                  "Ready" when none is, or a behaviour's name while that behaviour has interrupted it
#
# Several threads use these (Flask, the voice agent, behaviours, tasks), so READ them freely but only CHANGE them
# with the methods below (set_task, set_routine, stop_task, stop_routine...), which use self.state_lock.
import time, sys, os, math, logging, threading
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from clawpiinterface import (ClawPi, SERVO_LIMITS, CAMERA_PAN_SERVO_ID, CAMERA_TILT_SERVO_ID, CAMERA_PAN_TRIM,
                             CAMERA_TILT_TRIM, CLAW_GRIP_SERVO_ID, CLAW_LIFT_SERVO_ID, CLAW_OPEN, CLAW_CLOSED,
                             CLAW_LIFT_UP, CLAW_LIFT_DOWN)
from camerainterface import CameraInterface
from lidarinterface import LidarInterface
from linefollowerinterface import LineFollowerInterface
from mappinginterface import MappingInterface
from loggerinterface import get_logger
from robot_behaviours import BEHAVIOURS

# RGB values used to show a detection colour on the robot's LEDs
LED_COLOURS = {
    'red': (255, 0, 0), 'green': (0, 255, 0), 'dark_green': (0, 100, 0), 'blue': (0, 0, 255),
    'yellow': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
}

# Speeds used by the dashboard buttons and the routines (routines use self.DRIVE_SPEED and self.TURN_SPEED)
DRIVE_SPEED = 0.15 # m/s
TURN_SPEED = 0.5   # rad/s

# DASHBOARD HOLD CONTROLS -----------------------------------------------------------
# A dashboard button keeps its action going while it is held: the browser re-sends the action every 0.5 s.
# If nothing is heard for HOLD_TIMEOUT seconds (lost wifi, crashed tab, missed mouse release) the robot stops.
HOLD_TIMEOUT = 3.0  # seconds
HOLD_RATE_HZ = 20   # how often a held action is updated
ORBIT_RADIUS = 0.3  # metres, distance to the point the robot orbits around (in front of it)
DIAGONAL = DRIVE_SPEED * math.sqrt(0.5)

# Held movements: (forward m/s, left m/s, anticlockwise rad/s)
HOLD_DRIVES = {
    'forward': (DRIVE_SPEED, 0, 0),         'reverse': (-DRIVE_SPEED, 0, 0),
    'slide_left': (0, DRIVE_SPEED, 0),      'slide_right': (0, -DRIVE_SPEED, 0),
    'diagonal_left': (DIAGONAL, DIAGONAL, 0), 'diagonal_right': (DIAGONAL, -DIAGONAL, 0),
    'turn_left': (0, 0, TURN_SPEED),        'turn_right': (0, 0, -TURN_SPEED),
    # Orbit: slide sideways while turning, so the robot keeps facing a point ORBIT_RADIUS in front of it
    'orbit_left': (0, DRIVE_SPEED, -DRIVE_SPEED / ORBIT_RADIUS),
    'orbit_right': (0, -DRIVE_SPEED, DRIVE_SPEED / ORBIT_RADIUS),
    # Drive: forward while turning (a curve)
    'drive_left': (DRIVE_SPEED, 0, TURN_SPEED), 'drive_right': (DRIVE_SPEED, 0, -TURN_SPEED),
}

# Held camera moves: (pan direction, tilt direction). Positive pan = left, positive tilt = up.
CAMERA_SPEED = 60   # degrees per second while held
CAMERA_LIMIT = 90   # never more than 90 degrees left/right/up/down
HOLD_CAMERA = {'camera_left': (1, 0), 'camera_right': (-1, 0), 'camera_up': (0, 1), 'camera_down': (0, -1)}

def _camera_range(servo_id, trim):
    """The camera angles (degrees) a gimbal servo can really reach, inside +/- CAMERA_LIMIT."""
    low, high = SERVO_LIMITS[servo_id]
    to_angle = lambda pulse: (pulse - 1500) / 1000 * 90 - trim
    return max(-CAMERA_LIMIT, to_angle(low)), min(CAMERA_LIMIT, to_angle(high))
PAN_RANGE = _camera_range(CAMERA_PAN_SERVO_ID, CAMERA_PAN_TRIM)
TILT_RANGE = _camera_range(CAMERA_TILT_SERVO_ID, CAMERA_TILT_TRIM)

# Held claw moves: (servo, direction, lowest pulse, highest pulse, pulse if not moved yet)
CLAW_SPEED = 500    # microseconds of servo pulse per second while held (about 1 s from closed to open)
HOLD_CLAW = {
    'claw_open':  (CLAW_GRIP_SERVO_ID, 1, CLAW_CLOSED, CLAW_OPEN, CLAW_OPEN),
    'claw_close': (CLAW_GRIP_SERVO_ID, -1, CLAW_CLOSED, CLAW_OPEN, CLAW_OPEN),
    'claw_raise': (CLAW_LIFT_SERVO_ID, 1, CLAW_LIFT_DOWN, CLAW_LIFT_UP, CLAW_LIFT_DOWN),
    'claw_lower': (CLAW_LIFT_SERVO_ID, -1, CLAW_LIFT_DOWN, CLAW_LIFT_UP, CLAW_LIFT_DOWN),
}
HOLD_ACTIONS = set(HOLD_DRIVES) | set(HOLD_CAMERA) | set(HOLD_CLAW)

class SilentSound():
    """
    Stands in for the voice agent if it can't be loaded. It has the same basic methods, so calls like
    ROBOT.SOUND.say("Hello") keep working; say() prints the message instead of speaking it.
    """
    def say(self, message, block=False):
        print(f"[sound off] {message}")
    def load_mp3(self, song): pass
    def play_music(self, times=-1): pass
    def pause_music(self): pass
    def unpause_music(self): pass
    def stop_music(self): pass
    def set_volume(self, v=0.8): pass
    def start_listening(self, *args, **kwargs): print("[sound off] can't listen, the voice agent didn't start")
    def stop_listening(self): pass
    def stop(self): pass

class RobotInterface(ClawPi):
    DRIVE_SPEED = DRIVE_SPEED
    TURN_SPEED = TURN_SPEED
    # The names of the tasks and routines that start_task / start_routine (and the voice agent) can start come from
    # TASKS in robot.py and ROUTINES in robot_routines/__init__.py.

    def __init__(self):
        """
        Initializes the Robot Interface, sets up sound and camera modules,
        and waits for hardware to warm up.
        """
        # What the robot is doing (see the top of this file). Change them only with the methods below.
        self.task = None
        self.routine = "Ready"
        self.state_lock = threading.Lock() # protects self.task, self.routine and the shutdown flags
        self.is_shutdown = False           # True once shutdown() has started: nothing new can start after that
        self.shutdown_reason = None        # why emergency_shutdown() shut the robot down
        self._worker = None                # the thread running a task or routine started with start_task/start_routine

        self.behaviours = {}   # behaviours running in the background, by name (see add_behaviour)
        self._movement_owner = None # the thread that has taken over the wheels (see take_movement_control)
        self._movement_lock = threading.Lock()
        self._hold = (None, 0)  # (held dashboard action, time it must stop by)
        self._hold_lock = threading.Lock()
        self._hold_quit = threading.Event()
        self.camera_pan, self.camera_tilt = 0, 0
        self.logger = get_logger('Robot') # logs/robot.log

        # Speaker, music and voice commands through the WonderEcho Pro (see voiceagentinterface.py).
        # It only listens for "Hello Hiwonder" once ROBOT.SOUND.start_listening() is called.
        try:
            from voiceagentinterface import VoiceAgentInterface
            self.SOUND = VoiceAgentInterface(robot=self, name='Wally')
        except Exception as e:
            self.logger.error(f"Voice agent failed to start, sound is off: {e}")
            self.SOUND = SilentSound()

        super().__init__() # connects to the robot through ROS 2

        self.CAMERA = CameraInterface(node=self.node) # shares the robot's ROS connection
        self.CAMERA.start()

        # 4 channel line follower under the front of the robot (read over I2C, see linefollowerinterface.py).
        # If it can't be used, LINE_FOLLOWER.available is False and its readings are None.
        self.LINE_FOLLOWER = LineFollowerInterface() # logs/linefollower.log

        # The lidar and mapping interfaces are only created the first time they are used (see LIDAR and MAPPING)
        self._lidar = None
        self._mapping = None
        self._interface_lock = threading.Lock() # Flask threads could otherwise create two at once
        self.starttime = time.time() # when did robot get created

        # Safety loop for the dashboard's hold-down buttons (see hold_action)
        self._hold_thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._hold_thread.start()

        # Let the voice agent start this robot's tasks and routines (see TASKS and ROUTINES)
        if hasattr(self.SOUND, 'add_robot_commands'):
            self.SOUND.add_robot_commands(self)
        return

    @property
    def LIDAR(self):
        """The lidar interface (scans, distances, scan picture), sharing the robot's ROS connection."""
        with self._interface_lock:
            if self._lidar is None:
                self._lidar = LidarInterface(node=self.node)
            return self._lidar

    @property
    def MAPPING(self):
        """The mapping interface (SLAM map, position, object tagging), using the robot's lidar and camera."""
        lidar = self.LIDAR
        with self._interface_lock:
            if self._mapping is None:
                self._mapping = MappingInterface(lidar=lidar, camera=self.CAMERA, node=self.node, robot=self)
            return self._mapping

    # Lidar shortcuts, so robot.get_distance_ahead() works without going through robot.LIDAR
    def get_distance(self, angle=0, width=20):
        """Nearest obstacle in metres inside a cone (see LidarInterface.get_distance)."""
        return self.LIDAR.get_distance(angle, width)

    def get_distance_ahead(self, width=20):
        """Nearest obstacle straight ahead in metres."""
        return self.LIDAR.get_distance_ahead(width)

    def get_distance_behind(self, width=20):
        """Nearest obstacle behind in metres."""
        return self.LIDAR.get_distance_behind(width)

    def get_distance_left(self, width=20):
        """Nearest obstacle on the left in metres."""
        return self.LIDAR.get_distance_left(width)

    def get_distance_right(self, width=20):
        """Nearest obstacle on the right in metres."""
        return self.LIDAR.get_distance_right(width)

    # Line follower shortcuts, so robot.get_line_sensors() works without going through robot.LINE_FOLLOWER
    def get_line_sensors(self):
        """[far left, centre left, centre right, far right], True where the line is seen (None if unreadable)."""
        return self.LINE_FOLLOWER.read()

    def get_line_position(self):
        """Where the line is under the robot: -1.0 far left to 1.0 far right, None if it can't be seen."""
        return self.LINE_FOLLOWER.get_line_position()

    #---TASK AND ROUTINE STATE------------------------------------------------------------
    # Every change to self.task and self.routine happens inside "with self.state_lock:", so two threads can never
    # change them at the same moment (e.g. the dashboard's Stop and a routine resuming after a behaviour).
    def set_task(self, name):
        """Marks a task as running. Call it at the top of a task, e.g. self.set_task('automated_search')."""
        with self.state_lock:
            if self.is_shutdown:
                raise RuntimeError("the robot has been shut down")
            self.task = name
        self.logger.info(f"Task started: {name}")

    def set_routine(self, name):
        """Marks a routine as running. Call it at the top of a routine, e.g. self.set_routine('face_tracking')."""
        with self.state_lock:
            if self.is_shutdown:
                raise RuntimeError("the robot has been shut down")
            self.routine = name

    def stop_task(self):
        """
        Stops everything the robot is doing: the task, the routine, any behaviour's manoeuvre and any held
        dashboard button. The dashboard's Stop button and the voice command "stop" call this.
        Behaviours stay on, but they only act while the robot is doing something (see is_active).
        """
        with self.state_lock:
            stopped = self.task
            self.task = None
        if stopped:
            self.logger.info(f"Task stopped: {stopped}")
        self.stop_routine()
        with self._hold_lock:
            self._hold = (None, 0) # ends any held dashboard button
        if not self.is_shutdown:
            self.cancel_motion() # cuts short a behaviour's reverse-and-turn

    def stop_routine(self):
        """
        Ends the current routine (its loop sees self.routine change) and stops the wheels. A routine calls this in
        its 'finally' block when it finishes. It doesn't end a held dashboard button or cut short a behaviour's
        manoeuvre - stop_task() does those.
        """
        if self.is_shutdown:
            return # shutdown() has already stopped everything
        with self.state_lock:
            self.routine = "Ready"
        self.stop()

    def interrupt_routine(self, by):
        """
        Used by a behaviour that takes over: sets self.routine to the behaviour's name, so the running routine
        knows it has been interrupted.

        Returns:
            str: What self.routine was before.
        """
        with self.state_lock:
            if self.is_shutdown:
                raise RuntimeError("the robot has been shut down")
            previous = self.routine
            self.routine = by
            return previous

    def resume_routine(self, name, by):
        """
        Sets self.routine back to name, but ONLY if it is still 'by' (the behaviour that interrupted it).
        Checking and changing inside one lock means a Stop pressed at the same moment can't be lost.

        Returns:
            bool: True if the routine was resumed, False if something else changed it first (e.g. Stop).
        """
        with self.state_lock:
            if self.routine != by:
                return False
            self.routine = name
            return True

    @property
    def is_busy(self):
        """True while a task or routine is running."""
        return self.task is not None or self.routine not in ("Ready", "Shutdown")

    @property
    def is_active(self):
        """True while the robot is doing something: a task, a routine or a held dashboard drive button.
        Behaviours only act while it is active, so they never move a robot that has been stopped."""
        return self.is_busy or self._hold[0] in HOLD_DRIVES

    #---STARTING TASKS AND ROUTINES ON THEIR OWN THREAD---------------------------------
    # A task or routine is a loop that runs until it is finished. Run straight from a Flask request or the voice
    # agent, it would block them (the dashboard would freeze and "stop" couldn't be heard), so these run it on a
    # new thread instead. Only one runs at a time: starting a new one stops the old one first.
    def start_task(self, name, when_done=None, **settings):
        """
        Starts a task (one of TASKS) on its own thread and returns straight away, e.g.
        ROBOT.start_task('automated_search', timelimit=60).

        Args:
            name (str): The task's name.
            when_done (function, optional): Called with no arguments when the task ends.
            **settings: Passed to the task.
        """
        tasks = getattr(self, 'TASKS', [])
        if name not in tasks:
            raise ValueError(f"No task called {name!r}. There are: {', '.join(tasks) or 'none'}")
        return self._start_worker(name, when_done, settings)

    def start_routine(self, name, when_done=None, **settings):
        """
        Starts a routine (one of ROUTINES) on its own thread and returns straight away, e.g.
        ROBOT.start_routine('face_tracking', timelimit=20).
        """
        routines = getattr(self, 'ROUTINES', [])
        if name not in routines:
            raise ValueError(f"No routine called {name!r}. There are: {', '.join(routines) or 'none'}")
        return self._start_worker(name, when_done, settings)

    def _start_worker(self, name, when_done, settings):
        if self.is_shutdown:
            raise RuntimeError("the robot has been shut down")
        function = getattr(self, name)
        self.stop_task() # a new command replaces the current one
        old = self._worker
        if old is not None and old.is_alive() and old is not threading.current_thread():
            old.join(timeout=3) # let the old one finish its clean-up first
        def run():
            try:
                function(**settings)
            except Exception as e:
                self.logger.error(f"{name} ended with an error: {e}") # it has already called emergency_shutdown
            finally:
                if when_done:
                    when_done()
        self._worker = threading.Thread(target=run, name=name, daemon=True)
        self._worker.start()
        return self._worker

    #---MOVEMENT CONTROL----------------------------------------------------------------
    def set_velocity(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """
        Sends a velocity command (see ClawPi.set_velocity), unless a behaviour has taken over the wheels with
        take_movement_control(): then commands from every other thread are ignored until it hands them back.
        """
        owner = self._movement_owner
        if owner is not None and owner is not threading.current_thread():
            return
        super().set_velocity(linear_x, linear_y, angular_z)

    def stop(self):
        """Stops the wheels - unless a behaviour has taken over the wheels (it stops them itself when it is done).
        To stop the wheels whatever is driving them, use cancel_motion() or stop_task()."""
        owner = self._movement_owner
        if owner is not None and owner is not threading.current_thread():
            return
        super().stop()

    @property
    def movement_controlled(self):
        """True while a behaviour has taken over the wheels with take_movement_control()."""
        return self._movement_owner is not None

    @contextmanager
    def take_movement_control(self):
        """
        Lets the code inside a 'with' block be the only thing that can drive the wheels, e.g. a behaviour
        backing away from a line while a routine is still trying to drive forward:

            with robot.take_movement_control():
                robot.move_backward(0.5)
        """
        with self._movement_lock: # one behaviour at a time
            self._movement_owner = threading.current_thread()
            try:
                yield
            finally:
                self._movement_owner = None
                if not self.is_shutdown:
                    self.stop()

    #---BEHAVIOURS (each one lives in its own file in robot_behaviours/)-----------------
    def add_behaviour(self, name, **settings):
        """
        Turns on a behaviour: code that keeps running in the background on its own thread, whatever the robot's
        task or routine is. None are on by default. Adding one that is already on restarts it with the new settings.

            ROBOT.add_behaviour('stay_within_quadrant', degrees=-90)

        Args:
            name (str): The behaviour's name (see BEHAVIOURS in robot_behaviours/__init__.py).
            **settings: The behaviour's settings, e.g. degrees=180 for stay_within_quadrant.

        Returns:
            Behaviour: The running behaviour.
        """
        if self.is_shutdown:
            raise RuntimeError("the robot has been shut down")
        if name not in BEHAVIOURS:
            raise ValueError(f"No behaviour called {name!r}. There are: {', '.join(BEHAVIOURS)}")
        behaviour = BEHAVIOURS[name](self, **settings) # check the settings before stopping an old one
        self.remove_behaviour(name)
        self.behaviours[name] = behaviour
        behaviour.start()
        return behaviour

    def remove_behaviour(self, name):
        """Turns off a behaviour. Returns False if it wasn't on."""
        behaviour = self.behaviours.pop(name, None)
        if behaviour is None:
            return False
        behaviour.stop()
        return True

    def remove_all_behaviours(self):
        """Turns off every behaviour."""
        for name in list(self.behaviours):
            self.remove_behaviour(name)

    #---DASHBOARD HOLD CONTROLS-------------------------------------------------------
    def hold_action(self, action):
        """
        Starts (or keeps going) a dashboard hold-down action, e.g. 'forward', 'camera_up' or 'claw_open'.
        The action stops by itself HOLD_TIMEOUT seconds after the last call, so call this again
        (the dashboard does every 0.5 s) for as long as the button is held, and release_hold() when let go.
        A person pressing a button takes over: a running task or routine is stopped first. While a behaviour is
        manoeuvring (e.g. backing off a boundary line) presses are ignored until it has finished.

        Returns:
            bool: False if the action isn't known.
        """
        if action not in HOLD_ACTIONS or self.is_shutdown:
            return False
        if self.movement_controlled:
            return True # a behaviour is manoeuvring (e.g. backing off a line): it finishes first
        if self.is_busy:
            self.logger.info(f"Dashboard {action} pressed: stopping {self.task or self.routine}")
            self.stop_task()
        with self._hold_lock:
            if self._hold[0] != action: # log the press, not every 0.5 s repeat
                self.logger.info(f"Dashboard hold: {action}")
            self._hold = (action, time.time() + HOLD_TIMEOUT)
        return True

    def release_hold(self, deadline=None):
        """Stops the held dashboard action. Wheels stop; the camera and claw stay where they are."""
        with self._hold_lock:
            if deadline is not None and self._hold[1] != deadline:
                return # a newer press arrived, leave it running
            if self._hold[0] is not None:
                self.logger.info(f"Dashboard release: {self._hold[0]}")
            self._hold = (None, 0)
        if not self.is_shutdown:
            self.stop()

    def _hold_loop(self):
        """Runs in the background: moves whatever is held, and stops it if the dashboard goes quiet."""
        period = 1.0 / HOLD_RATE_HZ
        while not self._hold_quit.wait(period):
            with self._hold_lock:
                action, deadline = self._hold
            if action is None:
                continue
            if time.time() > deadline:
                self.logger.warning(f"{action} stopped: no word from the dashboard for {HOLD_TIMEOUT} s")
                self.release_hold(deadline)
                continue
            try:
                self._hold_step(action, period)
            except Exception as e:
                self.logger.error(f"{action} failed: {e}")
                self.release_hold()

    def _hold_step(self, action, period):
        """One small step of a held action."""
        if action in HOLD_DRIVES:
            self.set_velocity(*HOLD_DRIVES[action]) # resent every step so the wheels keep going
        elif action in HOLD_CAMERA:
            pan_dir, tilt_dir = HOLD_CAMERA[action]
            step = CAMERA_SPEED * period
            self.point_camera(self.camera_pan + pan_dir * step, self.camera_tilt + tilt_dir * step, duration=period)
        elif action in HOLD_CLAW:
            servo, direction, low, high, start = HOLD_CLAW[action]
            pulse = self.get_servo_position(servo) or start
            new_pulse = max(low, min(high, pulse + direction * CLAW_SPEED * period))
            if new_pulse != pulse:
                self.set_servo(servo, new_pulse, duration=period)

    def get_camera_angles(self):
        """The camera's current (pan, tilt) in whole degrees."""
        return round(self.camera_pan), round(self.camera_tilt)

    def point_camera(self, pan, tilt, duration=0.2):
        """
        Points the camera at (pan, tilt) degrees, kept inside the angles the gimbal can reach, and remembers it.
        Positive pan = left, positive tilt = up. Nothing is sent if the camera is already there (e.g. at a limit).

        Returns:
            tuple: The camera's new (pan, tilt) in degrees.
        """
        pan = max(PAN_RANGE[0], min(PAN_RANGE[1], pan))
        tilt = max(TILT_RANGE[0], min(TILT_RANGE[1], tilt))
        if (pan, tilt) != (self.camera_pan, self.camera_tilt):
            self.camera_pan, self.camera_tilt = pan, tilt
            self.set_camera(pan, tilt, duration=duration)
        return self.get_camera_angles()

    def centre_camera_view(self):
        """
        Points the camera straight ahead (pan 0, tilt 0) and remembers it, so the dashboard's
        hold-down pan/tilt buttons carry on from the centre. Ends any held dashboard action first.

        Returns:
            tuple: The camera's new (pan, tilt) in degrees.
        """
        self.release_hold()
        self.camera_pan, self.camera_tilt = 0, 0
        self.set_camera(0, 0, duration=0.2)
        return self.get_camera_angles()

    def set_led_colour(self, colour=None):
        """
        Shows a colour on the robot's two RGB LEDs.

        Args:
            colour (str, optional): A colour name such as 'red' or 'blue'. None turns the LEDs off.
        """
        self.turn_on_led(*LED_COLOURS.get(colour, (0, 0, 0)))

    #---SHUTDOWN------------------------------------------------------------------------
    def emergency_shutdown(self, reason):
        """
        Shuts the whole robot down because something went wrong, e.g. a routine crashed. Routines, tasks and
        behaviours call this from their 'except' block. It only runs once, even if several things fail.

        Args:
            reason (str): What went wrong. Kept in self.shutdown_reason, so the dashboard can show it.
        """
        with self.state_lock:
            if self.is_shutdown:
                return
        self.shutdown_reason = reason
        self.logger.error(f"EMERGENCY SHUTDOWN: {reason}")
        self.shutdown()

    def shutdown(self):
        """
        Safely powers down all systems: stops behaviours, the task and routine, the motors and the background
        camera threads, and clears the LEDs. Always call this when exiting the script to prevent hardware damage.
        Calling it again does nothing.
        """
        with self.state_lock:
            if self.is_shutdown:
                return
            self.is_shutdown = True # from now on nothing new can start, and clean-up code knows not to run
            self.task = None
            self.routine = "Shutdown"
        self.logger.info("Shutting down")
        self.cancel_motion() # any manoeuvre ends within one step
        # Behaviours first, before anything they use is stopped. Wait for each one properly.
        for name in list(self.behaviours):
            behaviour = self.behaviours.pop(name)
            if not behaviour.stop(timeout=3.0):
                self.logger.error(f"Behaviour {name} didn't stop within 3 s")
        self._hold_quit.set() # end the dashboard safety loop before the ROS connection goes
        if self._hold_thread is not threading.current_thread():
            self._hold_thread.join(timeout=1.0)
        self.SOUND.stop_listening() # no more voice commands once the robot is shutting down (it can still speak)
        # Stop in reverse order of creation: mapping uses the lidar and camera, which all use the ROS connection
        if self._mapping is not None:
            self._mapping.stop()
        if self._lidar is not None:
            self._lidar.stop()
        self.CAMERA.end_detection()
        self.CAMERA.stop()
        self.LINE_FOLLOWER.close()
        self.turn_off_led()
        super().shutdown() # releases the ROS connection
        return

# TEST ROBOT CODE (the routines and tasks are tested in robot.py)
if __name__ == '__main__':
    ROBOT = RobotInterface()

    try:
        ROBOT.stop()
        print("\033c")
        input("Press Enter to Start: ")

        ROBOT.center_camera()
        ROBOT.SOUND.say("Robot ready")
        print("Line follower:", ROBOT.get_line_sensors())

    except KeyboardInterrupt:
        # Failsafe if you cancel the script mid-movement!
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the robot stops moving and shuts down cleanly
        ROBOT.shutdown()
        sys.exit(0)
