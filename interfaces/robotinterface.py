# -*- coding: utf-8 -*-
# container for more advanced functionality
# use Ctrl + K, Ctrl + 0 to minimise all coding blocks in VSCODE
import time, sys, os, math, logging, threading
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from clawpiinterface import (ClawPi, SERVO_LIMITS, CAMERA_PAN_SERVO_ID, CAMERA_TILT_SERVO_ID, CAMERA_PAN_TRIM,
                             CAMERA_TILT_TRIM, CLAW_GRIP_SERVO_ID, CLAW_LIFT_SERVO_ID, CLAW_OPEN, CLAW_CLOSED,
                             CLAW_LIFT_UP, CLAW_LIFT_DOWN)
from camerainterface import CameraInterface
from lidarinterface import LidarInterface
from linefollowerinterface import LineFollowerInterface
from mappinginterface import MappingInterface
from loggerinterface import get_logger

# RGB values used to show a detection colour on the robot's LEDs
LED_COLOURS = {
    'red': (255, 0, 0), 'green': (0, 255, 0), 'dark_green': (0, 100, 0), 'blue': (0, 0, 255),
    'yellow': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
}

# Speeds used by the movement routines (defined in robot_routines/__init__.py)
from robot_routines import DRIVE_SPEED, TURN_SPEED
from robot_routines.move_toward_colour_detected import move_toward_colour_detected
from robot_routines.move_direction_until_detection import move_direction_until_detection
from robot_routines.face_tracking import face_tracking

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

    def __init__(self):
        """
        Initializes the Robot Interface, sets up sound and camera modules,
        and waits for hardware to warm up.
        """
        self.routine = "Ready" # keep track of which routine is running
        self._hold = (None, 0)  # (held dashboard action, time it must stop by)
        self._hold_lock = threading.Lock()
        self._hold_quit = threading.Event()
        self.camera_pan, self.camera_tilt = 0, 0
        self.show_camera = False # only turn this on to see the camera window, will slow down performance when using VNC
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

    def stop_routine(self):
        """
        Safely stops the robot's current movement process and resets the routine state.
        Call this anytime you need to abort a running while-loop instantly.
        """
        self.routine = "Ready"
        with self._hold_lock:
            self._hold = (None, 0)
        self.stop()
        return

    #---DASHBOARD HOLD CONTROLS-------------------------------------------------------
    def hold_action(self, action):
        """
        Starts (or keeps going) a dashboard hold-down action, e.g. 'forward', 'camera_up' or 'claw_open'.
        The action stops by itself HOLD_TIMEOUT seconds after the last call, so call this again
        (the dashboard does every 0.5 s) for as long as the button is held, and release_hold() when let go.

        Returns:
            bool: False if the action isn't known.
        """
        if action not in HOLD_ACTIONS:
            return False
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

    def show_camera_window(self):
        """
        Renders the OpenCV display window.
        If self.show_camera is False, it implements a small artificial sleep.
        This is a critical performance fix to prevent headless while-loops from
        spinning out of control and maxing out the CPU.

        Returns:
            bool: True normally, False if the user presses 'q' on the keyboard.
        """
        if not self.show_camera:
            # CRITICAL FIX: Add a small delay to prevent while-loops from running unthrottled
            time.sleep(0.03)
            return True

        frame = self.CAMERA.get_frame()
        cv2.imshow('Detection Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    #---ROUTINES (each one lives in its own file in robot_routines/)----------------
    move_toward_colour_detected = move_toward_colour_detected
    move_direction_until_detection = move_direction_until_detection
    face_tracking = face_tracking

    def shutdown(self):
        """
        Safely powers down all systems. Stops motors, stops the background camera threads,
        and clears the LEDs. Always call this when exiting the script to prevent hardware damage.
        """
        self.routine = "Shutdown"
        self._hold_quit.set() # end the dashboard safety loop before the ROS connection goes
        self._hold_thread.join(timeout=1.0)
        self.SOUND.stop_listening() # no more voice commands once the robot is shutting down (it can still speak)
        self.stop_routine()
        # Stop in reverse order of creation: mapping uses the lidar and camera, which all use the ROS connection
        if self._mapping is not None:
            self._mapping.stop()
        if self._lidar is not None:
            self._lidar.stop()
        self.CAMERA.stop()
        self.LINE_FOLLOWER.close()
        self.turn_off_led()
        super().shutdown() # releases the ROS connection
        return

# TEST ROBOT CODE
if __name__ == '__main__':
    ROBOT = RobotInterface()

    try:
        ROBOT.stop()
        print("\033c")
        input("Press Enter to Start: ")

        ROBOT.center_camera()
        ROBOT.SOUND.say("Robot ready")

        # Uncomment below to test movements
        data = ROBOT.move_direction_until_detection(movetype='forward', distanceto=50, detection_types=['lidar','colour'], confirmlevel=2, detection_colours=['red'], timelimit=10)
        print(data)

    except KeyboardInterrupt:
        # Failsafe if you cancel the script mid-movement!
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the robot stops moving and shuts down cleanly
        ROBOT.shutdown()
        sys.exit(0)
