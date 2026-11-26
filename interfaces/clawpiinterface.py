#!/usr/bin/env python3
# encoding: utf-8
"""
clawpiinterface.py - simple Python API for the Hiwonder ClawPi (mecanum + claw).

Talks to the robot through the ROS 2 topics published by the stock bringup
(ros_robot_controller, odom_publisher, imu filter, emotion_to_lcd), so the
bringup must already be running (it starts automatically on boot). The camera
is handled by camerainterface.py and the lidar by lidarinterface.py.

Run scripts inside the ClawPi container as the ubuntu user (running as root
cannot see the robot's sensor data), with the ROS environment sourced, e.g.

    docker exec -it -u ubuntu <container-name> zsh
    cd ~/shared/clawpitemplate && python3 my_script.py

The ClawPi has four PWM servos (pulse width in microseconds):

    1  camera tilt   (gimbal pitch)
    2  camera pan    (gimbal yaw)
    3  claw lift     (raises / lowers the claw)
    4  claw grip     (opens / closes the claw)

Servo calibration offsets (software/Servo_upper_computer/servo_config.yaml) are
applied by the controller board itself, so this file sends plain pulses.

Example:

    from interfaces.clawpiinterface import ClawPi

    with ClawPi() as robot:
        robot.move_forward(2)                    # 2 s forward
        robot.move_sideways(1, 'left')           # 1 s strafe left
        robot.rotate(1.5, 'clockwise')
        robot.pick_up()                          # close claw, then lift
        robot.put_down()                         # lower, then open claw
        robot.show_emotion('happy')
        robot.beep()
"""
import json
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String, UInt16
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from loggerinterface import get_logger

from ros_robot_controller_msgs.msg import (
    BuzzerState, ButtonState, LedState, MotorState, MotorsState,
    PWMServoState, RGBState, RGBStates, SetPWMServoState
)

# Topics provided by the stock ClawPi bringup.
CMD_VEL_TOPIC = '/controller/cmd_vel'
MOTOR_TOPIC = '/ros_robot_controller/set_motor'
RGB_TOPIC = '/ros_robot_controller/set_rgb'
LED_TOPIC = '/ros_robot_controller/set_led'
BUZZER_TOPIC = '/ros_robot_controller/set_buzzer'
PWM_SERVO_TOPIC = '/ros_robot_controller/pwm_servo/set_state'
BATTERY_TOPIC = '/ros_robot_controller/battery'
BUTTON_TOPIC = '/ros_robot_controller/button'
IMU_TOPIC = '/imu'                      # calibrated + complementary filtered
IMU_RAW_TOPIC = '/ros_robot_controller/imu_raw'
IMU_RPY_TOPIC = '/imu/rpy/filtered'
ODOM_TOPIC = '/odom'                    # EKF fused odometry
LCD_EMOTION_TOPIC = '/lcd_display/emotion'
LCD_TEXT_TOPIC = '/lcd_display/emotion_text'
BATTERY_FILE = '/home/ubuntu/shared/battery.txt'  # written by the driver every 5 s

# PWM servo IDs.
CAMERA_TILT_SERVO_ID = 1
CAMERA_PAN_SERVO_ID = 2
CLAW_LIFT_SERVO_ID = 3
CLAW_GRIP_SERVO_ID = 4

# Safe pulse range for each servo, from the stock
# driver/servo_controller/config/servo_controller.yaml.
SERVO_CENTER = 1500
SERVO_LIMITS = {
    CAMERA_TILT_SERVO_ID: (1000, 2400),
    CAMERA_PAN_SERVO_ID: (500, 2500),
    CLAW_LIFT_SERVO_ID: (1200, 2200),
    CLAW_GRIP_SERVO_ID: (1500, 2500),
}
DEFAULT_SERVO_LIMITS = (500, 2500)

# Positions used by Hiwonder's own pick-up demos
# (example/navigation_transport/automatic_pick.py).
CAMERA_TILT_HOME = 1200
CAMERA_PAN_HOME = 1500

# Extra software trim for the camera gimbal, in degrees, added on top of the
# board's calibration offsets. With these, pan 0 / tilt 0 points the camera
# straight ahead. If it is still off, nudge these values (positive pan/tilt
# moves the camera the same way as set_camera_pan(+x) / set_camera_tilt(+x)).
CAMERA_PAN_TRIM = 0.0
CAMERA_TILT_TRIM = 0.0
CLAW_LIFT_DOWN = 1200
CLAW_LIFT_UP = 1800
CLAW_CLOSED = 1700
CLAW_OPEN = 2200

# LCD face animations (lcd_display_pkg/config/emotions.yaml).
EMOTIONS = (
    'wink', 'proud', 'evil', 'dissatisfied', 'playful', 'money', 'happy',
    'sad', 'tsundere', 'cute', 'wronged', 'awkward', 'crying',
)

# Safety limits for velocity commands.
MAX_LINEAR_SPEED = 0.4    # m/s
MAX_ANGULAR_SPEED = 3.0   # rad/s
CMD_RATE_HZ = 20.0        # re-publish rate while a timed move is running


def _clamp(value, low, high):
    return max(low, min(high, value))


def _quaternion_to_rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(_clamp(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class ClawPi:
    """High level interface to the ClawPi robot.

    Movement functions block for the requested time and then stop the robot.
    Servo functions are non-blocking unless wait=True is passed.
    Sensor getters return the most recent message received (or None if no
    data has arrived yet).
    """

    def __init__(self, node_name='clawpi_interface', default_speed=0.2,
                 default_turn_speed=1.0):
        self.hardware_logger = get_logger('ClawPi') # logs/clawpi.log
        self.default_speed = default_speed
        self.default_turn_speed = default_turn_speed

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.node = Node(node_name)

        self._lock = threading.Lock()
        self._latest = {}
        self._events = {}
        self._button_callbacks = []
        self._servo_positions = {}  # last pulse sent to each servo

        # Publishers
        self._cmd_vel_pub = self.node.create_publisher(Twist, CMD_VEL_TOPIC, 1)
        self._motor_pub = self.node.create_publisher(MotorsState, MOTOR_TOPIC, 10)
        self._rgb_pub = self.node.create_publisher(RGBStates, RGB_TOPIC, 10)
        self._led_pub = self.node.create_publisher(LedState, LED_TOPIC, 10)
        self._buzzer_pub = self.node.create_publisher(BuzzerState, BUZZER_TOPIC, 10)
        self._servo_pub = self.node.create_publisher(SetPWMServoState, PWM_SERVO_TOPIC, 10)
        self._emotion_pub = self.node.create_publisher(String, LCD_EMOTION_TOPIC, 10)
        self._lcd_text_pub = self.node.create_publisher(String, LCD_TEXT_TOPIC, 10)

        # Subscribers: just cache the latest message of each.
        self._subscribe('imu', Imu, IMU_TOPIC)
        self._subscribe('imu_raw', Imu, IMU_RAW_TOPIC)
        self._subscribe('imu_rpy', Vector3Stamped, IMU_RPY_TOPIC)
        self._subscribe('odom', Odometry, ODOM_TOPIC)
        self._subscribe('battery', UInt16, BATTERY_TOPIC, qos=10)
        self._subscribe('button', ButtonState, BUTTON_TOPIC, qos=10,
                        callback=self._on_button)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Give DDS a moment to match publishers/subscribers.
        time.sleep(0.5)
        self.hardware_logger.info(f"Connected to the robot through ROS 2 node '{node_name}'")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def shutdown(self):
        """Stop the robot and release ROS resources."""
        try:
            self.stop()
            time.sleep(0.1)
        finally:
            # Let the spin thread finish any in-flight callback before the
            # node is destroyed, otherwise rclpy logs a "Destroyable" error.
            self._executor.shutdown(timeout_sec=1.0)
            self._spin_thread.join(timeout=1.0)
            self._executor.remove_node(self.node)
            self.node.destroy_node()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _subscribe(self, key, msg_type, topic, qos=qos_profile_sensor_data, callback=None):
        self._events[key] = threading.Event()

        def cb(msg):
            with self._lock:
                self._latest[key] = msg
            self._events[key].set()
            if callback:
                callback(msg)

        self.node.create_subscription(msg_type, topic, cb, qos)

    def _get(self, key, timeout):
        """Return the latest message for key, waiting up to timeout seconds."""
        if timeout and not self._events[key].is_set():
            self._events[key].wait(timeout)
        with self._lock:
            return self._latest.get(key)

    def _wait_fresh(self, key, timeout):
        """Wait for a message that arrives after this call."""
        self._events[key].clear()
        self._events[key].wait(timeout)
        with self._lock:
            return self._latest.get(key)

    # ------------------------------------------------------------------ #
    # Movement (mecanum chassis)
    # ------------------------------------------------------------------ #
    def set_velocity(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """Send a single velocity command (non-blocking).

        linear_x: forward(+)/backward(-) m/s
        linear_y: left(+)/right(-) m/s
        angular_z: anticlockwise(+)/clockwise(-) rad/s

        The robot keeps moving until stop() or another command is sent.
        """
        msg = Twist()
        msg.linear.x = float(_clamp(linear_x, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        msg.linear.y = float(_clamp(linear_y, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        msg.angular.z = float(_clamp(angular_z, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        self._cmd_vel_pub.publish(msg)

    def move(self, duration, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """Drive with the given velocities for duration seconds, then stop."""
        end = time.monotonic() + duration
        try:
            while time.monotonic() < end:
                self.set_velocity(linear_x, linear_y, angular_z)
                time.sleep(min(1.0 / CMD_RATE_HZ, max(0.0, end - time.monotonic())))
        finally:
            self.stop()

    def stop(self):
        """Stop all wheel motion."""
        self._cmd_vel_pub.publish(Twist())

    def move_forward(self, duration, direction='forward', speed=None):
        """Drive straight for duration seconds.

        direction: 'forward' or 'backward'
        speed: m/s (default self.default_speed)
        """
        speed = abs(speed if speed is not None else self.default_speed)
        if direction not in ('forward', 'backward'):
            raise ValueError("direction must be 'forward' or 'backward'")
        self.move(duration, linear_x=speed if direction == 'forward' else -speed)

    def move_backward(self, duration, speed=None):
        """Drive backwards for duration seconds."""
        self.move_forward(duration, 'backward', speed)

    def move_sideways(self, duration, direction='left', speed=None):
        """Strafe sideways (mecanum) for duration seconds.

        direction: 'left' or 'right'
        """
        speed = abs(speed if speed is not None else self.default_speed)
        if direction not in ('left', 'right'):
            raise ValueError("direction must be 'left' or 'right'")
        self.move(duration, linear_y=speed if direction == 'left' else -speed)

    def move_diagonal(self, duration, forward=True, left=True, speed=None):
        """Move diagonally (mecanum) for duration seconds."""
        speed = abs(speed if speed is not None else self.default_speed) / math.sqrt(2)
        self.move(duration,
                  linear_x=speed if forward else -speed,
                  linear_y=speed if left else -speed)

    def rotate(self, duration, direction='clockwise', speed=None):
        """Spin on the spot for duration seconds.

        direction: 'clockwise' or 'anticlockwise'
        speed: rad/s (default self.default_turn_speed)
        """
        speed = abs(speed if speed is not None else self.default_turn_speed)
        if direction not in ('clockwise', 'anticlockwise'):
            raise ValueError("direction must be 'clockwise' or 'anticlockwise'")
        self.move(duration, angular_z=-speed if direction == 'clockwise' else speed)

    def rotate_degrees(self, degrees, speed=None, timeout=10.0):
        """Rotate by an angle using the odometry yaw for feedback.

        Positive degrees = anticlockwise, negative = clockwise.
        """
        speed = abs(speed if speed is not None else self.default_turn_speed)
        start = self.get_yaw()
        if start is None:
            raise RuntimeError('No odometry received; is the bringup running?')
        target = math.radians(degrees)
        turned, last = 0.0, start
        end = time.monotonic() + timeout
        try:
            while abs(turned) < abs(target) and time.monotonic() < end:
                remaining = abs(target) - abs(turned)
                # slow down over the last ~20 degrees to limit overshoot
                w = max(0.3, min(speed, remaining * 2.0))
                self.set_velocity(angular_z=math.copysign(w, target))
                time.sleep(1.0 / CMD_RATE_HZ)
                yaw = self.get_yaw()
                turned += math.atan2(math.sin(yaw - last), math.cos(yaw - last))
                last = yaw
        finally:
            self.stop()

    def set_motor_speeds(self, m1, m2, m3, m4):
        """Set the four wheel motors directly in revolutions per second.

        Bypasses the kinematics; note the right-side motors are mounted
        mirrored (see controller/mecanum.py). Call stop_motors() to stop.
        """
        msg = MotorsState()
        msg.data = [MotorState(id=i + 1, rps=float(v)) for i, v in enumerate((m1, m2, m3, m4))]
        self._motor_pub.publish(msg)

    def stop_motors(self):
        """Set all four motors to zero rps."""
        self.set_motor_speeds(0, 0, 0, 0)

    # ------------------------------------------------------------------ #
    # IMU / odometry
    # ------------------------------------------------------------------ #
    def get_imu_sensor(self, timeout=1.0):
        """Return the filtered IMU reading as a dict, or None.

        {
          'accel': (x, y, z) m/s^2,
          'gyro': (x, y, z) rad/s,
          'orientation': (x, y, z, w) quaternion,
          'rpy': (roll, pitch, yaw) degrees,
        }
        """
        msg = self._get('imu', timeout)
        if msg is None:
            return None
        a, g, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        roll, pitch, yaw = _quaternion_to_rpy(q)
        return {
            'accel': (a.x, a.y, a.z),
            'gyro': (g.x, g.y, g.z),
            'orientation': (q.x, q.y, q.z, q.w),
            'rpy': (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
        }

    def get_imu_raw(self, timeout=1.0):
        """Return the uncalibrated IMU reading: {'accel': (...), 'gyro': (...)}."""
        msg = self._get('imu_raw', timeout)
        if msg is None:
            return None
        a, g = msg.linear_acceleration, msg.angular_velocity
        return {'accel': (a.x, a.y, a.z), 'gyro': (g.x, g.y, g.z)}

    def get_orientation(self, timeout=1.0):
        """Return (roll, pitch, yaw) in degrees from the IMU filter."""
        msg = self._get('imu_rpy', timeout)
        if msg is None:
            return None
        v = msg.vector
        return math.degrees(v.x), math.degrees(v.y), math.degrees(v.z)

    def get_odometry(self, timeout=1.0):
        """Return fused odometry as a dict, or None.

        {'x': m, 'y': m, 'yaw': degrees, 'vx': m/s, 'vy': m/s, 'wz': rad/s}
        """
        msg = self._get('odom', timeout)
        if msg is None:
            return None
        p, t = msg.pose.pose, msg.twist.twist
        _, _, yaw = _quaternion_to_rpy(p.orientation)
        return {
            'x': p.position.x, 'y': p.position.y, 'yaw': math.degrees(yaw),
            'vx': t.linear.x, 'vy': t.linear.y, 'wz': t.angular.z,
        }

    def get_yaw(self, timeout=1.0):
        """Return the odometry heading in radians, or None."""
        msg = self._get('odom', timeout)
        if msg is None:
            return None
        return _quaternion_to_rpy(msg.pose.pose.orientation)[2]

    # ------------------------------------------------------------------ #
    # Servos (generic)
    # ------------------------------------------------------------------ #
    def set_servos(self, positions, duration=0.5, wait=False):
        """Move several PWM servos together.

        positions: {servo_id: pulse_us, ...}. Each pulse is clamped to that
                   servo's safe range (see SERVO_LIMITS).
        duration: seconds the move should take.
        wait: if True, block until the move has had time to finish.
        """
        msg = SetPWMServoState()
        msg.duration = float(duration)
        for servo_id, pulse in positions.items():
            low, high = SERVO_LIMITS.get(int(servo_id), DEFAULT_SERVO_LIMITS)
            pulse = int(_clamp(pulse, low, high))
            state = PWMServoState()
            state.id = [int(servo_id)]
            state.position = [pulse]
            msg.state.append(state)
            self._servo_positions[int(servo_id)] = pulse
        self._servo_pub.publish(msg)
        if wait:
            time.sleep(duration)

    def set_servo(self, servo_id, pulse, duration=0.5, wait=False):
        """Move one PWM servo to pulse (microseconds, 1500 = centre)."""
        self.set_servos({servo_id: pulse}, duration, wait)

    def set_servo_angle(self, servo_id, angle, duration=0.5, wait=False):
        """Move a PWM servo to angle in degrees (-90..90, 0 = 1500 us centre).

        The result is still clamped to the servo's safe range, so some servos
        cannot reach the full -90..90.
        """
        angle = _clamp(angle, -90, 90)
        self.set_servo(servo_id, SERVO_CENTER + angle / 90.0 * 1000, duration, wait)

    def get_servo_position(self, servo_id):
        """Return the last pulse this interface sent to servo_id, or None.

        This is what was commanded, not a reading from the servo.
        """
        return self._servo_positions.get(int(servo_id))

    # ------------------------------------------------------------------ #
    # Camera gimbal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _camera_pulse(angle, trim):
        angle = _clamp(angle + trim, -90, 90)
        return SERVO_CENTER + angle / 90.0 * 1000

    def set_camera(self, pan=0.0, tilt=0.0, duration=0.5, wait=False):
        """Point the camera (degrees, 0/0 = straight ahead, trim applied)."""
        self.set_servos({
            CAMERA_PAN_SERVO_ID: self._camera_pulse(pan, CAMERA_PAN_TRIM),
            CAMERA_TILT_SERVO_ID: self._camera_pulse(tilt, CAMERA_TILT_TRIM),
        }, duration, wait)

    def set_camera_pan(self, angle, duration=0.5, wait=False):
        """Pan the camera left/right (degrees, 0 = straight ahead)."""
        self.set_servo(CAMERA_PAN_SERVO_ID,
                       self._camera_pulse(angle, CAMERA_PAN_TRIM), duration, wait)

    def set_camera_tilt(self, angle, duration=0.5, wait=False):
        """Tilt the camera up/down (degrees, 0 = level)."""
        self.set_servo(CAMERA_TILT_SERVO_ID,
                       self._camera_pulse(angle, CAMERA_TILT_TRIM), duration, wait)

    def center_camera(self, duration=0.5, wait=False):
        """Point the camera straight ahead (pan 0, tilt 0)."""
        self.set_camera(0, 0, duration, wait)

    def home_camera(self, duration=0.5, wait=False):
        """Point the camera straight ahead (same as center_camera)."""
        self.set_camera(0, 0, duration, wait)

    # ------------------------------------------------------------------ #
    # Claw
    # ------------------------------------------------------------------ #
    def open_claw(self, duration=0.5, wait=True):
        """Open the claw fully."""
        self.set_servo(CLAW_GRIP_SERVO_ID, CLAW_OPEN, duration, wait)

    def close_claw(self, duration=0.5, wait=True):
        """Close the claw (grip)."""
        self.set_servo(CLAW_GRIP_SERVO_ID, CLAW_CLOSED, duration, wait)

    def set_claw(self, amount, duration=0.5, wait=True):
        """Set how far open the claw is: 0.0 = closed, 1.0 = fully open."""
        amount = _clamp(amount, 0.0, 1.0)
        self.set_servo(CLAW_GRIP_SERVO_ID,
                       CLAW_CLOSED + amount * (CLAW_OPEN - CLAW_CLOSED), duration, wait)

    def lift_claw(self, duration=0.5, wait=True):
        """Raise the claw to its carrying height."""
        self.set_servo(CLAW_LIFT_SERVO_ID, CLAW_LIFT_UP, duration, wait)

    def lower_claw(self, duration=0.5, wait=True):
        """Lower the claw to the floor (pick-up height)."""
        self.set_servo(CLAW_LIFT_SERVO_ID, CLAW_LIFT_DOWN, duration, wait)

    def set_lift(self, amount, duration=0.5, wait=True):
        """Set the claw height: 0.0 = down (floor), 1.0 = up (carrying)."""
        amount = _clamp(amount, 0.0, 1.0)
        self.set_servo(CLAW_LIFT_SERVO_ID,
                       CLAW_LIFT_DOWN + amount * (CLAW_LIFT_UP - CLAW_LIFT_DOWN), duration, wait)

    def pick_up(self, duration=0.5):
        """Grab whatever is in front of the claw: lower, close, then lift."""
        self.lower_claw(duration)
        self.close_claw(duration)
        self.lift_claw(duration)

    def put_down(self, duration=0.5):
        """Release the held object: lower, open, then lift clear."""
        self.lower_claw(duration)
        self.open_claw(duration)
        self.lift_claw(duration)

    def reset_pose(self, duration=0.5, wait=True):
        """Move every servo to Hiwonder's start-up pose.

        Camera tilted forward and centred, claw down and open.
        """
        self.set_servos({
            CAMERA_TILT_SERVO_ID: CAMERA_TILT_HOME,
            CAMERA_PAN_SERVO_ID: CAMERA_PAN_HOME,
            CLAW_LIFT_SERVO_ID: CLAW_LIFT_DOWN,
            CLAW_GRIP_SERVO_ID: CLAW_OPEN,
        }, duration, wait)

    # ------------------------------------------------------------------ #
    # LCD screen
    # ------------------------------------------------------------------ #
    def show_emotion(self, emotion='happy'):
        """Play a face animation on the LCD screen.

        emotion: one of EMOTIONS, e.g. 'happy', 'sad', 'wink', 'cute'.
        """
        if emotion not in EMOTIONS:
            raise ValueError(f'emotion must be one of {", ".join(EMOTIONS)}')
        self._emotion_pub.publish(String(data=emotion))

    def stop_emotion(self):
        """Stop the face animation on the LCD screen."""
        self._emotion_pub.publish(String(data='stop'))

    def show_lcd_text(self, text, emotion='happy', duration=6.0):
        """Show a line of text on the LCD screen, over a face animation.

        The LCD service only draws text as a caption on top of a playing face;
        text sent with no face playing is silently held back. So this starts
        the given face first, looping for `duration` seconds, then sends the
        text for the same time.

        emotion: one of EMOTIONS to play behind the text.
        duration: seconds to keep the face and text on screen.
        """
        if emotion not in EMOTIONS:
            raise ValueError(f'emotion must be one of {", ".join(EMOTIONS)}')
        duration = float(duration)
        self._emotion_pub.publish(String(data=json.dumps(
            {'emotion': emotion, 'loop_count': 0, 'duration_seconds': duration})))
        time.sleep(0.3)  # let the face start, or the text is deferred
        self._lcd_text_pub.publish(String(data=json.dumps(
            {'text': str(text), 'duration_seconds': duration})))

    # ------------------------------------------------------------------ #
    # LEDs, buzzer
    # ------------------------------------------------------------------ #
    def set_rgb_led(self, index, red, green, blue):
        """Set one RGB LED on the controller board (index 1 or 2).

        Not yet tested on the ClawPi board; may have no visible effect.
        """
        msg = RGBStates()
        msg.states = [RGBState(index=int(index), red=int(red), green=int(green), blue=int(blue))]
        self._rgb_pub.publish(msg)

    def turn_on_led(self, red=255, green=255, blue=255, index=None):
        """Turn on the RGB LEDs. index=None sets both LEDs.

        Not yet tested on the ClawPi board; may have no visible effect.
        """
        indices = [1, 2] if index is None else [index]
        msg = RGBStates()
        msg.states = [RGBState(index=i, red=int(red), green=int(green), blue=int(blue))
                      for i in indices]
        self._rgb_pub.publish(msg)

    def turn_off_led(self, index=None):
        """Turn off the RGB LEDs. index=None turns off both."""
        self.turn_on_led(0, 0, 0, index)

    def blink_status_led(self, on_time=0.2, off_time=0.2, repeat=3, led_id=1):
        """Blink a single-colour status LED on the controller board (id 1 or 2)."""
        msg = LedState()
        msg.id = int(led_id)
        msg.on_time = float(on_time)
        msg.off_time = float(off_time)
        msg.repeat = int(repeat)
        self._led_pub.publish(msg)

    def beep(self, frequency=1900, on_time=0.1, off_time=0.1, repeat=1):
        """Sound the buzzer (non-blocking)."""
        msg = BuzzerState()
        msg.freq = int(frequency)
        msg.on_time = float(on_time)
        msg.off_time = float(off_time)
        msg.repeat = int(repeat)
        self._buzzer_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Battery / button
    # ------------------------------------------------------------------ #
    def get_battery_voltage(self, timeout=3.0):
        """Return the battery voltage in volts, or None.

        The board reports about once a second. If no report arrives in time,
        falls back to the value the driver saves to BATTERY_FILE (if recent).
        """
        msg = self._get('battery', timeout)
        if msg is not None:
            return msg.data / 1000.0
        try:
            if time.time() - os.path.getmtime(BATTERY_FILE) < 30:
                with open(BATTERY_FILE) as f:
                    return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
        return None

    def get_button(self, timeout=0.0):
        """Return the last button event as (button_id, state), or None.

        state: 0 released, 1 pressed, 2 long press, 3 long press repeat,
               4 released after long press, 5 click, 6 double click,
               7 triple click
        """
        msg = self._get('button', timeout)
        return None if msg is None else (msg.id, msg.state)

    def wait_for_button(self, button_id=None, timeout=None):
        """Block until a button is clicked. Returns (button_id, state) or None."""
        end = None if timeout is None else time.monotonic() + timeout
        while end is None or time.monotonic() < end:
            remaining = None if end is None else max(0.0, end - time.monotonic())
            msg = self._wait_fresh('button', remaining if remaining is not None else 3600)
            if msg is not None and msg.state in (0, 5) and \
                    (button_id is None or msg.id == button_id):
                return msg.id, msg.state
        return None

    def on_button(self, callback):
        """Register callback(button_id, state) called on every button event."""
        self._button_callbacks.append(callback)

    def _on_button(self, msg):
        for cb in self._button_callbacks:
            try:
                cb(msg.id, msg.state)
            except Exception as e:  # don't let user code kill the executor
                self.hardware_logger.error(f'Button {msg.id} callback {getattr(cb, "__name__", cb)} failed: {e}')


if __name__ == '__main__':
    # Self-test: read sensors, move the servos, beep, then run a driving
    # sequence. The robot WILL drive - give it about 1 m of clear space.
    #
    # Movement calls block until finished and stop the robot themselves, so
    # the only extra wait needed is the 1 s pause between commands.
    with ClawPi() as robot:
        def step(label, action):
            """Show label on the LCD + console, run action, then pause 1 s."""
            print(label)
            robot.show_lcd_text(label)
            action()
            robot.stop()
            time.sleep(1.0)

        print('battery  :', robot.get_battery_voltage(), 'V')
        print('imu      :', robot.get_imu_sensor())
        print('rpy      :', robot.get_orientation())
        print('odometry :', robot.get_odometry())

        input("Press Enter to test...")

        # --- buzzer ---
        step('Beep',
             lambda: robot.beep())
 
        # --- servos ---
        '''step('Reset pose',
             lambda: robot.reset_pose())
        step('Lift claw',
             lambda: robot.lift_claw())
        step('Open claw',
             lambda: robot.open_claw())
        step('Close claw',
             lambda: robot.close_claw())
        step('Lower claw',
             lambda: robot.lower_claw())'''
        step('Camera home',
             lambda: robot.home_camera(wait=True))
        time.sleep(1)
        step('Camera pan +30',
             lambda: robot.set_camera_pan(30, wait=True))
        step('Camera pan -30',
             lambda: robot.set_camera_pan(-30, wait=True))
        step('Camera home',
             lambda: robot.home_camera(wait=True))
        time.sleep(1)
        step('Camera tilt +30',
             lambda: robot.set_camera_tilt(30, wait=True))
        step('Camera tilt -30',
             lambda: robot.set_camera_tilt(-30, wait=True))
        step('Camera home',
             lambda: robot.home_camera(wait=True))
        time.sleep(1)
        '''
        # --- driving ---
        step('Forward 2s',
             lambda: robot.move_forward(2))
        step('Rotate 2s',
             lambda: robot.rotate(2, 'clockwise'))
        step('Rotate 90 deg',
             lambda: robot.rotate_degrees(90))          # + = anticlockwise
        step('Diagonal 2s',
             lambda: robot.move_diagonal(2, forward=True, left=True))
        # Steering: forward speed + turn rate = drive along an arc (left).
        step('Steer 2s',
             lambda: robot.move(2, linear_x=0.2, angular_z=0.5))
        # Orbiting: strafe + turn so the robot stays facing a point
        # r = 0.3 m in front of it (angular_z = -linear_y / r).
        step('Orbit 2s',
             lambda: robot.move(2, linear_y=0.15, angular_z=-0.15 / 0.3))

        step('Test done',
             lambda: None)

        # --- LCD face: last, so nothing overwrites it ---
        print('Happy face')
        robot.show_emotion('happy')
        time.sleep(3.0)                  # leave it on screen for a while
        #robot.stop_emotion()
        '''