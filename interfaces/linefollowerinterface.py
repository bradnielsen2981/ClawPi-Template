# -*- coding: utf-8 -*-
"""
linefollowerinterface.py - the ClawPi's 4 channel line follower sensor (the infrared sensor under the front).

Based on Hiwonder's driver (ros2_ws/src/driver/sdk/sdk/FourInfrared.py). The sensor is read straight over
I2C (bus 1, address 0x78), not through ROS. One byte holds all four sensors, one bit each.

The sensors are numbered left to right as seen from behind the robot:

    [0] far left   [1] centre left   [2] centre right   [3] far right

True means that sensor can see the line. The sensor itself only tells dark (little infrared reflected) from
light, so the mode says which one is the line:

    'dark_line'   a dark line on a light floor (the default, e.g. black tape on a white table)
    'light_line'  a light line on a dark floor (e.g. white tape on a dark mat)

    sensor = LineFollowerInterface(mode='light_line')   # or later: sensor.set_mode('light_line')

Example:

    from linefollowerinterface import LineFollowerInterface

    sensor = LineFollowerInterface()
    print(sensor.read())               # [False, True, True, False] = on the line, centred
    print(sensor.get_line_position())  # 0.0 = centred, -1 = line under the far left sensor, 1 = far right

The ubuntu user needs permission to use /dev/i2c-1. If it doesn't have it, the sensor reports
available = False and read() returns None instead of stopping the robot from loading.
"""
import threading

from loggerinterface import get_logger

try:
    import smbus
except ImportError:
    import smbus2 as smbus

LINE_SENSOR_BUS = 1          # /dev/i2c-1
LINE_SENSOR_ADDRESS = 0x78
LINE_SENSOR_REGISTER = 0x01
SENSOR_BITS = (0x01, 0x02, 0x04, 0x08)   # sensor 0 (far left) .. sensor 3 (far right)
SENSOR_POSITIONS = (-1.0, -1 / 3, 1 / 3, 1.0)  # where each sensor sits, -1 = far left, 1 = far right
# Crossing or T-junction: all four see the line, or three including both centre sensors
# (Hiwonder's junction patterns from self_driving_line_follower.py)
JUNCTIONS = ([True, True, True, True], [True, True, True, False], [False, True, True, True])
MODES = ('dark_line', 'light_line') # what the line looks like (see the top of this file)


def line_position(sensors):
    """-1.0 (far left) to 1.0 (far right) from the sensors that see the line, or None if none do."""
    seen = [position for position, on in zip(SENSOR_POSITIONS, sensors) if on]
    return sum(seen) / len(seen) if seen else None


class LineFollowerInterface():

    def __init__(self, bus=LINE_SENSOR_BUS, address=LINE_SENSOR_ADDRESS, mode='dark_line'):
        """
        Opens the I2C bus and checks the sensor answers.

        Args:
            bus (int, optional): I2C bus number. Defaults to 1 (/dev/i2c-1).
            address (int, optional): The sensor's I2C address. Defaults to 0x78.
            mode (str, optional): 'dark_line' (dark line on a light floor) or 'light_line' (light line on a
                                  dark floor). Defaults to 'dark_line'.
        """
        self.mode = 'dark_line'
        self.set_mode(mode)
        self.bus_number = bus
        self.address = address
        self.logger = get_logger('LineFollower') # logs/linefollower.log
        self.available = False   # True once the sensor has answered
        self.error = None        # why the sensor can't be used, if it can't
        self.last_reading = None # the latest [far left, centre left, centre right, far right]
        self._lock = threading.Lock() # Flask threads may read at the same time
        self._bus = None
        self._read_failed = False # so a failing sensor is only logged once, not on every read

        try:
            self._bus = smbus.SMBus(bus)
            self._bus.read_byte_data(address, LINE_SENSOR_REGISTER) # does it answer?
            self.available = True
            self.logger.info(f"Line follower found at {hex(address)} on I2C bus {bus}")
        except PermissionError:
            self.error = f"no permission to use /dev/i2c-{bus} (the user needs access to the I2C bus)"
        except (OSError, IOError):
            self.error = f"no line follower found at {hex(address)} on I2C bus {bus}"
        if self.error:
            self.logger.error(f"Line follower unavailable: {self.error}")
            self.close()

    def set_mode(self, mode):
        """
        Chooses what counts as the line.

        Args:
            mode (str): 'dark_line' (dark line on a light floor) or 'light_line' (light line on a dark floor).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode

    def read(self):
        """
        Reads all four sensors.

        Returns:
            list: [far left, centre left, centre right, far right], True where the line is seen.
                  None if the sensor can't be read.
        """
        if not self.available:
            return None
        try:
            with self._lock:
                value = self._bus.read_byte_data(self.address, LINE_SENSOR_REGISTER)
        except (OSError, IOError) as e:
            if not self._read_failed:
                self.logger.error(f"Line follower read failed: {e}")
                self._read_failed = True
            return None
        self._read_failed = False
        dark = [bool(value & bit) for bit in SENSOR_BITS] # the sensor sets a bit when it sees dark
        self.last_reading = dark if self.mode == 'dark_line' else [not d for d in dark]
        return self.last_reading

    def is_on_line(self):
        """True if any sensor can see the line, False if none can, None if the sensor can't be read."""
        sensors = self.read()
        return None if sensors is None else any(sensors)

    def get_line_position(self):
        """
        Where the line is under the robot, worked out from the sensors that can see it.

        Returns:
            float: -1.0 (under the far left sensor) to 1.0 (far right), 0.0 = centred.
                   None if no sensor can see the line or the sensor can't be read.
        """
        sensors = self.read()
        return None if sensors is None else line_position(sensors)

    def is_junction(self):
        """
        True at a crossing or a T-junction (see JUNCTIONS), None if the sensor can't be read.
        """
        sensors = self.read()
        return None if sensors is None else sensors in JUNCTIONS

    def get_line_data(self):
        """
        Everything in one reading (handy for the dashboard).

        Returns:
            dict: {'available', 'error', 'mode', 'sensors', 'on_line', 'position', 'junction'}
        """
        sensors = self.read()
        data = {'available': self.available, 'error': self.error, 'mode': self.mode, 'sensors': sensors,
                'on_line': None, 'position': None, 'junction': None}
        if sensors is not None:
            position = line_position(sensors)
            data['on_line'] = any(sensors)
            data['position'] = None if position is None else round(position, 2)
            data['junction'] = sensors in JUNCTIONS
        return data

    def close(self):
        """Releases the I2C bus."""
        with self._lock:
            if self._bus is not None:
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None
        self.available = False


# TEST THE SENSOR: run this file on its own and slide a line under the robot
if __name__ == '__main__':
    import time
    sensor = LineFollowerInterface()
    if not sensor.available:
        print("Line follower unavailable:", sensor.error)
    try:
        while sensor.available:
            sensors = sensor.read()
            marks = ''.join('#' if on else '.' for on in sensors) if sensors else '????'
            print(f"[{marks}]  position {sensor.get_line_position()}  junction {sensor.is_junction()}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
