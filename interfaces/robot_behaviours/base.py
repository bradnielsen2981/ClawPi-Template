# -*- coding: utf-8 -*-
# The base class every robot behaviour is built on.
#
# A behaviour is robot code that keeps running in the background on its own thread, on top of whatever task or
# routine the robot is doing (e.g. "never cross a line", "back away from walls"). Turn one on with
# ROBOT.add_behaviour('name', setting=value) and off with ROBOT.remove_behaviour('name').
#
# To make a new behaviour: copy stay_within_quadrant.py, give the class a new name, fill in check(), and add the
# class to BEHAVIOUR_CLASSES in __init__.py.

import threading

from loggerinterface import get_logger


class Behaviour():
    name = None   # The name used with ROBOT.add_behaviour, e.g. 'stay_within_quadrant'
    rate_hz = 50  # How many times a second check() is called

    def __init__(self, robot, **settings):
        """
        Args:
            robot (RobotInterface): The robot this behaviour controls.
            **settings: The behaviour's own settings, passed on from add_behaviour (see each behaviour's setup()).
        """
        self.robot = robot
        self.logger = get_logger('Robot.' + type(self).__name__) # logs/robot.log
        self._quit = threading.Event()
        self._thread = None
        self.setup(**settings)

    def setup(self, **settings):
        """Stores the behaviour's settings. Override this to give the behaviour its own settings."""
        if settings:
            raise TypeError(f"{self.name} has no settings called {', '.join(settings)}")

    def check(self):
        """Called rate_hz times a second on the behaviour's own thread. Override this to do the work."""
        raise NotImplementedError

    def start(self):
        """Starts the behaviour's thread."""
        self._quit.clear()
        self._thread = threading.Thread(target=self._loop, name=f"behaviour {self.name}", daemon=True)
        self._thread.start()
        self.logger.info(f"{self.name} started")

    def stop(self, timeout=3.0):
        """
        Asks the behaviour's thread to finish and waits for it (up to timeout seconds).

        Returns:
            bool: True if the thread has finished.
        """
        self._quit.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                return False
        self.logger.info(f"{self.name} stopped")
        return True

    @property
    def stopping(self):
        """True once stop() has been called: a long check() should finish as soon as it sees this."""
        return self._quit.is_set()

    @property
    def running(self):
        """True while the behaviour's thread is going."""
        return self._thread is not None and self._thread.is_alive() and not self._quit.is_set()

    def _loop(self):
        period = 1.0 / self.rate_hz
        while not self._quit.wait(period):
            try:
                self.check()
            except Exception as e:
                # A safety behaviour that has stopped working is dangerous, so the whole robot shuts down
                self._quit.set()
                self.robot.emergency_shutdown(f"behaviour {self.name} crashed: {e}")
                return
