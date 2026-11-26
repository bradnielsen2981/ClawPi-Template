# -*- coding: utf-8 -*-
"""
loggerinterface.py - one place that sets up logging for the whole robot.

Each interface writes to its own file in the logs folder:

    flask.log         the web app (app.py), Flask and anything else not listed below
    robot.log         RobotInterface and robot.py (movement routines, dashboard hold buttons)
    clawpi.log        ClawPi hardware (motors, servos, sensors)
    camera.log        CameraInterface and the detection tasks (colour, line, object, pose)
    lidar.log         LidarInterface
    map.log           MappingInterface, plus the output of the SLAM program it starts
    database.log      Database: SQL errors with the query that caused them
    linefollower.log  LineFollowerInterface (the 4 channel line sensor)
    voice.log         VoiceAgentInterface
    url.log           urlrequestinterface

A log line looks like:

    2026-11-26 10:15:02 ERROR   Database ViewQuery:44 - no such table: userz | SELECT * FROM userz | ()

    (time, level, who logged it, the function and line number, then the message)

Using it in an interface:

    from loggerinterface import get_logger
    logger = get_logger('Camera')                   # the name picks the file from LOG_FILES
    logger = get_logger('Camera.ObjectDetector')    # a part of the camera: goes to camera.log too

Turning logging off (app.py does this with its LOGGING setting):

    from interfaces.loggerinterface import setup_logging
    setup_logging(enabled=False)

Each log file is emptied the first time it is written to after the program starts.
Warnings and errors are also printed in the terminal.
"""
import logging, os, sys

# app.py imports this file as 'interfaces.loggerinterface', the interfaces import it as 'loggerinterface'.
# Make both names the same module, so there is only one set of settings.
for _name in ('loggerinterface', 'interfaces.loggerinterface'):
    sys.modules.setdefault(_name, sys.modules[__name__])

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Which file each interface logs to. A dotted name such as 'Camera.ObjectDetector' uses its first part.
LOG_FILES = {
    'Robot': 'robot.log',
    'ClawPi': 'clawpi.log',
    'Camera': 'camera.log',
    'Lidar': 'lidar.log',
    'Mapping': 'map.log',
    'Database': 'database.log',
    'LineFollower': 'linefollower.log',
    'Voice': 'voice.log',
    'URL': 'url.log',
}
FLASK_LOG_FILE = 'flask.log' # everything else: Flask, app.py and other libraries

LOG_FORMAT = '%(asctime)s %(levelname)-7s %(name)s %(funcName)s:%(lineno)d - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
CONSOLE_LEVEL = logging.WARNING # warnings and errors are printed in the terminal as well

_settings = {'enabled': True, 'level': logging.INFO}
_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)


def log_path(filename):
    """The full path of a file in the logs folder."""
    return os.path.join(LOG_DIR, filename)


def logging_enabled():
    """True unless setup_logging(enabled=False) has been called."""
    return _settings['enabled']


def _add_handlers(logger, filename):
    """A file in the logs folder plus the terminal (warnings and errors only)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    # delay=True: the file is only opened (and emptied) when the first message arrives,
    # so no empty files are made while logging is off
    file_handler = logging.FileHandler(log_path(filename), mode='w', delay=True)
    file_handler.setFormatter(_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LEVEL)
    console_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(name):
    """
    Returns the logger for an interface, writing to its file from LOG_FILES.

    Args:
        name (str): The interface name, e.g. 'Camera', or a part of one, e.g. 'Camera.ObjectDetector'.

    Returns:
        logging.Logger
    """
    top = name.split('.')[0]
    parent = logging.getLogger(top)
    if not getattr(parent, '_robot_log_ready', False):
        parent.setLevel(_settings['level'])
        parent.propagate = False # keep its messages out of flask.log
        _add_handlers(parent, LOG_FILES.get(top, top.lower() + '.log'))
        parent._robot_log_ready = True
    return logging.getLogger(name) # a dotted name passes its messages up to the parent's file


def setup_logging(enabled=True, level=logging.INFO):
    """
    Turns all logging on or off and sends everything without its own file to flask.log.
    Call it once, at the start of the program (app.py does).

    Args:
        enabled (bool, optional): False turns off every log file and the terminal warnings. Defaults to True.
        level (int, optional): The least important messages kept: logging.DEBUG (everything, including
                               every SQL query), logging.INFO, logging.WARNING or logging.ERROR. Defaults to INFO.
    """
    _settings['enabled'], _settings['level'] = enabled, level
    for top in LOG_FILES: # loggers made before this call follow the new level too
        existing = logging.getLogger(top)
        if getattr(existing, '_robot_log_ready', False):
            existing.setLevel(level)

    root = logging.getLogger()
    if not getattr(root, '_robot_log_ready', False):
        root.setLevel(level)
        _add_handlers(root, FLASK_LOG_FILE)
        root._robot_log_ready = True
    # Flask's web server logs every request (the dashboard polls several times a second), so only its problems
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # logging.disable switches off every logger in the program at once, including ones made later
    logging.disable(logging.NOTSET if enabled else logging.CRITICAL)
