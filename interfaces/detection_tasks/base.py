# -*- coding: utf-8 -*-
# The base class every detection task is built on, plus small helpers they share.
# To make a new detection task: copy one of the detect_*.py files, give the class a new name (e.g.
# 'detect_face'), fill in detect() and draw(), and add the class to DETECTOR_CLASSES in __init__.py.

import math
import cv2
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # Used by erode/dilate to remove speckles from masks


def largest_contour(contours, min_area):
    """
    Finds the largest contour above a minimum area.

    Returns:
        tuple: (contour, area), or (None, 0) if no contour is big enough.
    """
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area and area > best_area:
            best, best_area = c, area
    return best, best_area

def json_safe(value):
    """
    Returns a fresh copy of detection data made only of JSON types (dict, list, str, int, float, bool, None),
    so it can be sent with json.dumps / Flask's jsonify. Tuples become lists, numpy numbers and arrays become
    plain numbers and lists, and inf / nan become None (JSON has no infinity).
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)

def keep_or_expire(old, new, now, expire_time):
    """
    Makes a detection sticky: returns the new data if something was found, otherwise keeps the old find until
    expire_time seconds after it was last seen (stops flickering). Both dicts need 'found' and 'time'.
    """
    if new.get('found'):
        return new
    if old.get('found') and now - old.get('time', 0) < expire_time:
        return old
    return new

def keep_or_expire_items(old_items, new_items, now, expire_time, key):
    """
    Makes each item in a list of detections sticky on its own (e.g. each object class), so losing one doesn't
    remove the others. Items in old_items whose key isn't in new_items are kept until expire_time seconds after
    they were last seen. Each item needs a 'time'.

    Returns:
        list: The new items followed by the old items still being kept.
    """
    new_keys = {item.get(key) for item in new_items}
    kept = [item for item in old_items
            if item.get(key) not in new_keys and now - item.get('time', 0) < expire_time]
    return list(new_items) + kept


#-------------------------------------------
# The Detector base class
class Detector():
    name = None          # Task name used with CameraInterface.add_detection_task, e.g. 'detect_line'
    needs_depth = False  # True subscribes the camera to the depth picture when the task is added

    def __init__(self, camera):
        """
        Args:
            camera (CameraInterface): The camera this detector belongs to (for lab_colours, get_distance_at,
                                      detection_data_expire_time and the newest frames).
        """
        self.camera = camera

    @property
    def expire_time(self):
        """Seconds to keep a detection after it was last seen: the camera's detection_data_expire_time when
        sticky_detections is on, otherwise 0 (detections disappear as soon as they are lost)."""
        camera = self.camera
        return camera.detection_data_expire_time if getattr(camera, 'sticky_detections', True) else 0.0

    def is_stale(self, result, now):
        """True if a background model's 'found' result is older than it may be kept, e.g. because the model has
        stopped producing results. Never less than a second, so a slow model's newest result isn't dropped."""
        return result.get('found') and now - result.get('time', now) > max(self.expire_time, 1.0)

    def prepare(self):
        """Called when the task is added, e.g. to load a model. Returns False if the task can't run."""
        return True

    def detect(self, rgb, depth, previous, now):
        """
        Runs the detection on one frame. Called by the camera's processing thread for every new frame.

        Args:
            rgb (numpy.ndarray): The newest BGR colour frame.
            depth (numpy.ndarray): The matching uint16 depth frame in millimetres, or None.
            previous (dict): What this detector returned last frame ({} the first time).
            now (float): The frame's time (time.time()).

        Returns:
            dict: The detection data (stored in get_detection_data() under this detector's name),
                  or None if there is no result yet. Include 'found' (bool) and 'time', and use JSON-style
                  values (numbers, strings, lists, dicts). The camera passes it through json_safe() before
                  publishing, so tuples and numpy numbers are fine here and come out as lists and plain numbers.
        """
        raise NotImplementedError

    def draw(self, frame, data):
        """Draws this detector's results onto the frame (in place)."""

    def describe(self, data):
        """One line of text summarising the results, shown on the video when output text is on."""
        return f"{self.name}: {'found' if data.get('found') else 'none'}"

    def reset(self):
        """Forgets anything remembered between frames (called when the detection data is cleared)."""

    def background_loop(self):
        """
        Optional slow work in its own thread (e.g. a neural network), so it never holds up the other detectors.
        Started by CameraInterface.start() for detectors that override it; should return when the camera stops.
        """
