# -*- coding: utf-8 -*-
# Nearest obstacle detection from the depth picture for the CameraInterface.
# Switched off by default (the ClawPi has no depth camera) - see __init__.py to turn it back on.

import time
import cv2
import numpy as np

from .base import Detector


class DepthDetector(Detector):
    name = 'detect_depth'
    needs_depth = True

    def __init__(self, camera):
        super().__init__(camera)
        self.band = (0.25, 0.70) # Rows (fractions of height) checked for obstacles, above the floor

    def detect(self, rgb, depth, previous, now):
        if depth is None: # No depth picture (yet) - report nothing found rather than leaving detect_once waiting
            return {'found': False, 'zones': {}, 'time': now}
        return self.detect_depth(depth)

    def detect_depth(self, depth):
        """
        Finds the nearest obstacle in front of the robot using the depth picture. Only a band of rows above
        the floor is checked (see self.band), split into left, centre and right zones.

        Args:
            depth (numpy.ndarray): uint16 depth frame in millimetres (0 = no reading).

        Returns:
            dict: 'nearest' (mm), 'point' (x, y) of the nearest obstacle, 'zones' {'left', 'centre', 'right'} in mm
                  (None where there is no reading) and 'time'.
        """
        h, w = depth.shape
        top = int(self.band[0] * h)
        band = cv2.medianBlur(np.ascontiguousarray(depth[top:int(self.band[1] * h):4, ::4]), 5) # every 4th pixel, de-speckled

        zones = {}
        for name, part in zip(('left', 'centre', 'right'), np.array_split(band, 3, axis=1)):
            valid = part[part > 0]
            zones[name] = int(np.percentile(valid, 5)) if valid.size > 20 else None

        data = {'found': False, 'zones': zones, 'time': time.time()}
        valid = band > 0
        if valid.any():
            masked = np.where(valid, band, np.iinfo(np.uint16).max)
            row, col = np.unravel_index(np.argmin(masked), masked.shape)
            data.update({'found': True, 'nearest': int(band[row, col]), 'point': (int(col * 4), int(row * 4 + top))})
        return data

    def draw(self, frame, data):
        if data.get('found'):
            cv2.circle(frame, data['point'], 8, (255, 255, 255), 2)
            cv2.putText(frame, f"{data['nearest']}mm", (data['point'][0] + 10, data['point'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def describe(self, data):
        return "Depth mm: " + "  ".join(f"{k} {v}" for k, v in data.get('zones', {}).items())
