# -*- coding: utf-8 -*-
# Line following detection (Hiwonder's method) for the CameraInterface.

import time, math
import cv2
import numpy as np

from .base import Detector, KERNEL, largest_contour, keep_or_expire


class LineDetector(Detector):
    name = 'detect_line'

    def __init__(self, camera):
        super().__init__(camera)
        self.colour = "black" # Line colour from the LAB colour file
        # Horizontal strips as (top, bottom, weight), fractions of the frame height.
        # The strip nearest the robot counts most when working out the steering angle.
        self.rois = ((0.90, 0.95, 0.7), (0.80, 0.85, 0.2), (0.70, 0.75, 0.1))
        self.search_top = 0.4        # Crossing lines are searched for from here down (fraction of height)
        self.crossing_fraction = 0.5 # A row this full of line colour counts as a crossing line

    def detect(self, rgb, depth, previous, now):
        _, data = self.detect_line(rgb, self.colour)
        return keep_or_expire(previous, data, now, self.camera.detection_data_expire_time)

    def detect_line(self, frame, colour=None):
        """
        Hiwonder's line following method. Looks for the line colour in three thin horizontal strips near the
        bottom of the frame and combines their centres (nearest strip counts most) into a steering angle.
        Much cheaper than searching the whole picture, and works on curves. Also looks for a crossing line
        (a line running across the path), useful for stopping at junctions.

        Args:
            frame (numpy.ndarray): The current BGR video frame.
            colour (str, optional): Line colour from the YAML file (e.g. 'black'). If None or unknown,
                                    a dark line on a light floor is assumed.

        Returns:
            tuple: (frame, data_dictionary). When found, data contains:
                   'angle'    - steering angle in degrees (+ = line is to the left, turn left; - = turn right)
                   'offset'   - pixels from the frame centre to the line (- = left, + = right)
                   'points'   - centre of the line in each strip, nearest first
                   'line'     - (nearest point, furthest point)
                   'crossing' - {'found': True, 'y': row} if a line runs across the path
        """
        h, w = frame.shape[:2]
        top = int(min(self.search_top, min(r[0] for r in self.rois)) * h)
        mask = self._line_mask(frame[top:], colour)

        points = []
        weighted_sum, weight_used = 0.0, 0.0
        for r_top, r_bottom, weight in self.rois:
            y0, y1 = int(r_top * h) - top, int(r_bottom * h) - top
            contours, _ = cv2.findContours(mask[y0:y1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour, _ = largest_contour(contours, 30)
            if contour is not None:
                (cx, cy), _, _ = cv2.minAreaRect(contour)
                points.append((int(cx), int(cy) + y0 + top))
                weighted_sum += cx * weight
                weight_used += weight

        data = {'found': False}
        if points:
            centre = weighted_sum / weight_used
            angle = math.degrees(-math.atan((centre - w / 2.0) / (h / 2.0)))
            data = {'found': True, 'angle': round(angle, 1), 'offset': int(centre - w / 2),
                    'points': points, 'line': (points[0], points[-1]), 'time': time.time()}

        # Crossing line: the lowest (nearest) row that is mostly line colour
        coverage = mask.mean(axis=1) / 255.0
        rows = np.where(coverage > self.crossing_fraction)[0]
        if rows.size:
            data.update({'found': True, 'crossing': {'found': True, 'y': int(rows.max()) + top}, 'time': time.time()})
            data.setdefault('points', [])

        return frame, data

    def _line_mask(self, region, colour):
        """Binary mask of the line colour in a region of the frame."""
        lab_colours = self.camera.lab_colours
        if colour and colour in lab_colours:
            lab = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2LAB), (3, 3), 3)
            mask = cv2.inRange(lab, tuple(lab_colours[colour]['min']), tuple(lab_colours[colour]['max']))
        else:
            grey = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.dilate(cv2.erode(mask, KERNEL), KERNEL)

    def draw(self, frame, data):
        if not data.get('found'):
            return
        points = data.get('points', [])
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], False, (255, 0, 0), 2)
        if 'angle' in data:
            cv2.putText(frame, f"angle {data['angle']}", (points[0][0] + 10, points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        crossing = data.get('crossing')
        if crossing:
            cv2.line(frame, (0, crossing['y']), (frame.shape[1], crossing['y']), (255, 0, 255), 2)

    def describe(self, data):
        text = f"angle {data['angle']} offset {data['offset']}" if 'angle' in data else "not found"
        if data.get('crossing'):
            text += f"  crossing y={data['crossing']['y']}"
        return f"Line: {text}"
