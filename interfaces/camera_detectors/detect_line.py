# -*- coding: utf-8 -*-
# Line following detection (Hiwonder's method) for the CameraInterface.

import time, math
import cv2
import numpy as np

from .base import Detector, KERNEL, largest_contour


class LineDetector(Detector):
    name = 'detect_line'

    def __init__(self, camera):
        super().__init__(camera)
        self.colour = "black" # Line colour from the LAB colour file
        # Horizontal strips as (top, bottom, weight), fractions of the frame height.
        # The strip nearest the robot counts most when working out the steering angle.
        self.rois = ((0.90, 0.95, 0.7), (0.80, 0.85, 0.2), (0.70, 0.75, 0.1))
        self.crossing_angle = 30     # Two straight lines must differ by at least this many degrees to count as crossing
        self.crossing_reach = 25     # Pixels a line may stop short of the crossing point and still count (T-junctions, tape width)
        self.find_line = True        # Look for the line to follow (angle, offset, points) and long straight lines
        # Straight line segments (Hough transform) anywhere in the picture
        self.min_segment = 0.2       # Shortest segment kept, as a fraction of the frame's smaller side
        self.segment_gap = 6         # Largest gap in pixels bridged within one segment (bigger joins up textures)
        self.segment_votes = 60      # Hough votes a segment needs (higher = fewer, stronger lines)
        self.find_crossing = True    # Look for crossing lines (intersections)
        # Sticky detections: each straight line and intersection is remembered with the time it was last seen
        self._segments = []          # [(segment, time), ...]
        self._intersections = []     # [((x, y), lines, time), ...]
        self._follow = {}            # The last angle/offset/points/line found, with its 'time'
        self._last_found = ([], [])  # (segments, intersections) from the latest detect_line()

    def reset(self):
        self._segments, self._intersections, self._follow = [], [], {}

    def detect(self, rgb, depth, previous, now):
        """
        Each straight line and intersection is sticky on its own: it stays for expire_time seconds after it was
        last seen, unless a new detection of the same line (or at the same spot) replaces it sooner.
        """
        _, data = self.detect_line(rgb, self.colour)
        new_segments, new_points = self._last_found
        new_segments = [tuple(s) for s in new_segments]
        expire = self.expire_time

        # Straight lines: new ones replace the old ones they lie along, the rest are kept until they expire
        self._segments = [(s, now) for s in new_segments] + [
            (s, t) for s, t in self._segments
            if now - t < expire and not any(self._same_line(s, n) or self._same_line(n, s) for n in new_segments)]

        # Intersections: new ones replace old ones at the same spot
        near = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) <= 2 * self.crossing_reach
        self._intersections = [(p, c, now) for p, c in new_points] + [
            (p, c, t) for p, c, t in self._intersections
            if now - t < expire and not any(near(p, n) for n, _ in new_points)]
        self._intersections.sort(key=lambda i: -i[0][1])  # Nearest the robot (lowest in the picture) first

        # Line following: the angle/offset are kept as one until they expire
        if 'angle' in data:
            self._follow = {k: data[k] for k in ('angle', 'offset', 'points', 'line')}
            self._follow['time'] = now
        elif self._follow and now - self._follow['time'] >= expire:
            self._follow = {}

        result = {'found': False}
        times = [t for _, t in self._segments] + [t for _, _, t in self._intersections]
        if self._follow:
            result.update({k: v for k, v in self._follow.items() if k != 'time'})
            times.append(self._follow['time'])
        if self._segments and self.find_line:
            result['segments'] = [list(s) for s, _ in sorted(self._segments, key=lambda i: -math.hypot(i[0][2] - i[0][0], i[0][3] - i[0][1]))]
        if self._intersections and self.find_crossing:
            (x, y), lines, t = self._intersections[0]
            result['intersections'] = [list(p) for p, _, _ in self._intersections]
            result['crossing'] = {'found': True, 'x': x, 'y': y, 'lines': lines, 'time': t}
        if len(result) > 1:
            result.update({'found': True, 'time': max(times)})
            result.setdefault('points', [])
        return result

    def detect_line(self, frame, colour=None):
        """
        Hiwonder's line following method. Looks for the line colour in three thin horizontal strips near the
        bottom of the frame and combines their centres (nearest strip counts most) into a steering angle.
        Much cheaper than searching the whole picture, and works on curves. Also finds long straight lines
        anywhere in the picture and where they cross, useful for stopping at junctions.

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
                   'segments' - long straight lines anywhere in the picture, [[x1, y1, x2, y2], ...] longest first
                   'crossing'      - the nearest intersection (lowest in the picture) of two straight lines:
                                     {'found': True, 'x', 'y', 'lines': [[x1, y1, x2, y2], [x1, y1, x2, y2]]}
                   'intersections' - every intersection as [x, y], nearest first
        """
        h, w = frame.shape[:2]
        top = int(min(r[0] for r in self.rois) * h)
        mask = self._line_mask(frame[top:], colour)

        points = []
        weighted_sum, weight_used = 0.0, 0.0
        for r_top, r_bottom, weight in (self.rois if self.find_line else ()):
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

        # Long straight lines anywhere in the picture (not joined to each other)
        segments = self._find_segments(frame) if (self.find_line or self.find_crossing) else []
        if segments and self.find_line:
            data.update({'found': True, 'segments': segments, 'time': time.time()})
            data.setdefault('points', [])

        # Intersections: where two of those straight lines cross (or one ends on the other, as in a T)
        intersections = self._find_intersections(segments) if self.find_crossing else []
        if intersections:
            (x, y), lines = intersections[0]
            data.update({'found': True, 'time': time.time(), 'intersections': [list(p) for p, _ in intersections],
                         'crossing': {'found': True, 'x': x, 'y': y, 'lines': lines, 'time': time.time()}})
            data.setdefault('points', [])
        self._last_found = (segments, intersections)  # Everything found this frame, for detect()'s sticky lists

        return frame, data

    def _find_intersections(self, segments):
        """
        Finds where pairs of straight lines cross. Nearly parallel pairs are skipped, and the crossing point must lie
        on both lines (within crossing_reach pixels, so a line ending on another - a T-junction - still counts).

        Returns:
            list: [((x, y), [line_a, line_b]), ...] nearest the robot (lowest in the picture) first.
        """
        found = []
        for i, a in enumerate(segments):
            for b in segments[i + 1:]:
                point = self._intersection(a, b)
                if point:
                    found.append((point, [a, b]))
        found.sort(key=lambda f: -f[0][1])
        # Several lines meeting at one spot give several nearby points: keep the first of each cluster
        kept = []
        for point, lines in found:
            if all(math.hypot(point[0] - k[0][0], point[1] - k[0][1]) > 2 * self.crossing_reach for k in kept):
                kept.append((point, lines))
        return kept

    def _intersection(self, a, b):
        """The (x, y) where segments a and b cross, or None if they are nearly parallel or don't reach each other."""
        ax, ay, bx, by = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
        len_a, len_b = math.hypot(ax, ay), math.hypot(bx, by)
        cross = ax * by - ay * bx
        if abs(cross) < len_a * len_b * math.sin(math.radians(self.crossing_angle)):
            return None
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = (dx * by - dy * bx) / cross  # Fraction along a
        u = (dx * ay - dy * ax) / cross  # Fraction along b
        reach_a, reach_b = self.crossing_reach / len_a, self.crossing_reach / len_b
        if not (-reach_a <= t <= 1 + reach_a and -reach_b <= u <= 1 + reach_b):
            return None
        return int(round(a[0] + t * ax)), int(round(a[1] + t * ay))

    def _find_segments(self, frame):
        """
        Finds long straight lines (edges) anywhere in the frame with a probabilistic Hough transform. Edges of
        the same line (e.g. both sides of a strip of tape) are merged so each line appears once.

        Returns:
            list: [[x1, y1, x2, y2], ...] in full-frame pixels, longest first.
        """
        h, w = frame.shape[:2]
        grey = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(grey, 50, 150)
        found = cv2.HoughLinesP(edges, 1, np.pi / 180, self.segment_votes,
                                minLineLength=int(self.min_segment * min(w, h)), maxLineGap=self.segment_gap)
        if found is None:
            return []
        segments = sorted((tuple(int(v) for v in s[0]) for s in found),
                          key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
        kept = []
        for s in segments:
            if not any(self._same_line(s, k) for k in kept):
                kept.append(s)
        return [list(s) for s in kept]

    @staticmethod
    def _same_line(s, k, angle_tol=5, dist_tol=25):
        """True if segment s lies along (nearly parallel, close to and overlapping) the longer segment k."""
        a = math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180
        b = math.degrees(math.atan2(k[3] - k[1], k[2] - k[0])) % 180
        if min(abs(a - b), 180 - abs(a - b)) > angle_tol:
            return False
        length = math.hypot(k[2] - k[0], k[3] - k[1])
        ux, uy = (k[2] - k[0]) / length, (k[3] - k[1]) / length
        mx, my = (s[0] + s[2]) / 2 - k[0], (s[1] + s[3]) / 2 - k[1]  # Midpoint of s relative to k's start
        along = mx * ux + my * uy
        return abs(mx * uy - my * ux) < dist_tol and -dist_tol < along < length + dist_tol

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
        # Each straight line on its own (not joined up); the steering angle/offset are in the data, not drawn
        for x1, y1, x2, y2 in data.get('segments', []):
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crossing = data.get('crossing')
        if crossing:
            for x1, y1, x2, y2 in crossing.get('lines', []):                      # The two lines of the nearest crossing
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            for x, y in data.get('intersections', [[crossing['x'], crossing['y']]]):
                cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)  # Red cross at each intersection
            cv2.circle(frame, (crossing['x'], crossing['y']), 12, (0, 0, 255), 2)   # Circle on the nearest one

    def describe(self, data):
        text = f"angle {data['angle']} offset {data['offset']}" if 'angle' in data else "not found"
        if data.get('segments'):
            text += f"  {len(data['segments'])} lines"
        if data.get('crossing'):
            text += f"  crossing y={data['crossing']['y']}"
        return f"Line: {text}"
