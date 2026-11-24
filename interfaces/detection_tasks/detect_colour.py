# -*- coding: utf-8 -*-
# Colour blob detection (LAB colour ranges from Hiwonder's lab_tool) for the CameraInterface.

import time, math, threading
import cv2

from .base import Detector, KERNEL, keep_or_expire


class ColourDetector(Detector):
    name = 'detect_colour'

    def __init__(self, camera):
        super().__init__(camera)
        self.colours = []               # Colour names to look for (use the camera's add_detection_colour etc.)
        self.colours_lock = threading.Lock()
        self.min_area = 800             # Pixels at the full 640x480 resolution
        self.process_size = (320, 240)  # Detection runs on a smaller copy of the frame (4x fewer pixels)
        self.last_centres = {}          # Used to keep following the same blob between frames

    def detect(self, rgb, depth, previous, now):
        """
        Returns:
            dict: colour -> data for each colour currently seen (see detect_colours), with 'distance' in mm
                  added when there is a depth picture. Empty if nothing is seen.
        """
        with self.colours_lock:
            colours = list(self.colours)
        found = {}
        for colour, data in self.detect_colours(rgb, colours).items():
            data = keep_or_expire(previous.get(colour, {}), data, now, self.expire_time)
            if data.get('found'):
                if depth is not None:
                    (x, y), (w, h), _ = data['rect']
                    data['distance'] = self.camera.get_distance_at(x, y, radius=max(3, min(w, h) // 4), depth=depth)
                found[colour] = data
        return found

    def detect_colours(self, frame, colours):
        """
        Detects several colours at once. The frame is shrunk and converted to LAB once, then each colour is
        masked, cleaned up (erode/dilate removes speckles) and the best blob is chosen. If a colour was seen
        last frame, the blob nearest to it is kept so tracking doesn't jump between objects.

        Args:
            frame (numpy.ndarray): The current BGR video frame.
            colours (list): Colour names from the YAML file.

        Returns:
            dict: colour -> data. When found, data contains 'area', 'rect' ((x, y), (width, height), angle)
                  in full-frame pixels, and 'time'.
        """
        results = {}
        if not colours:
            return results
        lab_colours = self.camera.lab_colours
        small = cv2.resize(frame, self.process_size)
        lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 5)
        scale_x = frame.shape[1] / small.shape[1]
        scale_y = frame.shape[0] / small.shape[0]
        min_area = self.min_area / (scale_x * scale_y)

        for colour in colours:
            if colour not in lab_colours:
                continue
            mask = cv2.inRange(lab, tuple(lab_colours[colour]['min']), tuple(lab_colours[colour]['max']))
            mask = cv2.dilate(cv2.erode(mask, KERNEL), KERNEL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = [(c, cv2.contourArea(c)) for c in contours]
            candidates = [(c, a) for c, a in candidates if a > min_area]
            if not candidates:
                results[colour] = {'found': False}
                self.last_centres.pop(colour, None)
                continue

            contour, area = max(candidates, key=lambda c_a: c_a[1])
            last = self.last_centres.get(colour)
            if last is not None:
                def dist(c_a):
                    (x, y), _ = cv2.minEnclosingCircle(c_a[0])
                    return math.hypot(x - last[0], y - last[1])
                nearest = min(candidates, key=dist)
                if dist(nearest) < 50: # pixels in the small frame
                    contour, area = nearest

            (bx, by), (bw, bh), angle = cv2.minAreaRect(contour)
            self.last_centres[colour] = (bx, by)
            target_rect = ((int(bx * scale_x), int(by * scale_y)), (int(bw * scale_x), int(bh * scale_y)), int(angle))
            results[colour] = {'found': True, 'area': int(area * scale_x * scale_y), 'rect': target_rect, 'time': time.time()}
        return results

    def draw(self, frame, data):
        for colour, colour_data in data.items():
            if colour_data.get('found'):
                box = cv2.boxPoints(colour_data['rect']).astype(int)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                (x, y), _, _ = colour_data['rect']
                label = colour if colour_data.get('distance') is None else f"{colour} {colour_data['distance']}mm"
                cv2.putText(frame, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def describe(self, data):
        found = [f"{c} {d.get('distance') or ''}" for c, d in data.items()]
        return "Colour: " + (", ".join(found) if found else "none")

    def reset(self):
        self.last_centres.clear()
