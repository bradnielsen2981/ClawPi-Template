# -*- coding: utf-8 -*-
# Face detection with Google's MediaPipe Face Detection for the CameraInterface. It finds every face in the
# picture and reports the biggest one (usually the nearest person) as 'largest'. It is fast (about 5 ms a frame
# on the Pi 5 on a 320 pixel wide copy of the frame), so unlike detect_pose it runs straight in detect().

import time
import cv2

from .base import Detector, keep_or_expire
from .detect_pose import import_mediapipe

DETECTION_WIDTH = 320 # The frame is shrunk to this width before detecting (much faster, still finds faces)


class FaceDetector(Detector):
    name = 'detect_face'

    def __init__(self, camera):
        super().__init__(camera)
        self.detector = None
        self.confidence_level = 0.5 # Minimum score (0-1) to count as a face
        self.full_range = False     # False: faces within about 2 m (more reliable). True: up to about 5 m.

    def prepare(self):
        """Loads MediaPipe the first time the detector is turned on."""
        return self.detector is not None or self.load()

    def load(self, confidence=None, full_range=None):
        """
        Loads (or reloads) the face detector.

        Args:
            confidence (float, optional): Minimum score (0-1) to count as a face.
            full_range (bool, optional): True finds faces further away (up to about 5 m).

        Returns:
            bool: True if it loaded.
        """
        if confidence is not None:
            self.confidence_level = confidence
        if full_range is not None:
            self.full_range = full_range
        try:
            mp = import_mediapipe()
            if self.detector is not None:
                self.detector.close()
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1 if self.full_range else 0, min_detection_confidence=self.confidence_level)
            return True
        except Exception as e:
            self.logger.error(f"Face detector could not be loaded: {e}")
            self.detector = None
            return False

    def detect(self, rgb, depth, previous, now):
        if self.detector is None:
            return None
        height, width = rgb.shape[:2]
        small = rgb
        if width > DETECTION_WIDTH:
            small = cv2.resize(rgb, (DETECTION_WIDTH, int(height * DETECTION_WIDTH / width)))
        results = self.detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        faces = []
        for detection in results.detections or []:
            box = detection.location_data.relative_bounding_box
            x1, y1 = max(0.0, box.xmin) * width, max(0.0, box.ymin) * height
            x2, y2 = min(1.0, box.xmin + box.width) * width, min(1.0, box.ymin + box.height) * height
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append({
                'rect': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), # x, y of the top left corner, width, height
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'area': int((x2 - x1) * (y2 - y1)),
                'score': round(float(detection.score[0]), 2),
            })
        faces.sort(key=lambda f: f['area'], reverse=True)

        new = {'found': bool(faces), 'faces': faces, 'largest': faces[0] if faces else None,
               'frame_size': (width, height), 'time': now}
        return keep_or_expire(previous, new, now, self.expire_time)

    def draw(self, frame, data):
        for i, face in enumerate(data.get('faces', [])):
            x, y, w, h = face['rect']
            colour = (0, 255, 0) if i == 0 else (0, 160, 255) # the largest face is green
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, f"face {face['score']}", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    def describe(self, data):
        return f"Faces: {len(data.get('faces', []))}"
