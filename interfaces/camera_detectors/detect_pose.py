# -*- coding: utf-8 -*-
# Human pose detection with Google's MediaPipe Pose Landmarker for the CameraInterface. It finds 33 body
# landmarks per person (nose, shoulders, elbows, wrists, hips, knees, ankles...) and works out a few simple
# gestures from them (arm raised, arms out, fallen over). Like detect_object, the network runs in its own
# background thread on the newest frame (about 18 fps on the Pi 5 CPU) and detect() picks up its latest result.
#
# 'left' and 'right' are the PERSON's own left and right, so when they face the robot their left arm is on the
# right of the picture.
#
# The model is Hiwonder's copy of pose_landmarker.task (in their mediapipe_example). A path to another
# .task file (e.g. Google's pose_landmarker_lite / _heavy) can be passed to load().

import os, sys, time, math, threading
import cv2
import numpy as np

from .base import Detector, keep_or_expire

POSE_MODEL_PATH = '/home/ubuntu/ros2_ws/src/example/example/mediapipe_example/model/pose_landmarker.task'
# MediaPipe is installed for the ubuntu user. Programs run as root (e.g. the VS Code debugger) don't see that
# folder, so it is added if the import fails.
UBUNTU_SITE_PACKAGES = '/home/ubuntu/.local/lib/python3.10/site-packages'

# The 33 landmarks, in MediaPipe's order
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
    'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

# Lines drawn between landmarks to make the skeleton
SKELETON = [
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'), ('left_heel', 'left_foot_index'), ('right_ankle', 'right_heel'),
    ('right_heel', 'right_foot_index'), ('left_ear', 'left_eye'), ('left_eye', 'nose'), ('nose', 'right_eye'),
    ('right_eye', 'right_ear'), ('mouth_left', 'mouth_right')]

GESTURES = ['left_arm_up', 'right_arm_up', 'both_arms_up', 'arms_out', 'fallen']


def import_mediapipe():
    """Imports MediaPipe (takes a second or two the first time)."""
    try:
        import mediapipe
    except ImportError:
        if UBUNTU_SITE_PACKAGES in sys.path:
            raise
        sys.path.append(UBUNTU_SITE_PACKAGES)
        import mediapipe
    return mediapipe


class PoseDetector(Detector):
    name = 'detect_pose'

    def __init__(self, camera):
        super().__init__(camera)
        self.landmarker = None
        self.model_path = None
        self.num_poses = 1               # Most people to look for. The model is most reliable with one person.
        self.confidence_level = 0.5      # Minimum score (0-1) to find a person
        self.visibility_threshold = 0.5  # Landmarks less visible than this (0-1) are ignored for gestures and drawing
        self.watch_gestures = list(GESTURES) # Gestures to report (see GESTURES). People are always found.
        self.framerate = 0.0
        self.model_lock = threading.Lock()
        self._mp = None
        self._result = None
        self._last_timestamp = 0
        self._gesture_times = {}  # gesture -> when it was last seen, so gestures are sticky

    def prepare(self):
        """Loads the model the first time the detector is turned on."""
        return self.landmarker is not None or self.load()

    def load(self, model_path=None, num_poses=None, confidence=None):
        """
        Loads the MediaPipe pose model (again, if the settings change).

        Args:
            model_path (str, optional): Path to a pose_landmarker .task file. Defaults to Hiwonder's copy.
            num_poses (int, optional): Most people to look for. Defaults to self.num_poses (1).
            confidence (float, optional): Minimum score (0-1) to find a person. Defaults to self.confidence_level.

        Returns:
            bool: True if the model is ready.
        """
        model_path = model_path or self.model_path or POSE_MODEL_PATH
        if num_poses is not None:
            self.num_poses = int(num_poses)
        if confidence is not None:
            self.confidence_level = confidence
        if not os.path.exists(model_path):
            self.logger.error(f"Pose model not found: {model_path}")
            return False
        try:
            mp = import_mediapipe()
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            options = vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO, # Tracks people between frames: faster and steadier
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.confidence_level,
                min_pose_presence_confidence=self.confidence_level,
                min_tracking_confidence=self.confidence_level)
            landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            self.logger.error(f"Failed to load pose model {model_path}: {e}")
            return False

        with self.model_lock:
            old, self.landmarker = self.landmarker, landmarker
            self._mp, self.model_path = mp, model_path
            self._last_timestamp = 0
        if old is not None:
            old.close()
        self._result = None
        return True

    def clear(self):
        """Removes the pose model from memory."""
        with self.model_lock:
            old, self.landmarker = self.landmarker, None
        if old is not None:
            old.close()
        self._result = None

    def background_loop(self):
        """Runs the pose model on the newest frame whenever 'detect_pose' is active."""
        camera = self.camera
        last_frame_id = -1
        while camera.status == "Running":
            with camera.detector_lock:
                active = self.name in camera.active_detectors
            if camera.paused or not active or self.landmarker is None:
                time.sleep(0.05)
                continue
            with camera.frame_lock:
                bgr, frame_id = camera.rgb_frame, camera.rgb_frame_id
            if bgr is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            starttime = time.time()
            try:
                with self.model_lock:
                    if self.landmarker is None:
                        continue
                    # VIDEO mode needs timestamps that always go up
                    timestamp = max(int(time.monotonic() * 1000), self._last_timestamp + 1)
                    self._last_timestamp = timestamp
                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) # MediaPipe takes RGB
                    image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                    result = self.landmarker.detect_for_video(image, timestamp)
                people = [self._read_person(landmarks, bgr.shape[1], bgr.shape[0]) for landmarks in result.pose_landmarks]
            except Exception as e:
                self.logger.error(f"Pose detection failed: {e}")
                time.sleep(1)
                continue
            elapsed = time.time() - starttime
            self.framerate = round(1 / elapsed, 1) if elapsed > 0 else 0.0
            people.sort(key=lambda p: p['size'], reverse=True) # Nearest (biggest) person first
            self._result = {'found': bool(people), 'people': people,
                            'gestures': people[0]['gestures'] if people else [], 'time': time.time()}

    def _read_person(self, landmarks, width, height):
        """Turns one person's normalised MediaPipe landmarks into pixel positions, a box and gestures (see detect)."""
        points = {}
        for name, lm in zip(LANDMARK_NAMES, landmarks):
            visibility = lm.visibility if lm.visibility is not None else 1.0
            points[name] = [int(round(lm.x * width)), int(round(lm.y * height)), round(float(visibility), 2)]

        seen = [p for p in points.values() if p[2] >= self.visibility_threshold] or list(points.values())
        xs = [min(max(p[0], 0), width - 1) for p in seen]
        ys = [min(max(p[1], 0), height - 1) for p in seen]
        x, y = min(xs), min(ys)
        w, h = max(1, max(xs) - x), max(1, max(ys) - y)
        return {'rect': [x, y, w, h],
                'centre': [x + w // 2, y + h // 2],
                'size': round(h / height, 2),   # Fraction of the picture's height - bigger means closer
                'score': round(sum(p[2] for p in points.values()) / len(points), 2),
                'landmarks': points,
                'gestures': [g for g in self._gestures(points) if g in self.watch_gestures]}

    def _gestures(self, points):
        """
        Works out simple gestures from the landmarks:
            'left_arm_up' / 'right_arm_up' - wrist clearly above the shoulder
            'both_arms_up'                 - both of the above
            'arms_out'                     - both arms stretched out sideways at about shoulder height
            'fallen'                       - the body (shoulders to hips) is closer to lying flat than upright
        """
        def visible(*names):
            return all(points[n][2] >= self.visibility_threshold for n in names)

        def midpoint(a, b):
            return ((points[a][0] + points[b][0]) / 2, (points[a][1] + points[b][1]) / 2)

        gestures = []
        if not visible('left_shoulder', 'right_shoulder'):
            return gestures
        shoulder_width = max(1.0, math.dist(points['left_shoulder'][:2], points['right_shoulder'][:2]))
        if visible('left_hip', 'right_hip'):
            torso = math.dist(midpoint('left_shoulder', 'right_shoulder'), midpoint('left_hip', 'right_hip'))
        else:
            torso = shoulder_width * 1.5 # About right for most people
        torso = max(torso, shoulder_width)

        if visible('left_hip', 'right_hip'):
            sx, sy = midpoint('left_shoulder', 'right_shoulder')
            hx, hy = midpoint('left_hip', 'right_hip')
            if abs(hx - sx) > abs(hy - sy): # Torso more than 45 degrees from upright
                return ['fallen'] # Arm gestures are measured up/down the picture, so they mean nothing lying down

        arms_up = []
        for side in ('left', 'right'):
            if visible(f'{side}_wrist') and points[f'{side}_wrist'][1] < points[f'{side}_shoulder'][1] - 0.3 * torso:
                arms_up.append(f'{side}_arm_up')
        gestures += arms_up
        if len(arms_up) == 2:
            gestures.append('both_arms_up')

        if visible('left_wrist', 'right_wrist', 'left_elbow', 'right_elbow'):
            level = all(abs(points[f'{s}_wrist'][1] - points[f'{s}_shoulder'][1]) < 0.35 * torso for s in ('left', 'right'))
            spread = abs(points['left_wrist'][0] - points['right_wrist'][0]) > 2.5 * shoulder_width
            if level and spread:
                gestures.append('arms_out')
        return gestures

    def detect(self, rgb, depth, previous, now):
        """
        Returns:
            dict: 'found', 'time', 'people' and 'gestures', or None until the model has produced its first result.
                  'people' is a list, nearest (biggest) first. Each person is
                  {'rect' [x, y, width, height] (top-left corner), 'centre' [x, y], 'size' (height as a fraction of
                  the picture, bigger = closer), 'score' (0-1), 'gestures' (list, see _gestures),
                  'landmarks' {name: [x, y, visibility 0-1]} (see LANDMARK_NAMES), 'distance' (mm, with a depth picture)}.
                  'gestures' at the top is the nearest person's gestures, e.g. ['left_arm_up']. These are sticky:
                  each one stays for the camera's expire time after it was last seen (the ones in 'people' aren't).
        """
        result = self._result
        if result is None:
            return None
        if self.is_stale(result, now): # The model has stopped producing results
            result = {'found': False, 'people': [], 'gestures': [], 'time': now}
        data = keep_or_expire(previous, result, now, self.expire_time)

        for gesture in result['gestures']:
            self._gesture_times[gesture] = max(self._gesture_times.get(gesture, 0), result['time'])
        gestures = [g for g in GESTURES if g in self.watch_gestures and (g in result['gestures']
                    or now - self._gesture_times.get(g, 0) < self.expire_time)]
        data = dict(data, gestures=gestures, found=data['found'] or bool(gestures))
        if depth is not None:
            for person in data.get('people', []):
                x, y, w, h = person['rect']
                person['distance'] = self.camera.get_distance_at(x + w // 2, y + h // 2, radius=max(3, min(w, h) // 6), depth=depth)
        return data

    def draw(self, frame, data):
        for person in data.get('people', []):
            points = person['landmarks']
            for a, b in SKELETON:
                if points[a][2] >= self.visibility_threshold and points[b][2] >= self.visibility_threshold:
                    cv2.line(frame, tuple(points[a][:2]), tuple(points[b][:2]), (255, 255, 255), 2)
            for x, y, visibility in points.values():
                if visibility >= self.visibility_threshold:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            x, y, w, h = person['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)
            label = ", ".join(person['gestures']) or "person"
            if person.get('distance') is not None:
                label += f" {person['distance']}mm"
            cv2.putText(frame, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def describe(self, data):
        if self.landmarker is None:
            return "Pose: no model loaded"
        people = data.get('people', [])
        text = f"Pose ({self.framerate} fps): {len(people)} people"
        if data.get('gestures'):
            text += " - " + ", ".join(data['gestures'])
        return text

    def reset(self):
        self._result = None
        self._gesture_times = {}
