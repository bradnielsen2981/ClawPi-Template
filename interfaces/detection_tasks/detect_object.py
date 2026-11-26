# -*- coding: utf-8 -*-
# Object detection with Hiwonder's YOLO26 models (Ultralytics, accelerated with OpenVINO on the Pi's CPU)
# for the CameraInterface. The network is slow compared with the other detectors, so it runs in its own
# background thread on the newest frame and never holds them up. detect() just picks up its latest result.
#
# Models (in /home/ubuntu/third_party_ros2/yolo/yolo26, exported for 320 and 640 pixel input):
#   'yolo26n'                - 80 everyday objects (person, cup, chair, bottle...)
#   'best_traffic_26'        - 6 traffic signs
#   'best_garbage_26'        - 12 kinds of rubbish, with rotated boxes
#   'best_FourLineFollowing' - 8 classes for Hiwonder's line following track (640 only)
# A path to your own .pt file or exported _openvino_model folder works too.

import os, sys, time, threading
import cv2
import numpy as np
import yaml

from .base import Detector, keep_or_expire_items

YOLO_DIR = '/home/ubuntu/third_party_ros2/yolo/yolo26'
YOLO_EXAMPLE_DIR = '/home/ubuntu/ros2_ws/src/example/example/yolo_detect/models/26'
# Ultralytics and OpenVINO are installed for the ubuntu user. Programs run as root (e.g. the VS Code
# debugger) don't see that folder, so it is added if the import fails.
UBUNTU_SITE_PACKAGES = '/home/ubuntu/.local/lib/python3.10/site-packages'


def import_yolo():
    """Imports Ultralytics' YOLO class (takes a few seconds the first time)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        if UBUNTU_SITE_PACKAGES in sys.path:
            raise
        sys.path.append(UBUNTU_SITE_PACKAGES)
        from ultralytics import YOLO
    return YOLO

def find_model(model, size):
    """
    Finds a model by name. The OpenVINO export for the requested size is preferred (much faster), then any other
    OpenVINO export, then the plain PyTorch .pt file.

    Returns:
        str: Path to the _openvino_model folder or .pt file, or None if it can't be found.
    """
    if os.path.exists(model):
        return model
    name = model[:-3] if model.endswith('.pt') else model
    other = 640 if size == 320 else 320
    candidates = [os.path.join(YOLO_DIR, f'{s}_size', f'{name}_openvino_model') for s in (size, other)]
    candidates += [os.path.join(YOLO_EXAMPLE_DIR, f'{name}_openvino_model'), os.path.join(YOLO_DIR, f'{name}.pt')]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def read_metadata(path):
    """
    Reads an exported model's metadata.yaml.

    Returns:
        tuple: (task, image size, labels) - task is 'detect' or 'obb' (rotated boxes); None for anything unknown.
    """
    try:
        with open(os.path.join(path, 'metadata.yaml'), 'r') as f:
            metadata = yaml.safe_load(f) or {}
    except (OSError, NotADirectoryError):
        return None, None, None
    imgsz = metadata.get('imgsz')
    size = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz) if imgsz else None
    return metadata.get('task'), size, label_list(metadata.get('names'))

def label_list(names):
    """Class names as a list, from Ultralytics' {id: name} dictionary or a list."""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names] if names else []


class ObjectDetector(Detector):
    name = 'detect_object'

    def __init__(self, camera):
        super().__init__(camera)
        self.model = None
        self.model_name = None
        self.model_path = None
        self.task = 'detect'        # 'detect' (normal boxes) or 'obb' (rotated boxes)
        self.labels = []
        self.confidence_level = 0.5
        self.overlap_threshold = 0.45 # Boxes overlapping more than this are merged (non-maximum suppression)
        self.size = 320             # 320 is about 4x faster than 640, but 640 sees smaller/further objects
        self.framerate = 0.0
        self.model_lock = threading.Lock()
        self._result = None

    def prepare(self):
        """Loads the default model the first time the task is added."""
        return self.model is not None or self.load()

    def load(self, model="yolo26n", size=None, confidence=None):
        """
        Loads one of Hiwonder's YOLO26 models (see the list at the top of this file).

        Args:
            model (str, optional): Model name, e.g. 'yolo26n', 'best_traffic_26', 'best_garbage_26', or a path to
                                   a .pt file / _openvino_model folder. Defaults to 'yolo26n'.
            size (int, optional): Image size the model works at, 320 or 640. Defaults to self.size.
            confidence (float, optional): Minimum score (0-1) to report an object.

        Returns:
            bool: True if the model is ready.
        """
        if confidence is not None:
            self.confidence_level = confidence
        size = int(size or self.size)
        if self.model is not None and self.model_name == model and self.size == size:
            return True

        path = find_model(model, size)
        if path is None:
            self.logger.error(f"YOLO model '{model}' not found in {YOLO_DIR}")
            return False
        task, model_size, labels = read_metadata(path)
        try:
            YOLO = import_yolo()
            yolo = YOLO(path, task=task or 'detect')
            labels = labels or label_list(getattr(yolo, 'names', None))
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model {path}: {e}")
            return False
        if model_size and model_size != size:
            self.logger.warning(f"YOLO model {model} is only available at size {model_size} - using that")

        with self.model_lock:
            self.model, self.model_name, self.model_path = yolo, model, path
            self.task, self.labels = task or yolo.task, labels
            self.size = model_size or size # An OpenVINO export only works at the size it was exported at
        self._result = None
        self.logger.info(f"YOLO model {model} loaded from {path} at size {self.size} ({len(labels)} classes)")
        return True

    def clear(self):
        """Removes the neural network model and its labels from memory."""
        with self.model_lock:
            self.model, self.model_name, self.model_path, self.labels = None, None, None, []
        self._result = None

    def background_loop(self):
        """Runs the neural network on the newest frame whenever 'detect_object' is active."""
        camera = self.camera
        last_frame_id = -1
        while camera.status == "Running":
            with camera.task_lock:
                active = self.name in camera.detection_tasks
            with self.model_lock:
                model, labels, task, size = self.model, self.labels, self.task, self.size
            if camera.paused or not active or model is None:
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
                result = model(bgr, device='cpu', imgsz=size, conf=self.confidence_level,
                               iou=self.overlap_threshold, verbose=False)[0] # Ultralytics takes BGR like OpenCV
                targets = self._read_result(result, labels, task)
            except Exception as e:
                self.logger.error(f"Object detection failed: {e}")
                time.sleep(1)
                continue
            elapsed = time.time() - starttime
            self.framerate = round(1 / elapsed, 1) if elapsed > 0 else 0.0
            now = time.time()
            for target in targets:
                target['time'] = now
            self._result = {'found': bool(targets), 'targets': targets, 'time': now}

    def _read_result(self, result, labels, task):
        """Turns an Ultralytics result into a list of targets (see detect)."""
        height, width = result.orig_shape

        def label(class_id):
            return labels[class_id] if class_id < len(labels) else f"id_{class_id}"

        def rect(x1, y1, x2, y2):
            """[x, y, w, h], kept inside the picture so frame[y:y+h, x:x+w] always cuts out the object."""
            x1, x2 = int(max(0, min(x1, width - 1))), int(max(0, min(x2, width)))
            y1, y2 = int(max(0, min(y1, height - 1))), int(max(0, min(y2, height)))
            return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

        targets = []
        if task == 'obb' and result.obb is not None:
            corners = result.obb.xyxyxyxy.cpu().numpy()
            angles = np.degrees(result.obb.xywhr[:, 4].cpu().numpy())
            for points, angle, class_id, score in zip(corners, angles, result.obb.cls.tolist(), result.obb.conf.tolist()):
                (x1, y1), (x2, y2) = points.min(axis=0), points.max(axis=0)
                targets.append({'class': label(int(class_id)), 'score': round(score, 2), 'rect': rect(x1, y1, x2, y2),
                                'angle': round(float(angle), 1), 'corners': points.astype(int).tolist()})
        elif result.boxes is not None:
            for (x1, y1, x2, y2), class_id, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                targets.append({'class': label(int(class_id)), 'score': round(score, 2), 'rect': rect(x1, y1, x2, y2)})
        return targets

    def detect(self, rgb, depth, previous, now):
        """
        Returns:
            dict: 'found', 'time' and 'targets', or None until the network has produced its first result.
                  Each target is {'class', 'score', 'rect' [x, y, width, height] (top-left corner),
                  'time' (when it was last seen), 'distance' (mm, if there is a depth picture)}, plus 'angle'
                  (degrees) and 'corners' (4 points) for models with rotated boxes (e.g. best_garbage_26).
                  Each class is sticky on its own: one that is lost stays for the camera's expire time.
        """
        result = self._result
        if result is None:
            return None
        if self.is_stale(result, now): # The model has stopped producing results
            result = {'found': False, 'targets': [], 'time': now}
        targets = keep_or_expire_items(previous.get('targets', []), [dict(t) for t in result.get('targets', [])],
                                       now, self.expire_time, key='class')
        data = {'found': bool(targets), 'targets': targets,
                'time': max(t['time'] for t in targets) if targets else result['time']}
        if depth is not None:
            for target in targets:
                x, y, w, h = target['rect']
                target['distance'] = self.camera.get_distance_at(x + w // 2, y + h // 2, radius=max(3, min(w, h) // 4), depth=depth)
        return data

    def draw(self, frame, data):
        for target in data.get('targets', []):
            x, y, w, h = target['rect']
            if 'corners' in target:
                cv2.polylines(frame, [np.array(target['corners'], dtype=np.int32)], True, (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = f"{target['class']} {target['score']:.2f}"
            if target.get('distance') is not None:
                label += f" {target['distance']}mm"
            cv2.putText(frame, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def describe(self, data):
        if self.model is None:
            return "Objects: no model loaded"
        targets = data.get('targets', [])
        return f"Objects ({self.model_name} {self.framerate} fps): " + (", ".join(t['class'] for t in targets) or "none")

    def reset(self):
        self._result = None
