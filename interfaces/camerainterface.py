# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Camera interface for the Hiwonder MentorPi depth camera (ascamera).
# Frames come straight from the ROS 2 topics published by the robot's bringup (no video server needed).
# A background thread runs the requested detection tasks on the newest frame, draws the results and keeps
# the latest detection data ready for the robot code. Each detection task (line, colour, YOLO26 objects, MediaPipe pose)
# lives in its own file in detection_tasks/ - add a new one there and it can be used here straight away.
# Run inside the MentorPi container, where ROS 2 is available.

import sys, os, time, threading, json
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from detection_tasks import Detector, DETECTOR_CLASSES, DETECTION_TASKS, json_safe

RGB_IMAGE_TOPIC = '/ascamera/camera_publisher/rgb0/image'
DEPTH_IMAGE_TOPIC = '/ascamera/camera_publisher/depth0/image_raw'
# Reliable, keep only the newest frame. The camera publishes RELIABLE and its large depth frames
# never arrive on a best-effort (sensor data) subscription.
CAMERA_QOS = 1

# Hiwonder's colour calibration (edit with their lab_tool). 'Stereo' is the table for the ascamera depth camera.
LAB_CONFIG_PATH = '/home/ubuntu/software/lab_tool/lab_config.yaml'
LAB_CAMERA_TYPE = 'Stereo'

# Default folder for saved pictures (<project>/static/photos)
PHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'photos')



def image_msg_to_array(msg):
    """
    Converts a ROS sensor_msgs/Image into a numpy array without going through cv_bridge.

    Args:
        msg (sensor_msgs.msg.Image): A colour (bgr8/rgb8) or depth (16UC1) image message.

    Returns:
        numpy.ndarray: BGR uint8 image for colour, uint16 millimetres for depth (0 = no reading).
    """
    if msg.encoding in ('bgr8', 'rgb8'):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img
    if msg.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def photo_path(prefix, ext):
    """Timestamped file path in static/photos, e.g. camera_20261024_203015_123.jpg."""
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}'
    return os.path.join(PHOTO_DIR, f'{prefix}_{stamp}{ext}')

#-------------------------------------------
# The Camera Object
class CameraInterface():

    def __init__(self, timelimit=20, node=None, use_depth=False):
        """
        Initializes the Camera Interface, loads colour calibration data and subscribes to the camera topics.

        Args:
            timelimit (int, optional): General timelimit setting. Defaults to 20.
            node (rclpy.node.Node, optional): An existing ROS node to share (e.g. ClawPi().node, which is
                                              already being spun). If None, the camera creates and spins its own.
            use_depth (bool, optional): Subscribe to the depth picture so detections include distances. Only for a
                                        fitted depth camera - the ClawPi has none by default. Defaults to False.
        """
        self.timelimit = timelimit
        np.set_printoptions(suppress=True) # Disable scientific notation for clarity

        self.thread = None
        self.status = "Init"
        self.mode = "rgb" # 'rgb' shows the colour picture, 'depth' shows the colour-mapped depth picture
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8) # Latest processed frame with drawings
        self.frame_id = 0
        self.rgb_frame = None   # Newest raw colour frame (BGR)
        self.depth_frame = None # Newest raw depth frame (uint16 millimetres, 0 = no reading)
        self.rgb_frame_id = 0
        self.framerate = 0.0

        self.detection_data = {}
        self.detection_tasks = []
        self.detect_once = False
        # Sticky detections: each detection stays in the detection data until detection_data_expire_time seconds
        # after it was last seen, then disappears (stops flickering). Set with set_sticky_detections().
        self.sticky_detections = True
        self.detection_data_expire_time = 1.0
        self.clear_temp_detection_data = False

        # Depth
        self.use_depth = False
        self.depth_display_range = (200, 4000) # Millimetres mapped from red (near) to blue (far) in depth mode

        # One detector per detection task (see detection_tasks/). Their settings live on them, e.g.
        # CAMERA.detectors['detect_line'].rois or CAMERA.detectors['detect_object'].confidence_level
        self.detectors = {cls.name: cls(self) for cls in DETECTOR_CLASSES}
        self.background_threads = []

        self.output_text = True
        self.output_message = ""
        self.paused = False
        self.drawing = True

        # Thread Locks to prevent data corruption between the main program and the background processing threads
        self.dict_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Condition(self.frame_lock) # Notified every time a new processed frame is ready
        self.task_lock = threading.Lock()
        self.clear_lock = threading.Lock()
        self._jpeg = None
        self._jpeg_key = None

        # Load the colours for detection
        try:
            with open(LAB_CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.lab_colours = yaml.safe_load(f)['lab'][LAB_CAMERA_TYPE]
        except Exception as e:
            print(f"Could not load lab colours, using empty dictionary. Error: {e}")
            self.lab_colours = {}

        # Connect to the camera topics
        self._rgb_msg = None
        self._depth_msg = None
        self._new_rgb = threading.Event()
        self._subscriptions = []
        self._depth_subscription = None
        self._executor = None
        self._owns_node = node is None
        self._owns_rclpy = False
        try:
            if self._owns_node:
                if not rclpy.ok():
                    rclpy.init()
                    self._owns_rclpy = True
                self.node = Node('camera_interface')
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self.node)
                self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
                self._spin_thread.start()
            else:
                self.node = node
            self._subscriptions.append(self.node.create_subscription(Image, RGB_IMAGE_TOPIC, self._rgb_callback, CAMERA_QOS))
            if use_depth:
                self.enable_depth()
            if not self._new_rgb.wait(timeout=5):
                raise TimeoutError("no camera frames received - is the robot's bringup running?")
            self.status = "Ready"
        except Exception as e:
            self.status = "Fail"
            print(f"Video capture could not be accessed: {e}")

    def _rgb_callback(self, msg):
        """ROS callback: keeps only the newest colour message. Converting it is left to the processing thread."""
        self._rgb_msg = msg
        self._new_rgb.set()

    def _depth_callback(self, msg):
        """ROS callback: keeps only the newest depth message."""
        self._depth_msg = msg

    def enable_depth(self):
        """Subscribes to the depth picture (if not already). Needed for depth mode and distances (needs a depth camera)."""
        if self._depth_subscription is None and self.node is not None:
            self._depth_subscription = self.node.create_subscription(Image, DEPTH_IMAGE_TOPIC, self._depth_callback, CAMERA_QOS)
            self._subscriptions.append(self._depth_subscription)
        self.use_depth = True

    def start(self):
        """
        Starts the background camera threads if the camera initialized successfully.
        """
        if self.status == "Ready":
            self.status = "Running"
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            # Slow detectors (e.g. the neural network) run in their own threads and never hold up the others
            self.background_threads = []
            for detector in self.detectors.values():
                if type(detector).background_loop is not Detector.background_loop:
                    thread = threading.Thread(target=detector.background_loop, daemon=True)
                    thread.start()
                    self.background_threads.append(thread)

    def update(self):
        """
        The core loop executed by the background thread. Waits for each new camera frame, runs the active
        detection tasks, draws visual overlays and updates the shared data dictionaries safely.
        """
        local_detection_data = {}
        framecount = 0
        fps_start = time.time()

        while self.status == "Running":
            # Sleep until the camera delivers a new frame (no busy waiting)
            if not self._new_rgb.wait(timeout=0.5):
                continue
            self._new_rgb.clear()
            if self.paused:
                continue

            try:
                rgb = image_msg_to_array(self._rgb_msg)
                depth_msg = self._depth_msg if self.use_depth else None
                depth = image_msg_to_array(depth_msg) if depth_msg is not None else None
            except Exception as e:
                print(f"Bad camera frame: {e}")
                continue

            with self.frame_lock:
                self.rgb_frame = rgb
                self.depth_frame = depth
                self.rgb_frame_id += 1

            # Frame rate, averaged over each second
            currenttime = time.time()
            framecount += 1
            if currenttime - fps_start >= 1.0:
                self.framerate = round(framecount / (currenttime - fps_start), 1)
                framecount, fps_start = 0, currenttime

            # Clear the temporary detection data when required
            with self.clear_lock:
                if self.clear_temp_detection_data:
                    local_detection_data.clear()
                    for detector in self.detectors.values():
                        detector.reset()
                    self.clear_temp_detection_data = False

            with self.task_lock:
                current_tasks = list(self.detection_tasks)

            # Run each active detector (in the order of DETECTOR_CLASSES). None means no result yet.
            for name, detector in self.detectors.items():
                if name in current_tasks:
                    data = detector.detect(rgb, depth, local_detection_data.get(name, {}), currenttime)
                    if data is not None:
                        local_detection_data[name] = data

            # Remove data for tasks that have been switched off
            for task in list(local_detection_data):
                if task not in current_tasks:
                    del local_detection_data[task]

            # Detect once: stop after one full round of results (slow detectors like the model answer later)
            if self.detect_once and current_tasks and all(t in local_detection_data for t in current_tasks):
                with self.task_lock:
                    self.detection_tasks.clear()
                self.detect_once = False

            # Choose the picture to show and draw on
            if self.mode == "depth":
                display = self.colourise_depth(depth) if depth is not None else np.zeros_like(rgb)
            else:
                display = rgb.copy()

            if self.drawing and current_tasks:
                self._draw_detections(display, local_detection_data)

            if self.output_text:
                self._draw_text(display, local_detection_data, current_tasks)

            # Update state variables safely. json_safe makes a fresh copy using only JSON types.
            published = json_safe(local_detection_data)
            with self.dict_lock:
                self.detection_data = published
            with self.frame_ready:
                self.frame = display
                self.frame_id += 1
                self.frame_ready.notify_all()

    #-------------------------------------------
    # Detection tasks (each one lives in detection_tasks/). These run a detection directly on a frame.
    def detect_line(self, frame, colour=None):
        """Line following detection on one frame (see detection_tasks/detect_line.py). Returns (frame, data)."""
        return self.detectors['detect_line'].detect_line(frame, colour)

    def detect_colours(self, frame, colours):
        """Detects several colours on one frame (see detection_tasks/detect_colour.py). Returns {colour: data}."""
        return self.detectors['detect_colour'].detect_colours(frame, colours)

    def detect_color(self, frame, minC, maxC):
        """
        Detects one LAB colour range in a frame (kept for older code; detect_colours is faster for several colours).

        Returns:
            tuple: (frame, data_dictionary) containing detection success, area, bounding rect and timestamp.
        """
        self.lab_colours['_custom'] = {'min': list(minC), 'max': list(maxC)}
        data = self.detect_colours(frame, ['_custom'])['_custom']
        del self.lab_colours['_custom']
        return frame, data

    def get_distance_at(self, x, y, radius=5, depth=None):
        """
        Returns the distance to whatever is at pixel (x, y), using the median of the depth readings around it.
        The depth picture lines up with the colour picture, so detection coordinates can be used directly.

        Args:
            x, y (int): Pixel coordinates in the 640x480 frame.
            radius (int, optional): Half-size of the square of pixels used. Defaults to 5.

        Returns:
            int: Distance in millimetres, or None if there is no depth reading there.
        """
        if depth is None:
            with self.frame_lock:
                depth = self.depth_frame
        if depth is None:
            return None
        h, w = depth.shape
        x, y, radius = int(x), int(y), int(radius)
        window = depth[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
        valid = window[window > 0]
        return int(np.median(valid)) if valid.size else None

    def colourise_depth(self, depth):
        """
        Turns a depth frame into a colour picture for viewing: near = red, far = blue, no reading = black.
        Uses a fixed range (self.depth_display_range) so colours don't flicker between frames.
        """
        near, far = self.depth_display_range
        scaled = (np.clip(depth, near, far).astype(np.float32) - near) * (255.0 / (far - near))
        image = cv2.applyColorMap((255 - scaled).astype(np.uint8), cv2.COLORMAP_JET)
        image[depth == 0] = 0
        return image

    def load_detection_model(self, model="yolo26n", size=None, confidence=None):
        """
        Loads one of Hiwonder's YOLO26 models for detect_object (see detection_tasks/detect_object.py).

        Args:
            model (str, optional): 'yolo26n' (80 everyday objects - person, cup, chair...), 'best_traffic_26',
                                   'best_garbage_26', or a path to a .pt file / _openvino_model folder. Defaults to 'yolo26n'.
            size (int, optional): Image size the model works at (320 or 640).
            confidence (float, optional): Minimum score (0-1) to report a target.

        Returns:
            bool: True if the model is ready.
        """
        return self.detectors['detect_object'].load(model, size, confidence)

    def clear_detection_model(self):
        """Removes the neural network model and its labels from memory."""
        self.detectors['detect_object'].clear()

    def _draw_detections(self, frame, data):
        """Lets each detector draw its results onto the frame."""
        for name, detector in self.detectors.items():
            if name in data:
                detector.draw(frame, data[name])

    def _draw_text(self, frame, data, tasks):
        """Writes the frame rate, mode, custom message and each active detector's summary onto the frame."""
        lines = [f"FPS: {self.framerate}  Mode: {self.mode}  {self.output_message}"]
        self.output_message = ""
        for name, detector in self.detectors.items():
            if name in tasks:
                lines.append(detector.describe(data.get(name, {})))
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 15 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    def get_frame(self):
        """
        Thread-safe method to get the current processed video frame (colour or depth picture, depending on the mode).

        Returns:
            numpy.ndarray: The current image frame (or a blank frame if unavailable).
        """
        with self.frame_lock:
            # Guarantees a frame format is returned, never None
            if self.frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.frame.copy()

    def get_rgb_frame(self):
        """Returns the newest raw colour frame (BGR, no drawings), or None."""
        with self.frame_lock:
            return None if self.rgb_frame is None else self.rgb_frame.copy()

    def get_depth_frame(self):
        """Returns the newest raw depth frame (uint16 millimetres, 0 = no reading), or None."""
        with self.frame_lock:
            return None if self.depth_frame is None else self.depth_frame.copy()

    def get_jpeg_frame(self, quality=50):
        """
        Thread-safe method to retrieve the current frame encoded as JPEG bytes.
        Ideal for passing to a Flask or web server for remote streaming. If this frame has already been
        sent, it waits for the next one, so a streaming loop runs at the camera's rate instead of
        encoding the same picture over and over.

        Args:
            quality (int, optional): JPEG encoding quality (0-100). Defaults to 50.

        Returns:
            bytes: The JPEG encoded image.
        """
        with self.frame_ready:
            if self._jpeg_key == (self.frame_id, quality) and self.status == "Running":
                self.frame_ready.wait(timeout=0.5)
            frame_to_encode, key = self.frame, (self.frame_id, quality)

        if key != self._jpeg_key or self._jpeg is None:
            ret, jpeg = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret: # Absolute fallback if encoding fails (should be rare)
                _, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            self._jpeg, self._jpeg_key = jpeg.tobytes(), key
        return self._jpeg

    def _save_image(self, image, path):
        """Writes an image, creating its folder if needed. Returns the path, or None on failure."""
        if image is None:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, image) else None

    def save_frame_as_image(self, path=None, drawings=True):
        """
        Saves the current video frame as a JPEG.

        Args:
            path (str, optional): Where to save. Defaults to a timestamped file in static/photos.
            drawings (bool, optional): True saves the frame as shown (detections drawn, depth picture in depth mode),
                                       False saves the plain colour picture. Defaults to True.

        Returns:
            str: The saved path, or None if there was no frame.
        """
        image = self.get_frame() if drawings else self.get_rgb_frame()
        return self._save_image(image, path or photo_path('camera', '.jpg'))

    def save_depth_image(self, path=None, colourise=False):
        """
        Saves the current depth frame as a PNG.

        Args:
            path (str, optional): Where to save. Defaults to a timestamped file in static/photos.
            colourise (bool, optional): False saves the raw 16-bit depth (millimetres) losslessly, for measuring later.
                                        True saves a colour picture for viewing (near = red, far = blue). Defaults to False.

        Returns:
            str: The saved path, or None if there was no depth frame.
        """
        depth = self.get_depth_frame()
        if depth is not None and colourise:
            depth = self.colourise_depth(depth)
        return self._save_image(depth, path or photo_path('depth_colour' if colourise else 'depth', '.png'))

    def set_mode(self, mode):
        """
        Chooses which picture is shown and streamed. Detection always uses the colour picture.

        Args:
            mode (str): 'rgb' for the normal colour picture, 'depth' for the colour-mapped depth picture.
        """
        if mode not in ('rgb', 'depth'):
            print("Mode must be 'rgb' or 'depth'")
            return
        if mode == 'depth':
            self.enable_depth()
        self.mode = mode

    def turn_on_depth_mode(self):
        """Shows the depth picture (near = red, far = blue) instead of the colour picture."""
        self.set_mode('depth')

    def turn_off_depth_mode(self):
        """Goes back to showing the colour picture."""
        self.set_mode('rgb')

    def toggle_mode(self):
        """Switches between the colour and depth pictures. Returns the new mode."""
        self.set_mode('rgb' if self.mode == 'depth' else 'depth')
        return self.mode

    def get_detection_data(self):
        """
        Thread-safe method to retrieve the most recent detection data.

        Returns:
            dict: A copy of the detection data, {task name: that detector's data}, e.g.
                  {'detect_line': {'found': True, 'angle': -8.1, ...}, 'detect_colour': {'red': {...}}}.
                  Only JSON types are used, so it can be passed straight to json.dumps or Flask's jsonify.
        """
        with self.dict_lock:
            return json_safe(self.detection_data)

    def get_detection_json(self):
        """The most recent detection data (see get_detection_data) as a JSON string."""
        return json.dumps(self.get_detection_data())

    def clear_detection_data(self):
        """Clears out the active detection dictionary securely."""
        with self.dict_lock:
            self.detection_data.clear()
        with self.clear_lock:
            self.clear_temp_detection_data = True
        for detector in self.detectors.values(): # Background detectors straight away, so an old result isn't picked up
            if type(detector).background_loop is not Detector.background_loop:
                detector.reset()

    def set_sticky_detections(self, sticky=True, expire_time=None):
        """
        Turns sticky detections on or off. When on, a detection stays in the detection data after it is lost
        until expire_time seconds after it was last seen, so one missed frame doesn't make it flicker away.
        When off, detections disappear on the first frame they aren't seen.

        Args:
            sticky (bool, optional): Keep detections for a while after they are lost. Defaults to True.
            expire_time (float, optional): Seconds to keep them. Defaults to the current setting (1 second).
        """
        self.sticky_detections = bool(sticky)
        if expire_time is not None:
            self.detection_data_expire_time = float(expire_time)

    def add_detection_task(self, task, detect_once=False):
        """
        Adds a specific vision task to the active background polling cycle.

        Args:
            task (str): A detection task from detection_tasks/: 'detect_line', 'detect_colour', 'detect_object' or 'detect_pose'.
            detect_once (bool, optional): If True, removes the tasks after the first round of results. Defaults to False.
        """
        if task not in self.detectors:
            print("Task must be one of: " + ", ".join(self.detectors))
            return
        detector = self.detectors[task]
        if detector.needs_depth:
            self.enable_depth()
        detector.prepare() # e.g. loads the neural network the first time

        with self.task_lock:
            if task not in self.detection_tasks:
                self.detect_once = detect_once
                self.detection_tasks.append(task)

    def set_detection_tasks(self, tasks=None):
        """
        Overrides the current list of detection tasks with a new list.

        Args:
            tasks (list, optional): List of strings representing new tasks. Defaults to empty list.
        """
        self.clear_detection_tasks()
        for task in tasks or []:
            self.add_detection_task(task)

    def clear_detection_tasks(self):
        """Removes all currently active detection tasks from the polling cycle."""
        with self.task_lock:
            self.detection_tasks.clear()

    def detect_all(self, exclude_colours=None):
        """
        Helper method to activate all available detection tasks (lines, colours, objects, and any others in detection_tasks/).

        Args:
            exclude_colours (list, optional): Colours in the YAML file to ignore. Defaults to ['black', 'white'].
        """
        if exclude_colours is None:
            exclude_colours = ['black', 'white']

        for name, detector in self.detectors.items():
            if self.use_depth or not detector.needs_depth:
                self.add_detection_task(name)

        self.set_detection_colours([c for c in self.lab_colours if c not in exclude_colours])

    def turn_on_output_text(self):
        """Enables on-screen text overlays for framerate and data."""
        self.output_text = True

    def turn_off_output_text(self):
        """Disables on-screen text overlays."""
        self.output_text = False

    def set_output_message(self, message):
        """
        Sets a custom text string to display on the video feed.

        Args:
            message (str): The text message to display.
        """
        self.output_message = message

    def remove_detection_task(self, task):
        """
        Removes a single specific task from the active task list.

        Args:
            task (str): The name of the task to remove.
        """
        with self.task_lock:
            if task in self.detection_tasks:
                self.detection_tasks.remove(task)

    def end_detection(self):
        """Helper method that stops all tasks, wipes active data, and clears target colors simultaneously."""
        self.clear_detection_tasks()
        self.clear_detection_data()
        self.clear_detection_colours()

    def turn_on_drawing(self):
        """Enables drawing bounding boxes, lines, and rects on the frame."""
        self.drawing = True

    def turn_off_drawing(self):
        """Disables drawing visual elements on the frame to save processing time."""
        self.drawing = False

    def add_detection_colour(self, colour):
        """
        Adds a named color (from the YAML file) to the active searching rotation.

        Args:
            colour (str): The name of the color to search for (e.g., 'red', 'green').
        """
        detector = self.detectors['detect_colour']
        with detector.colours_lock:
            if colour not in detector.colours:
                detector.colours.append(colour)

    def set_detection_colours(self, colourlist=None):
        """
        Overrides the current color searching rotation with a specific list.

        Args:
            colourlist (list, optional): List of color name strings. Defaults to empty list.
        """
        detector = self.detectors['detect_colour']
        with detector.colours_lock:
            detector.colours = list(colourlist or [])

    def set_line_detection_colour(self, colour):
        """
        Sets a specific mask color for the line detection tracking.

        Args:
            colour (str): Color string name from the YAML settings.
        """
        self.detectors['detect_line'].colour = colour

    def remove_detection_colour(self, colour):
        """
        Removes a single specific color from the searching rotation.

        Args:
            colour (str): The string name of the color to stop tracking.
        """
        detector = self.detectors['detect_colour']
        with detector.colours_lock:
            if colour in detector.colours:
                detector.colours.remove(colour)

    def clear_detection_colours(self):
        """Stops the robot from tracking any specific colors by emptying the list."""
        detector = self.detectors['detect_colour']
        with detector.colours_lock:
            detector.colours.clear()

    def create_detection_window(self):
        """Opens a standard OpenCV GUI window named 'Detection Mode' on the host screen."""
        cv2.namedWindow('Detection Mode')
        cv2.resizeWindow('Detection Mode', 640, 480)

    def test_detection_performance(self, duration_per_task=5, show_window=True):
        """
        A standalone profiler to test the camera's performance without needing the full RobotInterface.
        Cycles through a baseline, each detection algorithm and depth mode, printing the frame rate of each.

        Args:
            duration_per_task (int): How long to run each task in seconds. Defaults to 5.
            show_window (bool): Whether to render the OpenCV window. Defaults to True.
        """
        tasks_to_test = ['baseline'] + DETECTION_TASKS + (['depth_mode'] if self.use_depth else [])

        # Pre-load dependencies
        self.set_detection_colours(['red', 'green', 'blue'])
        self.load_detection_model()
        self.turn_on_output_text()

        if show_window:
            self.create_detection_window()

        print("\n========================================")
        print(" STARTING STANDALONE CAMERA PROFILER")
        print("========================================\n")

        abort_test = False

        for task in tasks_to_test:
            if abort_test:
                break

            # Clear previous tasks and data
            self.clear_detection_tasks()
            self.clear_detection_data()
            self.set_mode('rgb')

            if task == 'baseline':
                print(f">>> Testing 'Baseline (No Detection)' for {duration_per_task} seconds...")
                message = "BASELINE - RAW FEED"
            elif task == 'depth_mode':
                print(f">>> Testing 'Depth mode' for {duration_per_task} seconds...")
                self.set_mode('depth')
                if 'detect_depth' in self.detectors:
                    self.add_detection_task('detect_depth')
                message = "DEPTH MODE"
            else:
                print(f">>> Testing '{task}' for {duration_per_task} seconds...")
                self.add_detection_task(task)
                message = task.upper()

            endtime = time.time() + duration_per_task

            # Run the specific task for the allotted time
            while (time.time() < endtime) and self.status == "Running":
                self.set_output_message(message)
                if show_window:
                    frame = self.get_frame()
                    if frame.size > 0:
                        cv2.imshow('Detection Mode', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        abort_test = True
                        break
                else:
                    # CPU saver for headless mode
                    time.sleep(0.03)

            model_fps = getattr(self.detectors.get(task), 'framerate', None)
            extra = f", model {model_fps} fps" if model_fps is not None else ""
            print(f"    frame rate: {self.framerate} fps{extra}")
            print(f"    data: {self.get_detection_data()}")

            if abort_test:
                print("\n[!] Test aborted by user.")
                break

        print("\n========================================")
        print(" PROFILER COMPLETE ")
        print("========================================\n")

        # Clean up
        self.set_output_message("")
        self.set_mode('rgb')
        self.end_detection()

        if show_window:
            cv2.destroyAllWindows()

        return

    def stop(self):
        """
        Initiates a graceful shutdown of the background camera threads and releases the camera subscriptions.
        """
        self.status = "Stop"
        with self.frame_ready:
            self.frame_ready.notify_all()
        for thread in [self.thread] + self.background_threads:
            if thread and thread is not threading.current_thread():
                thread.join(timeout=2)
        try:
            for subscription in self._subscriptions:
                self.node.destroy_subscription(subscription)
            self._subscriptions.clear()
            self._depth_subscription = None
            if self._owns_node and self.node is not None:
                self._executor.shutdown()
                self._spin_thread.join(timeout=2)
                self.node.destroy_node()
                self.node = None
                if self._owns_rclpy and rclpy.ok():
                    rclpy.shutdown()
        except Exception as e:
            print(f"Camera shutdown problem: {e}")

    def pause(self):
        """Temporarily halts the processing of new frames in the background thread."""
        print("Pausing Camera")
        self.paused = True

    def resume(self):
        """Resumes the processing of frames in the background thread."""
        print("Resuming Camera")
        self.paused = False

# TEST CAMERA CODE
if __name__ == '__main__':
    input("Please press enter to begin standalone camera test: ")
    CAMERA = CameraInterface()
    print("\033c")

    try:
        # 1. Start the background threads
        CAMERA.start()
        time.sleep(2) # Give the camera and threads a moment to initialize

        # 2. Let the built-in profiler take over! (show_window needs a display, e.g. VNC)
        CAMERA.test_detection_performance(duration_per_task=10, show_window=bool(os.environ.get('DISPLAY')))

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the camera releases the video feed when done
        CAMERA.end_detection()
        CAMERA.stop()
        cv2.destroyAllWindows()
        time.sleep(1)
        sys.exit(0)
