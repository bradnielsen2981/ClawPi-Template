# -*- coding: utf-8 -*-
# A robot routine, called as ROBOT.face_tracking(...). `self` is the robot.
import time

from .routine_tools import pause_for_behaviour

# How strongly the camera turns toward the face: degrees moved per pixel the face is off centre, each frame.
# Bigger = faster but can overshoot and wobble. Roughly 0.1 degrees per pixel would centre it in one step.
TRACKING_GAIN = 0.04
MAX_STEP = 6     # Most degrees to move in one frame, so a jump in detection doesn't jerk the camera
DEAD_ZONE = 15   # Pixels from centre that count as centred (stops the camera jittering)

def face_tracking(self, timelimit=30, recentre=True):
    """
    Moves the camera (pan and tilt) to keep the largest face in the middle of the picture.
    The robot's wheels don't move. If the face is lost, the camera stays where it is until one is seen again.

    Args:
        timelimit (int/float): Seconds to track for before stopping. Defaults to 30.
        recentre (bool): Point the camera straight ahead again when finished. Defaults to True.

    Returns:
        dict: The final detection data snapshot and timestamp.
    """
    self.set_routine("face_tracking")
    data = {'routine': "face_tracking", 'starttime': time.time()}
    detectors_before = self.CAMERA.get_detectors() # put back when finished, so other detectors aren't turned off

    try:
        self.CAMERA.add_detector("detect_face")
        endtime = time.time() + timelimit

        while time.time() < endtime:
            if self.routine != "face_tracking": # interrupted by a behaviour, or stopped
                paused = pause_for_behaviour(self, "face_tracking")
                if paused is None:
                    break # stopped, so end the routine
                endtime += paused # the pause doesn't count toward the time limit
                continue

            data.update(self.CAMERA.wait_for_detection()) # waits for the next camera frame
            face_data = data.get('detect_face', {})
            if not face_data.get('found'):
                continue

            x, y = face_data['largest']['center']
            width, height = face_data['frame_size']
            error_x = width / 2 - x  # positive = face is left of centre
            error_y = height / 2 - y # positive = face is above centre

            # Positive pan turns the camera left and positive tilt turns it up, toward the face
            pan_step = 0 if abs(error_x) < DEAD_ZONE else max(-MAX_STEP, min(MAX_STEP, error_x * TRACKING_GAIN))
            tilt_step = 0 if abs(error_y) < DEAD_ZONE else max(-MAX_STEP, min(MAX_STEP, error_y * TRACKING_GAIN))
            if pan_step or tilt_step:
                self.point_camera(self.camera_pan + pan_step, self.camera_tilt + tilt_step, duration=0.05)

        if recentre and self.routine == "face_tracking": # finished normally (not stopped)
            self.centre_camera_view()

    except Exception as e:
        self.emergency_shutdown(f"face_tracking crashed: {e}")
        raise
    finally:
        # Always runs - finished, stopped or crashed
        self.CAMERA.set_detectors(detectors_before)
        self.stop_routine()

    data['endtime'] = time.time()
    return data
