import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

from src.pose_estimation import PoseEstimation
from src.utils import wxyz_to_xyzw


class SliderObserver:
    def __init__(self, config, system):
        self.system      = system
        self.slider_name = config["slider_name"]
        self._vision     = PoseEstimation(config, system)

        self._vision_dt     = 1.0 / config.get("vision", {}).get("f_estimation", 30.0)
        self._vis_x         = None
        self._vis_lock      = threading.Lock()
        self._stop_event    = threading.Event()
        self._vision_thread = None

        self._frame_request = threading.Event()
        self._frame_ready   = threading.Event()
        self._pending_frame = None
        self._frame_lock    = threading.Lock()

    def start(self):
        self._stop_event.clear()
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    def stop(self):
        self._stop_event.set()
        self._frame_request.set()

    def service_frame_request(self):
        if not self._frame_request.is_set():
            return
        frame = self.system.get_camera_image("eye_in_hand", bgr=True)
        with self._frame_lock:
            self._pending_frame = frame
        self._frame_request.clear()
        self._frame_ready.set()

    def _request_frame(self):
        self._frame_ready.clear()
        self._frame_request.set()
        self._frame_ready.wait()
        with self._frame_lock:
            return self._pending_frame

    def _vision_loop(self):
        while not self._stop_event.is_set():
            t0    = time.perf_counter()
            frame = self._request_frame()

            if frame is not None and not self._stop_event.is_set():
                arm_state = self.system.get_state()["arm"]
                x = self._vision.estimate_from_frame(frame, arm_state)
                with self._vis_lock:
                    self._vis_x = x

            sleep = self._vision_dt - (time.perf_counter() - t0)
            if sleep > 0:
                self._stop_event.wait(timeout=sleep)

    def get_estimate(self):
        with self._vis_lock:
            return self._vis_x

    def get_gt_state(self):
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])