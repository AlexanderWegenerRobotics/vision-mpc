import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

from src.mpc import ControllerVariant
from src.pose_estimation import PoseEstimation
from src.utils import wxyz_to_xyzw


class SliderObserver:
    def __init__(self, config, system):
        self.system      = system
        self.slider_name = config["slider_name"]
        self.variant     = ControllerVariant[config["mpc"]["variant"]]
        self._vision     = PoseEstimation(config, system)

        f_estimation         = config.get("vision",{}).get("f_estimation", 30.0)
        self._vision_dt      = 1.0 / f_estimation
        self._vis_x          = None
        self._vis_lock       = threading.Lock()
        self._last_x         = None
        self._stop_event     = threading.Event()
        self._vision_thread  = None

    def start(self):
        self._stop_event.clear()
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    def stop(self):
        self._stop_event.set()

    def _vision_loop(self):
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            x = self._vision.get_pose_estimate()

            with self._vis_lock:
                self._vis_x = x

            elapsed = time.perf_counter() - t0
            sleep   = self._vision_dt - elapsed
            if sleep > 0:
                self._stop_event.wait(timeout=sleep)

    def get_state(self):
        with self._vis_lock:
            vis_x = self._vis_x

        if self.variant == ControllerVariant.BASELINE:
            x   = self._get_gt_state()
            cov = np.zeros((3, 3))
            self._last_x = x
            return x, cov, vis_x

        if vis_x is None:
            if self._last_x is not None:
                print("[warn] no detection, using last known state")
                x = self._last_x
            else:
                x = self._get_gt_state()
                print("[warn] no detection on first step, falling back to GT")
        else:
            x = vis_x

        cov = np.zeros((3, 3))
        self._last_x = x
        return x, cov, vis_x

    def _get_gt_state(self):
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

    def _get_vision_state(self):
        with self._vis_lock:
            return self._vis_x