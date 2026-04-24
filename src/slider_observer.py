import numpy as np
import threading
import time
from collections import deque
from scipy.spatial.transform import Rotation
from simcore import RobotSystem

from src.pose_estimation import PoseEstimation
from src.ekf import SliderEKF
from src.utils import wxyz_to_xyzw


class SliderObserver:
    def __init__(self, config: dict, system: RobotSystem, model=None):
        self.config                 = config
        self.system                 = system
        self.variant                = config["mpc"]["variant"]
        self.slider_name            = config["slider_name"]
        self._vision_lost_timeout   = config["vision"]["timeout"]

        self.estimator = PoseEstimation(config, system)
        self.est_dt    = 1.0 / config.get("vision", {}).get("f_estimation", 30.0)
        self.prop_dt   = config["mpc"]["dt"]

        self._ekf      = SliderEKF(model, config) if model is not None else None
        self._py       = 0.0

        self._est_lock  = threading.Lock()
        self._ekf_lock  = threading.Lock()
        self._new_meas  = False
        self._pending_z = None

        self.running      = False
        self._est_thread  = None
        self._prop_thread = None
        self._last_visual = None

        self._detection_times = deque()

    def start(self):
        if self.running: return
        self.running = True
        print("Started observer")
        self._est_thread  = threading.Thread(target=self._est_loop,  daemon=True)
        self._prop_thread = threading.Thread(target=self._prop_loop, daemon=True)
        self._est_thread.start()
        self._prop_thread.start()

    def stop(self):
        self.running = False

    def set_control(self, u: np.ndarray):
        if self._ekf is None: return
        with self._ekf_lock:
            self._ekf.set_control(u)

    def set_py(self, p_y: float):
        with self._ekf_lock:
            self._py = p_y

    def get_state(self) -> np.ndarray:
        if self.variant == "BASELINE":
            return self.get_gt_state()
        return self.get_est_state()

    def get_gt_state(self) -> np.ndarray:
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

    def get_est_state(self) -> np.ndarray:
        with self._ekf_lock:
            return self._ekf.mean if self._ekf is not None else None

    def get_covariance(self) -> np.ndarray:
        with self._ekf_lock:
            return self._ekf.covariance if self._ekf is not None else None

    def get_vision_state(self) -> np.ndarray:
        with self._ekf_lock:
            return self._pending_z if self._pending_z is not None else None

    def get_recent_detection_count(self, window_sec: float) -> int:
        now = time.time()
        with self._est_lock:
            while self._detection_times and self._detection_times[0] < now - window_sec:
                self._detection_times.popleft()
            return len(self._detection_times)

    def is_localised(self, n_required: int, window_sec: float, cov_thresh: float = 1e-3) -> bool:
        if self.variant == "BASELINE": return True
        if self.get_recent_detection_count(window_sec) < n_required:
            return False
        with self._ekf_lock:
            if self._ekf is None or not self._ekf.initialised():
                return False
            return float(np.trace(self._ekf.covariance)) < cov_thresh

    def reset(self, x0: np.ndarray = None):
        with self._ekf_lock:
            if self._ekf is not None:
                self._ekf.reset(x0=x0)
            self._py        = 0.0
            self._pending_z = None
        self._last_visual = None
        with self._est_lock:
            self._detection_times.clear()

    def has_visual(self) -> bool:
        if self.variant == "BASELINE": return True
        return self._last_visual is not None and (time.time() - self._last_visual) <= self._vision_lost_timeout

    def _prop_loop(self):
        # Propagate EKF at MPC rate; fuse pending vision measurement when available.
        while self.running:
            t0 = time.perf_counter()

            with self._ekf_lock:
                z  = self._pending_z
                self._pending_z = None
                py = self._py

            if self._ekf is not None:
                with self._ekf_lock:
                    if z is not None:
                        if not self._ekf.initialised():
                            self._ekf.reset(x0=z)
                        else:
                            self._ekf.update(z)
                    self._ekf.predict(py)

            sleep = self.prop_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _est_loop(self):
        # Run vision at camera rate; post measurements for the propagation loop.
        while self.running:
            t0 = time.perf_counter()
            z  = self.estimator.get_pose_estimate()

            if z is not None:
                with self._ekf_lock:
                    self._pending_z = z
                now = time.time()
                self._last_visual = now
                with self._est_lock:
                    self._detection_times.append(now)

            sleep = self.est_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)