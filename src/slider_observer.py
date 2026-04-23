import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation
from simcore import RobotSystem

from src.pose_estimation import PoseEstimation
from src.utils import wxyz_to_xyzw



class SliderObserver:
    def __init__(self, config: dict, system: RobotSystem):
        self.config = config
        self.system = system
        self.variant = config["mpc"]["variant"]
        self.slider_name = config["slider_name"]
        
        self.estimator = PoseEstimation(config, system)
        self.est_dt     = 1.0 / config.get("vision", {}).get("f_estimation", 30.0)
        self.est_thread = threading.Thread(target=self._est_loop, daemon=True)
        self.est_lock = threading.Lock()
        self.running = False

        self.xs_est = None

    def start(self):
        if self.running: return

        self.running = True
        print("Started observer")

        if self.est_thread is None or not self.est_thread.is_alive():
            self.est_thread = threading.Thread(target=self._est_loop, daemon=True)
            self.est_thread.start()

    def stop(self):
        self.running = False

    def get_state(self):
        if self.variant == "BASELINE":
            return self.get_gt_state()
        else:
            return self.get_est_state()

    def get_gt_state(self):
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

    def get_est_state(self):
        with self.est_lock:
            return None if self.xs_est is None else self.xs_est.copy()

    def reset(self):
        self.xs_est = None

    def _est_loop(self):
        print(f"Trying to run estimation loop: {self.running}")
        while self.running:
            t0 = time.perf_counter()
            x = self.estimator.get_pose_estimate()

            with self.est_lock:
                self.xs_est = x

            sleep = self.est_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
