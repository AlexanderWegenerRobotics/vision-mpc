import numpy as np
from scipy.spatial.transform import Rotation

from src.mpc import ControllerVariant
from src.pose_estimation import PoseEstimation
from src.utils import wxyz_to_xyzw


class SliderObserver:
    def __init__(self, config, system):
        self.system      = system
        self.slider_name = config["slider_name"]
        self.variant     = ControllerVariant[config["mpc"]["variant"]]
        self._last_x     = None

        if self.variant != ControllerVariant.BASELINE:
            self._vision = PoseEstimation(config, system)
        else:
            self._vision = None

    def get_state(self):
        if self.variant == ControllerVariant.BASELINE:
            x   = self._get_gt_state()
            cov = np.zeros((3, 3))
            self._last_x = x
            return x, cov

        x = self._get_vision_state()
        if x is None:
            if self._last_x is not None:
                print("[warn] no detection, using last known state")
                x = self._last_x
            else:
                x = self._get_gt_state()
                print("[warn] no detection on first step, falling back to GT")

        cov = np.zeros((3, 3))
        self._last_x = x
        return x, cov

    def _get_gt_state(self):
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

    def _get_vision_state(self):
        return self._vision.get_pose_estimate()