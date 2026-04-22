import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from src.utils import make_T, marker_obj_points, rvec_tvec_to_T


class PoseEstimation:
    ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    DETECTOR     = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    def __init__(self, config, system):
        self.system      = system
        vision_cfg       = config["vision"]
        self.MARKER_SIZE = vision_cfg["marker_size"]

        K, dist, _, _, _ = self.system.sim.get_camera_intrinsics(vision_cfg["cam_name"])
        self.K, self.dist = K, dist

        self.marker = {int(k): v for k, v in vision_cfg["marker"].items()}

        self.T_slider_tag = {}
        for marker_id, m in self.marker.items():
            R_st = Rotation.from_euler("xyz", m["rot"]).as_matrix()
            self.T_slider_tag[marker_id] = make_T(R_st, np.array(m["pos"]))

        cam_tf = vision_cfg["T_ee_cam"]
        self.T_ee_cam = make_T(Rotation.from_euler("xyz", cam_tf["euler_xyz"]).as_matrix(), np.array(cam_tf["pos"]))

        local_corners     = marker_obj_points(self.MARKER_SIZE)
        self.slider_corners = {}
        for marker_id, T_st in self.T_slider_tag.items():
            corners = (T_st[:3, :3] @ local_corners.T).T + T_st[:3, 3]
            self.slider_corners[marker_id] = corners.astype(np.float32)

    def get_pose_estimate(self):
        frame_bgr = self.system.get_camera_image("eye_in_hand", bgr=True)
        arm_state = self.system.get_state()["arm"]
        return self.estimate_from_frame(frame_bgr, arm_state)

    def estimate_from_frame(self, frame_bgr, arm_state):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.DETECTOR.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        obj_points, img_points = [], []
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self.slider_corners:
                continue
            obj_points.append(self.slider_corners[marker_id])
            img_points.append(corners[i][0])

        if not obj_points:
            return None

        obj_points = np.concatenate(obj_points, axis=0).astype(np.float32)
        img_points = np.concatenate(img_points, axis=0).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        cv2.solvePnPRefineLM(obj_points, img_points, self.K, self.dist, rvec, tvec)
        return rvec_tvec_to_T(rvec, tvec)

    def _transform_to_world(self, T_cam_slider, arm_state):
        ee_pose_base  = self.system.ctrl["arm"].kin_model.forward_kinematics(arm_state.q)
        ee_pose_world = self.system.ctrl["arm"].transform_base_to_world_frame(ee_pose_base)
        T_world_ee    = make_T(ee_pose_world.rotation_matrix, ee_pose_world.position)
        T_world_slider = T_world_ee @ self.T_ee_cam @ T_cam_slider
        xy    = T_world_slider[:2, 3]
        theta = np.arctan2(T_world_slider[1, 0], T_world_slider[0, 0])
        return np.array([xy[0], xy[1], theta])