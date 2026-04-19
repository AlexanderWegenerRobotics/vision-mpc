import numpy as np
import cv2
from simcore import RobotSystem
from src.utils import *
from scipy.spatial.transform import Rotation

class PoseEstimation:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    def __init__(self, config, system: RobotSystem):
        self.system = system
        vision_cfg = config["vision"]
        cam_name = vision_cfg["cam_name"]
        self.MARKER_SIZE = vision_cfg["marker_size"]

        K, dist, width, height, fovy = self.system.sim.get_camera_intrinsics(cam_name)
        self.K, self.dist = K, dist
        self.marker = {int(k): v for k, v in vision_cfg["marker"].items()}

        self.T_slider_tag = {}
        for marker_id, m in self.marker.items():
            R_st = Rotation.from_euler("xyz", m["rot"]).as_matrix()
            t_st = np.array(m["pos"])
            self.T_slider_tag[marker_id] = make_T(R_st, t_st)

        cam_tf = vision_cfg["T_ee_cam"]
        R_ec = Rotation.from_euler("xyz", cam_tf["euler_xyz"]).as_matrix()
        t_ec = np.array(cam_tf["pos"])
        self.T_ee_cam = make_T(R_ec, t_ec)

        local_corners = marker_obj_points(self.MARKER_SIZE)
        self.slider_corners = {}
        for marker_id, T_st in self.T_slider_tag.items():
            R_st = T_st[:3, :3]
            t_st = T_st[:3, 3]
            corners_in_slider = (R_st @ local_corners.T).T + t_st
            self.slider_corners[marker_id] = corners_in_slider.astype(np.float32)


    def get_pose_estimate(self):
        frame_bgr = self.system.get_camera_image("eye_in_hand", bgr=True)
        T_cam_slider = self.detect_slider_pose(frame_bgr)
        if T_cam_slider is None:
            return None
        arm_state = self.system.get_state()["arm"]
        return self.transform_to_world(T_cam_slider, arm_state)

    def detect_slider_pose(self, frame_bgr):
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

    def transform_to_world(self, T_cam_slider, arm_state):
        ee_pose_base  = self.system.ctrl["arm"].kin_model.forward_kinematics(arm_state.q)
        ee_pose_world = self.system.ctrl["arm"].transform_base_to_world_frame(ee_pose_base)
        T_world_ee    = make_T(ee_pose_world.rotation_matrix, ee_pose_world.position)
        T_world_slider = T_world_ee @ self.T_ee_cam @ T_cam_slider
        xy    = T_world_slider[:2, 3]
        theta = np.arctan2(T_world_slider[1, 0], T_world_slider[0, 0])
        return np.array([xy[0], xy[1], theta])


if __name__ == "__main__":
    from simcore import load_yaml
    config = load_yaml("configs/global_config.yaml")

    system = RobotSystem(config)
    system.set_controller_mode("arm", "position")
    system.set_target("arm", {"q":[0.1, -0.785, 0.5, -2.356, 0.0, 1.571, 0.785]})

    system.sim.start()
    system.running = True

    task_config = load_yaml(config.get("task_config"))
    estimator = PoseEstimation(task_config, system)

    try:
        while True:
            system.step()
            frame = system.get_camera_image("eye_in_hand", bgr=True)

            estimate = estimator.get_pose_estimate()
            slider   = system.get_object_states()["box"]

            if estimate is not None:
                gt_xy    = slider["pos"][:2]
                gt_theta = np.arctan2(
                    2 * (slider["quat"][0] * slider["quat"][3] + slider["quat"][1] * slider["quat"][2]),
                    1 - 2 * (slider["quat"][2]**2 + slider["quat"][3]**2)
                )
                err_xy    = estimate[:2] - gt_xy
                err_theta = estimate[2] - gt_theta
                print(f"est=({estimate[0]:+.4f},{estimate[1]:+.4f},{np.degrees(estimate[2]):+6.2f}deg)  "
                      f"gt=({gt_xy[0]:+.4f},{gt_xy[1]:+.4f},{np.degrees(gt_theta):+6.2f}deg)  "
                      f"err_xy=({err_xy[0]*1000:+.2f},{err_xy[1]*1000:+.2f})mm  "
                      f"err_theta={np.degrees(err_theta):+.2f}deg")
            else:
                print("no detection")

            corners, ids, _ = estimator.DETECTOR.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("eye_in_hand", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        system.stop()
        cv2.destroyAllWindows()