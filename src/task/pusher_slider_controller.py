from enum import Enum, auto
import numpy as np
import time
from scipy.spatial.transform import Rotation

from src.task.trajectory import TrajectoryPlanner
from simcore.common.pose import Pose

class Phase(Enum):
    IDLE = auto()
    REACHING = auto()
    PUSHING = auto()
    DONE = auto()
    FAILED = auto()


class PusherSliderController:
    def __init__(self, system=None, config=None):
        self.phase = Phase.IDLE
        self.config = config
        self.system = system
        self.running = False
        self.trajectory = TrajectoryPlanner()
        self.slider_name = config.get("slider_name")
        self.arm_name = config.get("arm_name")
        self.pusher_length = config.get("pusher_length")
        self.z_goal = config["surface_height_world"] + config["pusher_clearance"]
        self.dt = self.system.get_timestep()


    def run(self):
        self.running = True
        #self.system.set_controller_mode("arm", "dynamic_impedance")
        target = Pose(position=[0.46, -0.07, 0.62], quaternion=[0, 1, 0, 0])
        #self.system.set_target("arm", {"x": target})
        time.sleep(0.2)

        x_slider, x_pusher = self._get_mpc_state()
        goal_pos = np.array([0.75, 0.2])
        face = self._select_contact_face(x_slider, goal_pos)
        contact_point = self._get_contact_point_world(x_slider, face)
        ee_pos_target = self._tip_target_to_ee_target(contact_point)

        target = Pose(position=ee_pos_target.tolist(), quaternion=[0, 1, 0, 0])
        #self.system.set_target("arm", {"x": target})

        start_time = time.time()
        while(self.running):
            match self.phase:
                case Phase.IDLE:
                    pass
                case Phase.REACHING:
                    pass
                case Phase.PUSHING:
                    pass
            
            if self.phase == Phase.DONE or self.phase == Phase.FAILED:
                self.running = False
                print(f"Stopping system after {time.time() - start_time:.3f} seconds")

            self._tick()

    def _get_mpc_state(self) -> tuple[np.ndarray, np.ndarray]:
        slider = self.system.get_object_states()[self.slider_name]
        pos_s  = slider["pos"][:2]
        quat_s = slider["quat"][[1,2,3,0]]  # convert wxyz to xyzw for SciPy
        theta_s = Rotation.from_quat(quat_s).as_euler("xyz")[2]  # yaw

        x_slider = np.array([pos_s[0], pos_s[1], theta_s])

        arm_state = self.system.get_state()[self.arm_name]
        ee_pose   = self.system.ctrl[self.arm_name].get_ee_pose_world(arm_state)
        x_pusher  = ee_pose.position[:2]                         # x, y only

        return x_slider, x_pusher
    

    def _select_contact_face(self, x_slider: np.ndarray, goal_pos: np.ndarray) -> int:
        slider_pos = x_slider[:2]
        slider_theta = x_slider[2]
        d_world = goal_pos - slider_pos
        R = np.array([[np.cos(slider_theta),  np.sin(slider_theta)], [-np.sin(slider_theta), np.cos(slider_theta)]])
        d_body = R @ d_world
        normals = np.array([[0, -1], [0,  1], [-1, 0], [1,  0],])
        dots = normals @ d_body
        return int(np.argmax(dots))
    
    def _get_contact_point_world(self, x_slider: np.ndarray, face: int) -> np.ndarray:
        """Returns 3D world-frame contact point on slider surface."""
        a, b = self.config.get("slider_half_x", 0.025), self.config.get("slider_half_y", 0.025)
        r_pusher = self.config.get("pusher_radius", 0.005)
        offsets_body = np.array([
            [ 0.0, -(b + r_pusher)],  # 0: bottom (-y)
            [ 0.0,  (b + r_pusher)],  # 1: top    (+y)
            [-(a + r_pusher),  0.0],  # 2: left   (-x)
            [ (a + r_pusher),  0.0],  # 3: right  (+x)
        ])

        theta = x_slider[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

        p_contact_world_2d = x_slider[:2] + R @ offsets_body[face]
        return np.array([p_contact_world_2d[0], p_contact_world_2d[1], self.z_goal])
    
    def _execute_segment(self, p_start, q_start, p_end, q_end, max_speed) -> None:
        self.trajectory.plan_with_speed(p_start, q_start, p_end, q_end, max_speed=max_speed)
        while not self.trajectory.is_done():
            step = self.trajectory.step(self.dt)
            target_pose = Pose(position=step["pos"], quaternion=step["quat"])
            self.system.set_target(self.device_name, {"x": target_pose, "xd": np.concatenate([step["vel"], step["omega"]])})
            self._tick()

    def _get_pusher_tip_world(self) -> np.ndarray:
        arm_state = self.system.get_state()[self.arm_name]
        ee_pose   = self.system.ctrl[self.arm_name].get_ee_pose_world(arm_state)
        quat_xyzw = ee_pose.quaternion[[1,2,3,0]]  # wxyz -> xyzw
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        offset_local = np.array([0.0, 0.0, self.pusher_length])
        tip_world = ee_pose.position + R @ offset_local
        return tip_world
    
    def _tip_target_to_ee_target(self, tip_target_world: np.ndarray) -> np.ndarray:
        arm_state = self.system.get_state()[self.arm_name]
        ee_pose = self.system.ctrl[self.arm_name].get_ee_pose_world(arm_state)
        R = Rotation.from_quat(ee_pose.quaternion[[1,2,3,0]]).as_matrix()
        return tip_target_world - R @ np.array([0.0, 0.0, self.pusher_length])

    def _tick(self):
        if self.headless:
            self.system.step()
        else:
            time.sleep(self.dt)
        self._sim_time += self.dt