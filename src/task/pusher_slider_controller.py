from enum import Enum, auto
import numpy as np
import time
from scipy.spatial.transform import Rotation

from src.task.trajectory import TrajectoryPlanner
from src.task.mpc import PusherSliderModel, PusherSliderNMPC
from simcore.common.pose import Pose


class Phase(Enum):
    APPROACH = auto()
    PUSHING  = auto()
    DONE     = auto()
    FAILED   = auto()

def rotation_body_to_world(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def rotation_world_to_body(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, s],
                     [-s, c]])


class PusherSliderController:
    def __init__(self, system=None, config=None):
        self.phase       = Phase.APPROACH
        self.config      = config
        self.system      = system
        self.running     = False
        self.trajectory  = TrajectoryPlanner()
        self.slider_name = config["slider_name"]
        self.device_name = config["arm_name"]
        self.pusher_length = config["pusher_length"]
        self.z_goal      = config["surface_height_world"] + config["pusher_clearance"]
        self.q_init      = config["q_init"]
        self.dt          = self.system.get_timestep()
        self.ee_quat_ref = np.array([0, 1, 0, 0])
        self._rng        = np.random.default_rng()
        self.goal_pos    = None
        self.goal_theta  = None

        # State set by run_approach, consumed by run_pushing
        self.contact_face = None

        # init mpc
        self._ps_model = PusherSliderModel(self.config)
        self._nmpc = PusherSliderNMPC(self._ps_model, self.config)

        self.body_normals  = np.array([[ 0, -1], [ 0,  1], [-1,  0], [ 1,  0]], dtype=float)
        self.body_tangents = np.array([[-1,  0], [ 1,  0], [ 0, -1], [ 0,  1]], dtype=float)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def loop(self):
        iter = self.config.get("iterations", 10)
        for _ in range(iter):
            self.run()

    def run(self):
        self.reset()
        time.sleep(0.1)

        start_time = time.time()
        while self.running:
            match self.phase:
                case Phase.APPROACH:
                    self.phase = self.run_approach()
                case Phase.PUSHING:
                    self.phase = self.run_pushing()

            if self.phase in (Phase.DONE, Phase.FAILED):
                self.running = False
                print(f"Stopping after {time.time() - start_time:.3f}s")

            time.sleep(self.dt)

    # ------------------------------------------------------------------ #
    #  Phase handlers                                                      #
    # ------------------------------------------------------------------ #

    def run_approach(self):
        x_slider = self._get_slider_state()
        self.contact_face = self._select_contact_face(x_slider, self.goal_pos)
        contact_point = self._get_contact_point_world(x_slider, self.contact_face)
        ee_pos_target = self._tip_target_to_ee_target(contact_point)

        p_start = self._get_ee_pose()
        z_1 = ee_pos_target[2] + np.abs(ee_pos_target[2] - p_start.position[2]) / 2
        p_mid = np.concatenate([ee_pos_target[:2], [z_1]])

        self._execute_segment(p_start.position, self.ee_quat_ref, p_mid, self.ee_quat_ref, 0.2)
        self._execute_segment(p_mid, self.ee_quat_ref, ee_pos_target, self.ee_quat_ref, 0.1)

        return Phase.PUSHING

    def run_pushing(self) -> Phase:
        x_slider = self._get_slider_state()
        py = self._compute_py(x_slider, self.contact_face)
        x0_mpc = np.array([x_slider[0], x_slider[1], x_slider[2], py])
        effective_goal_theta = PusherSliderNMPC._normalize_goal_theta(x_slider[2], self.goal_theta)
        x_goal_mpc = np.array([self.goal_pos[0], self.goal_pos[1], effective_goal_theta, 0.0])
        x_ref = PusherSliderNMPC.make_linear_reference(x0_mpc, x_goal_mpc, self._nmpc.T)
        u_opt = self._nmpc.solve(x0_mpc, x_ref)
        ee_vel_world = self._contact_vel_to_ee_vel(u_opt, x_slider, py, self.contact_face)

        ee_pose_cmd = self._get_ee_pose()
        ee_pose_cmd.quaternion = self.ee_quat_ref
        ee_pose_cmd.position = self._tip_target_to_ee_target(np.array([ee_pose_cmd.position[0], ee_pose_cmd.position[1], self.z_goal]))
        self.system.set_target(self.device_name,{"x": ee_pose_cmd, "xd": np.concatenate([ee_vel_world, np.zeros(3)])})
        pos_err   = np.linalg.norm(x_slider[:2] - self.goal_pos)
        theta_err = abs(x_slider[2] - self.goal_theta)
        if pos_err < self.config.get("goal_pos_tol", 0.01) and theta_err < self.config.get("goal_theta_tol", 0.1):
            return Phase.DONE

        return Phase.PUSHING

    # ------------------------------------------------------------------ #
    #  Helpers                                                         #
    # ------------------------------------------------------------------ #

    def _compute_py(self, x_slider: np.ndarray, face: int) -> float:
        tip_world = self._get_pusher_tip_world()
        d_world   = tip_world[:2] - x_slider[:2]
        theta     = x_slider[2]
        d_body = rotation_world_to_body(theta) @ d_world
        # Faces 0,1: normal along body-y, tangent is body-x -> p_y = d_body[0]
        # Faces 2,3: normal along body-x, tangent is body-y -> p_y = d_body[1]
        return float(d_body[0] if face in (0, 1) else d_body[1])

    def _contact_vel_to_ee_vel(self, u, x_slider, py, face):
        v_n, v_t = u[0], u[1]
        theta = x_slider[2]
        n_body, t_body = self.body_normals[face], self.body_tangents[face]

        x0 = np.array([x_slider[0], x_slider[1], theta, py])
        xdot = self._ps_model.evaluate(x0, u)  # [x^dot_s, y^dot_s, theta^dot, p^dot_y] in world frame
        slider_vel_world = xdot[:2]
        rel_vel_body = -v_n * n_body + v_t * t_body
        rel_vel_world = rotation_body_to_world(theta) @ rel_vel_body

        vel_world_2d = slider_vel_world + rel_vel_world
        return np.array([vel_world_2d[0], vel_world_2d[1], 0.0])

    def _get_slider_state(self) -> tuple[np.ndarray, np.ndarray]:
        slider  = self.system.get_object_states()[self.slider_name]
        pos_s   = slider["pos"][:2]
        quat_s  = slider["quat"][[1, 2, 3, 0]]
        theta_s = Rotation.from_quat(quat_s).as_euler("zyx")[0]
        return np.array([pos_s[0], pos_s[1], theta_s])

    def _select_contact_face(self, x_slider: np.ndarray, goal_pos: np.ndarray) -> int:
        slider_pos   = x_slider[:2]
        d_world = slider_pos - goal_pos
        d_body = rotation_world_to_body(x_slider[2]) @ d_world
        normals = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        return int(np.argmax(normals @ d_body))

    def _get_contact_point_world(self, x_slider: np.ndarray, face: int) -> np.ndarray:
        a      = self.config.get("slider_half_x")
        b      = self.config.get("slider_half_y")
        r      = self.config.get("pusher_radius")
        margin = self.config.get("pusher_standoff", 0.005)
        offsets_body = np.array([
            [ 0.0, -(b + r + margin)],
            [ 0.0,  (b + r + margin)],
            [-(a + r + margin),  0.0],
            [ (a + r + margin),  0.0],
        ])
        p2d = x_slider[:2] + rotation_body_to_world(x_slider[2]) @ offsets_body[face]
        return np.array([p2d[0], p2d[1], self.z_goal])

    def _execute_segment(self, p_start, q_start, p_end, q_end, max_speed) -> None:
        self.trajectory.plan_with_speed(p_start, q_start, p_end, q_end, max_speed=max_speed)
        while not self.trajectory.is_done():
            step = self.trajectory.step(self.dt)
            target_pose = Pose(position=step["pos"], quaternion=step["quat"])
            self.system.set_target(self.device_name, {"x": target_pose, "xd": np.concatenate([step["vel"], step["omega"]])})
            time.sleep(self.dt)

    def _get_pusher_tip_world(self) -> np.ndarray:
        ee_pose   = self._get_ee_pose()
        R         = Rotation.from_quat(ee_pose.quaternion[[1, 2, 3, 0]]).as_matrix()
        return ee_pose.position + R @ np.array([0.0, 0.0, self.pusher_length])
    
    def _get_ee_pose(self):
        arm_state = self.system.get_state()[self.device_name]
        return self.system.ctrl[self.device_name].get_ee_pose_world(arm_state)

    def _tip_target_to_ee_target(self, tip_target_world: np.ndarray) -> np.ndarray:
        arm_state = self.system.get_state()[self.device_name]
        ee_pose   = self.system.ctrl[self.device_name].get_ee_pose_world(arm_state)
        R         = Rotation.from_quat(ee_pose.quaternion[[1, 2, 3, 0]]).as_matrix()
        return tip_target_world - R @ np.array([0.0, 0.0, self.pusher_length])

    def reset(self):
        self.phase     = Phase.APPROACH
        self.running   = True
        self._sim_time = 0.0

        ws             = self.config["workspace"]
        x_lo, x_hi    = ws["x"]
        y_lo, y_hi    = ws["y"]
        d_min          = self.config["min_start_goal_dist"]

        slider_xy    = self._rng.uniform([x_lo, y_lo], [x_hi, y_hi])
        slider_theta = self._rng.uniform(*self.config["slider_init"]["theta_range"])

        for _ in range(1000):
            goal_xy = self._rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            if np.linalg.norm(goal_xy - slider_xy) >= d_min:
                break
        else:
            raise RuntimeError("Could not sample valid goal within workspace constraints.")

        goal_theta = self._rng.uniform(*self.config["goal"]["theta_range"])

        slider_z    = self.config["surface_height_world"] + self.config["slider_half_z"]
        slider_pos  = np.array([slider_xy[0], slider_xy[1], slider_z])
        slider_quat = Rotation.from_euler("z", slider_theta).as_quat()
        slider_quat = np.array([slider_quat[3], *slider_quat[:3]])

        self.goal_pos   = goal_xy
        self.goal_theta = goal_theta

        self.system.sim.reset_device_state(self.device_name, self.q_init)
        self.system.sim.reset_object_pose(self.slider_name, slider_pos, slider_quat)

        goal_pos      = np.array([goal_xy[0], goal_xy[1], slider_z])
        goal_quat_xyzw = Rotation.from_euler("z", goal_theta).as_quat()
        goal_quat_wxyz = np.array([goal_quat_xyzw[3], *goal_quat_xyzw[:3]])
        self.system.sim.reset_object_pose("target", goal_pos, goal_quat_wxyz)

        ee_pose = self._get_ee_pose()
        self.system.set_controller_mode("arm", "dynamic_impedance")
        self.system.set_target("arm", {"x": Pose(position=ee_pose.position, quaternion=self.ee_quat_ref)})
        self.system.sim.forward()

