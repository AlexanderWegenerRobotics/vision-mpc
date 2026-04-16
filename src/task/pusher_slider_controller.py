from enum import Enum, auto
import numpy as np
import time
from scipy.spatial.transform import Rotation

from src.task.trajectory import TrajectoryPlanner
from src.task.mpc import PusherSliderModel, PusherSliderNMPC
from src.task.utils import update_vel_arrow, update_slider_frame
from simcore.common.pose import Pose


class Phase(Enum):
    APPROACH = auto()
    PUSHING  = auto()
    DONE     = auto()
    FAILED   = auto()


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
        self.z_contact   = config["surface_height_world"] + config["pusher_clearance"]
        self.q_init      = config["q_init"]
        self.dt          = self.system.get_timestep()
        self.ee_quat_ref = np.array([0, 1, 0, 0])
        self.goal_pos    = np.array(config["goal"]["xy"], dtype=float)
        self.goal_theta  = config["goal"]["theta"]
        self._step_count = 0
        self._ee_ref_pos = None
        self._last_push_time = time.time()
        self._x_ref = None

        self._ps_model = PusherSliderModel(self.config)
        self._nmpc     = PusherSliderNMPC(self._ps_model, self.config)

    def loop(self):
        for _ in range(self.config.get("iterations", 1)):
            self.run()

    def run(self):
        self.reset()
        time.sleep(0.1)
        t0 = time.time()
        prev_phase = self.phase

        while self.running:
            if self.phase == Phase.APPROACH:
                self.phase = self._run_approach()
            elif self.phase == Phase.PUSHING:
                self.phase = self._run_pushing()

            if self.phase == Phase.PUSHING and prev_phase != Phase.PUSHING:
                self.system.set_controller_params(self.device_name, {"K_cart": [1000, 1000, 600, 160, 160, 160]})
            prev_phase = self.phase

            if self.phase in (Phase.DONE, Phase.FAILED):
                self.running = False
                print(f"Phase {self.phase.name} after {time.time() - t0:.2f}s")

            time.sleep(self.dt)

    def _run_approach(self):
        """Move end-effector to contact point on the -x_body face."""
        x_slider = self._get_slider_state()
        contact_world = self._get_contact_point_world(x_slider)
        ee_target = self._tip_to_ee(contact_world)

        p_start = self._get_ee_pose()
        z_mid = ee_target[2] + abs(ee_target[2] - p_start.position[2]) / 2
        p_mid = np.array([ee_target[0], ee_target[1], z_mid])

        self._move_to(p_start.position, p_mid, 0.2)
        self._move_to(p_mid, ee_target, 0.1)

        return Phase.PUSHING
    
    def _run_pushing(self):
        x_slider = self._get_slider_state()
        update_slider_frame(self.system, self.config, x_slider)

        ee_pose = self._get_ee_pose()

        p_y = np.clip(self._compute_py(x_slider), -self.config["slider_half_y"], self.config["slider_half_y"])
        x0  = np.array([x_slider[0], x_slider[1], x_slider[2], p_y])

        eff_goal_theta = PusherSliderNMPC.normalize_goal_theta(x_slider[2], self.goal_theta)
        x_goal = np.array([self.goal_pos[0], self.goal_pos[1], eff_goal_theta, 0.0])
        x_ref  = PusherSliderNMPC.make_linear_reference(x0, x_goal, self._nmpc.T)

        for ref_point in x_ref:
            self.system.set_trail("ee_trail", np.array([ref_point[0], ref_point[1], self.z_contact]))

        u_opt        = self._nmpc.solve(x0, x_ref)
        ee_vel_world = self._mpc_vel_to_world(u_opt, x_slider[2])
        update_vel_arrow(self.system, ee_pose, ee_vel_world, self.z_contact)

        if self._ee_ref_pos is None:
            self._ee_ref_pos     = ee_pose.position.copy()
            self._last_push_time = time.time()
        self._ee_ref_pos = self._ee_ref_pos + ee_vel_world * self.dt

        ref_pose = Pose(
            position=self._tip_to_ee(np.array([self._ee_ref_pos[0], self._ee_ref_pos[1], self.z_contact])),
            quaternion=self.ee_quat_ref
        )

        self.system.set_target(self.device_name, {
            "x": ref_pose,
            "xd": np.concatenate([ee_vel_world, np.zeros(3)])
        })

        pos_err   = np.linalg.norm(x_slider[:2] - self.goal_pos)
        theta_err = abs((x_slider[2] - self.goal_theta + np.pi) % (2 * np.pi) - np.pi)
        if pos_err < self.config["goal_pos_tol"] and theta_err < self.config["goal_theta_tol"]:
            return Phase.DONE
        return Phase.PUSHING

    def _compute_py(self, x_slider):
        """Compute p_y: tangential offset of pusher along -x_body face (y_body coord)."""
        tip_world = self._get_pusher_tip()
        d_world = tip_world[:2] - x_slider[:2]
        theta = x_slider[2]
        c, s = np.cos(theta), np.sin(theta)
        d_body_y = -s * d_world[0] + c * d_world[1]
        return float(d_body_y)

    def _mpc_vel_to_world(self, u, theta):
        """Convert MPC control [v_n, v_t] to world-frame EE velocity.

        v_n drives pusher along +x_body (inward normal of -x face).
        v_t drives pusher along +y_body (tangent along face).
        Pusher velocity in body frame = [v_n, v_t].
        """
        c, s = np.cos(theta), np.sin(theta)
        vx_world = c * u[0] - s * u[1]
        vy_world = s * u[0] + c * u[1]
        return np.array([vx_world, vy_world, 0.0])

    def _get_contact_point_world(self, x_slider):
        """Contact point on -x_body face in world coordinates."""
        half_x = self.config["slider_half_x"]
        r      = self.config["pusher_radius"]
        margin = self.config.get("pusher_standoff", 0.005)
        d = half_x + r + margin
        theta = x_slider[2]
        c, s = np.cos(theta), np.sin(theta)
        p2d = x_slider[:2] + np.array([-c * d, -s * d])
        return np.array([p2d[0], p2d[1], self.z_contact])

    def _get_slider_state(self):
        """Returns [x, y, theta] of slider in world frame."""
        slider = self.system.get_object_states()[self.slider_name]
        pos = slider["pos"][:2]
        quat_xyzw = slider["quat"][[1, 2, 3, 0]]
        theta = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

    def _get_ee_pose(self):
        arm_state = self.system.get_state()[self.device_name]
        return self.system.ctrl[self.device_name].get_ee_pose_world(arm_state)

    def _get_pusher_tip(self):
        ee_pose = self._get_ee_pose()
        R = Rotation.from_quat(ee_pose.quaternion[[1, 2, 3, 0]]).as_matrix()
        return ee_pose.position + R @ np.array([0.0, 0.0, self.pusher_length])

    def _tip_to_ee(self, tip_world):
        ee_pose = self._get_ee_pose()
        R = Rotation.from_quat(ee_pose.quaternion[[1, 2, 3, 0]]).as_matrix()
        return tip_world - R @ np.array([0.0, 0.0, self.pusher_length])

    def _move_to(self, p_start, p_end, max_speed):
        """Execute a straight-line trajectory segment."""
        self.trajectory.plan_with_speed(p_start, self.ee_quat_ref, p_end, self.ee_quat_ref, max_speed=max_speed)
        while not self.trajectory.is_done():
            step = self.trajectory.step(self.dt)
            pose = Pose(position=step["pos"], quaternion=step["quat"])
            vel  = np.concatenate([step["vel"], step["omega"]])
            self.system.set_target(self.device_name, {"x": pose, "xd": vel})
            time.sleep(self.dt)

    def reset(self):
        self.phase   = Phase.APPROACH
        self.running = True
        self._step_count = 0
        self._ee_ref_pos = None
        self._x_ref = None
        self._last_push_time = None
        self.system.clear_trail("ee_trail")

        start_xy    = np.array(self.config["slider_start"]["xy"], dtype=float)
        start_theta = self.config["slider_start"]["theta"]

        slider_z   = self.config["surface_height_world"] + self.config["slider_half_z"]
        slider_pos = np.array([start_xy[0], start_xy[1], slider_z])
        q_euler    = Rotation.from_euler("z", start_theta).as_quat()
        slider_quat = np.array([q_euler[3], *q_euler[:3]])

        self.system.sim.reset_device_state(self.device_name, self.q_init)
        self.system.sim.reset_object_pose(self.slider_name, slider_pos, slider_quat)

        goal_pos = np.array([self.goal_pos[0], self.goal_pos[1], slider_z])
        g_euler  = Rotation.from_euler("z", self.goal_theta).as_quat()
        goal_quat = np.array([g_euler[3], *g_euler[:3]])
        self.system.sim.reset_object_pose("target", goal_pos, goal_quat)

        ee_pose = self._get_ee_pose()
        self.system.set_controller_mode("arm", "dynamic_impedance")
        self.system.set_target("arm", {"x": Pose(position=ee_pose.position, quaternion=self.ee_quat_ref)})
        self.system.sim.forward()

        x_slider = self._get_slider_state()
        update_slider_frame(self.system, self.config, x_slider)
