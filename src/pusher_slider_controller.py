from enum import Enum, auto
import numpy as np
import time
from scipy.spatial.transform import Rotation

from src.trajectory import TrajectoryPlanner
from src.mpc import PusherSliderModel, PusherSliderNMPC, Face, ControllerVariant
from src.path_planner import StraightLinePlanner, choose_face
from src.slider_observer import SliderObserver
from src.logger import EpisodeLogger
from src.utils import (update_vel_arrow, update_slider_frame, wxyz_to_xyzw, xyzw_to_wxyz,
                       EpisodeMetrics, append_result, load_fixed_scenario, sample_scenario,
                       quat_wxyz_from_tool_yaw, wrap_to_pi)
from simcore.common.pose import Pose


class Phase(Enum):
    APPROACH = auto()
    PUSHING  = auto()
    DONE     = auto()
    FAILED   = auto()


class PusherSliderController:
    def __init__(self, system=None, config=None):
        self.config        = config
        self.system        = system
        self.slider_name   = config["slider_name"]
        self.device_name   = config["arm_name"]
        self.pusher_length = config["pusher_length"]
        self.z_contact     = config["surface_height_world"] + config["pusher_clearance"]
        self.q_init        = config["q_init"]
        self.ee_quat_ref   = np.array([0, 1, 0, 0])
        self.sim_dt        = self.system.get_timestep()
        self.mpc_dt        = config["mpc"]["dt"]
        self.timeout       = config["timeout"]
        self.variant       = ControllerVariant[config["mpc"]["variant"]]

        self.start_xy    = np.array(config["scenario"]["start"]["xy"], dtype=float)
        self.start_theta = float(config["scenario"]["start"]["theta"])
        self.goal_xy     = np.array(config["scenario"]["goal"]["xy"], dtype=float)
        self.goal_theta  = float(config["scenario"]["goal"]["theta"])

        self._ps_model   = PusherSliderModel(self.config)
        self._nmpc       = PusherSliderNMPC(self._ps_model, self.config)
        self._planner    = StraightLinePlanner(v_max=config["planner"]["v_max"], a_max=config["planner"]["a_max"])
        self._trajectory = TrajectoryPlanner()
        self._observer   = SliderObserver(self.config, self.system, self._ps_model)
        self._logger     = EpisodeLogger(self.config)

        self.phase        = Phase.APPROACH
        self.running      = False
        self._path_ref    = None
        self._step_idx    = 0
        self._episode_idx = 0
        self._contact_face = None
        self._tip_ref_xy  = None
        self._last_obs_x  = None
        self._metrics     = None
        self._rng         = np.random.default_rng()

        self._ee_yaw_vis = None

    def loop(self):
        self._observer.start()
        try:
            for _ in range(self.config.get("iterations", 1)):
                self.run()
        finally:
            self._observer.stop()
            print("[observer] vision thread stopped")

    def run(self):
        self.reset()
        time.sleep(0.1)
        t0 = time.time()

        while self.running:
            if self.phase == Phase.APPROACH:
                self.phase = self._run_approach()
            elif self.phase == Phase.PUSHING:
                self.phase = self._run_pushing()

            if (time.time() - t0) > self.timeout:
                print("[fail]: System stopped due to timeout")
                self.phase = Phase.FAILED

            if self.phase in (Phase.DONE, Phase.FAILED):
                self.running = False
                success = self.phase == Phase.DONE
                print(f"Phase {self.phase.name} after {time.time() - t0:.2f}s")

                x_slider = self._observer.get_gt_state()
                row = self._metrics.summarise(x_slider, success)
                append_result(row, csv_path=self.config.get("results_csv", "results.csv"))
                self._logger.save(success)
                self._episode_idx += 1

            time.sleep(self.mpc_dt)

    def _run_approach(self):
        x_slider = self._observer.get_state()
        self._contact_face = choose_face(x_slider[:2], x_slider[2], self.goal_xy)
        self._nmpc.set_face(self._contact_face)
        self.ee_quat_ref = self._compute_visibility_quat(x_slider, self._contact_face)

        contact_world = self._contact_point_world(x_slider, self._contact_face)
        ee_target     = self._tip_to_ee(contact_world)

        p_start = self._get_ee_pose().position
        z_mid   = ee_target[2] + abs(ee_target[2] - p_start[2]) / 2
        p_mid   = np.array([ee_target[0], ee_target[1], z_mid])

        self._move_to(p_start, p_mid,     max_speed=0.2)
        self._move_to(p_mid,   ee_target, max_speed=0.1)

        x_slider     = self._observer.get_state()
        seated_world = self._seated_contact_point_world(x_slider, self._contact_face)
        seated_ee    = self._tip_to_ee(seated_world)
        self._move_to(ee_target, seated_ee, max_speed=0.02)

        x_slider = self._observer.get_state()
        self.ee_quat_ref = self._compute_visibility_quat(x_slider, self._contact_face)

        self._plan_path(x_slider)
        self._tip_ref_xy = self._get_pusher_tip()[:2].copy()
        self.system.set_controller_params(self.device_name, {"K_cart": [1000, 1000, 600, 160, 160, 160]})

        ref_win  = self._planner.window(self._path_ref, 0, self._nmpc.T)
        self._nmpc.reset(x0_world=x_slider, ref_world=ref_win)

        return Phase.PUSHING

    def _run_pushing(self):
        x_slider = self._observer.get_state()
        update_slider_frame(self.system, self.config, x_slider)
        self.ee_quat_ref = self._compute_visibility_quat(x_slider, self._contact_face)

        p_y = self._compute_py(x_slider, self._contact_face)
        p_y = np.clip(p_y, -self.config["slider_half_y"], self.config["slider_half_y"])
        x0  = np.array([x_slider[0], x_slider[1], x_slider[2], p_y])

        ref_win = self._planner.window(self._path_ref, self._step_idx, self._nmpc.T)

        t_solve       = time.perf_counter()
        u_opt, status = self._nmpc.solve(x0, ref_win)
        solve_time_ms = (time.perf_counter() - t_solve) * 1000.0
        self._observer.set_control(u_opt)
        self._observer.set_py(p_y)

        if status != 0:
            print(f"[warn] solver status {status} at step {self._step_idx}")

        ee_vel_world = self._canonical_vel_to_world(u_opt, x_slider[2], self._contact_face)

        self._tip_ref_xy = self._tip_ref_xy + ee_vel_world[:2] * self.mpc_dt
        tip_ref_world    = np.array([self._tip_ref_xy[0], self._tip_ref_xy[1], self.z_contact])
        ee_ref_pose      = Pose(position=self._tip_to_ee(tip_ref_world), quaternion=self.ee_quat_ref)

        self.system.set_target(self.device_name, {
            "x":  ee_ref_pose,
            "xd": np.concatenate([ee_vel_world, np.zeros(3)]),
        })

        ee_pose    = self._get_ee_pose()
        pusher_tip = self._get_pusher_tip()
        update_vel_arrow(self.system, ee_pose, ee_vel_world, self.z_contact)

        gt_state = self._observer.get_gt_state()
        est_state = self._observer.get_est_state()
        obs_cov = self._observer.get_covariance()

        self._logger.record(
            gt_state      = gt_state,
            obs_state     = x_slider,
            vis_state     = est_state,
            obs_cov       = obs_cov,
            control       = u_opt,
            ref_state     = ref_win[0],
            p_y           = p_y,
            solver_status = status,
            solve_time_ms = solve_time_ms,
            ee_pos        = ee_pose.position,
            ee_vel        = ee_vel_world[:2],
            pusher_tip    = pusher_tip[:2],
        )

        self._step_idx += 1

        pos_err   = np.linalg.norm(x_slider[:2] - self.goal_xy)
        theta_err = abs((x_slider[2] - self.goal_theta + np.pi) % (2 * np.pi) - np.pi)
        if pos_err < self.config["goal_pos_tol"] and theta_err < self.config["goal_theta_tol"]:
            return Phase.DONE

        grace_steps = int(5.0 / self.mpc_dt)
        if self._step_idx >= len(self._path_ref) + grace_steps:
            print(f"[fail] plan ended {grace_steps} steps ago, pos_err={pos_err*1000:.1f}mm  theta_err={np.degrees(theta_err):.1f}deg")
            return Phase.FAILED

        if np.linalg.norm(x_slider[:2] - self._tip_ref_xy) > 0.2:
            print("[fail]: Pusher left slider")
            return Phase.FAILED

        return Phase.PUSHING

    def _plan_path(self, x_slider):
        start = np.array([x_slider[0], x_slider[1], x_slider[2], 0.0])
        goal  = np.array([self.goal_xy[0], self.goal_xy[1], self.goal_theta, 0.0])

        self._path_ref = self._planner.plan(start, goal, n_steps=0, dt=self.mpc_dt)
        self._step_idx = 0

        self.system.clear_trail("ee_trail")
        for ref_point in self._path_ref:
            self.system.set_trail("ee_trail", np.array([ref_point[0], ref_point[1], self.z_contact]))

    def _compute_py(self, x_slider, face):
        tip_world = self._get_pusher_tip()
        d_world   = tip_world[:2] - x_slider[:2]
        c, s      = np.cos(x_slider[2]), np.sin(x_slider[2])
        d_body    = np.array([c * d_world[0] + s * d_world[1], -s * d_world[0] + c * d_world[1]])
        return float({Face.NEG_X:  d_body[1],
                      Face.POS_Y:  d_body[0],
                      Face.POS_X: -d_body[1],
                      Face.NEG_Y: -d_body[0]}[face])

    def _canonical_vel_to_world(self, u, theta, face):
        face_angle = {Face.NEG_X: 0.0, Face.POS_Y: -np.pi / 2, Face.POS_X: np.pi, Face.NEG_Y: np.pi / 2}[face]
        ca, sa = np.cos(face_angle), np.sin(face_angle)
        u_body = np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]])
        c, s   = np.cos(theta), np.sin(theta)
        return np.array([c * u_body[0] - s * u_body[1], s * u_body[0] + c * u_body[1], 0.0])

    def _contact_point_world(self, x_slider, face):
        a, b   = self.config["slider_half_x"], self.config["slider_half_y"]
        r      = self.config["pusher_radius"]
        margin = self.config.get("pusher_standoff", 0.005)
        offset = {Face.NEG_X: np.array([-(a + r + margin), 0.0]),
                  Face.POS_Y: np.array([0.0,  (b + r + margin)]),
                  Face.POS_X: np.array([ (a + r + margin), 0.0]),
                  Face.NEG_Y: np.array([0.0, -(b + r + margin)])}[face]
        c, s = np.cos(x_slider[2]), np.sin(x_slider[2])
        p2d  = x_slider[:2] + np.array([[c, -s], [s, c]]) @ offset
        return np.array([p2d[0], p2d[1], self.z_contact])

    def _seated_contact_point_world(self, x_slider, face):
        a, b      = self.config["slider_half_x"], self.config["slider_half_y"]
        r         = self.config["pusher_radius"]
        overshoot = self.config.get("pusher_seat_depth", 0.02)
        offset = {Face.NEG_X: np.array([-(a + r - overshoot), 0.0]),
                  Face.POS_Y: np.array([0.0,  (b + r - overshoot)]),
                  Face.POS_X: np.array([ (a + r - overshoot), 0.0]),
                  Face.NEG_Y: np.array([0.0, -(b + r - overshoot)])}[face]
        c, s = np.cos(x_slider[2]), np.sin(x_slider[2])
        p2d  = x_slider[:2] + np.array([[c, -s], [s, c]]) @ offset
        return np.array([p2d[0], p2d[1], self.z_contact])

    def _get_ee_pose(self):
        arm_state = self.system.get_state()[self.device_name]
        return self.system.ctrl[self.device_name].get_ee_pose_world(arm_state)

    def _get_pusher_tip(self):
        ee_pose = self._get_ee_pose()
        R = Rotation.from_quat(wxyz_to_xyzw(ee_pose.quaternion)).as_matrix()
        return ee_pose.position + R @ np.array([0.0, 0.0, self.pusher_length])

    def _tip_to_ee(self, tip_world):
        ee_pose = self._get_ee_pose()
        R = Rotation.from_quat(wxyz_to_xyzw(ee_pose.quaternion)).as_matrix()
        return tip_world - R @ np.array([0.0, 0.0, self.pusher_length])

    def _move_to(self, p_start, p_end, max_speed):
        self._trajectory.plan_with_speed(p_start, self.ee_quat_ref, p_end, self.ee_quat_ref, max_speed=max_speed)
        while not self._trajectory.is_done():
            step = self._trajectory.step(self.sim_dt)
            pose = Pose(position=step["pos"], quaternion=step["quat"])
            vel  = np.concatenate([step["vel"], step["omega"]])
            self.system.set_target(self.device_name, {"x": pose, "xd": vel})
            time.sleep(self.sim_dt)

    def reset(self):
        self.phase         = Phase.APPROACH
        self.running       = True
        self._path_ref     = None
        self._step_idx     = 0
        self._contact_face = None
        self._tip_ref_xy   = None
        self._last_obs_x   = None
        self._ee_yaw_vis = None
        self._nmpc.reset()
        self.system.clear_trail("ee_trail")
        update_vel_arrow(self.system, Pose(position=np.zeros(3), quaternion=np.array([0, 0, 0, 1])), np.ones(3), 0.0)

        if self.config["scenario"].get("mode", "fixed") == "random":
            self.start_xy, self.start_theta, self.goal_xy, self.goal_theta = \
                sample_scenario(self.config, self._rng)
        else:
            self.start_xy, self.start_theta, self.goal_xy, self.goal_theta = \
                load_fixed_scenario(self.config)

        self._metrics = EpisodeMetrics(
            variant=self.config["mpc"]["variant"],
            start_xy=self.start_xy, start_theta=self.start_theta,
            goal_xy=self.goal_xy,   goal_theta=self.goal_theta,
        )

        slider_z    = self.config["surface_height_world"] + self.config["slider_half_z"]
        slider_pos  = np.array([self.start_xy[0], self.start_xy[1], slider_z])
        slider_quat = xyzw_to_wxyz(Rotation.from_euler("z", self.start_theta).as_quat())
        goal_pos    = np.array([self.goal_xy[0], self.goal_xy[1], slider_z])
        goal_quat   = xyzw_to_wxyz(Rotation.from_euler("z", self.goal_theta).as_quat())

        self.system.sim.reset_device_state(self.device_name, self.q_init)
        self.system.sim.reset_object_pose(self.slider_name, slider_pos, slider_quat)
        self.system.sim.reset_object_pose("target", goal_pos, goal_quat)

        ee_pose = self._get_ee_pose()
        self.system.set_controller_mode(self.device_name, "dynamic_impedance")
        self.system.set_target(self.device_name, {"x": Pose(position=ee_pose.position, quaternion=self.ee_quat_ref)})
        self.system.sim.forward()

        x_slider = self._observer.get_state()
        update_slider_frame(self.system, self.config, x_slider)

        x0_ekf = np.array([self.start_xy[0], self.start_xy[1], self.start_theta])
        self._observer.reset(x0=x0_ekf)

        self._logger.reset(
            episode_idx = self._episode_idx,
            start_xy    = self.start_xy,
            start_theta = self.start_theta,
            goal_xy     = self.goal_xy,
            goal_theta  = self.goal_theta,
            face        = str(self._contact_face),
        )

    def _compute_visibility_quat(self, x_slider, face):
        yaw_cam_offset = np.deg2rad(-45.0)
        yaw_bias       = np.deg2rad(0.0)
        yaw_alpha      = 0.08
        yaw_rate_max   = np.deg2rad(25.0)
        yaw_nominal    = 0.0
        yaw_limit      = np.deg2rad(65.0)

        theta = x_slider[2]

        face_normal_body = {
            Face.NEG_X: np.array([-1.0,  0.0]),
            Face.POS_Y: np.array([ 0.0,  1.0]),
            Face.POS_X: np.array([ 1.0,  0.0]),
            Face.NEG_Y: np.array([ 0.0, -1.0]),
        }[face]

        c, s = np.cos(theta), np.sin(theta)
        R_ws = np.array([[c, -s],
                        [s,  c]])
        n_world = R_ws @ face_normal_body
        yaw_face_normal = np.arctan2(n_world[1], n_world[0])

        yaw_des = wrap_to_pi(yaw_face_normal + yaw_cam_offset + yaw_bias + np.pi)
        yaw_des = np.clip(wrap_to_pi(yaw_des - yaw_nominal), -yaw_limit, yaw_limit) + yaw_nominal
        yaw_des = wrap_to_pi(yaw_des)

        if self._ee_yaw_vis is None:
            self._ee_yaw_vis = yaw_des
        else:
            yaw_err = wrap_to_pi(yaw_des - self._ee_yaw_vis)
            yaw_step_max = yaw_rate_max * self.sim_dt
            yaw_step = np.clip(yaw_alpha * yaw_err, -yaw_step_max, yaw_step_max)
            self._ee_yaw_vis = wrap_to_pi(self._ee_yaw_vis + yaw_step)

        return quat_wxyz_from_tool_yaw(self._ee_yaw_vis)
