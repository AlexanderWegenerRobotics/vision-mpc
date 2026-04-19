from enum import Enum, auto
import numpy as np
import time
from scipy.spatial.transform import Rotation

from src.task.trajectory import TrajectoryPlanner
from src.task.mpc import PusherSliderModel, PusherSliderNMPC, Face
from src.task.path_planner import StraightLinePlanner, choose_face
from src.task.utils import (update_vel_arrow, update_slider_frame, wxyz_to_xyzw, xyzw_to_wxyz, 
                            EpisodeMetrics, append_result, load_fixed_scenario, sample_scenario)
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

        self.start_xy    = np.array(config["scenario"]["start"]["xy"], dtype=float)
        self.start_theta = float(config["scenario"]["start"]["theta"])
        self.goal_xy     = np.array(config["scenario"]["goal"]["xy"], dtype=float)
        self.goal_theta  = float(config["scenario"]["goal"]["theta"])

        self._ps_model = PusherSliderModel(self.config)
        self._nmpc     = PusherSliderNMPC(self._ps_model, self.config)
        self._planner  = StraightLinePlanner(v_max=config["planner"]["v_max"], a_max=config["planner"]["a_max"])
        self._trajectory = TrajectoryPlanner()

        self.phase       = Phase.APPROACH
        self.running     = False
        self._path_ref   = None
        self._step_idx   = 0
        self._contact_face = None
        self._tip_ref_xy = None
        self._metrics      = None
        self._rng        = np.random.default_rng()
        
    def loop(self):
        for _ in range(self.config.get("iterations", 1)):
            self.run()

    def run(self):
        # One episode: reset -> approach -> push until DONE/FAILED.
        self.reset()
        time.sleep(0.1)
        t0 = time.time()

        while self.running:
            if self.phase == Phase.APPROACH:
                self.phase = self._run_approach()
            elif self.phase == Phase.PUSHING:
                self.phase = self._run_pushing()

            if (time.time() - t0) > self.timeout:
                print("[fail]: System stoped due to timeout")
                self.phase = Phase.FAILED

            if self.phase in (Phase.DONE, Phase.FAILED):
                self.running = False
                success = self.phase == Phase.DONE
                print(f"Phase {self.phase.name} after {time.time() - t0:.2f}s")

                row = self._metrics.summarise(self._get_slider_state(), success)
                append_result(row, csv_path=self.config.get("results_csv", "results.csv"))

            time.sleep(self.mpc_dt)

    def _run_approach(self):
        # Pick face, tell MPC about it, move EE above contact point, descend, then
        # seat the pusher inward along the face normal to remove the standoff gap
        # so the first MPC step is actually in contact with the slider.
        x_slider = self._get_slider_state()
        self._contact_face = choose_face(x_slider[:2], x_slider[2], self.goal_xy)
        self._nmpc.set_face(self._contact_face)

        contact_world = self._contact_point_world(x_slider, self._contact_face)
        ee_target     = self._tip_to_ee(contact_world)

        p_start = self._get_ee_pose().position
        z_mid   = ee_target[2] + abs(ee_target[2] - p_start[2]) / 2
        p_mid   = np.array([ee_target[0], ee_target[1], z_mid])

        self._move_to(p_start, p_mid,     max_speed=0.2)
        self._move_to(p_mid,   ee_target, max_speed=0.1)

        # Seat against the face: move inward along the face normal by (standoff + overshoot)
        # so the pusher lightly presses on the slider.
        seated_world = self._seated_contact_point_world(x_slider, self._contact_face)
        seated_ee    = self._tip_to_ee(seated_world)
        self._move_to(ee_target, seated_ee, max_speed=0.02)

        self._plan_path(x_slider)
        self._tip_ref_xy = self._get_pusher_tip()[:2].copy()
        self.system.set_controller_params(self.device_name, {"K_cart": [1000, 1000, 600, 160, 160, 160]})

        return Phase.PUSHING

    def _run_pushing(self):
        x_slider = self._get_slider_state()
        update_slider_frame(self.system, self.config, x_slider)

        p_y   = self._compute_py(x_slider, self._contact_face)
        p_y   = np.clip(p_y, -self.config["slider_half_y"], self.config["slider_half_y"])
        x0    = np.array([x_slider[0], x_slider[1], x_slider[2], p_y])

        ref_win = self._planner.window(self._path_ref, self._step_idx, self._nmpc.T)

        u_opt, status = self._nmpc.solve(x0, ref_win)
        if status != 0:
            print(f"[warn] solver status {status} at step {self._step_idx}")

        # Convert canonical (v_n, v_t) into world-frame pusher velocity.
        ee_vel_world = self._canonical_vel_to_world(u_opt, x_slider[2], self._contact_face)

        # Integrate the commanded velocity to a position target for the impedance controller.
        self._tip_ref_xy = self._tip_ref_xy + ee_vel_world[:2] * self.mpc_dt
        tip_ref_world   = np.array([self._tip_ref_xy[0], self._tip_ref_xy[1], self.z_contact])
        ee_ref_pose     = Pose(position=self._tip_to_ee(tip_ref_world), quaternion=self.ee_quat_ref)

        self.system.set_target(self.device_name, {
            "x":  ee_ref_pose,
            "xd": np.concatenate([ee_vel_world, np.zeros(3)]),
        })

        update_vel_arrow(self.system, self._get_ee_pose(), ee_vel_world, self.z_contact)
        self._step_idx += 1

        pos_err   = np.linalg.norm(x_slider[:2] - self.goal_xy)
        theta_err = abs((x_slider[2] - self.goal_theta + np.pi) % (2 * np.pi) - np.pi)
        if pos_err < self.config["goal_pos_tol"] and theta_err < self.config["goal_theta_tol"]:
            return Phase.DONE

        # Grace period: after the plan ends, the window holds at the goal reference.
        # Let the controller settle for up to `grace_steps` more steps before giving up.
        grace_steps = int(5.0 / self.mpc_dt)
        if self._step_idx >= len(self._path_ref) + grace_steps:
            print(f"[fail] plan ended {grace_steps} steps ago, pos_err={pos_err*1000:.1f}mm  theta_err={np.degrees(theta_err):.1f}deg")
            return Phase.FAILED
    
        if np.linalg.norm(x_slider[:2] - self._tip_ref_xy) > 0.2:
            print(f"[fail]: Pusher left slider")
            return Phase.FAILED

        return Phase.PUSHING

    def _plan_path(self, x_slider):
        # Plan the full reference once, visualize as a trail in the sim.
        # StraightLinePlanner ignores n_steps and sizes the plan from its own profile.
        start = np.array([x_slider[0], x_slider[1], x_slider[2], 0.0])
        goal  = np.array([self.goal_xy[0], self.goal_xy[1], self.goal_theta, 0.0])

        self._path_ref = self._planner.plan(start, goal, n_steps=0, dt=self.mpc_dt)
        self._step_idx = 0

        self.system.clear_trail("ee_trail")
        for ref_point in self._path_ref:
            self.system.set_trail("ee_trail", np.array([ref_point[0], ref_point[1], self.z_contact]))

    def _compute_py(self, x_slider, face):
        # p_y is position along canonical +y_S axis, expressed in real body frame.
        # Canonical +y_S = real +y rotated by +alpha. So p_y = component of d_body
        # along the rotated unit vector.
        tip_world = self._get_pusher_tip()
        d_world   = tip_world[:2] - x_slider[:2]
        c, s = np.cos(x_slider[2]), np.sin(x_slider[2])
        d_body = np.array([ c * d_world[0] + s * d_world[1],
                           -s * d_world[0] + c * d_world[1]])
        # Canonical +y_S in real body frame for each face:
        #   NEG_X: (0, +1)  -> p_y = +d_body[1]
        #   POS_Y: (+1, 0)  -> p_y = +d_body[0]
        #   POS_X: (0, -1)  -> p_y = -d_body[1]
        #   NEG_Y: (-1, 0)  -> p_y = -d_body[0]
        py_pick = {Face.NEG_X:  d_body[1],
                   Face.POS_Y:  d_body[0],
                   Face.POS_X: -d_body[1],
                   Face.NEG_Y: -d_body[0]}
        return float(py_pick[face])

    def _canonical_vel_to_world(self, u, theta, face):
        # MPC returns u = (v_n, v_t) in canonical frame where pusher is on -x_S_canonical.
        # Canonical axes = real body axes rotated by +alpha, where alpha is the
        # face angle (same signs as mpc.py _FACE_ANGLES). To get real body-frame
        # velocity from canonical-frame velocity, apply R(+alpha) to u.
        face_angle = {Face.NEG_X: 0.0,
                      Face.POS_Y: -np.pi / 2,
                      Face.POS_X:  np.pi,
                      Face.NEG_Y:  np.pi / 2}[face]
        ca, sa = np.cos(face_angle), np.sin(face_angle)
        u_body = np.array([ca * u[0] - sa * u[1],
                           sa * u[0] + ca * u[1]])
        c, s = np.cos(theta), np.sin(theta)
        v_world = np.array([c * u_body[0] - s * u_body[1],
                            s * u_body[0] + c * u_body[1]])
        return np.array([v_world[0], v_world[1], 0.0])

    def _contact_point_world(self, x_slider, face):
        # World-frame position where the pusher tip should sit to contact the chosen face.
        a = self.config["slider_half_x"]
        b = self.config["slider_half_y"]
        r = self.config["pusher_radius"]
        margin = self.config.get("pusher_standoff", 0.005)
        # Pusher sits outside the face's outward normal, offset by (half-width + radius + margin).
        offsets_body = {Face.NEG_X: np.array([-(a + r + margin),  0.0]),
                        Face.POS_Y: np.array([ 0.0,  (b + r + margin)]),
                        Face.POS_X: np.array([ (a + r + margin),  0.0]),
                        Face.NEG_Y: np.array([ 0.0, -(b + r + margin)])}[face]
        c, s = np.cos(x_slider[2]), np.sin(x_slider[2])
        R = np.array([[c, -s], [s, c]])
        p2d = x_slider[:2] + R @ offsets_body
        return np.array([p2d[0], p2d[1], self.z_contact])

    def _seated_contact_point_world(self, x_slider, face):
        # Where the pusher tip should end up to be in light contact: half-width + radius
        # minus a small overshoot so we intentionally press into the slider face.
        a = self.config["slider_half_x"]
        b = self.config["slider_half_y"]
        r = self.config["pusher_radius"]
        overshoot = self.config.get("pusher_seat_depth", 0.02)
        offsets_body = {Face.NEG_X: np.array([-(a + r - overshoot),  0.0]),
                        Face.POS_Y: np.array([ 0.0,  (b + r - overshoot)]),
                        Face.POS_X: np.array([ (a + r - overshoot),  0.0]),
                        Face.NEG_Y: np.array([ 0.0, -(b + r - overshoot)])}[face]
        c, s = np.cos(x_slider[2]), np.sin(x_slider[2])
        R = np.array([[c, -s], [s, c]])
        p2d = x_slider[:2] + R @ offsets_body
        return np.array([p2d[0], p2d[1], self.z_contact])

    def _get_slider_state(self):
        slider    = self.system.get_object_states()[self.slider_name]
        pos       = slider["pos"][:2]
        quat_xyzw = wxyz_to_xyzw(slider["quat"])
        theta     = Rotation.from_quat(quat_xyzw).as_euler("zyx")[0]
        return np.array([pos[0], pos[1], theta])

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
        self._nmpc.reset()
        self.system.clear_trail("ee_trail")
        update_vel_arrow(self.system, Pose(position=np.zeros(3), quaternion=np.array([0,0,0,1])), np.ones(3), 0.0)

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

        goal_pos  = np.array([self.goal_xy[0], self.goal_xy[1], slider_z])
        goal_quat = xyzw_to_wxyz(Rotation.from_euler("z", self.goal_theta).as_quat())

        self.system.sim.reset_device_state(self.device_name, self.q_init)
        self.system.sim.reset_object_pose(self.slider_name, slider_pos, slider_quat)
        self.system.sim.reset_object_pose("target", goal_pos, goal_quat)

        ee_pose = self._get_ee_pose()
        self.system.set_controller_mode(self.device_name, "dynamic_impedance")
        self.system.set_target(self.device_name, {"x": Pose(position=ee_pose.position, quaternion=self.ee_quat_ref)})
        self.system.sim.forward()

        update_slider_frame(self.system, self.config, self._get_slider_state())