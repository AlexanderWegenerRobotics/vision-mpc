import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import time, csv
import cv2

def wxyz_to_xyzw(q):
    return q[[1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def quat_wxyz_from_tool_yaw(yaw):
    base_rot = Rotation.from_quat(wxyz_to_xyzw(np.array([0, 1, 0, 0])))
    yaw_rot  = Rotation.from_euler("z", yaw)
    quat_xyzw = (yaw_rot * base_rot).as_quat()
    return xyzw_to_wxyz(quat_xyzw)

def sample_scenario(config: dict, rng: np.random.Generator):
    # Reject-samples a start/goal pair within the workspace with a minimum
    # xy distance between them. Returns (start_xy, start_theta, goal_xy, goal_theta).
    sc     = config["scenario"]
    ws     = sc["workspace"]
    d_min  = float(sc["min_distance"])
    x_lim  = ws["x"]
    y_lim  = ws["y"]
    th_lim = ws["theta"]

    for _ in range(1000):
        start_xy    = rng.uniform([x_lim[0], y_lim[0]], [x_lim[1], y_lim[1]])
        start_theta = rng.uniform(th_lim[0], th_lim[1])
        goal_xy     = rng.uniform([x_lim[0], y_lim[0]], [x_lim[1], y_lim[1]])
        goal_theta  = rng.uniform(th_lim[0], th_lim[1])
        if np.linalg.norm(goal_xy - start_xy) >= d_min:
            return start_xy, start_theta, goal_xy, goal_theta
    raise RuntimeError(f"Could not sample start/goal with min_distance={d_min} after 1000 tries.")


def load_fixed_scenario(config: dict):
    sc = config["scenario"]
    return (np.array(sc["start"]["xy"], dtype=float),
            float(sc["start"]["theta"]),
            np.array(sc["goal"]["xy"],  dtype=float),
            float(sc["goal"]["theta"]))


def update_slider_frame(system, config, x_slider):
    return
    # Visualization: keeps the slider's coordinate-frame axes rendered at its current pose.
    slider_z  = config["surface_height_world"] + config["slider_half_z"]
    pos       = np.array([x_slider[0], x_slider[1], slider_z])
    quat_wxyz = xyzw_to_wxyz(Rotation.from_euler("z", x_slider[2]).as_quat())
    system.sim.reset_object_pose("slider_frame", pos, quat_wxyz)


def update_vel_arrow(system, ee_pose, ee_vel_world, z_contact):
    return
    # Visualization: renders the commanded EE velocity as an arrow in the scene.
    vel_2d = ee_vel_world[:2]
    speed  = np.linalg.norm(vel_2d)
    if speed < 1e-4:
        return

    half_len = np.clip(speed * 1.0, 0.02, 0.2)
    geom_id  = system.sim.mj_model.geom("vel_arrow/vel_arrow_shaft").id
    system.sim.mj_model.geom_size[geom_id] = [0.005, half_len, 0]

    pos    = ee_pose.position.copy()
    pos[2] = z_contact

    vel_dir = np.array([vel_2d[0], vel_2d[1], 0.0]) / speed
    z_axis  = np.array([0.0, 0.0, 1.0])
    axis    = np.cross(z_axis, vel_dir)
    angle   = np.arccos(np.clip(np.dot(z_axis, vel_dir), -1, 1))

    if np.linalg.norm(axis) < 1e-6:
        quat_xyzw = np.array([0, 0, 0, 1])
    else:
        axis      = axis / np.linalg.norm(axis)
        quat_xyzw = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])

    # Offset origin to the base of the arrow so it grows outward from the EE.
    pos += vel_dir * half_len
    system.sim.reset_object_pose("vel_arrow", pos, xyzw_to_wxyz(quat_xyzw))

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def marker_obj_points(marker_size):
    s = marker_size / 2
    return np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0],
    ], dtype=np.float32)

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


class EpisodeMetrics:
    """Accumulates per-step data during an episode and computes summary statistics."""

    def __init__(self, variant: str, start_xy, start_theta, goal_xy, goal_theta):
        self.variant       = variant
        self.start_xy      = np.array(start_xy)
        self.start_theta   = float(start_theta)
        self.goal_xy       = np.array(goal_xy)
        self.goal_theta    = float(goal_theta)
        self._t0           = time.time()
        self._xy_errors    = []
        self._solver_fails = 0
        self._n_solves     = 0

    def record_step(self, x_slider: np.ndarray, ref_xy: np.ndarray, solver_status: int):
        self._xy_errors.append(float(np.linalg.norm(x_slider[:2] - ref_xy)))
        self._n_solves     += 1
        self._solver_fails += int(solver_status != 0)

    def summarise(self, x_slider_final: np.ndarray, success: bool) -> dict:
        pos_err   = float(np.linalg.norm(x_slider_final[:2] - self.goal_xy))
        theta_err = float(abs((x_slider_final[2] - self.goal_theta + np.pi) % (2 * np.pi) - np.pi))
        errors    = np.array(self._xy_errors)
        return {
            "variant":           self.variant,
            "success":           int(success),
            "duration_s":        round(time.time() - self._t0, 3),
            "path_rms_mm":       round(float(np.sqrt(np.mean(errors**2))) * 1000, 3) if len(errors) else 0.0,
            "path_max_mm":       round(float(errors.max()) * 1000, 3) if len(errors) else 0.0,
            "final_pos_mm":      round(pos_err * 1000, 3),
            "final_theta_deg":   round(np.degrees(theta_err), 3),
            "solver_fail_rate":  round(self._solver_fails / max(self._n_solves, 1), 4),
            "start_x":           round(float(self.start_xy[0]), 4),
            "start_y":           round(float(self.start_xy[1]), 4),
            "start_theta":       round(self.start_theta, 4),
            "goal_x":            round(float(self.goal_xy[0]), 4),
            "goal_y":            round(float(self.goal_xy[1]), 4),
            "goal_theta":        round(self.goal_theta, 4),
        }


_CSV_FIELDS = [
    "variant", "success", "duration_s",
    "path_rms_mm", "path_max_mm",
    "final_pos_mm", "final_theta_deg",
    "solver_fail_rate",
    "start_x", "start_y", "start_theta",
    "goal_x",  "goal_y",  "goal_theta",
]


def append_result(row: dict, csv_path: str = "results.csv"):
    path   = Path(csv_path)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in _CSV_FIELDS})