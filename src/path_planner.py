import numpy as np
from abc import ABC, abstractmethod
from src.mpc import Face

class PathPlanner(ABC):
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        ...

    def window(self, full_ref: np.ndarray, step: int, horizon: int) -> np.ndarray:
        n = len(full_ref)
        if n == 0:
            raise ValueError("full_ref is empty")
        step = min(step, n - 1)
        end  = min(step + horizon + 1, n)
        win  = full_ref[step:end]
        if len(win) < horizon + 1:
            pad = np.tile(win[-1], (horizon + 1 - len(win), 1))
            win = np.vstack([win, pad])
        return win
    
    def nearest_idx(self, full_ref: np.ndarray, x_slider: np.ndarray) -> int:
        dists = np.linalg.norm(full_ref[:, :2] - x_slider[:2], axis=1)
        return int(np.argmin(dists))


class StraightLinePlanner(PathPlanner):
    def __init__(self, v_max: float = 0.05, a_max: float = 0.10, symmetry: float = 2 * np.pi, theta_delay_frac: float = 0.0):
        self.v_max            = float(v_max)
        self.a_max            = float(a_max)
        self.symmetry         = float(symmetry)
        self.theta_delay_frac = float(np.clip(theta_delay_frac, 0.0, 1.0))

    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        start_xy, goal_xy = start[:2], goal[:2]
        start_th, goal_th = start[2], self._nearest_goal_theta(start[2], goal[2], self.symmetry)

        distance = float(np.linalg.norm(goal_xy - start_xy))
        if distance < 1e-9:
            ref = np.tile(np.array([start_xy[0], start_xy[1], start_th, 0.0]), (2, 1))
            return ref

        s_profile = self._trapezoidal_progress(distance, dt)

        theta_s = np.where(
            s_profile < self.theta_delay_frac,
            0.0,
            (s_profile - self.theta_delay_frac) / (1.0 - self.theta_delay_frac + 1e-9)
        )
        theta_s = np.clip(theta_s, 0.0, 1.0)

        ref = np.zeros((len(s_profile), 4))
        ref[:, 0] = start_xy[0] + s_profile * (goal_xy[0] - start_xy[0])
        ref[:, 1] = start_xy[1] + s_profile * (goal_xy[1] - start_xy[1])
        ref[:, 2] = start_th   + theta_s   * (goal_th    - start_th)
        ref[:, 3] = 0.0
        return ref

    def _trapezoidal_progress(self, distance: float, dt: float) -> np.ndarray:
        v_max, a_max = self.v_max, self.a_max
        d_accel      = v_max**2 / (2 * a_max)

        if 2 * d_accel <= distance:
            # trapezoidal: accel / cruise / decel
            t_accel  = v_max / a_max
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / v_max
            t_total  = 2 * t_accel + t_cruise
        else:
            # triangular: accel up to v_peak < v_max, immediately decelerate
            v_peak   = np.sqrt(a_max * distance)
            t_accel  = v_peak / a_max
            t_cruise = 0.0
            t_total  = 2 * t_accel

        n_steps = max(int(np.ceil(t_total / dt)), 2)
        t       = np.linspace(0.0, t_total, n_steps + 1)
        d_of_t  = np.zeros_like(t)

        if t_cruise > 0:
            accel_mask  = t <= t_accel
            cruise_mask = (t > t_accel) & (t <= t_accel + t_cruise)
            decel_mask  = t > t_accel + t_cruise
            d_of_t[accel_mask]  = 0.5 * a_max * t[accel_mask]**2
            d_of_t[cruise_mask] = d_accel + v_max * (t[cruise_mask] - t_accel)
            t_d = t[decel_mask] - (t_accel + t_cruise)
            d_of_t[decel_mask]  = distance - 0.5 * a_max * (t_accel - t_d)**2
        else:
            v_peak     = np.sqrt(a_max * distance)
            accel_mask = t <= t_accel
            decel_mask = t > t_accel
            d_of_t[accel_mask] = 0.5 * a_max * t[accel_mask]**2
            t_d = t[decel_mask] - t_accel
            d_of_t[decel_mask] = distance - 0.5 * a_max * (t_accel - t_d)**2

        d_of_t = np.clip(d_of_t, 0.0, distance)
        return d_of_t / distance

    @staticmethod
    def _nearest_goal_theta(start_th: float, goal_th: float, symmetry: float) -> float:
        if symmetry >= 2 * np.pi - 1e-9:
            diff = (goal_th - start_th + np.pi) % (2 * np.pi) - np.pi
            return start_th + diff

        g_canon    = goal_th - symmetry * np.floor(goal_th / symmetry)
        n_wraps    = int(np.ceil(2 * np.pi / symmetry)) + 2
        base       = start_th - np.pi
        k0         = int(np.floor((base - g_canon) / symmetry))
        candidates = [g_canon + (k0 + k) * symmetry for k in range(n_wraps)]
        diffs      = [abs(c - start_th) for c in candidates]
        return candidates[int(np.argmin(diffs))]


class CircularPlanner(PathPlanner):

    def __init__(self, center: np.ndarray, radius: float, direction: str = "ccw"):
        assert direction in ("cw", "ccw")
        self.center    = np.asarray(center, dtype=float)
        self.radius    = float(radius)
        self.direction = direction

    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float, speed: float = 0.02) -> np.ndarray:
        start_angle = np.arctan2(start[1] - self.center[1], start[0] - self.center[0])
        sign        = 1.0 if self.direction == "ccw" else -1.0
        darc        = sign * speed * dt / self.radius

        angles = start_angle + darc * np.arange(n_steps + 1)
        ref    = np.zeros((n_steps + 1, 4))
        ref[:, 0] = self.center[0] + self.radius * np.cos(angles)
        ref[:, 1] = self.center[1] + self.radius * np.sin(angles)
        ref[:, 2] = angles + sign * np.pi / 2
        ref[:, 3] = 0.0
        return ref

def choose_face(slider_xy: np.ndarray, slider_theta: float, goal_xy: np.ndarray) -> Face:
    # Picks the face whose outward normal is most opposite to the push direction,
    # i.e. the face the pusher should sit behind. Returns a Face enum.
    # Face outward normals in slider body frame, keyed by Face.value:
    #   NEG_X (0): (-1, 0), POS_Y (1): (0, 1), POS_X (2): (1, 0), NEG_Y (3): (0, -1)
    body_normals = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=float)
    d_world = goal_xy - slider_xy
    c, s = np.cos(slider_theta), np.sin(slider_theta)
    R_world_to_body = np.array([[c, s], [-s, c]])
    d_body = R_world_to_body @ d_world
    face_idx = int(np.argmax(-body_normals @ d_body))
    return Face(face_idx)