import numpy as np
from abc import ABC, abstractmethod


class PathPlanner(ABC):
    # Emits a reference trajectory of shape [N+1, 4] with columns
    # [x_S, y_S, theta_S, p_y_ref] sampled at dt. p_y_ref is typically 0;
    # MPC weights it lightly so the solver picks the actual offset.

    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        ...

    def window(self, full_ref: np.ndarray, step: int, horizon: int) -> np.ndarray:
        # Returns a horizon+1 slice starting at `step`. If step is past the end
        # of the plan, the window is fully padded with the last reference row
        # (i.e. "hold at goal"). Never raises on out-of-range step.
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


class StraightLinePlanner(PathPlanner):
    # Straight line with trapezoidal velocity profile: accelerate from rest,
    # cruise at v_max, decelerate to rest at the goal. Falls back to a triangular
    # profile if the distance is too short to reach v_max. Position and theta
    # interpolate at the same normalized progress s(t) in [0, 1] so orientation
    # slows into the goal alongside position.

    def __init__(self, v_max: float = 0.05, a_max: float = 0.10):
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        # n_steps is ignored: the trapezoidal profile determines how many steps
        # the path needs. Returned reference always has length n_profile+1.
        start_xy, goal_xy = start[:2], goal[:2]
        start_th, goal_th = start[2], self._unwrap_goal_theta(start[2], goal[2])

        distance = float(np.linalg.norm(goal_xy - start_xy))
        if distance < 1e-9:
            ref = np.tile(np.array([start_xy[0], start_xy[1], start_th, 0.0]), (2, 1))
            return ref

        s_profile = self._trapezoidal_progress(distance, dt)  # [0..1], length N+1
        ref = np.zeros((len(s_profile), 4))
        ref[:, 0] = start_xy[0] + s_profile * (goal_xy[0] - start_xy[0])
        ref[:, 1] = start_xy[1] + s_profile * (goal_xy[1] - start_xy[1])
        ref[:, 2] = start_th    + s_profile * (goal_th    - start_th)
        ref[:, 3] = 0.0
        return ref

    def _trapezoidal_progress(self, distance: float, dt: float) -> np.ndarray:
        # Returns normalized progress s(t) in [0,1] sampled at dt. Handles
        # both trapezoidal (distance large enough to reach v_max) and
        # triangular (distance too short) cases.
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
    def _unwrap_goal_theta(start_th: float, goal_th: float) -> float:
        # Choose the representative of goal_th that is closest to start_th.
        diff = (goal_th - start_th + np.pi) % (2 * np.pi) - np.pi
        return start_th + diff


class CircularPlanner(PathPlanner):
    # Constant-speed circular arc. Center and radius fixed at construction;
    # start/goal passed to plan() are used only to pick the starting angle on
    # the circle and travel direction. Slider heading is kept tangent to the path.

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


from src.task.mpc import Face

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