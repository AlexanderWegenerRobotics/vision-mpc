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



class DubinsPlanner(PathPlanner):
    def __init__(self, R_min: float = 0.10, v_max: float = 0.05, a_max: float = 0.10):
        self.R_min = float(R_min)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def plan(self, start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        start_xy, start_th = start[:2], start[2]
        goal_xy,  goal_th  = goal[:2],  goal[2]

        path = self._shortest_dubins(start_xy, start_th, goal_xy, goal_th, self.R_min)
        if path is None:
            return self._fallback_straight(start, goal, dt)

        total_length = path["length"]
        s_profile = self._trapezoidal_progress(total_length, dt) * total_length

        ref = np.zeros((len(s_profile), 4))
        for i, s in enumerate(s_profile):
            x, y, th = self._sample_path(path, s)
            ref[i, 0] = x
            ref[i, 1] = y
            ref[i, 2] = th
            ref[i, 3] = 0.0
        return ref

    def _shortest_dubins(self, p0, th0, p1, th1, R):
        # Try all 6 Dubins word combinations, return the shortest valid one.
        candidates = []
        for word in ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL"):
            seg = self._dubins_word(p0, th0, p1, th1, R, word)
            if seg is not None:
                candidates.append(seg)
        if not candidates:
            return None
        return min(candidates, key=lambda s: s["length"])

    def _dubins_word(self, p0, th0, p1, th1, R, word):
        # Compute segment lengths for one Dubins word using normalized geometry.
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        D = np.hypot(dx, dy) / R
        alpha = (th0 - np.arctan2(dy, dx)) % (2 * np.pi)
        beta  = (th1 - np.arctan2(dy, dx)) % (2 * np.pi)

        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta),  np.cos(beta)

        if word == "LSL":
            tmp = 2 + D*D - 2*np.cos(alpha - beta) + 2*D*(sa - sb)
            if tmp < 0: return None
            t = (-alpha + np.arctan2(cb - ca, D + sa - sb)) % (2*np.pi)
            p = np.sqrt(tmp)
            q = (beta - np.arctan2(cb - ca, D + sa - sb)) % (2*np.pi)
            segs = [("L", t), ("S", p), ("L", q)]
        elif word == "RSR":
            tmp = 2 + D*D - 2*np.cos(alpha - beta) + 2*D*(sb - sa)
            if tmp < 0: return None
            t = (alpha - np.arctan2(ca - cb, D - sa + sb)) % (2*np.pi)
            p = np.sqrt(tmp)
            q = (-beta % (2*np.pi) + np.arctan2(ca - cb, D - sa + sb)) % (2*np.pi)
            segs = [("R", t), ("S", p), ("R", q)]
        elif word == "LSR":
            tmp = -2 + D*D + 2*np.cos(alpha - beta) + 2*D*(sa + sb)
            if tmp < 0: return None
            p = np.sqrt(tmp)
            t = (-alpha + np.arctan2(-ca - cb, D + sa + sb) - np.arctan2(-2.0, p)) % (2*np.pi)
            q = (-beta % (2*np.pi) + np.arctan2(-ca - cb, D + sa + sb) - np.arctan2(-2.0, p)) % (2*np.pi)
            segs = [("L", t), ("S", p), ("R", q)]
        elif word == "RSL":
            tmp = D*D - 2 + 2*np.cos(alpha - beta) - 2*D*(sa + sb)
            if tmp < 0: return None
            p = np.sqrt(tmp)
            t = (alpha - np.arctan2(ca + cb, D - sa - sb) + np.arctan2(2.0, p)) % (2*np.pi)
            q = (beta - np.arctan2(ca + cb, D - sa - sb) + np.arctan2(2.0, p)) % (2*np.pi)
            segs = [("R", t), ("S", p), ("L", q)]
        elif word == "RLR":
            tmp = (6 - D*D + 2*np.cos(alpha - beta) + 2*D*(sa - sb)) / 8.0
            if abs(tmp) > 1: return None
            p = (2*np.pi - np.arccos(tmp)) % (2*np.pi)
            t = (alpha - np.arctan2(ca - cb, D - sa + sb) + p/2.0) % (2*np.pi)
            q = (alpha - beta - t + p) % (2*np.pi)
            segs = [("R", t), ("L", p), ("R", q)]
        elif word == "LRL":
            tmp = (6 - D*D + 2*np.cos(alpha - beta) + 2*D*(sb - sa)) / 8.0
            if abs(tmp) > 1: return None
            p = (2*np.pi - np.arccos(tmp)) % (2*np.pi)
            t = (-alpha + np.arctan2(-ca + cb, D + sa - sb) + p/2.0) % (2*np.pi)
            q = ((beta % (2*np.pi)) - alpha - t + p) % (2*np.pi)
            segs = [("L", t), ("R", p), ("L", q)]
        else:
            return None

        length = (segs[0][1] + segs[1][1] + segs[2][1]) * R
        return {"p0": np.array(p0), "th0": th0, "R": R, "segs": segs, "length": length}

    def _sample_path(self, path, s):
        # Walk segments, find which one s falls in, sample pose at arc length s.
        R = path["R"]
        x, y, th = path["p0"][0], path["p0"][1], path["th0"]
        s_left = s
        for kind, seg_len in path["segs"]:
            seg_arc = seg_len * R if kind != "S" else seg_len * R
            # For S: seg_len is normalized (D-units), arc length is seg_len * R.
            # For L/R: seg_len is angle in rad, arc length is seg_len * R.
            if s_left <= seg_arc + 1e-12:
                ds = s_left
                if kind == "S":
                    x += ds * np.cos(th)
                    y += ds * np.sin(th)
                elif kind == "L":
                    dth = ds / R
                    cx = x - R * np.sin(th)
                    cy = y + R * np.cos(th)
                    th_new = th + dth
                    x = cx + R * np.sin(th_new)
                    y = cy - R * np.cos(th_new)
                    th = th_new
                elif kind == "R":
                    dth = ds / R
                    cx = x + R * np.sin(th)
                    cy = y - R * np.cos(th)
                    th_new = th - dth
                    x = cx - R * np.sin(th_new)
                    y = cy + R * np.cos(th_new)
                    th = th_new
                return x, y, ((th + np.pi) % (2*np.pi)) - np.pi
            # advance to end of this segment
            if kind == "S":
                x += seg_arc * np.cos(th)
                y += seg_arc * np.sin(th)
            elif kind == "L":
                dth = seg_len
                cx = x - R * np.sin(th)
                cy = y + R * np.cos(th)
                th_new = th + dth
                x = cx + R * np.sin(th_new)
                y = cy - R * np.cos(th_new)
                th = th_new
            elif kind == "R":
                dth = seg_len
                cx = x + R * np.sin(th)
                cy = y - R * np.cos(th)
                th_new = th - dth
                x = cx - R * np.sin(th_new)
                y = cy + R * np.cos(th_new)
                th = th_new
            s_left -= seg_arc
        return x, y, ((th + np.pi) % (2*np.pi)) - np.pi

    def _trapezoidal_progress(self, distance: float, dt: float) -> np.ndarray:
        # Same trapezoidal profile as StraightLinePlanner, normalized to [0, 1].
        if distance < 1e-9:
            return np.array([0.0, 0.0])
        v_max, a_max = self.v_max, self.a_max
        d_accel = v_max**2 / (2 * a_max)
        if 2 * d_accel <= distance:
            t_accel  = v_max / a_max
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / v_max
            t_total  = 2 * t_accel + t_cruise
        else:
            v_peak   = np.sqrt(a_max * distance)
            t_accel  = v_peak / a_max
            t_cruise = 0.0
            t_total  = 2 * t_accel
        n = max(int(np.ceil(t_total / dt)), 2)
        t = np.linspace(0.0, t_total, n + 1)
        d = np.zeros_like(t)
        if t_cruise > 0:
            am = t <= t_accel
            cm = (t > t_accel) & (t <= t_accel + t_cruise)
            dm = t > t_accel + t_cruise
            d[am] = 0.5 * a_max * t[am]**2
            d[cm] = d_accel + v_max * (t[cm] - t_accel)
            td = t[dm] - (t_accel + t_cruise)
            d[dm] = distance - 0.5 * a_max * (t_accel - td)**2
        else:
            am = t <= t_accel
            dm = t > t_accel
            d[am] = 0.5 * a_max * t[am]**2
            td = t[dm] - t_accel
            d[dm] = distance - 0.5 * a_max * (t_accel - td)**2
        return np.clip(d, 0.0, distance) / distance

    def _fallback_straight(self, start, goal, dt):
        # Degenerate case (poses coincide etc.); emit a 2-row hold reference.
        return np.tile(np.array([start[0], start[1], start[2], 0.0]), (2, 1))