import time
import numpy as np
import h5py
from pathlib import Path


class EpisodeLogger:
    def __init__(self, config):
        self.config   = config
        self.log_dir  = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.variant  = config["mpc"]["variant"]
        self._max_steps = int(config["timeout"] / config["mpc"]["dt"]) + int(5.0 / config["mpc"]["dt"]) + 10
        self._n    = 0
        self._t0   = None
        self._buf  = None

    def reset(self, episode_idx, start_xy, start_theta, goal_xy, goal_theta, face):
        self._episode_idx = episode_idx
        self._t0          = time.time()
        self._n           = 0
        self._meta = {
            "variant":     self.variant,
            "face":        face,
            "start_xy":    np.array(start_xy),
            "start_theta": float(start_theta),
            "goal_xy":     np.array(goal_xy),
            "goal_theta":  float(goal_theta),
        }

        N = self._max_steps
        self._buf = {
            "t":               np.zeros(N),
            "gt_state":        np.zeros((N, 3)),
            "obs_state":       np.zeros((N, 3)),
            "est_state":       np.zeros((N, 3)),
            "vis_state":       np.full((N, 3), np.nan),
            "detection_valid": np.zeros(N, dtype=bool),
            "est_cov":         np.zeros((N, 3, 3)),
            "control":         np.zeros((N, 2)),
            "ref_state":       np.zeros((N, 4)),
            "p_y":             np.zeros(N),
            "solver_status":   np.zeros(N, dtype=np.int32),
            "solve_time_ms":   np.zeros(N),
            "ee_pos":          np.zeros((N, 3)),
            "ee_vel":          np.zeros((N, 2)),
            "pusher_tip":      np.zeros((N, 2)),
        }

    def record(self, gt_state, obs_state, est_state, vis_state, est_cov, control, ref_state, p_y,
               solver_status, solve_time_ms, ee_pos, ee_vel, pusher_tip):
        if self._buf is None or self._n >= self._max_steps:
            return
        i = self._n
        detected = vis_state is not None
        self._buf["t"][i]               = time.time() - self._t0
        self._buf["gt_state"][i]        = gt_state
        self._buf["obs_state"][i]       = obs_state
        self._buf["est_state"][i]       = est_state
        self._buf["detection_valid"][i] = detected
        self._buf["vis_state"][i]       = vis_state if detected else np.full(3, np.nan)
        self._buf["est_cov"][i]         = est_cov
        self._buf["control"][i]         = control
        self._buf["ref_state"][i]       = ref_state
        self._buf["p_y"][i]             = p_y
        self._buf["solver_status"][i]   = solver_status
        self._buf["solve_time_ms"][i]   = solve_time_ms
        self._buf["ee_pos"][i]          = ee_pos
        self._buf["ee_vel"][i]          = ee_vel
        self._buf["pusher_tip"][i]      = pusher_tip
        self._n += 1

    def save(self, success: bool):
        if self._buf is None or self._n == 0:
            return

        n    = self._n
        path = self.log_dir / f"episode_{self._episode_idx:04d}.h5"

        with h5py.File(path, "w") as f:
            f.attrs["variant"]     = self._meta["variant"]
            f.attrs["face"]        = self._meta["face"]
            f.attrs["start_xy"]    = self._meta["start_xy"]
            f.attrs["start_theta"] = self._meta["start_theta"]
            f.attrs["goal_xy"]     = self._meta["goal_xy"]
            f.attrs["goal_theta"]  = self._meta["goal_theta"]
            f.attrs["success"]     = int(success)
            f.attrs["n_steps"]     = n

            for key, arr in self._buf.items():
                f.create_dataset(key, data=arr[:n], compression="gzip")

        print(f"[log] episode {self._episode_idx:04d} saved to {path} ({n} steps)")
        self._buf = None