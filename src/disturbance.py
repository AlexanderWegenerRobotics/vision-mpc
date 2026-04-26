import numpy as np
from collections import deque


class DisturbanceModel:
    def __init__(self, config: dict, rng: np.random.Generator):
        self.sigma_xy        = float(config.get("sigma_xy", 0.0))
        self.sigma_theta     = float(config.get("sigma_theta", 0.0))
        self.drop_prob       = float(config.get("drop_prob", 0.0))
        self.latency_frames  = int(config.get("latency_frames", 0))
        self._rng            = rng
        self._buffer         = deque()

    def reset(self):
        self._buffer.clear()

    def process(self, z: np.ndarray):
        if z is None:
            return None
        if self._rng.random() < self.drop_prob:
            return self._release(None)
        z_noisy    = z.copy()
        z_noisy[0] += self._rng.normal(0.0, self.sigma_xy)
        z_noisy[1] += self._rng.normal(0.0, self.sigma_xy)
        z_noisy[2] += self._rng.normal(0.0, self.sigma_theta)
        return self._release(z_noisy)

    def _release(self, z):
        self._buffer.append(z)
        if len(self._buffer) <= self.latency_frames:
            return None
        return self._buffer.popleft()