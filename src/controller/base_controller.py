from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    def __init__(self, config):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def compute_control(self, state, target):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def validate_torques(self, tau, tau_max, verbose=False):
        tau_clipped = np.clip(tau, -tau_max, tau_max)
        if not np.allclose(tau, tau_clipped):
            exceeded = np.abs(tau) > tau_max
            if verbose:
                print(f"Warning: Torque limits exceeded on joints {np.where(exceeded)[0]}")
        return tau_clipped
    
    def is_initialized(self):
        return self._initialized