import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.controller.base_controller import BaseController

class JointPositionController(BaseController):
    def __init__(self, config, robot_kinematics=None):
        super().__init__(config)
        self.robot_kin = robot_kinematics
        self.kp = np.array(config['kp'])
        self.kd = np.array(config['kd'])
        self.tau_max = np.array(config['tau_max'])
        self._initialized = True
    
    def compute_control(self, state, target):
        q = state.q
        qd = state.qd
        q_desired = target['q']
        
        tau_pd = self.kp * (q_desired - q) - self.kd * qd
        
        if self.robot_kin is not None:
            tau_gravity = self.robot_kin.get_gravity_torques(q)
            tau = tau_pd + tau_gravity
        else:
            tau = tau_pd
        
        return self.validate_torques(tau, self.tau_max)
    
    def reset(self):
        pass