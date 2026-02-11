import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.controller.base_controller import BaseController
from src.common.utils import load_yaml

class ImpedanceController(BaseController):
    def __init__(self, config, robot_kinematics):
        super().__init__(config)
        self.robot_kin = robot_kinematics
        self.K_cart = np.diag(config['K_cart'])
        self.D_cart = np.diag(config['D_cart'])
        self.K_null = config['K_null']
        self.tau_max = np.array(config.get('tau_max', [87]*7))
        self.q_nominal = np.array(config.get('q_nominal', [0, -0.785, 0, -2.356, 0, 1.571, 0.785]))
        self._initialized = True
    
    def compute_control(self, state, target):
        q = state['q']
        qd = state['qd']
        x_desired = target['x']
        xd_desired = target.get('xd', np.zeros(6))
        
        x_current = self.robot_kin.forward_kinematics(q)
        xd_current = self.robot_kin.get_ee_velocity(q, qd)
        J = self.robot_kin.get_jacobian(q)
        
        F = self.K_cart @ (x_desired - x_current) + self.D_cart @ (xd_desired - xd_current)
        
        tau_task = J.T @ F
        
        J_pinv = np.linalg.pinv(J)
        null_projector = np.eye(7) - J_pinv @ J
        tau_null = self.K_null * (self.q_nominal - q)
        tau = tau_task + null_projector @ tau_null
        
        return self.validate_torques(tau, self.tau_max)
    
    def reset(self):
        pass