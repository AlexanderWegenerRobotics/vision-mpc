import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
import pinocchio as pin
from src.controller.base_controller import BaseController


class ImpedanceController(BaseController):
    def __init__(self, config, robot_kinematics=None):
        super().__init__(config)
        
        if robot_kinematics is None:
            raise ValueError("ImpedanceController requires a kinematic model")
        
        self.robot_kin = robot_kinematics
        self.K_cart = np.diag(config['K_cart'])
        self.D_cart = np.diag(config['D_cart'])
        self.K_null = config['K_null']
        self.tau_max = np.array(config['tau_max'])
        self.q_nominal = np.array(config.get('q_nominal', np.zeros(len(config['kp']) if 'kp' in config else 7)))
        self.gravity_comp = config.get('gravity_compensation', True)
        self._initialized = True
    
    def compute_control(self, state, target):
        q = state.q
        qd = state.qd
        x_desired = target['x']      # Pose object
        xd_desired = target.get('xd', np.zeros(6))
        
        x_current = self.robot_kin.forward_kinematics(q)
        xd_current = self.robot_kin.get_ee_velocity(q, qd)
        J = self.robot_kin.get_jacobian(q)
        
        # Position error (base frame)
        e_pos = x_desired.position - x_current.position
        
        # Orientation error via SO(3) logarithmic map
        R_current = x_current.rotation_matrix
        R_desired = x_desired.rotation_matrix
        R_error = R_desired @ R_current.T
        e_rot = pin.log3(R_error)
        
        e = np.concatenate([e_pos, e_rot])
        
        F = self.K_cart @ e + self.D_cart @ (xd_desired - xd_current)
        tau_task = J.T @ F
        
        J_pinv = np.linalg.pinv(J)
        null_projector = np.eye(len(q)) - J_pinv @ J
        tau_null = self.K_null * (self.q_nominal - q)
        
        tau = tau_task + null_projector @ tau_null
        
        if self.gravity_comp:
            tau += self.robot_kin.get_gravity_torques(q)
        
        return self.validate_torques(tau, self.tau_max)
    
    def reset(self):
        pass