import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.controller.base_controller import BaseController
from src.common.utils import load_yaml

class JointPositionController(BaseController):
    def __init__(self, config, robot_kinematics):
        super().__init__(config)
        self.kp = np.array(config['kp'])
        self.kd = np.array(config['kd'])
        self.tau_max = np.array(config.get('tau_max', [87]*7))
        self._initialized = True
        self.robot_kin = robot_kinematics
    
    def compute_control(self, state, target):
        q = state.q
        qd = state.qd
        q_desired = target
        tau_pd = self.kp * (q_desired - q) - self.kd * qd
        tau_gravity = self.robot_kin.get_gravity_torques(q)
        tau = tau_pd + tau_gravity
        return self.validate_torques(tau, self.tau_max)
    
    def reset(self):
        pass

if __name__ == "__main__":
    config = load_yaml("configs/controller_config.yaml")
    ctrl = JointPositionController(config=config.get("position"))