import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
import numpy as np
from src.controller.base_controller import BaseController
from src.controller.joint_position import JointPositionController
from src.controller.impedance import ImpedanceController

class ControllerManager:
    def __init__(self, config, robot_kinematics, logger=None):
        self.config = config
        self.logger = logger
        self.mode = config.get('default_mode', 'position')
        
        self.controllers = {
            'position': JointPositionController(config['position'], robot_kinematics),
            'impedance': ImpedanceController(config['impedance'], robot_kinematics),
            'idle': None
        }
        
        self.active_controller = self.controllers[self.mode]
    
    def set_mode(self, mode):
        if mode not in self.controllers:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.controllers.keys())}")
        
        if self.active_controller is not None:
            self.active_controller.reset()
        
        self.mode = mode
        self.active_controller = self.controllers[mode]
    
    def get_current_mode(self) -> str:
        return self.mode
    
    def compute_control(self, state, target) -> np.ndarray:
        if self.mode == 'idle':
            tau = np.zeros(7)
        else:
            tau = self.active_controller.compute_control(state, target)
        
        if self.logger is not None:
            self.logger.log_bundle('control', {
                'tau_cmd': tau,
                'q': state['q'],
                'qd': state['qd'],
                'q_target': target.get('q', np.zeros(7)),
                'x_target': target.get('x', np.zeros(6)),
                'mode': np.array([list(self.controllers.keys()).index(self.mode)])
            })
        
        return tau