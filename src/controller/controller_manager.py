import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
import numpy as np
from src.controller.joint_position import JointPositionController
from src.controller.impedance import ImpedanceController

class ControllerManager:
    def __init__(self, config):
        self.device_name = config["name"]
        self.base_pose = config["base_pose"]
        self.kin_model = config.get("kinematic_model", None)
        
        ctrl_params = config["control_param"]
        self.mode = ctrl_params.get("default_mode", "idle")
        
        available_controllers = ctrl_params
        ctrl_names = list(available_controllers.keys())
        
        self.controllers = {"idle": None}
        
        if "position" in ctrl_names:
            self.controllers["position"] = JointPositionController(
                available_controllers["position"], 
                self.kin_model
            )
        
        if "impedance" in ctrl_names:
            self.controllers["impedance"] = ImpedanceController(
                available_controllers["impedance"], 
                self.kin_model
            )
        
        if self.mode not in self.controllers:
            print(f"Warning: Default mode '{self.mode}' not available for {self.device_name}. Using 'idle'.")
            self.mode = "idle"
        
        self.active_controller = self.controllers[self.mode]
    
    def set_mode(self, mode):
        if mode not in self.controllers:
            raise ValueError(f"Mode '{mode}' not available for {self.device_name}. Available: {list(self.controllers.keys())}")
        
        if self.active_controller is not None:
            self.active_controller.reset()
        
        self.mode = mode
        self.active_controller = self.controllers[mode]
    
    def get_current_mode(self) -> str:
        return self.mode
    
    def compute_control(self, state, target) -> np.ndarray:
        if self.mode == "idle" or self.active_controller is None:
            return np.zeros(len(state.q))
        
        return self.active_controller.compute_control(state, target)