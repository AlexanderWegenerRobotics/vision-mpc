import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.controller.joint_position import JointPositionController
from src.controller.impedance import ImpedanceController

class ControllerManager:
    def __init__(self, config):
        self.device_name = config["name"]
        self.base_pose = config["base_pose"]
        self.kin_model = config.get("kinematic_model", None)

        pos = np.array(self.base_pose["position"])
        quat = self.base_pose["orientation"]  # [w,x,y,z]

        quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
        self.R_world_base = R.from_quat(quat_scipy)
        self.R_base_world = self.R_world_base.inv()
        self.p_world_base = pos
        
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
        
        # Transform Cartesian target from world to base frame if needed
        if 'x' in target and self.mode == 'impedance':
            target_base = target.copy()
            target_base['x'] = self.transform_world_to_base_frame(target['x'])
            
            if 'xd' in target:
                # Velocity only needs rotation (no translation offset)
                xd_world = target['xd']
                xd_base = np.concatenate([
                    self.R_base_world.apply(xd_world[:3]),  # Linear velocity
                    self.R_base_world.apply(xd_world[3:])   # Angular velocity
                ])
                target_base['xd'] = xd_base
            
            return self.active_controller.compute_control(state, target_base)
        
        return self.active_controller.compute_control(state, target)
    
    def transform_world_to_base_frame(self, pose):
        """Transform 6D pose [x,y,z,rx,ry,rz] from world to base frame"""
        pos_world = pose[:3]
        rot_world = pose[3:]
        
        # Position: p_base = R_base_world * (p_world - p_world_base)
        pos_base = self.R_base_world.apply(pos_world - self.p_world_base)
        
        # Rotation: R_base = R_base_world * R_world
        R_world_ee = R.from_rotvec(rot_world)
        R_base_ee = self.R_base_world * R_world_ee
        rot_base = R_base_ee.as_rotvec()
        
        return np.concatenate([pos_base, rot_base])

    def transform_base_to_world_frame(self, pose):
        """Transform 6D pose [x,y,z,rx,ry,rz] from base to world frame"""
        pos_base = pose[:3]
        rot_base = pose[3:]
        
        # Position: p_world = R_world_base * p_base + p_world_base
        pos_world = self.R_world_base.apply(pos_base) + self.p_world_base
        
        # Rotation: R_world = R_world_base * R_base
        R_base_ee = R.from_rotvec(rot_base)
        R_world_ee = self.R_world_base * R_base_ee
        rot_world = R_world_ee.as_rotvec()
        
        return np.concatenate([pos_world, rot_world])