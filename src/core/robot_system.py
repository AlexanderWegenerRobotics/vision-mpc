import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
import time, threading
from typing import Dict

from src.simulation.sim_model import SimulationModel
from src.simulation.sim_display import SimulationDisplay
from src.controller.controller_manager import ControllerManager
from src.common.robot_kinematics import RobotKinematics
from src.common.data_logger import DataLogger
from src.common.video_logger import VideoLogger
from src.common.utils import load_yaml


class RobotSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.sim_cfg = load_yaml(self.config.get("scene_config"))

        log_dir = Path(self.config.get("logging_path", "log/"))
        trial_name = self.config.get("trial_name", f"trial_{int(time.time())}")
        trial_dir = log_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = DataLogger(trial_name, str(log_dir))

        self.video_logger = None
        if self.sim_cfg.get('video_logging', {}).get('enabled', False):
            fps = self.sim_cfg['video_logging'].get('frequency', 10)
            self.video_logger = VideoLogger(trial_dir, fps=fps)
            print(f"Video logging enabled at {fps} fps")
        
        self.sim = SimulationModel(config=self.sim_cfg, logger=self.logger)
        self.display = SimulationDisplay(sim=self.sim, config=self.sim_cfg, video_logger=self.video_logger)
        
        self.ctrl, kin_model = {}, {}
        self._target = {}
        for device in self.sim_cfg["devices"]:
            device_name = device["name"]
            base_pose = device["base_pose"]
            dof = device["dof"]
            q0 = device["q0"]
            if q0 and len(q0) >= dof:
                q_init = np.array(q0[:dof])
            else:
                q_init = np.zeros(dof)

            self._target[device_name] = {
                'q': q_init,
                'x': np.zeros(6),
                'xd': np.zeros(6)
            }
            urdf_path = device.get("urdf_path", None)
            urdf_ee_name = device.get("urdf_ee_name", None)
            robot_kin = None
            if urdf_path and urdf_ee_name:
                if urdf_path not in kin_model:
                    robot_kin = RobotKinematics(urdf_path=urdf_path, ee_frame_name=urdf_ee_name)
                    kin_model[urdf_path] = robot_kin
                else:
                    robot_kin = kin_model[urdf_path]
            param_path = device.get("ctrl_param")
            if not param_path:
                print(f"Warning! For {device_name} there is no control parameter file! Cant initialize model like this")
                continue

            ctrl_param = load_yaml(param_path)
            cfg = {
                "name":device_name,
                "base_pose":base_pose,
                "kinematic_model":robot_kin,
                "control_param":ctrl_param
            }
                
            self.ctrl[device_name] = ControllerManager(config=cfg)

        self.control_rate = self.sim_cfg.get("control_rate", 200.0)
        self.dt = 1.0 / self.control_rate
        self.running = False
        
        self._lock = threading.Lock()
    
    def run(self):
        """Start all subsystems and block on display"""
        self.running = True
        
        # Start physics thread
        self.sim.start()
        
        # Start control loop in separate thread
        self.control_thread = threading.Thread(target=self._loop, daemon=False)
        self.control_thread.start()
        
        # Run display on main thread (blocks here)
        self.display.run()
        
        # When display exits, stop everything
        self.stop()
    
    def stop(self):
        """Shutdown all subsystems - only close video logger ONCE"""
        if not self.running:
            return
        
        print("Shutting down robot system...")
        self.running = False
        
        if hasattr(self, 'control_thread') and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        self.display.stop()
        
        self.sim.stop()
        
        if self.video_logger is not None:
            self.video_logger.close()
        
        self.logger.save()
        
        print("Robot system stopped")
    
    def _loop(self):
        """Main control loop"""
        print("Control loop started")
        
        while self.running:
            start_time = time.time()
            
            # Get current state from simulation
            states = self.sim.get_state()
            
            # Get target
            with self._lock:
                target = self._target.copy()
            
            # Compute control & push to simulation
            for name, ctrl in self.ctrl.items():
                ctrl_vec = ctrl.compute_control(states[name], target[name])
                self.sim.set_command(tau=ctrl_vec, device_name=name)
            
            # Timing
            elapsed = time.time() - start_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Control loop overrun: {-sleep_time:.4f}s")
        
        print("Control loop stopped")
    
    def set_target(self, device_name: str, target: Dict):
        """Set control target for a specific device"""
        with self._lock:
            if device_name not in self._target:
                raise ValueError(f"Unknown device: {device_name}")
            
            # Update only provided keys
            for key in ['q', 'x', 'xd']:
                if key in target:
                    self._target[device_name][key] = target[key]
    
    def get_state(self) -> Dict:
        """Get current state"""
        return self.sim.get_state()
    
    def set_controller_mode(self, device_name: str, mode: str):
        """Switch controller mode"""
        if device_name in self.ctrl:
            self.ctrl[device_name].set_mode(mode)

if __name__ == "__main__":
    cfg = load_yaml("configs/global_config.yaml")
    system = RobotSystem(cfg)

    try:
        system.run() 
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        system.stop()