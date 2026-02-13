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
        self.ctrl_cfg = load_yaml(self.config.get("controller_config"))
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
        
        self.robot_kin = RobotKinematics("assets/urdf/panda/panda.urdf")
        self.ctrl = ControllerManager(config=self.ctrl_cfg, robot_kinematics=self.robot_kin)
        
        self.control_rate = 200
        self.dt = 1.0 / self.control_rate
        self.running = False
        
        self._target = {'q': np.zeros(7)}
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

        self._target = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        while self.running:
            start_time = time.time()
            
            # Get current state from simulation
            state = self.sim.get_state()
            
            # Get target
            with self._lock:
                target = self._target.copy()
            
            # Compute control
            ctrl_vec = self.ctrl.compute_control(state, target)
            
            # Send to simulation
            self.sim.set_command(tau=ctrl_vec, device_name="arm")
            
            # Timing
            elapsed = time.time() - start_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Control loop overrun: {-sleep_time:.4f}s")
        
        print("Control loop stopped")
    
    def set_target(self, target: Dict):
        """Set control target from outside"""
        with self._lock:
            self._target = target
    
    def get_state(self) -> Dict:
        """Get current state"""
        return self.sim.get_state()
    
    def set_controller_mode(self, mode: str):
        """Switch controller mode"""
        self.ctrl.set_mode(mode)


if __name__ == "__main__":
    cfg = load_yaml("configs/global_config.yaml")
    system = RobotSystem(cfg)

    try:
        system.run() 
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        system.stop()