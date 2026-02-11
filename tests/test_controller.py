import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))
import numpy as np
import time
from src.env.sim_model import SimulationModel
from src.common.robot_kinematics import RobotKinematics
from src.controller.controller_manager import ControllerManager
from src.common.utils import load_yaml
from src.common.data_logger import DataLogger

def test_position_control(logger=None):
    print("=== Testing Position Control ===")
    
    config = load_yaml("configs/global_config.yaml")
    controller_config = load_yaml(config['controller_config'])
    
    sim = SimulationModel(config)
    robot_kin = RobotKinematics(config['pin_model'])
    controller = ControllerManager(controller_config, robot_kin, logger=logger)
    
    controller.set_mode('position')
    
    q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    q_test = np.array([0.5, -0.5, 0, -2.0, 0, 1.5, 0.8])
    
    print(f"Mode: {controller.get_current_mode()}")
    print(f"Moving to home position...")
    
    def control_loop():
        for i in range(4000):
            state = sim.get_state()
            
            if i < 4000:
                target = {'q': q_home}
            else:
                target = {'q': q_test}
            
            tau = controller.compute_control(state, target)
            sim.set_control(tau)
            
            if i % 50 == 0:
                q_current = state['q']
                error = np.linalg.norm(target['q'] - q_current)
                print(f"Step {i}, Error: {error:.4f}")
            
            time.sleep(0.005)
    
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()
    
    sim.start()
    control_thread.join()
    sim.stop()

def test_impedance_control(logger=None):
    print("\n=== Testing Impedance Control ===")
    
    config = load_yaml("configs/scene_config.yaml")
    controller_config = load_yaml(config['controller_config'])
    
    sim = SimulationModel(config)
    robot_kin = RobotKinematics(config['pin_model'])
    controller = ControllerManager(controller_config['controller'], robot_kin, logger=logger)
    
    controller.set_mode('impedance')
    
    x_hover = np.array([0.4, 0.0, 0.3, np.pi, 0, 0])
    
    print(f"Mode: {controller.get_current_mode()}")
    print(f"Moving to hover position...")
    
    def control_loop():
        for i in range(500):
            state = sim.get_state()
            
            target = {'x': x_hover}
            
            tau = controller.compute_control(state, target)
            sim.set_control(tau)
            
            if i % 50 == 0:
                x_current = robot_kin.forward_kinematics(state['q'])
                error = np.linalg.norm(x_hover - x_current)
                print(f"Step {i}, Task Error: {error:.4f}")
            
            time.sleep(0.002)
    
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()
    
    sim.start()
    control_thread.join()
    sim.stop()

if __name__ == "__main__":
    import threading
    logger = DataLogger(trial_name='test_controller', output_dir='log/')

    try:
        test_position_control(logger=logger)
        time.sleep(2)
        test_impedance_control(logger=logger)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        logger.save()
        print(logger.get_summary())