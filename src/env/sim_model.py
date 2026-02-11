import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
import numpy as np
import mujoco as mj
import mujoco.viewer
import cv2
import time, threading
from src.common.utils import load_yaml

class SimulationModel:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.running = False
        self.model = mj.MjModel.from_xml_path(config.get("world_model"))
        self.data = mj.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        self._lock = threading.Lock()
        self._command = np.zeros(self.model.nu)
        
        self.robot_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.robot_joint_ids = [int(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)) 
                                for name in self.robot_joint_names]
        self.robot_qpos_addrs = [int(self.model.jnt_qposadr[jid]) for jid in self.robot_joint_ids]
        self.robot_qvel_addrs = [int(self.model.jnt_dofadr[jid]) for jid in self.robot_joint_ids]
        
        self.box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'box')
        
        self.viewer = None
        self.renderer_main = mj.Renderer(self.model, height=480, width=640)
        self.renderer_wrist = mj.Renderer(self.model, height=480, width=640)
        
        self.main_cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "frontview")
        self.wrist_cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
    
    def start(self):
        if not self.running:
            self.running = True
            self._physics_loop()
    
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
    
    def set_control(self, ctrl):
        with self._lock:
            self._command[:] = ctrl
    
    def get_state(self):
        with self._lock:
            q_robot = np.array([self.data.qpos[addr] for addr in self.robot_qpos_addrs])
            qd_robot = np.array([self.data.qvel[addr] for addr in self.robot_qvel_addrs])
            
            box_pos = self.data.xpos[self.box_body_id].copy()
            box_quat = self.data.xquat[self.box_body_id].copy()
            
            return {
                'q': q_robot,
                'qd': qd_robot,
                'box_pos': box_pos,
                'box_quat': box_quat,
                'time': self.data.time
            }
    
    def _physics_loop(self):
        while self.running:
            with self._lock:
                self.data.ctrl[:] = self._command

            mj.mj_step(self.model, self.data)

            if self.logger is not None:
                q_robot = np.array([self.data.qpos[addr] for addr in self.robot_qpos_addrs])
                qd_robot = np.array([self.data.qvel[addr] for addr in self.robot_qvel_addrs])
                box_pos = self.data.xpos[self.box_body_id].copy()
                box_quat = self.data.xquat[self.box_body_id].copy()
                
                self.logger.log_bundle('physics', {
                    'q': q_robot,
                    'qd': qd_robot,
                    'tau_cmd': self._command.copy(),
                    'box_pos': box_pos,
                    'box_quat': box_quat
                })
            
            self.renderer_main.update_scene(self.data, camera=self.main_cam_id)
            frame_main = self.renderer_main.render()
            
            self.renderer_wrist.update_scene(self.data, camera=self.wrist_cam_id)
            frame_wrist = self.renderer_wrist.render()
            
            combined = np.hstack([frame_main, frame_wrist])
            
            cv2.imshow("Main View | Wrist View", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            
            time.sleep(self.dt)

if __name__ == "__main__":
    from src.common.data_logger import DataLogger
    logger = DataLogger(trial_name='test_sim_model', output_dir='log/')

    config = load_yaml("configs/global_config.yaml")
    sim = SimulationModel(config=config, logger=logger)
    
    try:
        sim.start()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        sim.stop()
        logger.save()
        print(logger.get_summary())