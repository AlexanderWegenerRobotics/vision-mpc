import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
import numpy as np
import mujoco as mj
import mujoco.viewer
import cv2
import time, threading
from typing import Dict, List, Optional, Tuple

from src.common.utils import load_yaml
from src.common.robot_state import RobotState
from src.common.video_logger import VideoLogger

class DeviceInfo:
    """Stores metadata about a device (robot/sensor/object)."""
    
    def __init__(self, name: str, device_type: str, prefix: str):
        self.name = name
        self.type = device_type
        self.prefix = prefix
        
        # Joint information
        self.joint_ids: List[int] = []
        self.joint_names: List[str] = []
        self.dof_ids: List[int] = []
        
        # Actuator information
        self.actuator_ids: List[int] = []
        self.actuator_names: List[str] = []
        
        # Body information
        self.body_ids: List[int] = []
        self.body_names: List[str] = []
        
        # Initial configuration
        self.q0: Optional[np.ndarray] = None

class SimulationModel:

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
        self.mj_model, self.devices, self.objects = self._build_model_from_config(config)
        self.mj_data = mj.MjData(self.mj_model)

        self.cameras = {}
        self.renderers = {}
        self._setup_cameras()

        self._lock = threading.Lock()
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._command = np.zeros(self.mj_model.nu)

        self.running = False
        self.dt = self.mj_model.opt.timestep

        self.log_config = config.get('logging', {})
        self.log_frequency = self.log_config.get('frequency', 30)
        self._log_counter = 0
        self._log_interval = max(1, int((1.0 / self.log_frequency) / self.dt))


    def _build_model_from_config(self, config:Dict):
        """
        Build MuJoCo model by composing base world + devices + objects.
        """
        world_spec = mj.MjSpec.from_file(config['world_model'])
        world_spec.copy_during_attach = True

        devices, objects = {}, {}

        # Add devices (robots, sensors)
        for device_cfg in config.get('devices', []):
            if not device_cfg.get('enabled', True):
                continue
        
            name = device_cfg['name']
            device_type = device_cfg['type']
            model_path = device_cfg['model_path']
            base_pose = device_cfg.get('base_pose', {})

            # Load device model
            device_spec = mj.MjSpec.from_file(model_path)

            # Extract pose
            pos = base_pose.get('position', [0, 0, 0])
            quat = base_pose.get('orientation', [1, 0, 0, 0])  # [w, x, y, z]

            # Create attachment frame in world
            attach_frame = world_spec.worldbody.add_frame(pos=pos, quat=quat)
            world_spec.attach(device_spec, frame=attach_frame, prefix=f"{name}/")

            device_info = DeviceInfo(name, device_type, prefix=f"{name}/")
            device_info.q0 = np.array(device_cfg.get('q0', []))
            devices[name] = device_info

        # Add objects (static/dynamic props)
        for obj_cfg in config.get('objects', []):
            if not obj_cfg.get('enabled', True):
                continue
                
            name = obj_cfg['name']
            obj_type = obj_cfg['type']
            model_path = obj_cfg['model_path']
            pose = obj_cfg.get('pose', {})
            
            # Load object model
            obj_spec = mj.MjSpec.from_file(model_path)
            
            # Extract pose
            pos = pose.get('position', [0, 0, 0])
            quat = pose.get('orientation', [1, 0, 0, 0])  # [w, x, y, z]
            
            # Create attachment frame in world
            attach_frame = world_spec.worldbody.add_frame(pos=pos, quat=quat)            
            world_spec.attach(obj_spec, frame=attach_frame, prefix=f"{name}/")
            
            # Store object info
            obj_info = DeviceInfo(name, obj_type, prefix=f"{name}/")
            objects[name] = obj_info

        # Add cameras (workspace observer / sensors etc)
        for cam_cfg in config.get("cameras",[]):
            if not cam_cfg.get("enabled",True):
                continue
            cam_name = cam_cfg.get("name")
            cam_type = cam_cfg.get("type", "fixed")

            if cam_type == "fixed":
                world_spec.worldbody.add_camera(
                    name=cam_name,
                    pos=cam_cfg.get("pos", [0,0,1]),
                    quat=cam_cfg.get("quat",[1,0,0,0]),
                    fovy=cam_cfg.get("fovy",60)
                )
                print(f"Added scene camera: {cam_name}")
            elif cam_type == "tracking":
                target_body = cam_cfg.get('target_body', 'world')
                world_spec.worldbody.add_camera(
                    name=cam_name,
                    mode='trackcom',  # Track center of mass
                    pos=cam_cfg.get('pos', [0, 0, 1]),
                    quat=cam_cfg.get('quat', [1, 0, 0, 0]),
                    fovy=cam_cfg.get('fovy', 60)
                )
                print(f"Added tracking camera: {cam_name} -> {target_body}")

        # Compile final model
        compiled_model = world_spec.compile()
        
        # Extract IDs from compiled model
        all_entities = {**devices, **objects}
        self._extract_device_ids(model=compiled_model, devices=devices)

        return compiled_model, devices, objects
    
    def _extract_device_ids(self, model: mj.MjModel, devices: Dict[str, DeviceInfo]):
        for device_info in devices.values():
            prefix = device_info.prefix
            # Find joints
            for joint_id in range(model.njnt):
                joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name and joint_name.startswith(prefix):
                    device_info.joint_ids.append(joint_id)
                    device_info.joint_names.append(joint_name)
                    dof_adr = model.jnt_dofadr[joint_id]
                    device_info.dof_ids.append(dof_adr)

            # Find actuators
            for act_id in range(model.nu):
                act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id)
                if act_name and act_name.startswith(prefix):
                    device_info.actuator_ids.append(act_id)
                    device_info.actuator_names.append(act_name)
            
            # Find bodies
            for body_id in range(model.nbody):
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
                if body_name and body_name.startswith(prefix):
                    device_info.body_ids.append(body_id)
                    device_info.body_names.append(body_name)

    def _setup_cameras(self):
        """Setup camera renderers based on config and available cameras in model."""

        # Find all cameras in the compiled model
        for cam_id in range(self.mj_model.ncam):
            cam_name = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
            if cam_name:
                self.cameras[cam_name] = cam_id

        render_configs = self.config.get('render_cameras', [])
        for cam_cfg in render_configs:
            if not cam_cfg.get("enabled",True):
                continue

            cam_name = cam_cfg.get("name")
            attachment = cam_cfg.get("attachment")
            if attachment and attachment != False:
                prefixed_name = f"{attachment}/{cam_name}"
            else:
                prefixed_name = cam_name
            
            if prefixed_name in self.cameras:
                width = cam_cfg.get('width', 640)
                height = cam_cfg.get('height', 480)
                self.renderers[cam_name] = mj.Renderer(self.mj_model, height=height, width=width)
                self.renderers[cam_name]._cam_id = self.cameras[prefixed_name]
                print(f"Created renderer for camera: {cam_name} (model name: {prefixed_name})")
            else:
                print(f"Camera '{cam_name}' not found in model (tried: {prefixed_name})")

    def start(self):
        self.running = True
        self.physics_thread.start()

    def stop(self):
        """Stop simulation gracefully"""
        self.running = False
        if self.physics_thread.is_alive():
            self.physics_thread.join()

    def get_state(self, device_name:str="arm") -> RobotState:
        device = self.devices[device_name]
        with self._lock:
            q = self.mj_data.qpos[device.dof_ids].copy()
            qd = self.mj_data.qvel[device.dof_ids].copy()
            qdd = self.mj_data.qacc[device.dof_ids].copy()
            tau = self.mj_data.actuator_force[device.actuator_ids].copy()
        return RobotState(q=q, qd=qd, qdd=qdd, tau=tau)

    def set_command(self, tau: np.array, device_name):
        if device_name not in self.devices:
            raise ValueError(f"Device '{device_name}' not found. Available: {list(self.devices.keys())}")
        device = self.devices[device_name]
        with self._lock:
            self._command[device.actuator_ids] = tau

    def get_camera_image(self, camera_name: str) -> Optional[np.ndarray]:
        """Render image from specified camera."""
        if camera_name not in self.renderers:
            print(f"Warning! Referenced camera name {camera_name} not existent. Options: {self.cameras.keys()}")
        
        renderer = self.renderers[camera_name]
        cam_id = renderer._cam_id
        
        # Update renderer with current state
        renderer.update_scene(self.mj_data, camera=cam_id)
        
        # Render and return pixels (RGB format)
        return renderer.render()

    def _physics_loop(self):
        last_time = time.time()

        while self.running:
            with self._lock:
                self.mj_data.ctrl[:] = self._command

            mj.mj_step(self.mj_model, self.mj_data)

            if self.logger is not None:
                self._log_counter += 1
                if self._log_counter >= self._log_interval:
                    self._log_step()
                    self._log_counter = 0
                        
            elapsed = time.time() - last_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Simulation loop overrun: {-sleep_time:.4f}s")
            last_time = time.time()

    def _log_step(self):
        """Log current simulation state based on configuration."""
        if self.logger is None:
            return
            
        bundles_to_log = self.log_config.get('bundles', [])
        
        for bundle_cfg in bundles_to_log:
            bundle_name = bundle_cfg.get('name')
            bundle_type = bundle_cfg.get('type')

            if bundle_type == 'robot_state':
                device_name = bundle_cfg.get('device', 'arm')
                state = self.get_state(device_name)
                
                self.logger.log_bundle(bundle_name, 
                                       {
                    'q': state.q,
                    'qd': state.qd,
                    'qdd': state.qdd,
                    'tau': state.tau,
                    'command': self._command[self.devices[device_name].actuator_ids].copy()
                })

        
if __name__ == "__main__":
    from src.simulation.sim_display import SimulationDisplay
    
    global_cfg = load_yaml("configs/global_config.yaml")
    sim_config = load_yaml(global_cfg.get("scene_config"))
    sim = SimulationModel(sim_config)
    display = SimulationDisplay(sim, sim_config)

    try:
        print("Starting simulation")
        sim.start() 
        display.run() 
    except KeyboardInterrupt:
        print("Stopping simulation")
    finally:
        display.stop()
        sim.stop()