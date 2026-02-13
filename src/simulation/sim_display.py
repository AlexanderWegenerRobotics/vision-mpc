import cv2
import numpy as np
import time
from typing import Dict, TYPE_CHECKING

from src.simulation.sim_model import SimulationModel

class SimulationDisplay:

    def __init__(self, sim: 'SimulationModel', config: Dict, video_logger=None):
        self.sim = sim
        self.config = config
        self.video_logger = video_logger
        self.running = False
        self.stop_request = False
        self.fps = 1.0 / 30.0
        
        if self.video_logger is not None:
            self._setup_video_logging()

    def _setup_video_logging(self):
        """Initialize video logger with configured cameras"""
        video_config = self.config.get('video_logging', {})
        cameras_to_log = video_config.get('cameras', [])
        
        for cam_name in cameras_to_log:
            if cam_name in self.sim.renderers:
                renderer = self.sim.renderers[cam_name]
                self.video_logger.add_camera(cam_name, renderer.width, renderer.height)
            else:
                print(f"Warning: Camera {cam_name} not found for video logging")

    def run(self):
        """Run display loop on main thread"""
        self.running = True
        
        if not self.sim.renderers:
            print("No renderers configured. Display loop exiting.")
            return
        
        while self.running and self.sim.running:
            self._update()
            time.sleep(self.fps)
        
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()

    def _update(self):
        """Update CV2 display with current camera views."""
        if not self.sim.renderers:
            return
        
        frames = {}
        
        with self.sim._lock:
            for cam_name, renderer in self.sim.renderers.items():
                renderer.update_scene(self.sim.mj_data, camera=renderer._cam_id)

        for cam_name, renderer in self.sim.renderers.items():  
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames[cam_name] = frame_bgr
            
            if self.video_logger is not None:
                video_config = self.config.get('video_logging', {})
                if cam_name in video_config.get('cameras', []):
                    self.video_logger.log_frame(cam_name, frame_bgr)
        
        display_frames = []
        for cam_name, frame in frames.items():
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            display_frames.append(annotated_frame)
        
        display_cfg = self.config.get('display', {})
        layout = display_cfg.get('layout', 'horizontal')
        
        if len(display_frames) == 1:
            combined = display_frames[0]
        elif layout == 'horizontal':
            combined = np.hstack(display_frames)
        elif layout == 'vertical':
            combined = np.vstack(display_frames)
        else:
            cols = display_cfg.get('grid_cols', 2)
            rows = (len(display_frames) + cols - 1) // cols
            
            while len(display_frames) < rows * cols:
                display_frames.append(np.zeros_like(display_frames[0]))
            
            grid_rows = [np.hstack(display_frames[i*cols:(i+1)*cols]) for i in range(rows)]
            combined = np.vstack(grid_rows)
        
        window_name = display_cfg.get('window_name', 'Camera Views')
        cv2.imshow(window_name, combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_request = True
            self.running = False
            print("User requested to stop simualtion")