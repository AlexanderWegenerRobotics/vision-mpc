import cv2
import numpy as np
import time
from typing import Dict, TYPE_CHECKING

from src.simulation.sim_env import SimulationModel

class SimulationDisplay:

    def __init__(self, sim: 'SimulationModel', config: Dict):
        self.sim = sim
        self.config = config
        self.running = False
        self.fps = 1.0 / 30.0

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
        
        frames = []
        
        # Lock while reading simulation state
        with self.sim._lock:
            for cam_name, renderer in self.sim.renderers.items():
                renderer.update_scene(self.sim.mj_data, camera=renderer._cam_id)

        for cam_name, renderer in self.sim.renderers.items():  
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)        
            cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            frames.append(frame)
        
        # Combine frames based on display config
        display_cfg = self.config.get('display', {})
        layout = display_cfg.get('layout', 'horizontal')
        
        if len(frames) == 1:
            combined = frames[0]
        elif layout == 'horizontal':
            combined = np.hstack(frames)
        elif layout == 'vertical':
            combined = np.vstack(frames)
        else:  # grid
            cols = display_cfg.get('grid_cols', 2)
            rows = (len(frames) + cols - 1) // cols
            
            # Pad with black frames if needed
            while len(frames) < rows * cols:
                frames.append(np.zeros_like(frames[0]))
            
            # Create grid
            grid_rows = [np.hstack(frames[i*cols:(i+1)*cols]) for i in range(rows)]
            combined = np.vstack(grid_rows)
        
        # Display
        window_name = display_cfg.get('window_name', 'Camera Views')
        cv2.imshow(window_name, combined)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False
            self.sim.running = False
            print("User stoped simualtion")