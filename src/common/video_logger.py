import cv2
import numpy as np
from pathlib import Path
import time

class VideoLogger:
    """Video logger for camera streams - single threaded, called from display thread"""
    
    def __init__(self, trial_dir: Path, fps: int = 30):
        self.trial_dir = Path(trial_dir)
        self.fps = fps
        self.writers = {}
        self.frame_counts = {}
        
    def add_camera(self, camera_name: str, width: int, height: int):
        """Initialize video writer for a camera"""
        filepath = self.trial_dir / f"{camera_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(filepath),
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not writer.isOpened():
            print(f"Warning: Could not open video writer for {camera_name}")
            return
        
        self.writers[camera_name] = writer
        self.frame_counts[camera_name] = 0
        
        print(f"Video logger initialized: {camera_name} -> {filepath}")
    
    def log_frame(self, camera_name: str, frame: np.ndarray):
        """Log a single frame (expects BGR format)"""
        if camera_name not in self.writers:
            return
        
        self.writers[camera_name].write(frame)
        self.frame_counts[camera_name] += 1
    
    def close(self):
        """Release all video writers"""
        print("Closing video logger...")
        
        for camera_name, writer in self.writers.items():
            try:
                writer.release()
                print(f"Saved {self.frame_counts[camera_name]} frames for {camera_name}")
            except Exception as e:
                print(f"Error releasing writer for {camera_name}: {e}")
        
        self.writers.clear()
        self.frame_counts.clear()
        
        print("Video logger closed")