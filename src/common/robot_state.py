from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class RobotState:
    
    # Joint space
    q: np.ndarray           # Joint positions [rad or m]
    qd: np.ndarray          # Joint velocities [rad/s or m/s]
    qdd: Optional[np.ndarray] = None  # Joint accelerations [rad/s² or m/s²]
    tau: Optional[np.ndarray] = None  # Joint torques/forces [Nm or N]
    
    # Task space (might not be used for all systems)
    x: Optional[np.ndarray] = None    # End-effector pose SE3 in base frame
    xd: Optional[np.ndarray] = None   # End-effector velocity [vx, vy, vz, wx, wy, wz]
    
    # External forces (if available)
    f_ext: Optional[np.ndarray] = None  # External wrench at EE [fx, fy, fz, mx, my, mz] in base frame
    
    # Auxiliary
    time: float = 0.0       # Simulation/system time
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        self.q = np.asarray(self.q)
        self.qd = np.asarray(self.qd)
        if self.qdd is not None:
            self.qdd = np.asarray(self.qdd)
        if self.tau is not None:
            self.tau = np.asarray(self.tau)
        if self.x is not None:
            self.x = np.asarray(self.x)
        if self.xd is not None:
            self.xd = np.asarray(self.xd)
        if self.f_ext is not None:
            self.f_ext = np.asarray(self.f_ext)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'q': self.q,
            'qd': self.qd,
            'qdd': self.qdd,
            'tau': self.tau,
            'x': self.x,
            'xd': self.xd,
            'f_ext': self.f_ext,
            'time': self.time
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RobotState':
        """Create from dictionary"""
        return cls(
            q=data['q'],
            qd=data['qd'],
            qdd=data.get('qdd'),
            tau=data.get('tau'),
            x=data.get('x'),
            xd=data.get('xd'),
            f_ext=data.get('f_ext'),
            time=data.get('time', 0.0)
        )