import numpy as np
import pinocchio as pin
from src.common.pose import Pose


class RobotKinematics:
    def __init__(self, urdf_path: str, ee_frame_name: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        try:
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        except:
            print(f"Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}")
            raise ValueError(f"Frame '{ee_frame_name}' not found in URDF")
        
    def forward_kinematics(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self.ee_frame_id]
        return Pose.from_matrix(pose.translation.copy(), pose.rotation.copy())
    
    def get_jacobian(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J
    
    def get_ee_velocity(self, q, qd):
        J = self.get_jacobian(q)
        return J @ qd
    
    def get_gravity_torques(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return tau_g[:7]
    
    def get_mass_matrix(self, q):
        pin.crba(self.model, self.data, q)
        return self.data.M[:7, :7]

    def get_coriolis_matrix(self, q, qd):
        pin.computeCoriolisMatrix(self.model, self.data, q, qd)
        return self.data.C[:7, :7]