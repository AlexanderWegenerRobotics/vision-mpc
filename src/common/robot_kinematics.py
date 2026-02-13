import numpy as np
import pinocchio as pin

class RobotKinematics:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("panda_hand")

        print("Loaded kinematics model")

    def forward_kinematics(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self.ee_frame_id]
        position = pose.translation
        rotation = pin.rpy.matrixToRpy(pose.rotation)
        return np.concatenate([position, rotation])
    
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