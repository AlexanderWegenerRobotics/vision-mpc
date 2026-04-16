import numpy as np
from scipy.spatial.transform import Rotation


def update_slider_frame(system, config, x_slider):
    """Move the debug slider_frame body to match current slider pose."""
    slider_z = config["surface_height_world"] + config["slider_half_z"]
    pos = np.array([x_slider[0], x_slider[1], slider_z])
    q = Rotation.from_euler("z", x_slider[2]).as_quat()
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]])
    system.sim.reset_object_pose("slider_frame", pos, quat_wxyz)


def update_vel_arrow(system, ee_pose, ee_vel_world, z_contact):
    """Update the velocity arrow visualisation in MuJoCo."""
    vel_2d = ee_vel_world[:2]
    speed  = np.linalg.norm(vel_2d)
    if speed < 1e-4:
        return

    half_len = np.clip(speed * 1.0, 0.02, 0.2)
    geom_id = system.sim.mj_model.geom("vel_arrow/vel_arrow_shaft").id
    system.sim.mj_model.geom_size[geom_id] = [0.005, half_len, 0]

    pos = ee_pose.position.copy()
    pos[2] = z_contact

    vel_dir = np.array([vel_2d[0], vel_2d[1], 0.0]) / speed
    z_axis  = np.array([0.0, 0.0, 1.0])
    axis    = np.cross(z_axis, vel_dir)
    angle   = np.arccos(np.clip(np.dot(z_axis, vel_dir), -1, 1))

    if np.linalg.norm(axis) < 1e-6:
        quat_xyzw = np.array([0, 0, 0, 1])
    else:
        axis = axis / np.linalg.norm(axis)
        quat_xyzw = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])

    quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
    pos += vel_dir * half_len
    system.sim.reset_object_pose("vel_arrow", pos, quat_wxyz)