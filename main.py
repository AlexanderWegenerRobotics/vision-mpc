from simcore import RobotSystem, Pose, load_yaml

cfg = load_yaml("configs/global_config.yaml")
system = RobotSystem(cfg)

#system.set_controller_mode("arm", "impedance")
#target = Pose(position=[0.5, 0.0, 0.8], quaternion=[0, 1, 0, 0])
#system.set_target("arm", {"x": target})

system.run()