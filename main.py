from simcore import RobotSystem, Pose, load_yaml

config = load_yaml("configs/global_config.yaml")
system = RobotSystem(config)
system.set_controller_mode("arm", "impedance")
target = Pose(position=[0.46, -0.07, 0.52], quaternion=[0, 1, 0, 0])
system.set_target("arm", {"x": target})

system.run()