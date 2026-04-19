from simcore import RobotSystem, load_yaml
import cv2
import numpy as np
# <camera name="eye_in_hand" pos="0 0.0555 0.043" euler="3.1416 0 3.1416" fovy="60"/>
'''
<body name="realsense_mount" pos="0 0 0.107" euler="0 0 -2.356">
                          <geom name="realsense_mount_geom" type="mesh" mesh="camera" group="2" contype="0" conaffinity="0" mass="0.05" rgba="0.35 0.35 0.35 1"/>
                          <camera name="eye_in_hand" pos="0 0 0" euler="3.1416 0 0" fovy="60"/>
                      </body>
'''

def main():
    config = load_yaml("configs/global_config.yaml")

    system = RobotSystem(config)
    system.set_controller_mode("arm", "position")
    system.set_target("arm", {"q":[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]})

    system.sim.start()   # start physics
    system.running = True

    K, dist, witdh, height, fovy = system.sim.get_camera_intrinsics("eye_in_hand")

    try:
        while True:
            system.step()  # advances simulation

            frame = system.get_camera_image("eye_in_hand", bgr=True)

            cv2.imshow("eye_in_hand", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        system.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()