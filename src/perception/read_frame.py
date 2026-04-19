from simcore import RobotSystem, load_yaml
import cv2
import numpy as np

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