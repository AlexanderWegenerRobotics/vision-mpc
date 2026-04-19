from simcore import RobotSystem, Pose, load_yaml
import threading
from src.pusher_slider_controller import PusherSliderController

def main():
    config = load_yaml("configs/global_config.yaml")
    system = RobotSystem(config)
    system.set_controller_mode("arm", "position")
    system.set_target("arm", {"q":[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]})

    task_config = load_yaml(config.get("task_config"))
    controller = PusherSliderController(system=system, config=task_config)

    task_thread = threading.Thread(target=controller.loop, daemon=True)
    task_thread.start()

    try:
        system.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()


if __name__ == "__main__":
    main()