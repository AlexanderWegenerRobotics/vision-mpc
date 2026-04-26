import argparse
import threading
from simcore import RobotSystem, Pose, load_yaml
from src.pusher_slider_controller import PusherSliderController


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",     default=None, choices=["BASELINE", "CERTAINTY_EQUIV", "UNCERTAINTY_AWARE"])
    p.add_argument("--disturbance", default="configs/disturbance_config.yaml")
    p.add_argument("--n-seeds",     default=None, type=int)
    return p.parse_args()


def main():
    args = parse_args()

    config      = load_yaml("configs/global_config.yaml")
    task_config = load_yaml(config["task_config"])
    study_cfg   = load_yaml(config["study_config"])
    dist_cfg    = load_yaml(args.disturbance)

    if args.variant is not None:
        task_config["mpc"]["variant"] = args.variant
    if args.n_seeds is not None:
        task_config["n_seeds_override"] = args.n_seeds

    task_config["disturbance_config"] = dist_cfg
    config["_task_resolved"]          = task_config
    config["_study_resolved"]         = study_cfg

    system = RobotSystem(config)
    system.set_controller_mode("arm", "position")
    system.set_target("arm", {"q": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]})

    controller = PusherSliderController(system=system, config=config)

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