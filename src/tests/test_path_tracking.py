import numpy as np
import matplotlib.pyplot as plt
from mpc import PusherSliderModel, PusherSliderNMPC, Face
from path_planner import StraightLinePlanner, CircularPlanner
from simcore import load_yaml


def rk4_step(model, x, u, dt):
    k1 = model.evaluate(x, u)
    k2 = model.evaluate(x + dt/2 * k1, u)
    k3 = model.evaluate(x + dt/2 * k2, u)
    k4 = model.evaluate(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def saturate_py(x, half_face):
    x[3] = np.clip(x[3], -half_face, half_face)
    return x


def run_episode(mpc: PusherSliderNMPC, model: PusherSliderModel, planner, start, goal, n_sim, face=Face.NEG_X):
    dt        = mpc.params["dt"]
    half_face = mpc.params["slider_half_y"]
    mpc.set_face(face)
    mpc.reset()

    full_ref = planner.plan(start=start, goal=goal, n_steps=n_sim, dt=dt)

    x = start.copy()
    xs, us, statuses, errors = [x.copy()], [], [], []

    for k in range(n_sim):
        ref_win   = planner.window(full_ref, step=k, horizon=mpc.T)
        u, status = mpc.solve(x, ref_win)
        x         = rk4_step(model, x, u, dt)
        x         = saturate_py(x, half_face)

        xs.append(x.copy())
        us.append(u.copy())
        statuses.append(status)
        errors.append(np.linalg.norm(x[:2] - full_ref[min(k + 1, len(full_ref) - 1), :2]))

    return np.array(xs), np.array(us), np.array(statuses), np.array(errors), full_ref


def plot_results(xs, us, ref, statuses, errors, dt, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(ref[:, 0], ref[:, 1], "k--", label="ref")
    ax.plot(xs[:, 0],  xs[:, 1],  "b-",  label="actual")
    ax.plot(xs[0, 0],  xs[0, 1],  "go",  label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.legend(); ax.axis("equal"); ax.grid(True)

    t = np.arange(len(xs)) * dt
    ax = axes[0, 1]
    ax.plot(t, np.degrees(ref[:len(xs), 2]), "k--", label="ref")
    ax.plot(t, np.degrees(xs[:, 2]), "b-", label="actual")
    ax.set_xlabel("t [s]"); ax.set_ylabel("theta [deg]")
    ax.set_title("Orientation"); ax.legend(); ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t, xs[:, 3], "b-")
    ax.set_xlabel("t [s]"); ax.set_ylabel("p_y [m]")
    ax.set_title("Contact position p_y"); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t[1:], errors)
    ax.set_xlabel("t [s]"); ax.set_ylabel("||e_xy|| [m]")
    ax.set_title("XY tracking error"); ax.grid(True)

    ax = axes[1, 1]
    t_u = t[:-1]
    ax.plot(t_u, us[:, 0], label="v_n")
    ax.plot(t_u, us[:, 1], label="v_t")
    ax.set_xlabel("t [s]"); ax.set_ylabel("velocity [m/s]")
    ax.set_title("Control inputs"); ax.legend(); ax.grid(True)

    ax = axes[1, 2]
    ax.step(t_u, statuses, where="post")
    ax.set_xlabel("t [s]"); ax.set_ylabel("status")
    ax.set_title("Solver status (0=OK, 2=max iter, 3=min step)")
    ax.set_yticks([0, 1, 2, 3, 4]); ax.grid(True)

    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=120)
    print(f"[INFO] saved {fname}")


def validate_path_tracking(mpc: PusherSliderNMPC, model: PusherSliderModel):
    dt    = mpc.params["dt"]
    N_sim = 150

    start = np.array([0.0, 0.015, np.deg2rad(8.0), 0.0])
    goal  = np.array([0.30, 0.0, 0.0, 0.0])
    planner = StraightLinePlanner()
    xs, us, statuses, errors, ref = run_episode(mpc, model, planner, start, goal, N_sim)
    plot_results(xs, us, ref, statuses, errors, dt, "Path Tracking Straight")
    print(f"Straight: mean_error={errors.mean()*1000:.2f}mm  max_error={errors.max()*1000:.2f}mm  "
          f"solver_failures={np.sum(statuses != 0)}/{N_sim}")

    R        = 0.20
    center   = np.array([0.0, 0.0])
    start    = np.array([0.0, -R, 0.0, 0.0])
    goal     = np.array([0.0,  R, np.pi, 0.0])
    planner  = CircularPlanner(center=center, radius=R, direction="ccw")
    xs, us, statuses, errors, ref = run_episode(mpc, model, planner, start, goal, N_sim)
    plot_results(xs, us, ref, statuses, errors, dt, "Path Tracking Circle")
    print(f"Circle:   mean_error={errors.mean()*1000:.2f}mm  max_error={errors.max()*1000:.2f}mm  "
          f"solver_failures={np.sum(statuses != 0)}/{N_sim}")


if __name__ == "__main__":
    config      = load_yaml("configs/global_config.yaml")
    task_config = load_yaml(config.get("task_config"))
    model       = PusherSliderModel(config=task_config)
    mpc         = PusherSliderNMPC(model=model, config=task_config)
    validate_path_tracking(mpc=mpc, model=model)