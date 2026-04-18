import numpy as np
import matplotlib.pyplot as plt
from mpc import PusherSliderModel, PusherSliderNMPC, Face
from simcore import load_yaml

def make_circle_ref(cx, cy, radius, n_steps, dt, speed=0.02, start_angle=-np.pi/2):
    # Circle reference: slider center on circle, theta tangent to path, p_y=0.
    arc_per_step = speed * dt
    total_arc    = n_steps * arc_per_step
    angles       = np.linspace(start_angle, start_angle + total_arc / radius, n_steps + 1)
    ref = np.zeros((n_steps + 1, 4))
    ref[:, 0] = cx + radius * np.cos(angles)
    ref[:, 1] = cy + radius * np.sin(angles)
    ref[:, 2] = angles + np.pi / 2
    ref[:, 3] = 0.0
    return ref

def make_straight_ref(x0_nominal, n_steps, speed_per_step=0.02):
    ref = np.zeros((n_steps + 1, 4))
    for i in range(n_steps + 1):
        ref[i] = x0_nominal.copy()
        ref[i, 0] = x0_nominal[0] + i * speed_per_step
    return ref

def rk4_step(model, x, u, dt):
    k1 = model.evaluate(x, u)
    k2 = model.evaluate(x + dt/2 * k1, u)
    k3 = model.evaluate(x + dt/2 * k2, u)
    k4 = model.evaluate(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def saturate_py(x, half_face):
    x[3] = np.clip(x[3], -half_face, half_face)
    return x

def run_sim(mpc: PusherSliderNMPC, model: PusherSliderModel, x0, full_ref, n_sim, half_face):
    dt     = mpc.params["dt"]
    T      = mpc.T
    x      = x0.copy()
    xs, us, statuses, errors = [x.copy()], [], [], []

    for k in range(n_sim):
        i_start = k
        i_end   = min(k + T + 1, len(full_ref))
        ref_win = full_ref[i_start:i_end]
        if len(ref_win) < T + 1:
            pad = np.tile(ref_win[-1], (T + 1 - len(ref_win), 1))
            ref_win = np.vstack([ref_win, pad])

        u, status = mpc.solve(x, ref_win)
        x         = rk4_step(model, x, u, dt)
        x         = saturate_py(x, half_face)

        xs.append(x.copy())
        us.append(u.copy())
        statuses.append(status)
        errors.append(np.linalg.norm(x[:2] - full_ref[min(k+1, len(full_ref)-1), :2]))

    return np.array(xs), np.array(us), np.array(statuses), np.array(errors)

def plot_results(xs, us, ref, statuses, errors, dt, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(ref[:, 0], ref[:, 1], "k--", label="ref")
    ax.plot(xs[:, 0],  xs[:, 1],  "b-",  label="actual")
    ax.plot(xs[0, 0],  xs[0, 1],  "go",  label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.legend(); ax.axis("equal"); ax.grid(True)

    ax = axes[0, 1]
    t  = np.arange(len(xs)) * dt
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

def validate_sim_tracking(mpc: PusherSliderNMPC, model: PusherSliderModel):
    dt        = mpc.params["dt"]
    half_face = mpc.params["slider_half_y"]
    N_sim     = 150

    # Straight line with lateral + angular perturbation (canonical face).
    mpc.set_face(Face.NEG_X)
    mpc.reset()
    x0_nominal   = np.array([0.0, 0.0, 0.0, 0.0])
    x0_straight  = np.array([0.0, 0.015, np.deg2rad(8.0), 0.0])
    ref_straight = make_straight_ref(x0_nominal, N_sim + mpc.T, speed_per_step=dt * 0.05)
    xs, us, statuses, errors = run_sim(mpc, model, x0_straight, ref_straight, N_sim, half_face)
    plot_results(xs, us, ref_straight, statuses, errors, dt, "Straight Line Tracking")
    print(f"Straight: mean_error={errors.mean()*1000:.2f}mm  max_error={errors.max()*1000:.2f}mm  "
          f"solver_failures={np.sum(statuses != 0)}/{N_sim}")

    # Circle with feasible radius. R_min ~= v / omega_max; for our geometry R=0.2 m
    # gives ~2.5x margin over the geometric minimum. Slider starts on the reference.
    mpc.reset()
    R        = 0.1
    v        = 0.02
    cx, cy   = 0.0, 0.0
    start_a  = -np.pi/2
    x0_circle = np.array([cx + R*np.cos(start_a), cy + R*np.sin(start_a), start_a + np.pi/2, 0.0])
    ref_circle = make_circle_ref(cx=cx, cy=cy, radius=R, n_steps=N_sim + mpc.T, dt=dt, speed=v, start_angle=start_a)
    xs, us, statuses, errors = run_sim(mpc, model, x0_circle, ref_circle, N_sim, half_face)
    plot_results(xs, us, ref_circle, statuses, errors, dt, "Circle Tracking")
    print(f"Circle:   mean_error={errors.mean()*1000:.2f}mm  max_error={errors.max()*1000:.2f}mm  "
          f"solver_failures={np.sum(statuses != 0)}/{N_sim}")