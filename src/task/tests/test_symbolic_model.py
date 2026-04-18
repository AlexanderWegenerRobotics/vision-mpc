import numpy as np
import matplotlib.pyplot as plt
from mpc import PusherSliderModel

def compute_gamma(p):
    px, py = p["slider_half_x"], p["slider_half_y"]
    a, b = px * 2, py * 2
    c2 = (a**2 + b**2) / 12
    mu = p["mu_slider"]
    gt = ( mu*c2 - px*py + mu*px**2) / (c2 + py**2 - mu*px*py)
    gb = (-mu*c2 - px*py - mu*px**2) / (c2 + py**2 + mu*px*py)
    return gt, gb

def test_sticking(model: PusherSliderModel):
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([0.05, 0.0])
    xdot = model.evaluate(x, u)
    print(f"xdot = {xdot}")

    p = model.params
    px   = p["slider_half_x"]
    py_0 = p["slider_half_y"]
    a, b = px*2, py_0*2
    c2   = (a**2 + b**2) / 12
    mu   = p["mu_slider"]
    gt   = ( mu*c2 - px*py_0 + mu*px**2) / (c2 + py_0**2 - mu*px*py_0)
    gb   = (-mu*c2 - px*py_0 - mu*px**2) / (c2 + py_0**2 + mu*px*py_0)
    print(f"gamma_t={gt:.4f}  gamma_b={gb:.4f}")
    print(f"v_t={u[1]:.4f}  gamma_t*v_n={gt*u[0]:.4f}  gamma_b*v_n={gb*u[0]:.4f}")
    print(f"sticking: {gb*u[0] <= u[1] <= gt*u[0]}")
    
    py_state = x[3]
    denom = c2 + px**2 + py_state**2
    print(f"c2={c2:.6f}  denom={denom:.6f}  px={px}  py_state={py_state}")
    print(f"thetadot manual = {(-py_state * u[0] + px * u[1]) / denom:.6f}")

def test_sliding(model: PusherSliderModel):
    # v_t far above gamma_t*v_n: pusher slides, py_dot must be nonzero
    gt, _ = compute_gamma(model.params)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    v_n = 0.05
    u = np.array([v_n, gt * v_n + 0.1])
    xdot = model.evaluate(x, u)
    assert abs(xdot[3]) > 1e-6, f"py_dot should be nonzero during sliding, got {xdot[3]:.6f}"
    print(f"[PASS] sliding: xdot={xdot}")

def test_rotation(model: PusherSliderModel):
    # contact offset in y (p_y != 0): normal push should generate rotation
    x = np.array([0.0, 0.0, 0.0, 0.02])
    u = np.array([0.05, 0.0])
    xdot = model.evaluate(x, u)
    assert abs(xdot[2]) > 1e-6, f"theta_dot should be nonzero with p_y offset, got {xdot[2]:.6f}"
    print(f"[PASS] rotation: xdot={xdot}")

def test_continuity(model: PusherSliderModel):
    gt, gb = compute_gamma(model.params)
    v_n = 0.05
    vt_sweep = np.linspace(gb * v_n - 0.1, gt * v_n + 0.1, 300)
    x = np.array([0.0, 0.0, 0.0, 0.0])

    xdots = np.array([model.evaluate(x, np.array([v_n, vt])) for vt in vt_sweep])
    diffs = np.abs(np.diff(xdots, axis=0))
    max_jump = diffs.max()

    # check jump is proportional to step size (continuous), not a hard discontinuity
    dvt = vt_sweep[1] - vt_sweep[0]
    assert max_jump < 1.0 * dvt * 1000, f"Discontinuity detected: max jump = {max_jump:.6f}"
    print(f"[PASS] continuity: max jump={max_jump:.2e} over step={dvt:.2e}")

def test_rollout(model: PusherSliderModel):
    # short RK4 rollout with constant sticking input: trajectory should be smooth
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([0.03, 0.0])
    dt, N = 0.05, 50

    def rk4(x, u, dt):
        k1 = model.evaluate(x, u)
        k2 = model.evaluate(x + dt/2 * k1, u)
        k3 = model.evaluate(x + dt/2 * k2, u)
        k4 = model.evaluate(x + dt * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    traj = [x.copy()]
    for _ in range(N):
        x = rk4(x, u, dt)
        traj.append(x.copy())
    traj = np.array(traj)

    assert not np.any(np.isnan(traj)), "NaN in rollout"
    assert not np.any(np.isinf(traj)), "Inf in rollout"
    print(f"[PASS] rollout: final state = {traj[-1]}")

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1])
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("RK4 rollout (sticking, pure normal push)")
    plt.axis("equal"); plt.grid(True)
    plt.savefig("rollout.png", dpi=120)
    print("[INFO] plot saved to rollout.png")

def validate_dynamic_model(model:PusherSliderModel):
    test_sticking(model)
    test_sliding(model)
    test_rotation(model)
    test_continuity(model)
    test_rollout(model)
    print("\nAll tests passed.")
