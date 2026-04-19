import numpy as np
from mpc import PusherSliderModel, PusherSliderNMPC

def make_straight_ref(x0, N, v=0.01, dt=0.05):
    """Build a straight-line reference trajectory of length N+1 moving in x."""
    ref = np.zeros((N + 1, 4))
    for i in range(N + 1):
        ref[i] = x0.copy()
        ref[i, 0] += i * v * dt
    return ref

def test_solver_builds(mpc:PusherSliderNMPC, model: PusherSliderModel):
    """Solver instantiation should complete without errors and produce a JSON file."""
    assert mpc._solver is not None
    print("[PASS] solver builds")

def test_trivial_solve(mpc: PusherSliderNMPC, model: PusherSliderModel):
    """Already at goal: x0 == x_ref everywhere. Expect status 0 and near-zero state error."""
    x0  = np.zeros(4)
    ref = np.tile(x0, (mpc.T + 1, 1))
    u0, status = mpc.solve(x0, ref)
    assert status == 0, f"Solver failed with status {status}"

    # apply u0 one step and check state stays near goal
    dt   = mpc.params["dt"]
    xnew = x0 + dt * model.evaluate(x0, u0)
    assert np.linalg.norm(xnew[:3]) < 0.01, f"State drifted too far: {xnew}"
    print(f"[PASS] trivial solve: u0={u0}, status={status}, xnew={xnew}")

def test_straight_line_tracking(mpc:PusherSliderNMPC, model: PusherSliderModel):
    """Start offset from a straight-line reference. Expect v_n > 0 and status 0."""
    x0  = np.array([-0.02, 0.0, 0.0, 0.0])
    ref = make_straight_ref(np.zeros(4), mpc.T)
    u0, status = mpc.solve(x0, ref)
    assert status == 0,  f"Solver failed with status {status}"
    assert u0[0] > 0,    f"Expected v_n > 0 to close x error, got {u0[0]:.4f}"
    print(f"[PASS] straight line tracking: u0={u0}, status={status}")

def test_input_constraints(mpc: PusherSliderNMPC, model: PusherSliderModel):
    x0  = np.array([-0.05, 0.02, 0.1, 0.01])
    ref = make_straight_ref(np.zeros(4), mpc.T)
    u0, status = mpc.solve(x0, ref)
    v_n_max = mpc.params["v_n_max"]
    v_t_max = mpc.params["v_t_max"]
    v_n_min = mpc.params["v_n_min"]
    assert u0[0] >= v_n_min - 1e-4, f"v_n below min: {u0[0]:.6f}"
    assert u0[0] <= v_n_max + 1e-4, f"v_n above max: {u0[0]:.6f}"
    assert abs(u0[1]) <= v_t_max + 1e-4, f"|v_t| above max: {u0[1]:.6f}"
    print(f"[PASS] input constraints: u0={u0}")

def test_cost_decreases(mpc:PusherSliderNMPC, model: PusherSliderModel):
    """Over successive MPC calls from a fixed offset, cost should decrease monotonically."""
    x   = np.array([-0.05, 0.0, 0.0, 0.0])
    ref = make_straight_ref(np.zeros(4), mpc.T)
    dt = mpc.params["dt"]
    costs = []

    def rk4(x, u):
        k1 = model.evaluate(x, u)
        k2 = model.evaluate(x + dt/2 * k1, u)
        k3 = model.evaluate(x + dt/2 * k2, u)
        k4 = model.evaluate(x + dt * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    for _ in range(10):
        u0, status = mpc.solve(x, ref)
        assert status == 0, f"Solver failed with status {status}"
        costs.append(np.linalg.norm(x[:3]))
        x = rk4(x, u0)

    assert costs[-1] < costs[0], f"Cost did not decrease: {costs[0]:.4f} -> {costs[-1]:.4f}"
    print(f"[PASS] cost decreases: {costs[0]:.4f} -> {costs[-1]:.4f}")

def test_state_constraint(mpc: PusherSliderNMPC, model: PusherSliderModel):
    """Solver's predicted p_y must stay within bounds at all horizon nodes."""
    half_y = mpc.params["slider_half_y"]
    x0  = np.array([0.0, 0.0, 0.0, half_y * 0.2])
    ref = make_straight_ref(np.zeros(4), mpc.T)
    mpc.reset()
    _, status = mpc.solve(x0, ref)
    assert status == 0, f"Solver failed with status {status}"

    for t in range(1, mpc.T + 1):
        x_t = mpc._solver.get(t, "x")
        assert abs(x_t[3]) <= half_y + 1e-4, f"p_y={x_t[3]:.4f} out of bounds at node {t}"

    print(f"[PASS] state constraint: predicted p_y stayed within ±{half_y}")

def test_warm_start(mpc:PusherSliderNMPC, model: PusherSliderModel):
    """Second solve from same state should have equal or lower cost than first (cold) solve."""
    x0  = np.array([-0.03, 0.01, 0.05, 0.0])
    ref = make_straight_ref(np.zeros(4), mpc.T)

    mpc.reset()
    _, status1 = mpc.solve(x0, ref)
    cost1 = mpc._solver.get_cost()

    _, status2 = mpc.solve(x0, ref)
    cost2 = mpc._solver.get_cost()

    assert status1 == 0 and status2 == 0
    assert cost2 <= cost1 + 1e-6, f"Warm start cost {cost2:.4f} > cold cost {cost1:.4f}"
    print(f"[PASS] warm start: cold={cost1:.4f} warm={cost2:.4f}")

def test_reset(mpc:PusherSliderNMPC, model: PusherSliderModel):
    """After reset(), first solve should behave identically to a fresh cold start."""
    x0  = np.array([-0.03, 0.0, 0.0, 0.0])
    ref = make_straight_ref(np.zeros(4), mpc.T)

    u_first, s1 = mpc.solve(x0, ref)
    mpc.reset()
    u_reset, s2 = mpc.solve(x0, ref)

    assert s1 == 0 and s2 == 0
    assert np.allclose(u_first, u_reset, atol=1e-6), \
        f"Reset gave different result: {u_first} vs {u_reset}"
    print(f"[PASS] reset: u={u_reset}, status={s2}")

def validate_mpc(mpc: PusherSliderNMPC, model: PusherSliderModel):
    test_solver_builds(mpc=mpc, model=model)
    test_trivial_solve(mpc=mpc, model=model)
    test_straight_line_tracking(mpc=mpc, model=model)
    test_input_constraints(mpc=mpc, model=model)
    test_cost_decreases(mpc=mpc, model=model)
    test_state_constraint(mpc=mpc, model=model)
    test_warm_start(mpc=mpc, model=model)
    test_reset(mpc=mpc, model=model)
    print("\nAll MPC tests passed.")
