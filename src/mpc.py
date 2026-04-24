import casadi as ca
import numpy as np
from enum import Enum, auto
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from simcore import load_yaml

# Face convention:
#   MPC always operates as if the pusher contacts the -x_S face of the slider.
#   p_x = -slider_half_x  (signed, negative, constant)
#   p_y in [-slider_half_y, +slider_half_y]  (state, position along face)
#   Contact frame F_C: n_hat = +x_hat_S (into slider), t_hat = +y_hat_S
#   v_n >= 0 means pushing into the slider.
#   For non-canonical real faces, use set_face() + world_to_canonical() /
#   canonical_to_world() to rotate states and commands.

class ControllerVariant(Enum):
    BASELINE          = auto()
    CERTAINTY_EQUIV   = auto()
    UNCERTAINTY_AWARE = auto()

class Face(Enum):
    NEG_X = 0   # canonical, pusher on -x_S face, n_hat = +x_hat_S
    POS_Y = 1   # pusher on +y_S face,            n_hat = -y_hat_S
    POS_X = 2   # pusher on +x_S face,            n_hat = -x_hat_S
    NEG_Y = 3   # pusher on -y_S face,            n_hat = +y_hat_S

# Rotation that takes real body axes into canonical body axes (where the active
# face is -x_S). For each face, canonical +x_S should point in the direction the
# slider should translate under "push":
#   NEG_X (pusher on real -x): canonical +x = real +x,  alpha = 0
#   POS_Y (pusher on real +y): canonical +x = real -y,  alpha = -pi/2
#   POS_X (pusher on real +x): canonical +x = real -x,  alpha = +pi
#   NEG_Y (pusher on real -y): canonical +x = real +y,  alpha = +pi/2
# theta_canonical = theta_real + alpha.
# Keyed by enum .value (int) to avoid identity mismatches when Face is imported
# from a module loaded twice (e.g. mpc.py as __main__ and as `import mpc`).
_FACE_ANGLES = {0: 0.0, 1: -np.pi/2, 2: np.pi, 3: np.pi/2}


class PusherSliderModel:
    def __init__(self, config):
        self.params = {
            "slider_half_x": config["slider_half_x"],
            "slider_half_y": config["slider_half_y"],
            "mu_slider":     config["mpc"]["mu_slider"],
        }
        self._build_symbols()
        self._build_dynamics()

    def _build_symbols(self):
        self.x_sym = ca.MX.sym("x", 4)
        self.u_sym = ca.MX.sym("u", 2)
        self.x_s, self.y_s     = self.x_sym[0], self.x_sym[1]
        self.theta_s, self.p_y = self.x_sym[2], self.x_sym[3]
        self.v_n, self.v_t     = self.u_sym[0], self.u_sym[1]

    def _build_dynamics(self):
        # Canonical frame: pusher on -x_S face, so p_x is signed negative.
        p   = self.params
        p_x = -p["slider_half_x"]
        c2  = (4*p["slider_half_x"]**2 + 4*p["slider_half_y"]**2) / 12
        mu  = p["mu_slider"]
        py  = self.p_y

        gt = ( mu*c2 - p_x*py + mu*p_x**2) / (c2 + py**2 - mu*p_x*py)
        gb = (-mu*c2 - p_x*py - mu*p_x**2) / (c2 + py**2 + mu*p_x*py)

        beta = 800.0  # sharpness parameter
        w_upper = 0.5 * (1 + ca.tanh(beta * (self.v_t - gt * self.v_n)))
        w_lower = 0.5 * (1 + ca.tanh(beta * (gb * self.v_n - self.v_t)))
        w_stick = 1 - w_upper - w_lower
        vc_n = self.v_n
        vc_t = w_stick * self.v_t + w_upper * (gt * self.v_n) + w_lower * (gb * self.v_n)

        denom    = c2 + p_x**2 + py**2
        xdot_b   = ((c2 + p_x**2) * vc_n + p_x * py * vc_t) / denom
        ydot_b   = (p_x * py * vc_n + (c2 + py**2) * vc_t) / denom
        thetadot = (-py * vc_n + p_x * vc_t) / denom
        py_dot   = self.v_t - vc_t

        cos_th = ca.cos(self.theta_s)
        sin_th = ca.sin(self.theta_s)

        self.f_expr = ca.vertcat(
            cos_th * xdot_b - sin_th * ydot_b,
            sin_th * xdot_b + cos_th * ydot_b,
            thetadot,
            py_dot,
        )
        self.f_func = ca.Function("f", [self.x_sym, self.u_sym], [self.f_expr], ["x", "u"], ["xdot"])

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self.f_func(x, u)).flatten()


class PusherSliderNMPC:
    # Constrained state layout for chance-constraint tightening:
    #   idxbx = [0, 1, 3]  -> x_S, y_S, p_y
    #   theta is not box-constrained; keep-out bounds on x_S/y_S default to +-1e6
    #   (effectively non-binding) until a scenario sets tighter values.
    _CC_STATE_IDX = np.array([0, 1, 3], dtype=int)

    def __init__(self, model: PusherSliderModel, config: dict):
        self.model   = model
        mpc_cfg      = config["mpc"]
        self.variant = ControllerVariant[mpc_cfg["variant"]]
        self.params  = {
            "horizon":             mpc_cfg["horizon"],
            "dt":                  mpc_cfg["dt"],
            "Q":                   mpc_cfg["Q"],
            "R":                   mpc_cfg["R"],
            "Q_terminal_scale":    mpc_cfg["Q_terminal_scale"],
            "v_n_max":             mpc_cfg["v_n_max"],
            "v_t_max":             mpc_cfg["v_t_max"],
            "slider_half_x":       config["slider_half_x"],
            "slider_half_y":       config["slider_half_y"],
            "nlp_solver_max_iter": mpc_cfg.get("nlp_solver_max_iter", 20),
        }
        self.T  = self.params["horizon"]
        self.nx = model.nx
        self.nu = model.nu
        self._face = Face.NEG_X

        self._cc_beta       = float(mpc_cfg.get("cc_beta", 1.645))
        self._cc_min_half_y = float(mpc_cfg.get("cc_min_half_y", 0.01))
        self._ekf_Q         = np.array(mpc_cfg["ekf_Q"], dtype=float)

        self._lbx_nom = None
        self._ubx_nom = None
        self._build_solver()
        self._build_cc_jacobian()
        self._initialized = False

    def _build_solver(self):
        p            = self.params
        acados_model = AcadosModel()
        acados_model.name        = "pusher_slider"
        acados_model.x           = self.model.x_sym
        acados_model.u           = self.model.u_sym
        acados_model.f_expl_expr = self.model.f_expr

        ocp       = AcadosOcp()
        ocp.model = acados_model
        ocp.solver_options.N_horizon = self.T
        ocp.solver_options.tf        = self.T * p["dt"]

        Q_mat = np.diag(np.array(p["Q"], dtype=float))
        R_mat = np.diag(np.array(p["R"], dtype=float))
        ny    = self.nx + self.nu
        ny_e  = self.nx

        ocp.cost.cost_type   = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        Vx = np.zeros((ny, self.nx));  Vx[:self.nx, :] = np.eye(self.nx)
        Vu = np.zeros((ny, self.nu));  Vu[self.nx:, :] = np.eye(self.nu)
        ocp.cost.Vx   = Vx
        ocp.cost.Vu   = Vu
        ocp.cost.Vx_e = np.eye(ny_e)
        ocp.cost.W    = np.block([[Q_mat, np.zeros((self.nx, self.nu))],
                                   [np.zeros((self.nu, self.nx)), R_mat]])
        ocp.cost.W_e    = p["Q_terminal_scale"] * Q_mat
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.lbu   = np.array([0.0, -p["v_t_max"]])
        ocp.constraints.ubu   = np.array([p["v_n_max"],  p["v_t_max"]])
        ocp.constraints.idxbu = np.array([0, 1], dtype=int)

        # State box: [x_S, y_S, p_y]. x_S/y_S default to effectively unconstrained.
        big = 1e6
        half_y = p["slider_half_y"]
        lbx = np.array([-big, -big, -half_y])
        ubx = np.array([ big,  big,  half_y])
        ocp.constraints.lbx   = lbx
        ocp.constraints.ubx   = ubx
        ocp.constraints.idxbx = self._CC_STATE_IDX.copy()

        ocp.constraints.lbx_0 = np.zeros(self.nx, dtype=float)
        ocp.constraints.ubx_0 = np.zeros(self.nx, dtype=float)
        ocp.constraints.idxbx_0 = np.arange(self.nx, dtype=int)

        ocp.solver_options.integrator_type     = "ERK"
        ocp.solver_options.num_stages          = 4
        ocp.solver_options.num_steps           = 1
        ocp.solver_options.nlp_solver_type     = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = p["nlp_solver_max_iter"]
        ocp.solver_options.qp_solver           = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N    = max(1, self.T // 4)
        ocp.solver_options.hessian_approx      = "GAUSS_NEWTON"
        ocp.solver_options.print_level         = 0

        self._solver  = AcadosOcpSolver(ocp, json_file="pusher_slider_ocp.json")
        self._Q_nom   = Q_mat.copy()
        self._R_mat   = R_mat.copy()
        self._lbx_nom = lbx.copy()
        self._ubx_nom = ubx.copy()

    def _build_cc_jacobian(self):
        # Discrete-time 4x4 Jacobian F = d(x + dt*f)/dx for covariance propagation.
        dt     = self.params["dt"]
        x_next = self.model.x_sym + dt * self.model.f_expr
        F_expr = ca.jacobian(x_next, self.model.x_sym)
        self._F_full_fn = ca.Function("F_full", [self.model.x_sym, self.model.u_sym], [F_expr])

        self._Q3 = np.diag(self._ekf_Q)

    def set_face(self, face: Face):
        # Selects which slider face is in contact for this episode. MPC always
        # solves in the canonical (-x_S) frame; this just updates the rotation
        # used by world_to_canonical / canonical_to_world.
        self._face = face
        self._initialized = False

    def set_state_keepout(self, lbx_world: np.ndarray, ubx_world: np.ndarray):
        # Set nominal box bounds on [x_S, y_S, p_y] (world frame for xy).
        # Use +-1e6 entries to leave a dimension effectively unconstrained.
        self._lbx_nom = np.array(lbx_world, dtype=float).copy()
        self._ubx_nom = np.array(ubx_world, dtype=float).copy()
        self._reset_stage_constraints()

    def _face_angle(self) -> float:
        return _FACE_ANGLES[self._face.value]

    def world_to_canonical(self, x_world: np.ndarray) -> np.ndarray:
        # Rotate slider state so the active face becomes the -x_S face.
        # xy stays in world; only theta shifts by -face_angle. p_y is already
        # expressed in the slider frame and, by our convention, already refers
        # to position along the active face, so it passes through unchanged.
        x_can = x_world.copy()
        x_can[2] = x_world[2] + self._face_angle()
        return x_can

    def canonical_to_world(self, x_can: np.ndarray) -> np.ndarray:
        x_world = x_can.copy()
        x_world[2] = x_can[2] - self._face_angle()
        return x_world

    def ref_world_to_canonical(self, ref_world: np.ndarray) -> np.ndarray:
        ref_can = ref_world.copy()
        ref_can[:, 2] = ref_world[:, 2] + self._face_angle()
        return ref_can

    def solve(self, x0_world: np.ndarray, ref_world: np.ndarray, P: np.ndarray = None):
        x0_can  = self.world_to_canonical(x0_world)
        ref_can = self.ref_world_to_canonical(ref_world)

        self._solver.constraints_set(0, "lbx", x0_can)
        self._solver.constraints_set(0, "ubx", x0_can)

        for t in range(self.T):
            self._solver.cost_set(t, "yref", np.concatenate([ref_can[t], np.zeros(self.nu)]))
        self._solver.cost_set(self.T, "yref", ref_can[self.T])

        if self.variant == ControllerVariant.UNCERTAINTY_AWARE and P is not None:
            self._apply_chance_constraints(x0_can, P)
        else:
            self._reset_stage_constraints()

        if self._initialized:
            for t in range(self.T - 1):
                self._solver.set(t, "x", self._solver.get(t + 1, "x"))
                self._solver.set(t, "u", self._solver.get(t + 1, "u"))
            self._solver.set(self.T - 1, "u", self._solver.get(self.T - 2, "u"))
            self._solver.set(self.T,     "x", self._solver.get(self.T - 1, "x"))

        status = self._solver.solve()
        self._initialized = (status == 0)
        return self._solver.get(0, "u"), status

    def _apply_chance_constraints(self, x0_can: np.ndarray, P_ekf: np.ndarray):
        # Propagate 3x3 EKF covariance along horizon; tighten world-frame
        # [x_S, y_S] box bounds by beta * sigma per stage. p_y stays at nominal.
        beta = self._cc_beta
        xy_slots = np.array([0, 1], dtype=int)

        P = P_ekf.copy()
        F_xy = np.eye(4)

        for t in range(1, self.T):
            x_t = self._solver.get(t, "x") if self._initialized else x0_can
            u_t = self._solver.get(max(t - 1, 0), "u") if self._initialized else np.zeros(self.nu)
            F4  = np.array(self._F_full_fn(x_t, u_t))
            F_xy[:3, :3] = F4[:3, :3]
            P = F_xy[:3, :3] @ P @ F_xy[:3, :3].T + self._Q3

            sigma_xy = np.sqrt(np.clip(np.diag(P)[:2], 0.0, None))
            lbx_t = self._lbx_nom.copy()
            ubx_t = self._ubx_nom.copy()
            lbx_t[xy_slots] = self._lbx_nom[xy_slots] + beta * sigma_xy
            ubx_t[xy_slots] = self._ubx_nom[xy_slots] - beta * sigma_xy

            self._solver.constraints_set(t, "lbx", lbx_t)
            self._solver.constraints_set(t, "ubx", ubx_t)

    def _reset_stage_constraints(self):
        for t in range(1, self.T):
            self._solver.constraints_set(t, "lbx", self._lbx_nom)
            self._solver.constraints_set(t, "ubx", self._ubx_nom)

    def reset(self, x0_world: np.ndarray = None, ref_world: np.ndarray = None):
        self._initialized = False
        self._reset_stage_constraints()
        if x0_world is not None and ref_world is not None:
            self._seed_warm_start(x0_world, ref_world)

    def _seed_warm_start(self, x0_world: np.ndarray, ref_world: np.ndarray):
        # Seed solver with linear interpolation from x0 toward reference trajectory.
        x0_4d   = x0_world if len(x0_world) == 4 else np.append(x0_world, 0.0)
        x0_can  = self.world_to_canonical(x0_4d)
        ref_can = self.ref_world_to_canonical(ref_world)
        u_init  = np.array([self.params["v_n_max"] * 0.1, 0.0])
        for t in range(self.T + 1):
            alpha = t / self.T
            x_t   = (1 - alpha) * x0_can + alpha * ref_can[min(t, len(ref_can) - 1)]
            self._solver.set(t, "x", x_t)
        for t in range(self.T):
            self._solver.set(t, "u", u_init)
        self._initialized = True


if __name__ == "__main__":
    config      = load_yaml("configs/global_config.yaml")
    task_config = load_yaml(config.get("task_config"))
    model       = PusherSliderModel(config=task_config)
    mpc = PusherSliderNMPC(model=model, config=task_config)

    mpc.set_face(Face.POS_Y)

    #from tests.sim_tracking import validate_sim_tracking
    #validate_sim_tracking(mpc=mpc, model=model)

    from tests.test_path_tracking import validate_path_tracking
    validate_path_tracking(mpc=mpc, model=model)