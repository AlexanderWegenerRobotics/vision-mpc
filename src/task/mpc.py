import casadi as ca
import numpy as np
from enum import Enum, auto
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class ControllerVariant(Enum):
    BASELINE          = auto()
    CERTAINTY_EQUIV   = auto()
    UNCERTAINTY_AWARE = auto()


class PusherSliderModel:
    """Quasi-static pusher-slider dynamics following Hogan & Rodriguez / Federico et al.

    State:  x = [x_s, y_s, theta_s, p_y]
    Control: u = [v_n, v_t]

    Convention (pusher on -x_body face, pushing in +x_body):
      p_x  = fixed perpendicular distance from CoM to contact face = slider_half_x
      p_y  = tangential offset of pusher along the face (y_body direction, state variable)
      v_n  > 0  pushes into the slider (along +x_body, i.e. inward normal)
      v_t  > 0  slides the pusher along the face in +y_body direction
    """

    def __init__(self, config: dict):
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
        p   = self.params
        a   = p["slider_half_x"] * 2
        b   = p["slider_half_y"] * 2
        c2  = (a**2 + b**2) / 12.0
        mu  = p["mu_slider"]

        # p_x = perpendicular distance from CoM to contact face
        # The dynamics assume pusher on -x_body face, so p_x = slider_half_x
        p_x = p["slider_half_x"]
        p_y = self.p_y

        gamma_t = ( mu * c2 - p_x * p_y + mu * p_x**2) / (c2 + p_y**2 - mu * p_x * p_y)
        gamma_b = (-mu * c2 - p_x * p_y - mu * p_x**2) / (c2 + p_y**2 + mu * p_x * p_y)

        # Clamp v_t to motion cone boundaries
        vc_n = self.v_n
        vc_t = ca.if_else(self.v_t > gamma_t * self.v_n, self.v_n * gamma_t, self.v_t)
        vc_t = ca.if_else(self.v_t < gamma_b * self.v_n, self.v_n * gamma_b, vc_t)

        denom    = c2 + p_x**2 + p_y**2
        xdot_b   = ((c2 + p_x**2) * vc_n + p_x * p_y * vc_t) / denom
        ydot_b   = (p_x * p_y * vc_n + (c2 + p_y**2) * vc_t) / denom
        thetadot = (-p_y * vc_n + p_x * vc_t) / denom
        py_dot   = self.v_t - vc_t

        cos_th = ca.cos(self.theta_s)
        sin_th = ca.sin(self.theta_s)

        # xdot_b, ydot_b are slider CoM velocity in body frame
        # Rotate to world frame
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
    def __init__(self, model: PusherSliderModel, config: dict):
        self.model = model
        mpc_cfg    = config["mpc"]
        self.variant = ControllerVariant[mpc_cfg["variant"]]
        self.params = {
            "horizon":          mpc_cfg["horizon"],
            "dt":               mpc_cfg["dt"],
            "Q":                mpc_cfg["Q"],
            "R":                mpc_cfg["R"],
            "Q_terminal_scale": mpc_cfg["Q_terminal_scale"],
            "v_n_max":          mpc_cfg["v_n_max"],
            "v_t_max":          mpc_cfg["v_t_max"],
            "v_n_min":          mpc_cfg.get("v_n_min", 0.0),
            "slider_half_x":    config["slider_half_x"],
            "slider_half_y":    config["slider_half_y"],
        }
        self.T  = self.params["horizon"]
        self.nx = model.nx
        self.nu = model.nu
        self._build_solver()

    def _build_solver(self):
        p = self.params

        acados_model             = AcadosModel()
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
        ny   = self.nx + self.nu
        ny_e = self.nx

        ocp.cost.cost_type   = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        Vx = np.zeros((ny, self.nx)); Vx[:self.nx, :] = np.eye(self.nx)
        Vu = np.zeros((ny, self.nu)); Vu[self.nx:, :] = np.eye(self.nu)
        ocp.cost.Vx   = Vx
        ocp.cost.Vu   = Vu
        ocp.cost.Vx_e = np.eye(ny_e)

        W   = np.block([[Q_mat, np.zeros((self.nx, self.nu))], [np.zeros((self.nu, self.nx)), R_mat]])
        W_e = p["Q_terminal_scale"] * Q_mat

        ocp.cost.W      = W
        ocp.cost.W_e    = W_e
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.lbu   = np.array([p["v_n_min"], -p["v_t_max"]])
        ocp.constraints.ubu   = np.array([p["v_n_max"],  p["v_t_max"]])
        ocp.constraints.idxbu = np.array([0, 1])

        # p_y bounded by face half-length (slider_half_y for -x face, tangent is y_body)
        half_face = p["slider_half_y"]
        ocp.constraints.lbx   = np.array([-half_face])
        ocp.constraints.ubx   = np.array([ half_face])
        ocp.constraints.idxbx = np.array([3])

        ocp.constraints.x0 = np.zeros(self.nx)

        ocp.solver_options.integrator_type  = "ERK"
        ocp.solver_options.num_stages       = 4
        ocp.solver_options.num_steps        = 1
        ocp.solver_options.nlp_solver_type  = "SQP_RTI"
        ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = self.T // 4
        ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
        ocp.solver_options.print_level      = 0

        self._solver = AcadosOcpSolver(ocp, json_file="pusher_slider_ocp.json")
        self._Q_nom  = Q_mat.copy()
        self._R_mat  = R_mat.copy()

    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Solve the OCP and return the first control action."""
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        for t in range(self.T):
            self._solver.cost_set(t, "yref", np.concatenate([x_ref[t], np.zeros(self.nu)]))
        self._solver.cost_set(self.T, "yref", x_ref[self.T])

        status = self._solver.solve()
        if status not in (0, 2):
            print(f"[NMPC] acados status {status}")

        return self._solver.get(0, "u")

    @staticmethod
    def normalize_goal_theta(current_theta: float, goal_theta: float) -> float:
        """Snap goal_theta to nearest equivalent orientation mod pi/2 for a rectangle."""
        candidates = [goal_theta + k * (np.pi / 2) for k in range(-2, 3)]
        deltas = [(c - current_theta + np.pi) % (2 * np.pi) - np.pi for c in candidates]
        best = min(zip(deltas, candidates), key=lambda x: abs(x[0]))
        return best[1]

    @staticmethod
    def make_linear_reference(x_start, x_goal, T):
        """Linearly interpolate from x_start to x_goal over T+1 steps."""
        delta = x_goal - x_start
        delta[2] = (delta[2] + np.pi) % (2 * np.pi) - np.pi
        return x_start + np.outer(np.linspace(0, 1, T + 1), delta)