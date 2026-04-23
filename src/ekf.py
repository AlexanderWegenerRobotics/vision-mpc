import numpy as np
import casadi as ca
from src.utils import wrap_to_pi


class SliderEKF:
    def __init__(self, model, config):
        mpc_cfg  = config["mpc"]
        self.dt  = mpc_cfg["dt"]
        self._gate_thresh = mpc_cfg["ekf_gate_thresh"]
        self.nx  = 3
        self._P_max_diag = np.array([0.01, 0.01, 0.05])

        self.Q = np.diag(np.array(mpc_cfg["ekf_Q"], dtype=float))
        self.R = np.diag(np.array(mpc_cfg["ekf_R"], dtype=float))

        self._build_functions(model)
        self.reset()

    def _build_functions(self, model):
        # Declare fresh symbols, substitute into model expressions, compile RK4 and Jacobian.
        x3  = ca.MX.sym("x3", 3)
        py  = ca.MX.sym("py")
        u   = ca.MX.sym("u", 2)
        x4  = ca.vertcat(x3, py)

        f_sub = ca.substitute(model.f_expr, model.x_sym, x4)
        f_sub = ca.substitute(f_sub,        model.u_sym, u)

        dt = self.dt
        k1 = f_sub
        k2 = ca.substitute(f_sub, x4, x4 + dt/2 * k1)
        k3 = ca.substitute(f_sub, x4, x4 + dt/2 * k2)
        k4 = ca.substitute(f_sub, x4, x4 + dt   * k3)
        x4_next = x4 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        x3_next = x4_next[:3]
        F_expr  = ca.jacobian(x3_next, x3)

        self._rk4_fn = ca.Function("rk4",   [x3, py, u], [x3_next])
        self._F_fn   = ca.Function("F_jac", [x3, py, u], [F_expr])

    def reset(self, x0: np.ndarray = None, P0: np.ndarray = None):
        self._x = x0.copy() if x0 is not None else None
        self._P = P0.copy() if P0 is not None else np.eye(self.nx) * 1.0
        self._u = np.zeros(2)

    def set_control(self, u: np.ndarray):
        self._u = u.copy()

    def predict(self, p_y: float):
        # Propagate mean and covariance one dt step using current control.
        if self._x is None:
            return
        x_next    = np.array(self._rk4_fn(self._x, p_y, self._u)).flatten()
        x_next[2] = wrap_to_pi(x_next[2])
        F         = np.array(self._F_fn(self._x, p_y, self._u))
        self._P   = F @ self._P @ F.T + self.Q
        np.fill_diagonal(self._P, np.minimum(np.diag(self._P), self._P_max_diag))
        self._x   = x_next

    def update(self, z: np.ndarray):
        if self._x is None:
            self._x = z.copy()
            return
        inn    = z - self._x
        inn[2] = wrap_to_pi(inn[2])
        S      = self._P + self.R
        
        # Mahalanobis gate — reject outlier measurements
        mahal_sq = float(inn @ np.linalg.solve(S, inn))
        if mahal_sq > self._gate_thresh:
            return
        
        K          = self._P @ np.linalg.solve(S.T, np.eye(self.nx)).T
        self._x    = self._x + K @ inn
        self._x[2] = wrap_to_pi(self._x[2])
        self._P    = (np.eye(self.nx) - K) @ self._P

    def initialised(self) -> bool:
        return self._x is not None

    @property
    def mean(self) -> np.ndarray:
        return self._x.copy() if self._x is not None else None

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()