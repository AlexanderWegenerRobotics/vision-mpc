# pusher_slider_model.py
import casadi as ca
import numpy as np


class PusherSliderModel:
    """
    Symbolic CasADi model of the quasi-static pusher-slider system.

    State:  x = [x_s, y_s, theta_s, p_y]  (slider CoM x/y, yaw, pusher tangential position)
    Input:  u = [v_n, v_t]                 (pusher normal and tangential velocity in contact frame)

    All expressions are symbolic — no numerics are evaluated here.
    acados code-generates derivatives from these expressions at setup time.
    """

    def __init__(self, params: dict):
        """
        params keys:
            slider_half_x   : half-extent of slider in x (= p_x, face offset from CoM)
            slider_half_y   : half-extent of slider in y
            mu_slider       : friction coefficient between pusher and slider face
            mu_ground       : friction coefficient between slider and ground
            slider_mass     : mass of slider [kg]
            gravity         : gravitational acceleration [m/s^2]
        """
        self.params = params
        self._build_symbols()
        self._build_dynamics()

    # ------------------------------------------------------------------ #
    #  Symbolic variables                                                  #
    # ------------------------------------------------------------------ #

    def _build_symbols(self):
        # State and input as CasADi column vectors
        self.x_sym = ca.MX.sym("x", 4)   # [x_s, y_s, theta_s, p_y]
        self.u_sym = ca.MX.sym("u", 2)   # [v_n, v_t]

        # Unpack for readability — these are still symbolic scalars
        self.x_s     = self.x_sym[0]
        self.y_s     = self.x_sym[1]
        self.theta_s = self.x_sym[2]
        self.p_y     = self.x_sym[3]

        self.v_n = self.u_sym[0]
        self.v_t = self.u_sym[1]

    # ------------------------------------------------------------------ #
    #  Dynamics                                                            #
    # ------------------------------------------------------------------ #

    def _build_dynamics(self):
        p = self.params

        # p_x is fixed: pusher contacts the face at the face offset from CoM
        p_x = p["slider_half_x"]

        # Characteristic length of the limit surface ellipsoid (eq. 2 in paper)
        # c^2 = (a^2 + b^2) / 12  for a rectangular slider under uniform pressure
        a = p["slider_half_x"] * 2   # full side length
        b = p["slider_half_y"] * 2
        c2 = (a**2 + b**2) / 12.0    # numeric scalar — parameter, not symbolic
        c  = ca.sqrt(c2)

        mu   = p["mu_slider"]        # pusher-slider friction
        p_y  = self.p_y              # symbolic: varies along contact face

        # ── Motion cone slopes (eq. 6) ─────────────────────────────────
        # These bound the ratio v_t/v_n that keeps the pusher in sticking contact.
        # If v_t/v_n exceeds gamma_t the pusher slides upward on the face,
        # if it falls below gamma_b it slides downward.
        gamma_t = (mu * c2 - p_x * p_y + mu * p_x**2) / (c2 + p_y**2 - mu * p_x * p_y)
        gamma_b = (-mu * c2 - p_x * p_y - mu * p_x**2) / (c2 + p_y**2 + mu * p_x * p_y)

        # ── Contact velocity selection (eq. 7) ─────────────────────────
        # CasADi if_else is the symbolic counterpart of Python if/else.
        # It keeps the expression graph differentiable (as a smooth step)
        # so the SQP Jacobian is well-defined everywhere.
        #
        # Sticking:      vc = u                     (full velocity transmitted)
        # Sliding up:    vc = v_n * [1, gamma_t]^T  (along upper cone edge)
        # Sliding down:  vc = v_n * [1, gamma_b]^T  (along lower cone edge)

        vc_n_stick = self.v_n
        vc_t_stick = self.v_t

        vc_n_up = self.v_n
        vc_t_up = self.v_n * gamma_t

        vc_n_dn = self.v_n
        vc_t_dn = self.v_n * gamma_b

        # First branch: sliding up if v_t > gamma_t * v_n
        vc_n = ca.if_else(self.v_t > gamma_t * self.v_n, vc_n_up, vc_n_stick)
        vc_t = ca.if_else(self.v_t > gamma_t * self.v_n, vc_t_up, vc_t_stick)

        # Second branch: override with sliding down if v_t < gamma_b * v_n
        vc_n = ca.if_else(self.v_t < gamma_b * self.v_n, vc_n_dn, vc_n)
        vc_t = ca.if_else(self.v_t < gamma_b * self.v_n, vc_t_dn, vc_t)

        # ── Limit surface mapping: contact wrench → slider body velocity ─
        # Q matrix (eq. 4) maps the contact velocity to body-frame translational velocity.
        # The denominator (c^2 + p_x^2 + p_y^2) normalises by the inertial resistance.
        denom = c2 + p_x**2 + p_y**2

        Q11 = c2 + p_x**2
        Q12 = p_x * p_y
        Q21 = p_x * p_y
        Q22 = c2 + p_y**2

        # Body-frame slider velocities (eq. 3)
        xdot_b = (Q11 * vc_n + Q12 * vc_t) / denom
        ydot_b = (Q21 * vc_n + Q22 * vc_t) / denom
        thetadot = (-p_y * vc_n + p_x * vc_t) / denom

        # ── Rotate body-frame translation to world frame (eq. 5) ───────
        cos_th = ca.cos(self.theta_s)
        sin_th = ca.sin(self.theta_s)

        xdot_w = cos_th * xdot_b - sin_th * ydot_b
        ydot_w = sin_th * xdot_b + cos_th * ydot_b

        # ── Pusher tangential sliding on face (eq. 8) ──────────────────
        # In sticking mode vc_t = v_t so p_y_dot = 0.
        # In sliding modes the pusher drifts along the face.
        py_dot = self.v_t - vc_t

        # ── Assemble f(x, u) ───────────────────────────────────────────
        self.f_expr = ca.vertcat(xdot_w, ydot_w, thetadot, py_dot)

        # Build a callable CasADi Function for debugging / unit-testing
        # outside of acados (e.g. verify dynamics numerically)
        self.f_func = ca.Function(
            "f",
            [self.x_sym, self.u_sym],
            [self.f_expr],
            ["x", "u"],
            ["xdot"],
        )

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Numeric evaluation of f — useful for unit tests and debugging."""
        return np.array(self.f_func(x, u)).flatten()