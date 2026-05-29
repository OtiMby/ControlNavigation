"""
Perturbative control navigation — core classes and preset loaders.

Exports
-------
Mobility     — scalar mobility field wrapper
Force        — external force field wrapper
control_nav  — main perturbative solver
load_*       — preset factory functions
"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator

from numba_kernel import (
    kernel_T1_noncons,     kernel_T1_cons,
    kernel_dT1dTh_noncons, kernel_dT1dTh_cons,
    kernel_T2_dT2dTh_145,  kernel_T2_dT2dTh_151,
)


# ---------------------------------------------------------------------------
# Mobility
# ---------------------------------------------------------------------------

class Mobility:
    """
    Callable wrapper for a scalar mobility field D(X, Y).

    Parameters
    ----------
    func          : callable (X, Y) -> array
        The mobility field D(X, Y).
    dtheta_func   : callable (X, Y) -> array, optional
        Analytical ∂θ D.  Falls back to centred FD (h = 1e-4) if omitted.
    d2theta_func  : callable (X, Y) -> array, optional
        Analytical ∂²θ D.  Falls back to 2nd-order centred FD (h = 1e-3)
        if omitted.
    """

    def __init__(self, func, dtheta_func=None, d2theta_func=None):
        self._func         = func
        self._dtheta_func  = dtheta_func
        self._d2theta_func = d2theta_func

    def __call__(self, X, Y):
        return self._func(X, Y)

    def dtheta(self, X, Y):
        """∂θ D — analytical if provided, else centred FD (h = 1e-4)."""
        if self._dtheta_func is not None:
            return self._dtheta_func(X, Y)
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._func(R * np.cos(th + dh), R * np.sin(th + dh))
              - self._func(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def d2theta(self, X, Y):
        """∂²θ D — analytical if provided, else 2nd-order centred FD (h = 1e-3)."""
        if self._d2theta_func is not None:
            return self._d2theta_func(X, Y)
        dh    = 1e-3
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._func(R * np.cos(th + dh), R * np.sin(th + dh))
              - 2.0 * self._func(X, Y)
              + self._func(R * np.cos(th - dh), R * np.sin(th - dh))) / dh**2


# ---------------------------------------------------------------------------
# Force
# ---------------------------------------------------------------------------

class Force:
    """
    External force field.

    Three modes
    -----------
    conservative : derived from a potential φ, force = −∇φ
    cartesian    : given as (f_x, f_y) Cartesian component functions
    polar        : given as (f_R, f_θ) polar component functions

    fx() / fy() always return Cartesian components regardless of mode.

    Parameters
    ----------
    dtheta_fR_func  : callable (X, Y) -> array, optional
        Analytical ∂θ f_R.  Falls back to centred FD (h = 1e-4) if omitted.
    dtheta_fT_func  : callable (X, Y) -> array, optional
        Analytical ∂θ f_θ.  Falls back to centred FD (h = 1e-4) if omitted.
    d2theta_fR_func : callable (X, Y) -> array, optional
        Analytical ∂²θ f_R.  Falls back to 2nd-order centred FD (h = 1e-3)
        if omitted.
    _dpotdth        : callable (X, Y) -> array, optional
        Analytical ∂θ φ(r, θ) at fixed r (angular derivative of the
        potential).  Falls back to centred FD of φ (h = 1e-4) if omitted.
        Only used when conservative=True.
    """

    def __init__(self, Lx, Ly, Nx, Ny,
                 conservative=False,
                 f_x=None, f_y=None,
                 potential=None,
                 cartesian=True,
                 f_R=None, f_theta=None,
                 dtheta_fR_func=None,
                 _dpotdth=None,
                 dtheta_fT_func=None,
                 d2theta_fR_func=None):

        self.conservative     = conservative
        self.cartesian        = cartesian
        self._potential       = potential
        self._f_R             = f_R
        self._f_theta         = f_theta
        self._dtheta_fR_func  = dtheta_fR_func
        self._dtheta_fT_func  = dtheta_fT_func
        self._d2theta_fR_func = d2theta_fR_func
        self.dpot             = _dpotdth   # ∂θ φ  (kept for back-compat)

        if self._potential is not None:
            Xg, Yg = np.mgrid[-Lx/2:Lx/2:Nx*1j, -Ly/2:Ly/2:Ny*1j]
            pot    = potential(Xg, Yg)
            xs, ys = Xg[:, 0], Yg[0, :]
            kw     = dict(bounds_error=False, fill_value=None)
            if conservative:
                self._f_x = RegularGridInterpolator(
                    (xs, ys), -np.gradient(pot, xs, axis=0), **kw)
                self._f_y = RegularGridInterpolator(
                    (xs, ys), -np.gradient(pot, ys, axis=1), **kw)
            else:
                # divergence-free (rotational) field ∇⊥φ
                self._f_x = RegularGridInterpolator(
                    (xs, ys),  np.gradient(pot, ys, axis=1), **kw)
                self._f_y = RegularGridInterpolator(
                    (xs, ys), -np.gradient(pot, xs, axis=0), **kw)
        elif cartesian:
            self._f_x = f_x
            self._f_y = f_y
        else:   # polar
            self._f_x = lambda X, Y: (
                f_R(X, Y) * np.cos(np.arctan2(Y, X))
                - f_theta(X, Y) * np.sin(np.arctan2(Y, X))
            )
            self._f_y = lambda X, Y: (
                f_R(X, Y) * np.sin(np.arctan2(Y, X))
                + f_theta(X, Y) * np.cos(np.arctan2(Y, X))
            )

    # ── Cartesian components ─────────────────────────────────────────────────

    def fx(self, X, Y):
        if self._potential is not None:
            return self._f_x(np.stack([X, Y], axis=-1))
        return self._f_x(X, Y)

    def fy(self, X, Y):
        if self._potential is not None:
            return self._f_y(np.stack([X, Y], axis=-1))
        return self._f_y(X, Y)

    # ── Potential ────────────────────────────────────────────────────────────

    def potential(self, X, Y):
        return self._potential(X, Y)

    # ── Polar components ─────────────────────────────────────────────────────

    def fR(self, X, Y):
        """Radial component f_R at Cartesian (X, Y)."""
        if not self.cartesian and self._f_R is not None:
            return self._f_R(X, Y)
        th = np.arctan2(Y, X)
        return self.fx(X, Y) * np.cos(th) + self.fy(X, Y) * np.sin(th)

    def fT(self, X, Y):
        """Tangential component f_θ at Cartesian (X, Y)."""
        if not self.cartesian and self._f_theta is not None:
            return self._f_theta(X, Y)
        th = np.arctan2(Y, X)
        return -self.fx(X, Y) * np.sin(th) + self.fy(X, Y) * np.cos(th)

    # ── Angular derivatives of polar components ──────────────────────────────

    def dtheta_fR(self, X, Y):
        """∂θ f_R — analytical if provided, else centred FD (h = 1e-4)."""
        if self._dtheta_fR_func is not None:
            return self._dtheta_fR_func(X, Y)
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fR(R * np.cos(th + dh), R * np.sin(th + dh))
              - self.fR(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def dtheta_fT(self, X, Y):
        """∂θ f_θ — analytical if provided, else centred FD (h = 1e-4)."""
        if self._dtheta_fT_func is not None:
            return self._dtheta_fT_func(X, Y)
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fT(R * np.cos(th + dh), R * np.sin(th + dh))
              - self.fT(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def d2theta_fR(self, X, Y):
        """∂²θ f_R — analytical if provided, else 2nd-order centred FD (h = 1e-3)."""
        if self._d2theta_fR_func is not None:
            return self._d2theta_fR_func(X, Y)
        dh    = 1e-3
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fR(R * np.cos(th + dh), R * np.sin(th + dh))
              - 2.0 * self.fR(X, Y)
              + self.fR(R * np.cos(th - dh), R * np.sin(th - dh))) / dh**2

    def dtheta_phi(self, X, Y):
        """
        ∂φ/∂θ at fixed R — angular derivative of the potential.
        Uses analytical dpot if provided, else centred FD of φ (h = 1e-4).
        Only meaningful for conservative forces.
        """
        if self.dpot is not None:
            return self.dpot(X, Y)
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._potential(R * np.cos(th + dh), R * np.sin(th + dh))
              - self._potential(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)


# ---------------------------------------------------------------------------
# control_nav
# ---------------------------------------------------------------------------

class control_nav:
    """
    Perturbative control navigation solver.

    Computes T0, T1, T2, their angular derivatives, and the optimal control
    direction c = c0 + ε c1 + ε² c2 on a regular Cartesian grid.

    Typical usage
    -------------
    cn = control_nav(...)
    cn.compute_T1()
    cn.compute_dT1dTh()
    cn.compute_T2()          # also computes dT2dTh internally
    cn.compute_dT2dTh()      # no-op if compute_T2 was already called
    cn.compute_all_control_forces()
    # cn.T0, cn.T1, cn.T2, cn.c0x/y, cn.c1x/y, cn.c2x/y are now available.

    Parameters
    ----------
    Lx, Ly  : float    domain half-sizes  (domain = [−Lx/2, Lx/2] × [−Ly/2, Ly/2])
    Nx, Ny  : int      grid resolution
    D0      : float    leading-order (constant) mobility
    D1, D2  : Mobility first- and second-order mobility perturbations
    f1, f2  : Force    first- and second-order external forces
    epsilon : float    perturbation parameter
    """

    def __init__(self, Lx, Ly, Nx, Ny,
                 D0: float,
                 D1: Mobility, D2: Mobility,
                 f1: Force,    f2: Force,
                 epsilon: float = 0.1):

        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.D0          = D0
        self.D1, self.D2 = D1, D2
        self.f1, self.f2 = f1, f2
        self.epsilon     = epsilon

        # Nr must be odd for composite Simpson's rule
        Nr       = int(np.hypot(Nx, Ny))
        self.Nr  = Nr if Nr % 2 == 1 else Nr + 1

        # Cartesian grid  (Nx, Ny)
        self.X, self.Y = np.mgrid[-Lx/2:Lx/2:Nx*1j, -Ly/2:Ly/2:Ny*1j]

        # Radial distance — guard against exact zero
        self.R  = np.where(np.hypot(self.X, self.Y) == 0, 1e-12,
                           np.hypot(self.X, self.Y))

        # Leading-order travel time
        self.T0 = self.R / D0

        # Polar angle  (Nx, Ny)
        self._Th = np.arctan2(self.Y, self.X) % (2 * np.pi)

        # Unit radial / tangential vectors  (Nx, Ny)
        self._eRx =  np.cos(self._Th)
        self._eRy =  np.sin(self._Th)
        self._eTx = -np.sin(self._Th)
        self._eTy =  np.cos(self._Th)

        # Fields filled by compute_* methods
        self.T1     = None
        self.T2     = None
        self.dT1dTh = None
        self.dT2dTh = None
        self.c0x = self.c0y = None
        self.c1x = self.c1y = None
        self.c2x = self.c2y = None

        self._fields_ready = False

    # ── Field pre-computation ─────────────────────────────────────────────────

    def precompute_fields(self):
        """
        Evaluate all field functions on the 2-D Cartesian grid (Nx, Ny) and
        cache the results as contiguous float64 arrays for the Numba kernels.

        Called automatically by compute_T1(); can also be invoked explicitly
        to control when the (potentially slow) field evaluations happen.
        Subsequent calls are no-ops.
        """
        if self._fields_ready:
            return

        X, Y = self.X, self.Y

        # ── fields common to both conservative and non-conservative paths ─────
        self._pf = dict(
            D1       = np.ascontiguousarray(self.D1(X, Y),        dtype=np.float64),
            D2       = np.ascontiguousarray(self.D2(X, Y),        dtype=np.float64),
            f1R      = np.ascontiguousarray(self.f1.fR(X, Y),     dtype=np.float64),
            f1T      = np.ascontiguousarray(self.f1.fT(X, Y),     dtype=np.float64),
            f2R      = np.ascontiguousarray(self.f2.fR(X, Y),     dtype=np.float64),
            df1Rdth  = np.ascontiguousarray(self.f1.dtheta_fR(X, Y),  dtype=np.float64),
            df1Tdth  = np.ascontiguousarray(self.f1.dtheta_fT(X, Y),  dtype=np.float64),
            dD1dth   = np.ascontiguousarray(self.D1.dtheta(X, Y),     dtype=np.float64),
            d2D1dth2 = np.ascontiguousarray(self.D1.d2theta(X, Y),    dtype=np.float64),
            df2Rdth  = np.ascontiguousarray(self.f2.dtheta_fR(X, Y),  dtype=np.float64),
            dD2dth   = np.ascontiguousarray(self.D2.dtheta(X, Y),     dtype=np.float64),
        )

        # ── non-conservative only : ∂²θ f1R needed for the g2 accumulator ────
        if not self.f1.conservative:
            self._pf['d2f1Rdth2'] = np.ascontiguousarray(
                self.f1.d2theta_fR(X, Y), dtype=np.float64)

        # ── conservative only : potential and its angular derivative ──────────
        if self.f1.conservative:
            self._pf['phi']   = np.ascontiguousarray(self.f1.potential(X, Y), dtype=np.float64)
            self._pf['phi0']  = float(self.f1.potential(0.0, 0.0))
            self._pf['dphi']  = np.ascontiguousarray(self.f1.dtheta_phi(X, Y), dtype=np.float64)
            self._pf['dphi0'] = 0.0   # ∂θ φ = 0 at the origin (R = 0)

        # ── grid parameters for bilinear interpolation ────────────────────────
        xs = X[:, 0];  ys = Y[0, :]
        self._x0 = float(xs[0]);  self._dx = float(xs[1] - xs[0])
        self._y0 = float(ys[0]);  self._dy = float(ys[1] - ys[0])

        self._fields_ready = True

    def _gargs(self):
        """Return (x0, dx, y0, dy) tuple for the Numba kernels."""
        return self._x0, self._dx, self._y0, self._dy

    # ── T1 ────────────────────────────────────────────────────────────────────

    def compute_T1(self):
        """Compute and store T1 on the grid."""
        self.precompute_fields()
        pf = self._pf;  ga = self._gargs()

        if self.f1.conservative:
            self.T1 = kernel_T1_cons(
                self.X, self.Y, self.R, *ga,
                pf['D1'], pf['phi'], pf['phi0'],
                self.D0, self.Nr,
            )
        else:
            self.T1 = kernel_T1_noncons(
                self.X, self.Y, self.R, *ga,
                pf['D1'], pf['f1R'],
                self.D0, self.Nr,
            )

    # ── dT1/dθ ────────────────────────────────────────────────────────────────

    def compute_dT1dTh(self):
        """Compute and store dT1/dθ.  Must be called after compute_T1()."""
        pf = self._pf;  ga = self._gargs()

        if self.f1.conservative:
            self.dT1dTh = kernel_dT1dTh_cons(
                self.X, self.Y, self.R, *ga,
                pf['dD1dth'], pf['dphi'], pf['dphi0'],
                self.D0, self.Nr,
            )
        else:
            self.dT1dTh = kernel_dT1dTh_noncons(
                self.X, self.Y, self.R, *ga,
                pf['df1Rdth'], pf['dD1dth'],
                self.D0, self.Nr,
            )

    # ── T2 + dT2/dθ (joint) ──────────────────────────────────────────────────

    def _compute_T2_and_dT2dTh(self):
        """
        Compute T2 and dT2/dθ jointly in a single kernel pass.
        dT2/dθ is obtained analytically via the Leibniz rule — no FD in θ.
        Requires compute_T1() and compute_dT1dTh() to have been called.
        """
        pf = self._pf;  ga = self._gargs()

        if self.f1.conservative:
            self.T2, self.dT2dTh = kernel_T2_dT2dTh_151(
                self.X, self.Y, self.R, *ga,
                pf['D1'], pf['D2'], pf['f1R'], pf['f1T'], pf['f2R'],
                pf['df1Rdth'], pf['df1Tdth'], pf['dD1dth'], pf['d2D1dth2'],
                pf['df2Rdth'], pf['dD2dth'],
                self.D0, self.Nr,
            )
        else:
            self.T2, self.dT2dTh = kernel_T2_dT2dTh_145(
                self.X, self.Y, self.R, *ga,
                pf['D1'], pf['D2'], pf['f1R'], pf['f1T'], pf['f2R'],
                pf['df1Rdth'], pf['df1Tdth'], pf['dD1dth'],
                pf['d2f1Rdth2'], pf['d2D1dth2'],
                pf['df2Rdth'], pf['dD2dth'],
                self.D0, self.Nr,
            )

    def compute_T2(self):
        """Compute and store T2 (also computes dT2/dθ).
        Requires compute_T1() and compute_dT1dTh()."""
        if self.T2 is None:
            self._compute_T2_and_dT2dTh()

    def compute_dT2dTh(self):
        """Compute and store dT2/dθ.  No-op if compute_T2() was already called."""
        if self.dT2dTh is None:
            self._compute_T2_and_dT2dTh()

    # ── Control force ─────────────────────────────────────────────────────────

    def compute_control_force(self, order):
        """
        Return (cx, cy) for the given perturbative order (0, 1 or 2).

        Prerequisites
        -------------
        order >= 1 : compute_T1(), compute_dT1dTh()
        order == 2 : additionally compute_T2(), compute_dT2dTh()
        """
        eRx, eRy = self._eRx, self._eRy
        eTx, eTy = self._eTx, self._eTy

        if order == 0:
            return -eRx, -eRy

        if order == 1:
            expr = -self.dT1dTh * self.D0 / self.R
            return eTx * expr, eTy * expr

        # order == 2
        D1     = self.D1(self.X, self.Y)
        expr_r = self.dT1dTh**2 * self.D0**2 / (2 * self.R**2)
        expr_t = (
            self.dT1dTh * (self.f1.fR(self.X, self.Y) - D1)
            - self.D0 * self.dT2dTh
        ) / self.R
        return (eRx * expr_r + eTx * expr_t,
                eRy * expr_r + eTy * expr_t)

    def compute_all_control_forces(self):
        """Compute and store c0, c1, c2 for reuse by velocity_field."""
        self.c0x, self.c0y = self.compute_control_force(0)
        self.c1x, self.c1y = self.compute_control_force(1)
        self.c2x, self.c2y = self.compute_control_force(2)

    # ── Velocity field ────────────────────────────────────────────────────────

    def velocity_field(self, order):
        """
        Assemble the total drift velocity (vx, vy) at the given perturbative
        order.  Requires compute_all_control_forces() to have been called.
        """
        ep   = self.epsilon
        D0   = self.D0
        D1   = self.D1(self.X, self.Y)
        X, Y = self.X, self.Y

        vx = D0 * self.c0x
        vy = D0 * self.c0y

        if order >= 1:
            vx += ep * (D0 * self.c1x + D1 * self.c0x + self.f1.fx(X, Y))
            vy += ep * (D0 * self.c1y + D1 * self.c0y + self.f1.fy(X, Y))

        if order == 2:
            vx += ep**2 * (D0 * self.c2x + self.f2.fx(X, Y) + D1 * self.c1x)
            vy += ep**2 * (D0 * self.c2y + self.f2.fy(X, Y) + D1 * self.c1y)

        return vx, vy

    # ── Path integration ──────────────────────────────────────────────────────

    def path_RK4(self, order, start_coords, dt=0.05, steps=800, r_stop=0.1):
        """Trace an optimal path using RK4 integration of the velocity field."""
        vx, vy = self.velocity_field(order)
        xr, yr = self.X[:, 0], self.Y[0, :]
        kw     = dict(bounds_error=False, fill_value=0.)
        ifx    = RegularGridInterpolator((xr, yr), vx, **kw)
        ify    = RegularGridInterpolator((xr, yr), vy, **kw)

        def v(xi, yi):
            p = np.array([xi, yi])
            return float(ifx(p)), float(ify(p))

        x, y   = float(start_coords[0]), float(start_coords[1])
        xs, ys = [x], [y]
        for _ in range(steps):
            k1x, k1y = v(x, y)
            k2x, k2y = v(x + dt/2*k1x, y + dt/2*k1y)
            k3x, k3y = v(x + dt/2*k2x, y + dt/2*k2y)
            k4x, k4y = v(x + dt*k3x,   y + dt*k3y)
            x += dt / 6 * (k1x + 2*k2x + 2*k3x + k4x)
            y += dt / 6 * (k1y + 2*k2y + 2*k3y + k4y)
            xs.append(x);  ys.append(y)
            if abs(x) > self.Lx/2 or abs(y) > self.Ly/2 or np.hypot(x, y) < r_stop:
                break
        return np.array(xs), np.array(ys)

    # ── Validity radius ───────────────────────────────────────────────────────

    def compute_eps(self):
        """
        Compute the largest ε at each pixel such that the perturbation
        expansion remains valid.
        """
        self.dist  = np.hypot(self.X, self.Y).ravel()
        self.order = np.argsort(self.dist)
        T0 = np.abs(self.T0.ravel()[self.order])
        T1 = np.abs(self.T1.ravel()[self.order])
        T2 = np.abs(self.T2.ravel()[self.order])

        tol = 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            eps_T10 = np.where((T1 > tol) & (T0 > tol), T0 / T1, np.inf)

        disc = T1**2 + 4 * T2 * T0
        with np.errstate(divide='ignore', invalid='ignore'):
            root = np.where(T2 > tol,
                            (T1 + np.sqrt(np.maximum(disc, 0))) / (2 * T2),
                            np.where(T1 > tol, T0 / T1, np.inf))
        eps_tot = np.where(T0 > tol, root, np.inf)

        self.eps_T10 = np.clip(eps_T10, 0.0, 1.0)
        self.eps_tot  = np.clip(eps_tot,  0.0, 1.0)
        constraining = np.minimum(self.eps_T10, self.eps_tot)

        return self.eps_tot, self.eps_T10, constraining

    def radial_eps_validity(self):
        """
        For each radius (spaced by dr), return the most constraining ε
        inside the disk, i.e. the minimum over all pixels within that radius.
        """
        R        = np.linspace(0, np.hypot(self.Lx/2, self.Ly/2), self.Nr)
        dist_s   = self.dist[self.order]
        cumu_T10 = np.minimum.accumulate(self.eps_T10)
        cumu_tot  = np.minimum.accumulate(self.eps_tot)

        n_in = np.searchsorted(dist_s, R, side='right')
        idx  = np.maximum(n_in[1:] - 1, 0)

        epsesT10 = np.where(n_in[1:] > 0, cumu_T10[idx], 1.0)
        epsesTot  = np.where(n_in[1:] > 0, cumu_tot[idx],  1.0)
        constraining = np.minimum(epsesT10, epsesTot)

        return epsesTot, epsesT10, constraining



def load_compare(Lx, Ly, N):
    """
    Non-conservative polar force with spatially varying mobility.

    Analytical polar derivatives supplied for exact T2 (Eq. 1.45):
      f1R = cos(2θ)              -> d/dθ f1R    = -2 sin(2θ)
      f1θ = 1/(1+R)              -> d/dR(R f1θ) = 1/(1+R)^2
      D1  = 0.1 R sin(2θ)       -> d/dθ D1     = 0.2 R cos(2θ)
    """
    def f_R1(X, Y):
        return np.cos(2 * (np.arctan2(Y, X) % (2 * np.pi)))

    def f_T1(X, Y):
        return 1.0 / (1 + np.hypot(X, Y))

    def f_R2(X, Y):
        R, th = np.hypot(X, Y), np.arctan2(Y, X) % (2 * np.pi)
        return np.exp(-R**2 / 10) * np.sin(2 * th)

    def f_T2(X, Y):
        return np.zeros_like(X)

    def _D1(X, Y):
        R, th = np.hypot(X, Y), np.arctan2(Y, X) % (2 * np.pi)
        return 0.1 * R * np.sin(2 * th)

    def _D2(X, Y):
        R, th = np.hypot(X, Y), np.arctan2(Y, X) % (2 * np.pi)
        return np.exp(-(R - 3)**2) * np.cos(2 * th)

    def _dtheta_f1R(X, Y):
        return -2 * np.sin(2 * (np.arctan2(Y, X) % (2 * np.pi)))


    def _dtheta_D1(X, Y):
        R, th = np.hypot(X, Y), np.arctan2(Y, X) % (2 * np.pi)
        return 0.2 * R * np.cos(2 * th)

    f1 = Force(Lx, Ly, N, N, conservative=False,
               f_R=f_R1, f_theta=f_T1, cartesian=False,
               dtheta_fR_func=_dtheta_f1R)
    f2 = Force(Lx, Ly, N, N, conservative=False,
               f_R=f_R2, f_theta=f_T2, cartesian=False)
    D1 = Mobility(_D1, dtheta_func=_dtheta_D1)
    D2 = Mobility(_D2)
    return f1, f2, D1, D2, "compare"

def load_conservative1(Lx, Ly, N):
    def pot1(x, y):
        k = 2 * np.pi * (2 / Lx)
        return 2 * np.cos(k * x)**2 * np.cos(k * y)**2

    def _dpotdth(x,y):
        k = 2 * np.pi * (5 / Lx)
        return 4 * k * (np.sin(k*x)*np.cos(k*x)*np.cos(k*y)**2 +np.sin(k*y)*np.cos(k*y)*np.cos(k*x)**2 )
    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=True, potential=pot1, _dpotdth = None)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(lambda X, Y: np.zeros_like(X),
                  dtheta_func=lambda X, Y: np.zeros_like(X))
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "conservative_peaks2"

def load_conservative2(Lx, Ly, N):
    def pot1(x, y):
        k = 2 * np.pi * (5 / Lx)
        return -4 * np.cos(k * x)**2 * np.cos(k * y)**2

    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=True, potential=pot1)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(lambda X, Y: np.zeros_like(X),
                  dtheta_func=lambda X, Y: np.zeros_like(X))
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "conservative_wells"

def load_mob_only(Lx, Ly, N):
    """D1 = X = R cos(θ)  ->  d/dθ D1 = -R sin(θ) = -Y"""
    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(lambda X, Y: X.copy(), dtheta_func=lambda X, Y: -Y.copy())
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "only_mob"

def load_divless(Lx, Ly, N):

    def pot(x, y):
        k = 2 * np.pi * (6 / Lx)
        return np.cos(k * x)**4 * np.sin(k * y)**4

    
    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=False, potential=pot)
    f2 = Force(Lx, Ly, N, N, conservative=True, potential=_zero)
    D1 = Mobility(_zero,dtheta_func=_zero)
    D2 = Mobility(_zero,dtheta_func=_zero)
    return f1, f2, D1, D2, "div_less4"

def load_divless2(Lx, Ly, N):
    def pot(x, y):
        k = 2 * np.pi * (6 / Lx)
        return np.cos(k * x)**2* np.cos(k * y)**2

    
    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=False, potential=pot)
    f2 = Force(Lx, Ly, N, N, conservative=True, potential=_zero)
    D1 = Mobility(_zero,dtheta_func=_zero)
    D2 = Mobility(_zero,dtheta_func=_zero)
    return f1, f2, D1, D2, "div_less2"

