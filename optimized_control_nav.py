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
import sympy as _sp
import warnings as _warnings
import json
import random as rd
# ---------------------------------------------------------------------------
# Symbolic angular-derivative helpers
# ---------------------------------------------------------------------------
# When an analytic ∂θ / ∂²θ is NOT supplied, the field may be given as a
# symbolic expression (string or sympy) in the symbols  x, y, R, theta.
# The angular derivative at fixed R is then obtained EXACTLY with the operator
#
#       L = -y ∂/∂x + x ∂/∂y        (≡ ∂/∂θ at fixed R ;  L R = 0 ,  L θ = 1)
#
# and lambdified to NumPy. 
#
# A field that uses np.cos / np.exp / ... cannot be traced symbolically, so
# the symbolic path requires the closed form (string is simplest), e.g.
#     Mobility(expr="0.1*R*sin(2*theta)")
#     Force(..., cartesian=False, expr_R="cos(2*theta)", expr_T="1/(1+R)")
#     Force(..., conservative=False, expr_phi="cos(k*x)**2*cos(k*y)**2", ...)


# Creating Variables for analytical derivations of forces and mobility
_SX, _SY = _sp.symbols('x y', real=True)
_SR  = _sp.sqrt(_SX**2 + _SY**2)
_STH = _sp.atan2(_SY, _SX)
_SLOC = dict(x=_SX, y=_SY, X=_SX, Y=_SY, R=_SR, r=_SR,
             theta=_STH, th=_STH, Theta=_STH,
             pi=_sp.pi, E=_sp.E,
             sin=_sp.sin, cos=_sp.cos, tan=_sp.tan,
             asin=_sp.asin, acos=_sp.acos, atan=_sp.atan, atan2=_sp.atan2,
             sinh=_sp.sinh, cosh=_sp.cosh, tanh=_sp.tanh,
             exp=_sp.exp, log=_sp.log, sqrt=_sp.sqrt, Abs=_sp.Abs, sign=_sp.sign)


def _to_sym(expr, **consts):
    """Accept a sympy expression or a string in x, y, R, theta -> sympy expr in x, y.

    Extra named constants (e.g. k=...) may be supplied to resolve symbols that
    appear in a string expression.
    """
    if isinstance(expr, _sp.Basic):
        return expr
    loc = dict(_SLOC)
    loc.update(consts)
    return _sp.sympify(str(expr).replace('^', '**'), locals=loc)


def _Lth(e):
    """Angular derivative at fixed R:   ∂θ e = -y ∂x e + x ∂y e."""
    return -_SY * _sp.diff(e, _SX) + _SX * _sp.diff(e, _SY)


def _lambdify(e):
    """sympy expr (in x, y) -> NumPy callable f(X, Y); non-finite (origin) -> 0."""
    f = _sp.lambdify((_SX, _SY), e, 'numpy')

    def _f(X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.asarray(f(X, Y), dtype=float)
        shape = np.broadcast(X, Y).shape
        if out.shape != shape:
            out = np.broadcast_to(out, shape).astype(float)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return _f


class _SymField:
    """A field's symbolic form, exposing value / ∂θ / ∂²θ as cached NumPy callables."""

    def __init__(self, expr, **consts):
        self.expr = _to_sym(expr, **consts)
        self._f = self._d = self._d2 = self._dx = self._dy = None

    def f(self):
        if self._f is None:
            self._f = _lambdify(self.expr)
        return self._f

    def dth(self):
        if self._d is None:
            self._d = _lambdify(_Lth(self.expr))
        return self._d

    def d2th(self):
        if self._d2 is None:
            self._d2 = _lambdify(_Lth(_Lth(self.expr)))
        return self._d2
    
    def dx(self, x: float, y: float): 
        if self._dx is None:
            self._dx = _lambdify(_sp.diff(self.expr, _SX))
        return self._dx(x, y)

    def dy(self, x: float, y: float):
        if self._dy is None:
            self._dy = _lambdify(_sp.diff(self.expr, _SY))
        return self._dy(x, y)


from numba_kernel import *


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

    def __init__(self, func=None, dtheta_func=None, d2theta_func=None, expr=None,
                 **consts):
        # symbolic source of truth (optional)
        self._sym = _SymField(expr, **consts) if expr is not None else None
        if func is None:
            if self._sym is None:
                raise ValueError("Mobility needs either `func` or `expr`.")
            func = self._sym.f()

        self._func         = func
        self._dtheta_func  = dtheta_func
        self._d2theta_func = d2theta_func
        self._warned       = False

    def __call__(self, X, Y):
        return self._func(X, Y)

    def _warn_fd(self, what):
        if not self._warned:
            _warnings.warn(
                f"Mobility.{what}: no analytic or symbolic derivative supplied; "
                "falling back to finite differences (inaccurate, esp. for ∂²θ). "
                "Pass `expr=` or an explicit derivative function.",
                RuntimeWarning, stacklevel=3)
            self._warned = True

    def dtheta(self, X, Y):
        """∂θ D — analytic if provided, else exact symbolic, else centred FD."""
        if self._dtheta_func is not None:
            return self._dtheta_func(X, Y)
        if self._sym is not None:
            return self._sym.dth()(X, Y)
        self._warn_fd("dtheta")
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._func(R * np.cos(th + dh), R * np.sin(th + dh))
              - self._func(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def d2theta(self, X, Y):
        """∂²θ D — analytic if provided, else exact symbolic, else 2nd-order FD."""
        if self._d2theta_func is not None:
            return self._d2theta_func(X, Y)
        if self._sym is not None:
            return self._sym.d2th()(X, Y)
        self._warn_fd("d2theta")
        dh    = 1e-3
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._func(R * np.cos(th + dh), R * np.sin(th + dh))
              - 2.0 * self._func(X, Y)
              + self._func(R * np.cos(th - dh), R * np.sin(th - dh))) / dh**2
    
    def dx(self, X, Y):
        """∂ₓ D — exact si symbolique (expr=), sinon DF centrée."""
        if self._sym is not None:
            return self._sym.dx(X, Y)
        h = 1e-5
        return (self._func(X + h, Y) - self._func(X - h, Y)) / (2.0 * h)

    def dy(self, X, Y):
        """∂y D — exact si symbolique (expr=), sinon DF centrée."""
        if self._sym is not None:
            return self._sym.dy(X, Y)
        h = 1e-5
        return (self._func(X, Y + h) - self._func(X, Y - h)) / (2.0 * h)


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
                 d2theta_fR_func=None,
                 expr_R=None, expr_T=None, expr_phi=None,
                 **consts):

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

        # ── symbolic fields (optional) — exact ∂θ replaces FD ────────────────
        self._warned = False
        self._sym_fR = self._sym_fT = None
        self._sym_fx = self._sym_fy = self._sym_phi = None

        if expr_phi is not None:
            phi = _to_sym(expr_phi, **consts)
            self._sym_phi = _SymField(phi)
            if conservative:                       # F = -∇Φ
                fxs, fys = -_sp.diff(phi, _SX), -_sp.diff(phi, _SY)
            else:                                  # divergence-free  (∂yΦ, -∂xΦ)
                fxs, fys =  _sp.diff(phi, _SY), -_sp.diff(phi, _SX)
            self._sym_fx = _SymField(fxs)
            self._sym_fy = _SymField(fys)
            self._sym_fR = _SymField((fxs * _SX + fys * _SY) / _SR)
            self._sym_fT = _SymField((-fxs * _SY + fys * _SX) / _SR)

        if expr_R is not None:
            self._sym_fR = _SymField(expr_R, **consts)
        if expr_T is not None:
            self._sym_fT = _SymField(expr_T, **consts)

    # ── Cartesian components ─────────────────────────────────────────────────

    def fx(self, X, Y):
        if self._sym_fx is not None:
            return self._sym_fx.f()(X, Y)
        elif self._potential is not None:
            return self._f_x(np.stack([X, Y], axis=-1))
        return self._f_x(X, Y)

    def fy(self, X, Y):
        if self._sym_fy is not None:
            return self._sym_fy.f()(X, Y)
        if self._potential is not None:
            return self._f_y(np.stack([X, Y], axis=-1))
        return self._f_y(X, Y)

    # ── Potential ────────────────────────────────────────────────────────────

    def potential(self, X, Y):
        if self._potential is not None:
            return self._potential(X, Y)
        else:
            return self._sym_phi.f()(X, Y)

    # ── Polar components ─────────────────────────────────────────────────────

    def fR(self, X, Y):
        """Radial component f_R at Cartesian (X, Y)."""
        if self._sym_fR is not None:
            return self._sym_fR.f()(X, Y)
        if not self.cartesian and self._f_R is not None:
            return self._f_R(X, Y)
        th = np.arctan2(Y, X)
        return self.fx(X, Y) * np.cos(th) + self.fy(X, Y) * np.sin(th)

    def fT(self, X, Y):
        """Tangential component f_θ at Cartesian (X, Y)."""
        if self._sym_fT is not None:
            return self._sym_fT.f()(X, Y)
        if not self.cartesian and self._f_theta is not None:
            return self._f_theta(X, Y)
        th = np.arctan2(Y, X)
        return -self.fx(X, Y) * np.sin(th) + self.fy(X, Y) * np.cos(th)

    # ── Angular derivatives of polar components ──────────────────────────────

    def _warn_fd(self, what):
        if not self._warned:
            _warnings.warn(
                f"Force.{what}: no analytic or symbolic derivative supplied; "
                "falling back to finite differences (inaccurate, esp. for ∂²θ on "
                "grid-interpolated/potential fields). Pass expr_R / expr_T / "
                "expr_phi or an explicit derivative function.",
                RuntimeWarning, stacklevel=3)
            self._warned = True

    def dtheta_fR(self, X, Y):
        """∂θ f_R — analytic if provided, else exact symbolic, else centred FD."""
        if self._dtheta_fR_func is not None:
            return self._dtheta_fR_func(X, Y)
        if self._sym_fR is not None:
            return self._sym_fR.dth()(X, Y)
        self._warn_fd("dtheta_fR")
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fR(R * np.cos(th + dh), R * np.sin(th + dh))
              - self.fR(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def dtheta_fT(self, X, Y):
        """∂θ f_θ — analytic if provided, else exact symbolic, else centred FD."""
        if self._dtheta_fT_func is not None:
            return self._dtheta_fT_func(X, Y)
        if self._sym_fT is not None:
            return self._sym_fT.dth()(X, Y)
        self._warn_fd("dtheta_fT")
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fT(R * np.cos(th + dh), R * np.sin(th + dh))
              - self.fT(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)

    def d2theta_fR(self, X, Y):
        """∂²θ f_R — analytic if provided, else exact symbolic, else 2nd-order FD."""
        if self._d2theta_fR_func is not None:
            return self._d2theta_fR_func(X, Y)
        if self._sym_fR is not None:
            return self._sym_fR.d2th()(X, Y)
        self._warn_fd("d2theta_fR")
        dh    = 1e-3
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self.fR(R * np.cos(th + dh), R * np.sin(th + dh))
              - 2.0 * self.fR(X, Y)
              + self.fR(R * np.cos(th - dh), R * np.sin(th - dh))) / dh**2

    def dtheta_phi(self, X, Y):
        """
        ∂φ/∂θ at fixed R — angular derivative of the potential.
        Uses analytic dpot if provided, else exact symbolic (expr_phi),
        else centred FD of φ (h = 1e-4).  Only meaningful for conservative forces.
        """
        if self.dpot is not None:
            return self.dpot(X, Y)
        if self._sym_phi is not None:
            return self._sym_phi.dth()(X, Y)
        self._warn_fd("dtheta_phi")
        dh    = 1e-4
        R, th = np.hypot(X, Y), np.arctan2(Y, X)
        return (self._potential(R * np.cos(th + dh), R * np.sin(th + dh))
              - self._potential(R * np.cos(th - dh), R * np.sin(th - dh))) / (2.0 * dh)
    
    def dfx_dx(self, X, Y):
        if self._sym_fx is not None:
            return self._sym_fx.dx(X, Y)
        h = 1e-5
        return (self.fx(X + h, Y) - self.fx(X - h, Y)) / (2.0 * h)

    def dfx_dy(self, X, Y):
        if self._sym_fx is not None:
            return self._sym_fx.dy(X, Y)
        h = 1e-5
        return (self.fx(X, Y + h) - self.fx(X, Y - h)) / (2.0 * h)

    def dfy_dx(self, X, Y):
        if self._sym_fy is not None:
            return self._sym_fy.dx(X, Y)
        h = 1e-5
        return (self.fy(X + h, Y) - self.fy(X - h, Y)) / (2.0 * h)

    def dfy_dy(self, X, Y):
        if self._sym_fy is not None:
            return self._sym_fy.dy(X, Y)
        h = 1e-5
        return (self.fy(X, Y + h) - self.fy(X, Y - h)) / (2.0 * h)


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

    def compute_control_force(self, order, eps):
        """
        Return (cx, cy) for the given perturbative order (0, 1 or 2).
        """
        eRx, eRy = self._eRx, self._eRy
        eTx, eTy = self._eTx, self._eTy
        D0 = self.D0

        cr, cth = -1., 0.

        if order >= 1:
            cr -= eps * (self.f1.fR(self.X, self.Y) - self.D1(self.X, self.Y))/D0**2
            cth -= eps * self.dT1dTh / self.R
            
        
        if order == 2:
            cr -= eps**2 * ( (self.f1.fR(self.X, self.Y)/D0 - self.D1(self.X, self.Y)/D0)**2 + (self.f2.fR(self.X, self.Y) - self.D2(self.X, self.Y))/D0 )
            cr += eps**2 * 0.5* (self.f1.fT(self.X, self.Y)/D0 - D0 * self.dT1dTh/self.R)**2
            cth -= eps**2 * self.dT2dTh / self.R
        
        
        cx = cth * eTx + cr * eRx
        cy = cth * eTy + cr * eRy
        n = np.hypot(cx, cy)

        return cx/n, cy/n
    

    # ── Velocity field ────────────────────────────────────────────────────────

    def velocity_field(self, order, ep):
        """
        Assemble the total drift velocity (vx, vy) at the given perturbative
        order for given epsilon. 
        """
        D0   = self.D0
        D1   = self.D1(self.X, self.Y)
        X, Y = self.X, self.Y

        D = D0
        fx, fy = 0, 0
        cx, cy = self.compute_control_force(order, ep)

        if order >= 1:
            D += ep * D1
            fx, fy = fx + ep * self.f1.fx(X, Y), fy + ep * self.f1.fy(X, Y)

        if order == 2:
            D += ep**2 * self.D2(self.X, self.Y)
            fx, fy = fx + ep**2 * self.f2.fx(X, Y), fy + ep**2 * self.f2.fy(X, Y)

        vx, vy = D*cx + fx, D*cy+fy

        return vx, vy

    # ── Path integration ──────────────────────────────────────────────────────

    def path_RK4(self, order, start_coords, eps, dt=0.05, steps=800, r_stop=0.1):
        """Trace an optimal path using RK4 integration of the velocity field."""
        vx, vy = self.velocity_field(order, eps)
        xr, yr = self.X[:, 0], self.Y[0, :]
        kw     = dict(bounds_error=False, fill_value=0.)
        ifx    = RegularGridInterpolator((xr, yr), vx, **kw)
        ify    = RegularGridInterpolator((xr, yr), vy, **kw)
 
        def v(xi, yi):
            p = np.array([[xi, yi]])   # (1, 2) — requis depuis scipy 1.10
            return float(ifx(p)[0]), float(ify(p)[0])
 
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


    def trace_paths(self, vx, vy, starts, dt=0.05, steps=800, r_stop=0.1):
        """
        Trace many RK4 trajectories of a given drift field (vx, vy) in
        parallel across all CPU cores (numba prange).
 
        Parameters
        ----------
        vx, vy  : (Nx, Ny) arrays   drift field on the solver grid
        starts  : (M, 2) array-like start coordinates
        dt, steps, r_stop           same meaning as path_RK4
 
        Returns
        -------
        list of (xs, ys) tuples, one per start point — identical format to
        path_RK4, so existing plotting code works unchanged.
        """
        vx = np.ascontiguousarray(vx, dtype=np.float64)
        vy = np.ascontiguousarray(vy, dtype=np.float64)
        xr, yr = self.X[:, 0], self.Y[0, :]
        x0, dx = float(xr[0]), float(xr[1] - xr[0])
        y0, dy = float(yr[0]), float(yr[1] - yr[0])
        starts = np.ascontiguousarray(
            np.asarray(starts, dtype=np.float64).reshape(-1, 2))
 
        xs, ys, n = kernel_trace_paths(
            vx, vy, x0, dx, y0, dy, starts,
            float(dt), int(steps), float(r_stop),
            self.Lx / 2.0, self.Ly / 2.0)
 
        return list(zip(xs, ys))
 
    
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
                            (- T1 + np.sqrt(np.maximum(disc, 0))) / (2 * T2),
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

    def compare_paths(self, order, func, n_theta, eps, dt=0.002, r_stop=0.1, Npaths=0):
        """Compare les trajectoires perturbatives (RK4 sur velocity_field) aux
        caractéristiques optimales EXACTES de
            ẋ = D(x)·(cosθ, sinθ) + F(x),   |c| = 1.

        Caractéristiques obtenues par intégration RÉTROGRADE du flot optimal
        depuis la cible (origine), avec la MÊME loi `func` (make_theta_eq) que
        celle définissant le modèle — cohérence vitesse/pilotage à mobilité
        variable. Renvoie (err, dtau) par trajectoire (RMS spatial resamplé par
        abscisse curviligne, erreur temporelle relative). Les trajectoires non
        valides (jamais sorties de la boîte, ou RK4 n'atteignant pas la cible)
        valent NaN -> utiliser np.nanmean en aval.
        """
        box_x, box_y = 0.9*self.Lx/2.0, 0.9*self.Ly/2.0
        GAP          = 1e-3   # écart aux singularités cosθ=0, INDÉPENDANT de eps

        tf = np.concatenate([np.linspace(-np.pi/2 + GAP,   np.pi/2 - GAP,   n_theta//2),
                             np.linspace( np.pi/2 + GAP, 3*np.pi/2 - GAP,   n_theta//2)])
        M  = tf.size

        # --- caractéristiques exactes : rétrograde depuis la cible ----------
        Vc   = self.D0
        nmax = int(3.0*np.hypot(self.Lx, self.Ly)/(dt*Vc)) + 100
        Xh, Yh = np.zeros((nmax + 1, M)), np.zeros((nmax + 1, M))
        th = tf.copy()                                  # cap AU but, pas de +pi
        for k in range(nmax):
            xk, yk = Xh[k], Yh[k]
            V  = self.D0 + eps*self.D1(xk, yk) + eps**2*self.D2(xk, yk)
            fx = eps*self.f1.fx(xk, yk) + eps**2*self.f2.fx(xk, yk)
            fy = eps*self.f1.fy(xk, yk) + eps**2*self.f2.fy(xk, yk)
            c, s = np.cos(th), np.sin(th)
            Xh[k+1] = xk - dt*(fx + V*c)                # d/dτ = -(flot avant)
            Yh[k+1] = yk - dt*(fy + V*s)
            th      = th - dt*func(th, xk, yk, eps)

        outside = (np.abs(Xh) > box_x) | (np.abs(Yh) > box_y)
        left    = outside.any(0)
        e       = np.where(left, outside.argmax(0), nmax)
        if not left.all():
            _warnings.warn(
                f"compare_paths: {int((~left).sum())}/{M} caractéristiques "
                "n'ont pas atteint la boîte (nmax trop petit ou extrémale "
                "divergente) ; marquées NaN.", RuntimeWarning, stacklevel=2)
        sp = np.ascontiguousarray(
            np.column_stack([Xh[e, np.arange(M)], Yh[e, np.arange(M)]]),
            dtype=np.float64)

        # --- RK4 perturbatif depuis les lancements vers la cible ------------
        vx, vy = self.velocity_field(order, eps)
        vx = np.ascontiguousarray(vx, dtype=np.float64)
        vy = np.ascontiguousarray(vy, dtype=np.float64)
        xr, yr = self.X[:, 0], self.Y[0, :]
        x0, dx = float(xr[0]), float(xr[1] - xr[0])
        y0, dy = float(yr[0]), float(yr[1] - yr[0])
        xs, ys, nstep = kernel_trace_paths(
            vx, vy, x0, dx, y0, dy, sp,
            float(dt), int(5000), float(r_stop), self.Lx/2.0, self.Ly/2.0)

        def _resample(x, y, L):
            s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
            if s[-1] == 0.0:
                return np.full(L, x[0]), np.full(L, y[0])
            u = np.linspace(0.0, s[-1], L)
            return np.interp(u, s, x), np.interp(u, s, y)

        L    = 200
        err  = np.full(M, np.nan)
        dtau = np.full(M, np.nan)
        for m in range(M):
            if not left[m]:
                continue
            em    = int(e[m])
            rchar = np.hypot(Xh[:em+1, m], Yh[:em+1, m])
            if not (rchar >= r_stop).any():
                continue
            kin    = int(np.argmax(rchar >= r_stop))     # 1er passage à r_stop
            cx, cy = Xh[kin:em+1, m], Yh[kin:em+1, m]     # char : r_stop -> lancement

            nm = int(nstep[m])
            # le RK4 doit avoir atteint la cible (sinon extrémale/troncature KO)
            if np.hypot(xs[m, nm-1], ys[m, nm-1]) > 2.0*r_stop:
                continue
            rx, ry = xs[m, :nm][::-1], ys[m, :nm][::-1]   # RK4 : r_stop -> lancement

            ax, ay = _resample(cx, cy, L)
            bx, by = _resample(rx, ry, L)
            err[m] = np.sqrt(np.mean((ax - bx)**2 + (ay - by)**2))

            t_char  = (em - kin)*dt
            t_rk4   = (nm - 1)*dt
            dtau[m] = (t_rk4 - t_char)/t_char if t_char > 0 else np.nan

        if Npaths:
            valid = np.where(np.isfinite(err))[0]      # n'échantillonne que les valides
            if valid.size == 0:
                return err, dtau, [[], []]
            k   = min(Npaths, valid.size)
            ids = np.random.choice(valid, size=k, replace=False)

            anpath = []   # caractéristiques exactes : origine -> lancement
            rk4  = []   # chemins RK4 perturbatifs : lancement -> cible
            for j in ids:
                ej = int(e[j])                          # index de lancement
                anpath.append((Xh[:ej+1, j].copy(), Yh[:ej+1, j].copy()))
                nj = int(nstep[j])                      # nb de pas valides du RK4
                rk4.append((xs[j, :nj].copy(), ys[j, :nj].copy()))
            return err, dtau, [anpath, rk4]
        
        return err, dtau

    def make_theta_eq(self,th,x,y, e):
        """Loi de pilotage de Zermelo à VITESSE VARIABLE pour le modèle
            ẋ = D(x)·(cosθ, sinθ) + F(x),   |c| = 1,   D = D0 + ε D1 + ε² D2.

            θ̇ = sin²θ ∂ₓv + sinθcosθ(∂ₓu − ∂yv) − cos²θ ∂yu
                + (sinθ ∂ₓD − cosθ ∂yD)

        avec u=F_x=εf1x+ε²f2x, v=F_y=εf1y+ε²f2y, D=D0+εD1+ε²D2.
        """

        dudx = e*self.f1.dfx_dx(x, y) + e**2*self.f2.dfx_dx(x, y)
        dudy = e*self.f1.dfx_dy(x, y) + e**2*self.f2.dfx_dy(x, y)
        dvdx = e*self.f1.dfy_dx(x, y) + e**2*self.f2.dfy_dx(x, y)
        dvdy = e*self.f1.dfy_dy(x, y) + e**2*self.f2.dfy_dy(x, y)

        dDdx = e*self.D1.dx(x, y) + e**2*self.D2.dx(x, y)
        dDdy = e*self.D1.dy(x, y) + e**2*self.D2.dy(x, y)

        s, c     = np.sin(th), np.cos(th)
        classical = s*s*dvdx + s*c*(dudx - dvdy) - c*c*dudy
        mobility  = s*dDdx - c*dDdy
        return classical + mobility

def nul(x, y):
    return np.zeros_like(x)


def load_compare(Lx, Ly, N):
    """
    Non-conservative polar force with spatially varying mobility.

    Analytical polar derivatives supplied for exact T2 (Eq. 1.45):
      f1R = cos(2θ)              -> d/dθ f1R    = -2 sin(2θ)
      f1θ = 1/(1+R)              -> d/dR(R f1θ) = 1/(1+R)^2
      D1  = 0.1 R sin(2θ)        -> d/dθ D1     = 0.2 R cos(2θ)
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

def load_conservative(Lx, Ly, N):
    pot = f"cos(12*{1/Lx} * pi  * x)**2*cos(12*{1/Ly} * pi * y)**2"

    f1 = Force(Lx, Ly, N, N, conservative=True, expr_phi=pot)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=nul, f_y=nul, cartesian=True)
    D1 = Mobility(expr=nul,dtheta_func=nul)
    D2 = Mobility(expr=nul,dtheta_func=nul)
    return f1, f2, D1, D2, "conservative_peaks"

def load_mob_only(Lx, Ly, N):
        
    f1 = Force(Lx, Ly, N, N, expr_phi="0", conservative=False)
    f2 = Force(Lx, Ly, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="-x",dtheta_func=lambda x,y: y)
    D2 = Mobility(expr="0",dtheta_func=nul)


    return f1, f2, D1, D2, "only_mob"

def load_divless(Lx, Ly, N):
    pot = f"cos(12*{1/Lx} * pi  * x)**2*cos(12*{1/Ly} * pi * y)**2"    

    f1 = Force(Lx, Ly, N, N, conservative=False, expr_phi=pot)
    f2 = Force(Lx, Ly, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=nul)
    D2 = Mobility(expr="0",dtheta_func=nul)
    return f1, f2, D1, D2, "div_less"

def load_cisaillement(Lx, Ly, N):
    phi="-y**2/2"

    def _zero(x,y):
        return 0*x

    f1 = Force(Lx, Ly, N, N, expr_phi=phi, conservative=False)
    f2 = Force(Lx, Ly, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=_zero)
    D2 = Mobility(expr="0",dtheta_func=_zero)
    return f1, f2, D1, D2, "cisaillement"

def load_divless_shifted(Lx, Ly, N):


    f1 = Force(Lx, Ly, N, N, conservative=False, expr_phi=f"cos(12*{1/Lx} * pi  * x)**2*cos(12*{1/Ly} * pi * y - pi/6)**2", cartesian=True)
    f2 = Force(Lx, Ly, N, N, conservative=False, expr_phi=nul)
    D1 = Mobility(nul,dtheta_func=nul)
    D2 = Mobility(nul,dtheta_func=nul)
    return f1, f2, D1, D2, "shifted"

def load_divless_mob(Lx, Ly, N):
    pot = f"cos(12*{1/Lx} * pi  * x)**2*cos(12*{1/Ly} * pi * y - pi/6)**2"


    f1 = Force(Lx, Ly, N, N, conservative=False, expr_phi=pot)
    f2 = Force(Lx, Ly, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=nul)
    D2 = Mobility(expr="0",dtheta_func=nul)
    return f1, f2, D1, D2, "div_less_mob"

def load_multiples(dx):
    p1 = ("cos(12*1/15 * pi  * x)**2*cos(12*1/15 * pi * y - pi/6)**2", 
          False,
          15)
    p2 = ("cos(12*1/15 * pi  * x)**2*cos(12*1/15 * pi * y - pi/6)**2", 
          True,
          15)
    p3 = ("cos(12*1/15 * pi  * x)**2*cos(12*1/15 * pi * y)**2", 
          False,
          15)
    p4 = ("cos(48*1/40 * pi  * x)**2*cos(48*1/40 * pi * y)**2", 
          False,
          40)
    p5 = ("cos(12*1/15 * pi  * x)**2*cos(12*1/15 * pi * y)**2", 
          True,
          15)
    p6 = ("cos(48*1/40 * pi  * x)**2*cos(48*1/40 * pi * y)**2", 
          True,
          40)
    p7 = ("cos(12*1/15 * pi  * x)**3*cos(12*1/15 * pi * y)**3", 
          False,
          15)
    p8 = ("cos(20*1/20 * pi  * x)**3*cos(20*1/20 * pi * y - pi/6)**3", 
          False,
          20)
    p9 = ("cos(12*1/15 * pi  * x)**3*cos(12*1/15 * pi * y)**3", 
          True,
          15)

    p10 = ("cos(6*1/15 * pi  * x)**2*cos(6*1/15 * pi * y)**2", 
          True,
          15)
    p11 = ("cos(6*1/15 * pi  * x)**2*cos(6*1/15 * pi * y)**2", 
          False,
          15)
    
    p12 = ("cos(20*1/20 * pi  * x)+cos(20*1/20 * pi * y)", 
          True,
          20)
    p13 = ("cos(20*1/20 * pi  * x)+cos(20*1/20 * pi * y)", 
          False,
          20)
    
    p14 = ("cos(20*1/20 * pi  * x)+cos(20*1/20 * pi * y)", 
          True,
          20)
    p15 = ("cos(20*1/20 * pi  * x)+cos(20*1/20 * pi * y)", 
          False,
          20)
    
    p16 = ("cos(20*1/20 * pi  * x)*sin(20*1/20 * pi * y)", 
          True,
          20)
    p17 = ("cos(20*1/20 * pi  * x)*sin(20*1/20 * pi * y)", 
          False,
          20)
    
    args = []
    ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17]
    for i, (pot, conservative, L) in enumerate(ps):
        print(i)
        N = int(L/dx)
        f1 = Force(L, L, N, N, conservative=conservative, expr_phi=pot)
        f2 = Force(L, L, N, N, conservative=False, expr_phi="0")
        D1 = Mobility(expr="0",dtheta_func=nul)
        D2 = Mobility(expr="0",dtheta_func=nul)
        args.append([f1, f2, D1, D2, f"data{i}", pot, L, N])
        
    return args

def load_crossed_shear(Lx, Ly, N):
    phi="-y*x"

    def _zero(x,y):
        return 0*x

    f1 = Force(Lx, Ly, N, N, expr_phi=phi, conservative=False)
    f2 = Force(Lx, Ly, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=_zero)
    D2 = Mobility(expr="0",dtheta_func=_zero)
    return f1, f2, D1, D2, "crossed_cisaillement"

def load_bash(L, N):
    pot = f"cos(2*pi* x)**2*cos(2* pi * y)**2"


    f1 = Force(L, L, N, N, conservative=False, expr_phi=pot)
    f2 = Force(L, L, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=nul)
    D2 = Mobility(expr="0",dtheta_func=nul)
    return f1, f2, D1, D2, "bash"


if __name__ == "__main__":
    
    import sys
    L = sys.argv[1]
    pot = sys.argv[2]
    conservatif = sys.argv[3]
    saveStr = sys.argv[4]
    D0 = 1.
    L = 10.
    pot = "cos(2*pi*x)**2*cos(2*pi*y)**2"
    conservatif = True
    saveStr = "post_process_test"

    dx = 0.02
    N = int(L/dx)

    f1 = Force(L, L, N, N, conservative=conservatif, expr_phi=pot)
    f2 = Force(L, L, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=nul)
    D2 = Mobility(expr="0",dtheta_func=nul)
    
   
        
    cn = control_nav(Lx=L, Ly=L, Nx=N, Ny=N, D0=D0 ,D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.01)
    print("Computing T1 ...")
    cn.compute_T1()
    print("Computing dT1/dθ ...")
    cn.compute_dT1dTh()
    print("Computing T2 ...")
    cn.compute_T2()
    print("Computing dT2/dθ ...")
    cn.compute_dT2dTh()
    print("computing control forces contributions")

    data = {
        "L": L,
        "N": N,
        "pot":pot,
        "T0":cn.T0.tolist(),
        "T1":cn.T1.tolist(),
        "T2":cn.T2.tolist(),
        "dT1dTh":cn.dT1dTh.tolist(),
        "dT2dTh":cn.dT2dTh.tolist(),
    }

    with open(f"{saveStr}.json", "w") as f:
        json.dump(data, f)
        