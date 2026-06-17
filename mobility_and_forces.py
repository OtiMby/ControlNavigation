
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
