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


from mobility_and_forces import Force, Mobility
from numba_kernel import *  # For parallel computing



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
                 f1: Force,    f2: Force):

        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.D0          = D0
        self.D1, self.D2 = D1, D2
        self.f1, self.f2 = f1, f2

        # Nr is the number of points for the radial integration: it must be odd for composite Simpson's rule
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

def nul(x, y):
    return np.zeros_like(x)


if __name__ == "__main__":
    
    import sys

    if len(sys.argv) > 1:
        L = float(sys.argv[1])
        pot = sys.argv[2]
        conservatif = 0
        if sys.argv[3] == '1':
            conservatif = 1
        saveStr = sys.argv[4]
        expr_D1 = "0"
        expr_D2 = "0"

    else:
        L = 10.
        pot = "-x**2*y**2"
        conservatif = False
        expr_D1 = "0"
        expr_D2 = "0"
        saveStr = ""

    
    
    D0 = 1.

    dx = 0.01
    N = int(L/dx)
    saveStr=f'datas/{saveStr}'
    f1 = Force(L, L, N, N, conservative=conservatif, expr_phi=pot)
    f2 = Force(L, L, N, N, conservative=True, expr_phi="0")
    D1 = Mobility(expr=expr_D1,dtheta_func=nul)
    D2 = Mobility(expr=expr_D2,dtheta_func=nul)
    
        
    cn = control_nav(Lx=L, Ly=L, Nx=N, Ny=N, D0=D0 ,D1=D1, D2=D2, f1=f1, f2=f2)
    print("computing T1...")
    cn.compute_T1()
    print("computing dT1dTh...")
    cn.compute_dT1dTh()
    print("computing T2...")
    cn.compute_T2()
    print("computing dT2dTh...")
    cn.compute_dT2dTh()

    data = {
        "L": L,
        "N": N,
        'conservative':conservatif,
        "pot":pot,
        'D0': D0,
        "D1": expr_D1,
        "D2": expr_D2,
        "T0":cn.T0.tolist(),
        "T1":cn.T1.tolist(),
        "T2":cn.T2.tolist(),
        "dT1dTh":cn.dT1dTh.tolist(),
        "dT2dTh":cn.dT2dTh.tolist(),
    }

    with open(f"{saveStr}.json", "w") as f:
        json.dump(data, f)
        