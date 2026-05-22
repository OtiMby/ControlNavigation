"""
Solve the control navigation problem with numerical methods.

Finds the optimal control direction c to minimise travel time when problems
are too complex for the analytical approach.  The FMM is then used to
compute the optimal path.
"""

import os
import scipy as sp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsp
import warnings
import skfmm
from tqdm import tqdm
import time

np.seterr(divide='ignore', invalid='ignore', over='ignore')
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Mobility
# ---------------------------------------------------------------------------

class Mobility:
    """Callable wrapper for a scalar mobility field D(X, Y).

    Parameters
    ----------
    func : callable (X, Y) -> array
        The mobility field D(X, Y).
    dtheta_func : callable (X, Y) -> array, optional
        Analytical d/dtheta D in polar coordinates, expressed as a function
        of Cartesian (X, Y).  If *None* a centred finite-difference fallback
        is used automatically.
    """

    def __init__(self, func, dtheta_func=None):
        self._func = func
        self._dtheta_func = dtheta_func

    def __call__(self, X, Y):
        return self._func(X, Y)

    def dtheta(self, X, Y):
        """Return d/dtheta D at (X, Y) -- analytical if provided, else FD fallback."""
        if self._dtheta_func is not None:
            return self._dtheta_func(X, Y)
        dh = 1e-5
        R = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        Xp = R * np.cos(th + dh); Yp = R * np.sin(th + dh)
        Xm = R * np.cos(th - dh); Ym = R * np.sin(th - dh)
        return (self._func(Xp, Yp) - self._func(Xm, Ym)) / (2 * dh)


# ---------------------------------------------------------------------------
# Force
# ---------------------------------------------------------------------------

class Force:
    """
    External force field.  Three modes:
      - conservative: derived from a potential, -grad(phi)
      - cartesian:    given as (f_x, f_y) Cartesian component functions
      - polar:        given as (f_R, f_theta) polar component functions

    fx() / fy() always return Cartesian components regardless of mode.

    For the perturbative T2 computation the following analytical polar
    derivatives can optionally be supplied; if omitted a centred
    finite-difference fallback is used.  They are not needed for
    conservative forces (only d/dtheta D1 enters the conservative formula).

    Parameters
    ----------
    dtheta_fR_func : callable (X, Y) -> array, optional
        Analytical d/dtheta f_R in polar coordinates as a function of
        Cartesian (X, Y).
    dR_RfT_func : callable (X, Y) -> array, optional
        Analytical d/dR(R * f_theta) in polar coordinates as a function of
        Cartesian (X, Y).
    """

    def __init__(self, Lx, Ly, Nx, Ny, conservative=False, f_x=None, f_y=None,
                 potential=None, cartesian=True, f_R=None, f_theta=None,
                 dtheta_fR_func=None, dR_RfT_func=None):
        self.conservative = conservative
        self.cartesian = cartesian
        self._potential = potential
        self._f_R = f_R
        self._f_theta = f_theta
        self._dtheta_fR_func = dtheta_fR_func
        self._dR_RfT_func = dR_RfT_func

        if conservative:
            Xg, Yg = np.mgrid[-Lx/2:Lx/2:Nx*1j, -Ly/2:Ly/2:Ny*1j]
            pot = potential(Xg, Yg)
            kw = dict(bounds_error=False, fill_value=0.0)
            self._f_x = RegularGridInterpolator(
                (Xg[:, 0], Yg[0, :]), -np.gradient(pot, Xg[:, 0], axis=0), **kw)
            self._f_y = RegularGridInterpolator(
                (Xg[:, 0], Yg[0, :]), -np.gradient(pot, Yg[0, :], axis=1), **kw)
        elif cartesian:
            self._f_x = f_x
            self._f_y = f_y
        else:
            self._f_x = lambda X, Y: (
                f_R(X, Y) * np.cos(np.arctan2(Y, X) % (2 * np.pi))
                - f_theta(X, Y) * np.sin(np.arctan2(Y, X) % (2 * np.pi))
            )
            self._f_y = lambda X, Y: (
                f_R(X, Y) * np.sin(np.arctan2(Y, X) % (2 * np.pi))
                + f_theta(X, Y) * np.cos(np.arctan2(Y, X) % (2 * np.pi))
            )

    def fx(self, X, Y):
        if self.conservative:
            return self._f_x(np.stack([X, Y], axis=-1))
        return self._f_x(X, Y)

    def fy(self, X, Y):
        if self.conservative:
            return self._f_y(np.stack([X, Y], axis=-1))
        return self._f_y(X, Y)

    def potential(self, X, Y):
        return self._potential(X, Y)

    # ------------------------------------------------------------------
    # Internal polar-component helpers (used by FD fallbacks)
    # ------------------------------------------------------------------

    def _fR(self, X, Y):
        """Radial component f_R at Cartesian (X, Y)."""
        if not self.cartesian and self._f_R is not None:
            return self._f_R(X, Y)
        eRx = np.cos(np.arctan2(Y, X))
        eRy = np.sin(np.arctan2(Y, X))
        return self.fx(X, Y) * eRx + self.fy(X, Y) * eRy

    def _fT(self, X, Y):
        """Tangential component f_theta at Cartesian (X, Y)."""
        if not self.cartesian and self._f_theta is not None:
            return self._f_theta(X, Y)
        eTx = -np.sin(np.arctan2(Y, X))
        eTy =  np.cos(np.arctan2(Y, X))
        return self.fx(X, Y) * eTx + self.fy(X, Y) * eTy

    # ------------------------------------------------------------------
    # Analytical polar derivatives for T2
    # ------------------------------------------------------------------

    def dtheta_fR(self, X, Y):
        """Return d/dtheta f_R at (X, Y) -- analytical if provided, else FD fallback."""
        if self._dtheta_fR_func is not None:
            return self._dtheta_fR_func(X, Y)
        dh = 1e-5
        R = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        Xp = R * np.cos(th + dh); Yp = R * np.sin(th + dh)
        Xm = R * np.cos(th - dh); Ym = R * np.sin(th - dh)
        return (self._fR(Xp, Yp) - self._fR(Xm, Ym)) / (2 * dh)

    def dR_RfT(self, X, Y):
        """Return d/dR(R * f_theta) at (X, Y) -- analytical if provided, else FD fallback."""
        if self._dR_RfT_func is not None:
            return self._dR_RfT_func(X, Y)
        dh = 1e-5
        R = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        Rp = R + dh; Rm = np.maximum(R - dh, 1e-12)
        Xp = Rp * np.cos(th); Yp = Rp * np.sin(th)
        Xm = Rm * np.cos(th); Ym = Rm * np.sin(th)
        return (Rp * self._fT(Xp, Yp) - Rm * self._fT(Xm, Ym)) / (2 * dh)


# ---------------------------------------------------------------------------
# control_nav
# ---------------------------------------------------------------------------

class control_nav:
    def __init__(self, Lx, Ly, Nx, Ny, D0, D1, D2, f1, f2, epsilon=0.2):
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        self.Nr = 2 * int(np.hypot(Nx, Ny))

        self.X, self.Y = np.mgrid[-Lx/2:Lx/2:Nx*1j, -Ly/2:Ly/2:Ny*1j]
        self.D0, self.D1, self.D2 = D0, D1, D2
        self.f1, self.f2 = f1, f2
        self.epsilon = epsilon

        self.R = np.where(np.hypot(self.X, self.Y) == 0, 1e-12, np.hypot(self.X, self.Y))
        self.T0 = self.R / self.D0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _theta(X, Y):
        return np.arctan2(Y, X) % (2 * np.pi)

    def _r_line(self, x, y):
        r = np.linspace(np.zeros_like(np.hypot(x, y)), np.hypot(x, y), self.Nr)
        return np.where(r == 0, 1e-12, r)

    def eRa(self, X, Y):
        th = self._theta(X, Y)
        return np.cos(th), np.sin(th)

    def eTh(self, X, Y):
        th = self._theta(X, Y)
        return -np.sin(th), np.cos(th)

    def interp(self, field, method='linear'):
        return RegularGridInterpolator(
            (self.X[:, 0], self.Y[0, :]), field,method=method, bounds_error=False, fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Integrand for T1
    # ------------------------------------------------------------------

    def _f1_radial_integrand(self, Xc, Yc):
        val = -self.D1(Xc, Yc)
        if not self.f1.conservative:
            if self.f1.cartesian:
                eRx, eRy = self.eRa(Xc, Yc)
                val += self.f1.fx(Xc, Yc) * eRx + self.f1.fy(Xc, Yc) * eRy
            else:
                val += self.f1._f_R(Xc, Yc)
        return val

    # ------------------------------------------------------------------
    # T1
    # ------------------------------------------------------------------

    def compute_T1(self):
        R_line = self._r_line(self.X, self.Y)
        th = self._theta(self.X, self.Y)
        Xc, Yc = R_line * np.cos(th), R_line * np.sin(th)

        self.T1 = sp.integrate.simpson(self._f1_radial_integrand(Xc, Yc), x=R_line, axis=0) / self.D0

        if self.f1.conservative:
            self.T1 += (self.f1.potential(self.X, self.Y) - self.f1.potential(0, 0)) / self.D0**2

    # ------------------------------------------------------------------
    # T2  --  implements Eq. 1.48 (non-conservative) / Eq. 1.51 (conservative)
    #
    # The double-integral negative term common to both equations is
    #
    #   -1/(2 R\'2 D0^2) * [ integral_0^{R\'} inner_integrand dR\" ]^2
    #
    # where
    #   conservative    (Eq. 1.51): inner_integrand = d/dtheta D1
    #   non-conservative (Eq. 1.48): inner_integrand = d/dR(R f1theta) - d/dtheta f1R + d/dtheta D1
    #
    # These derivatives come from the analytical functions supplied to
    # Mobility (dtheta_func) and Force (dtheta_fR_func, dR_RfT_func),
    # or from centred finite differences when those are not provided.
    # ------------------------------------------------------------------

    def _t2_integrand(self, Xc, Yc, R_line):
        """Compute the T2 radial integrand at positions (Xc, Yc) along R_line."""
        eRx, eRy = self.eRa(Xc, Yc)
        eTx, eTy = self.eTh(Xc, Yc)
        D1 = self.D1(Xc, Yc)
        D2 = self.D2(Xc, Yc)

        if self.f1.cartesian:
            f1R = self.f1.fx(Xc, Yc) * eRx + self.f1.fy(Xc, Yc) * eRy
            f1T = self.f1.fx(Xc, Yc) * eTx + self.f1.fy(Xc, Yc) * eTy
        else:
            f1R = self.f1._f_R(Xc, Yc)
            f1T = self.f1._f_theta(Xc, Yc)

        if self.f2.cartesian:
            f2R = self.f2.fx(Xc, Yc) * eRx + self.f2.fy(Xc, Yc) * eRy
        else:
            f2R = self.f2._f_R(Xc, Yc)

        dth_D1 = self.D1.dtheta(Xc, Yc)

        if self.f1.conservative:
            inner_integrand = dth_D1
        else:
            inner_integrand = (self.f1.dR_RfT(Xc, Yc) *0
                               - self.f1.dtheta_fR(Xc, Yc) *0
                               - dth_D1)

        inner_integral = np.zeros_like(R_line)
        inner_integral[1:] = sp.integrate.cumulative_trapezoid(inner_integrand, x=R_line, axis=0)

        expr1 = (f2R - D2) / self.D0  * 0
        expr2 = ((f1R - D1)**2 + f1T**2 / 2) / self.D0**2  *0
        expr3 = -inner_integral**2 / (2 * R_line**2 * self.D0**2) 
        return expr1 + expr2 + expr3

    def compute_T2(self):
        R_line = self._r_line(self.X, self.Y)
        th = self._theta(self.X, self.Y)
        Xc, Yc = R_line * np.cos(th), R_line * np.sin(th)

        self.T2 = sp.integrate.simpson(self._t2_integrand(Xc, Yc, R_line), x=R_line, axis=0) / self.D0

    def compute_dT1(self):
        R_line = self._r_line(self.X, self.Y)
        th = self._theta(self.X, self.Y)
        Xc, Yc = R_line * np.cos(th), R_line * np.sin(th)

        # dT1/dtheta integrand for control force
        dth_D1 = self.D1.dtheta(Xc, Yc)
        if self.f1.conservative:
            dT1dth_integrand = -dth_D1
        else:
            dT1dth_integrand = self.f1.dtheta_fR(Xc, Yc) - dth_D1

        dT1dth_polar = np.zeros_like(R_line)
        dT1dth_polar[1:] = sp.integrate.cumulative_trapezoid(dT1dth_integrand, x=R_line, axis=0)
        dT1dth_polar /= self.D0
        self.dT1dth = dT1dth_polar[-1]

    def compute_dT2(self):
        # Differentiate under the radial integral: perturb theta by dh, recompute
        # the T2 integrand along the same radial line, then take centered differences.
        # More accurate than np.gradient on the 2D Cartesian T2 field.
        dh = 1e-5
        R_line = self._r_line(self.X, self.Y)
        th = self._theta(self.X, self.Y)
        Xcp = R_line * np.cos(th + dh)
        Ycp = R_line * np.sin(th + dh)
        Xcm = R_line * np.cos(th - dh)
        Ycm = R_line * np.sin(th - dh)
        integrand_p = self._t2_integrand(Xcp, Ycp, R_line)
        integrand_m = self._t2_integrand(Xcm, Ycm, R_line)
        d_integrand = (integrand_p - integrand_m) / (2 * dh)
        self.dT2dth = sp.integrate.simpson(d_integrand, x=R_line, axis=0) / self.D0
    # ------------------------------------------------------------------
    # Control force
    # ------------------------------------------------------------------

    def compute_control_force(self, order):
        eRx, eRy = self.eRa(self.X, self.Y)
        eTx, eTy = self.eTh(self.X, self.Y)

        if order == 0:
            return -eRx, -eRy

        if order == 1:
            expr = -self.dT1dth * self.D0 / self.R
            return eTx * expr, eTy * expr

        # order == 2
        D1  = self.D1(self.X, self.Y)
        f1R = (self.f1.fx(self.X, self.Y) * eRx
               + self.f1.fy(self.X, self.Y) * eRy)

        expr_r = self.dT1dth**2 * self.D0**2 / (2 * self.R**2)
        expr_t = (self.dT1dth * (f1R - D1) - self.D0 * self.dT2dth) / self.R

        return (eRx * expr_r + eTx * expr_t,
                eRy * expr_r + eTy * expr_t)


# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------

def plot_overview_figure(
    cn,
    starting_points,
    saveStr="simple_dynamics_overview",
    path_algorithm="FMM",
    forces=None,
    potential=None,
    mobilities=None,
    times=None,
    contour=1,
    c=None,
    path=None,
    time_overlay=True,
):
    
    forces     = [1, [1, 1], [1, 1]] if forces     is None else forces
    potential  = [1, 1]              if potential  is None else potential
    mobilities = [1, 1]              if mobilities is None else mobilities
    times      = [1, 1, 1, 1]       if times      is None else times
    c          = [1, 1, 1, 1]       if c          is None else c
    path       = [1, 1, 1, 1]       if path       is None else path

    X, Y   = cn.X, cn.Y
    ep     = cn.epsilon
    D0     = cn.D0
    EXTENT = [-cn.Lx / 2, cn.Lx / 2, -cn.Ly / 2, cn.Ly / 2]
    TICKS  = np.linspace(-cn.Lx / 2, cn.Lx / 2, 5)

    fig = plt.figure(figsize=(15, 22))
    gs  = gridsp.GridSpec(
        5, 4, figure=fig,
        height_ratios=[1, 1, 1, 1, 1.1],
        left=0.07, right=0.97,
        bottom=0.03, top=0.94,
        hspace=0.55, wspace=0.38,
    )

    fR1_plot  = fig.add_subplot(gs[0, 0])
    fT1_plot  = fig.add_subplot(gs[0, 1])
    fR2_plot  = fig.add_subplot(gs[0, 2])
    fT2_plot  = fig.add_subplot(gs[0, 3])
    fTot_plot = fig.add_subplot(gs[1, 0])
    phi1_plot = fig.add_subplot(gs[1, 1])
    m1_plot   = fig.add_subplot(gs[1, 2])
    m2_plot   = fig.add_subplot(gs[1, 3])
    tTot_plot = fig.add_subplot(gs[2, 0])
    t0_plot   = fig.add_subplot(gs[2, 1])
    t1_plot   = fig.add_subplot(gs[2, 2])
    t2_plot   = fig.add_subplot(gs[2, 3])
    cTot_plot = fig.add_subplot(gs[3, 0])
    c0_plot   = fig.add_subplot(gs[3, 1])
    c1_plot   = fig.add_subplot(gs[3, 2])
    c2_plot   = fig.add_subplot(gs[3, 3])
    traj_plot4 = fig.add_subplot(gs[4, 0])
    traj_plot1 = fig.add_subplot(gs[4, 1])
    traj_plot2 = fig.add_subplot(gs[4, 2])
    traj_plot3 = fig.add_subplot(gs[4, 3])

    SECTION_CONFIGS = [
        (fR1_plot,   "EXTERNAL FIELDS  --  forces / potential / mobilities", "#2c3e50"),
        (tTot_plot,  "OPTIMAL TRAVEL TIME  --  T0 + e T1 + e2 T2",          "#1a5276"),
        (cTot_plot,  "CONTROL DIRECTION  --  c0 / c1 / c2",                  "#145a32"),
        (traj_plot4, "TRAJECTORIES",                                           "#4a235a"),
    ]

    def _draw_section_banners():
        for anchor_ax, label, color in SECTION_CONFIGS:
            pos = anchor_ax.get_position()
            fig.text(
                0.5, pos.y1 + 0.005, label,
                transform=fig.transFigure,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="white",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                          edgecolor="none", alpha=1.0),
            )

    QSTRIDE = max(1, cn.Nx // 60)
    Xq, Yq  = X[::QSTRIDE, ::QSTRIDE], Y[::QSTRIDE, ::QSTRIDE]

    def _quiver(ax, ux, uy, color):
        ax.quiver(Xq, Yq,
                  ux[::QSTRIDE, ::QSTRIDE], uy[::QSTRIDE, ::QSTRIDE],
                  color=color, scale=None, scale_units="xy", angles="xy",
                  width=0.005, alpha=0.85)

    def _add_colorbar(ax, im):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))

    def _style(ax, title, left=False, bottom=False):
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_aspect("equal")
        ax.set_xlim(-cn.Lx / 2, cn.Lx / 2)
        ax.set_ylim(-cn.Ly / 2, cn.Ly / 2)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.set_xticklabels([f"{v:.1f}" for v in TICKS] if bottom else [], fontsize=6)
        ax.set_yticklabels([f"{v:.1f}" for v in TICKS] if left  else [], fontsize=6)
        ax.tick_params(length=2)
        ax.plot(0, 0, "D", ms=2, color="black", zorder=5)

    def _ishow(ax, data, cmap):
        im = ax.imshow(data, cmap=cmap, extent=EXTENT, origin="lower")
        _add_colorbar(ax, im)

    eRx, eRy = cn.eRa(X, Y)
    eTx, eTy = cn.eTh(X, Y)

    print("Plotting force / potential / mobility fields ...")

    def _decompose_force(force):
        if force.cartesian:
            fx, fy = force.fx(X, Y), force.fy(X, Y)
            fR_s = fx * eRx + fy * eRy
            fT_s = fx * eTx + fy * eTy
            fRx, fRy = fR_s * eRx, fR_s * eRy
            fTx, fTy = fT_s * eTx, fT_s * eTy
        else:
            fR_s, fT_s = force._f_R(X, Y), force._f_theta(X, Y)
            fRx, fRy = fR_s * eRx, fR_s * eRy
            fTx, fTy = fT_s * eTx, fT_s * eTy
            fx, fy   = fRx + fTx, fRy + fTy
        return fRx, fRy, fTx, fTy, fx, fy

    f1Rx, f1Ry, f1Tx, f1Ty, f1x, f1y = _decompose_force(cn.f1)
    f2Rx, f2Ry, f2Tx, f2Ty, f2x, f2y = _decompose_force(cn.f2)
    D1 = cn.D1(X, Y)
    D2 = cn.D2(X, Y)

    if forces[1][0]: _quiver(fR1_plot,  f1Rx, f1Ry, "#2980b9")
    if forces[1][1]: _quiver(fT1_plot,  f1Tx, f1Ty, "#2980b9")
    if forces[2][0]: _quiver(fR2_plot,  f2Rx, f2Ry, "#8e44ad")
    if forces[2][1]: _quiver(fT2_plot,  f2Tx, f2Ty, "#8e44ad")
    if forces[0]:
        _quiver(fTot_plot, ep * f1x + ep**2 * f2x, ep * f1y + ep**2 * f2y, "#2c3e50")

    if cn.f1.conservative and potential[0]:
        _ishow(phi1_plot, cn.f1.potential(X, Y), "RdPu")

    if mobilities[0]: _ishow(m1_plot, D1, "Blues")
    if mobilities[1]: _ishow(m2_plot, D2, "Blues")

    _style(fR1_plot,  r"$f_{1R}$",          left=True)
    _style(fT1_plot,  r"$f_{1\theta}$")
    _style(fR2_plot,  r"$f_{2R}$")
    _style(fT2_plot,  r"$f_{2\theta}$")
    _style(fTot_plot, r"$\vec{F}_{tot}$",    left=True, bottom=True)
    _style(phi1_plot, r"$\phi_1(\vec{x})$",  bottom=True)
    _style(m1_plot,   r"$D_1(\vec{x})$",     bottom=True)
    _style(m2_plot,   r"$D_2(\vec{x})$",     bottom=True)

    print("Plotting travel-time fields ...")

    T0, T1, T2 = cn.T0, cn.T1, cn.T2
    tTot = T0 + ep * T1 + ep**2 * T2

    if times[0]:
        _ishow(tTot_plot, tTot, "YlOrRd")
        if contour:
            line = T0 - ep * np.abs(T1) - ep**2 * np.abs(T2)
            print(np.min(line), np.max(line))
            if np.min(line) < 0 < np.max(line):
                cs = tTot_plot.contour(X, Y, line, levels=[0], colors="black", linewidths=0.6, linestyles="--")
                tTot_plot.contour(X, Y, line, levels=[1], colors="red", linewidths=0.6, linestyles="--")
                verts = np.concatenate([p.vertices for p in cs.collections[0].get_paths()])
                idx = np.linspace(0, len(verts) - 1, 6, dtype=int)[1:-1]
                grad_x, grad_y = np.gradient(line, X[:, 0], Y[0, :])
                igrad_x = RegularGridInterpolator((X[:, 0], Y[0, :]), grad_x, bounds_error=False, fill_value=0.)
                igrad_y = RegularGridInterpolator((X[:, 0], Y[0, :]), grad_y, bounds_error=False, fill_value=0.)
                for xi, yi in verts[idx]:
                    pt  = np.array([[xi, yi]])
                    gx, gy = float(igrad_x(pt)), float(igrad_y(pt))
                    mag = np.hypot(gx, gy)
                    if mag < 1e-10:
                        continue
                    scale = 0.07 * cn.Lx
                    tTot_plot.annotate(
                        "", xy=(xi + gx / mag * scale, yi + gy / mag * scale),
                        xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1, mutation_scale=3),
                    )

    if times[1]: _ishow(t0_plot, T0, "YlOrRd")
    if times[2]: _ishow(t1_plot, T1, "YlOrRd")
    if times[3]: _ishow(t2_plot, T2, "YlOrRd")

    _style(tTot_plot, rf"$\mathcal{{T}}^*$  ($\varepsilon={ep:.2g}$)", left=True, bottom=True)
    _style(t0_plot,   r"$\mathcal{T}^*_0$", bottom=True)
    _style(t1_plot,   r"$\mathcal{T}^*_1$", bottom=True)
    _style(t2_plot,   r"$\mathcal{T}^*_2$", bottom=True)

    c0x, c0y = cn.compute_control_force(0)
    c1x, c1y = cn.compute_control_force(1)
    c2x, c2y = cn.compute_control_force(2)

    if c[0]: _quiver(cTot_plot, c0x + ep*c1x + ep**2*c2x, c0y + ep*c1y + ep**2*c2y, "#1e8449")
    if c[1]: _quiver(c0_plot,   c0x, c0y, "#1a5276")
    if c[2]: _quiver(c1_plot,   c1x, c1y, "#922b21")
    if c[3]: _quiver(c2_plot,   c2x, c2y, "#1a5276")

    _style(cTot_plot, r"$\hat{c}$  to  $\mathcal{O}(\varepsilon^2)$", left=True, bottom=True)
    _style(c0_plot,   r"$\hat{c}_0$", bottom=True)
    _style(c1_plot,   r"$\hat{c}_1$", bottom=True)
    _style(c2_plot,   r"$\hat{c}_2$", bottom=True)

    def _rgi(field):
        return RegularGridInterpolator((X[:, 0], Y[0, :]), field, bounds_error=False, fill_value=0.)

    def RK4(order, start_coords, dt=0.05, steps=800, r_stop=0.1):
        vx = D0 * c0x; vy = D0 * c0y
        if order >= 1:
            vx += ep * (D0 * c1x + D1 * c0x)
            vy += ep * (D0 * c1y + D1 * c0y)
        if order == 2:
            vx += ep**2 * (D0 * c2x + f2x + D1 * c1x)
            vy += ep**2 * (D0 * c2y + f2y + D1 * c1y)
        
        ifx, ify = _rgi(vx), _rgi(vy)
        x, y = start_coords; xs, ys = [x], [y]
        for _ in range(steps):
            def v(xi, yi):
                p = np.array([xi, yi])
                return float(ifx(p)), float(ify(p))
            k1x, k1y = v(x, y)
            k2x, k2y = v(x + dt/2*k1x, y + dt/2*k1y)
            k3x, k3y = v(x + dt/2*k2x, y + dt/2*k2y)
            k4x, k4y = v(x + dt*k3x,   y + dt*k3y)
            x += dt / 6 * (k1x + 2*k2x + 2*k3x + k4x)
            y += dt / 6 * (k1y + 2*k2y + 2*k3y + k4y)
            xs.append(x); ys.append(y)
            if abs(x) > cn.Lx/2 or abs(y) > cn.Ly/2 or np.hypot(x, y) < r_stop:
                break
        return np.array(xs), np.array(ys)

    def FMM(order, start_coords, step_size, n_steps, r_stop):
        x0, y0 = start_coords
        xr = np.linspace(-cn.Lx/2, cn.Lx/2, cn.Nx)
        yr = np.linspace(-cn.Ly/2, cn.Ly/2, cn.Ny)
        vxf = D0 * c0x; vyf = D0 * c0y
        if order >= 1:
            vxf += ep * (D0 * c1x + D1 * c0x + f1x)
            vyf += ep * (D0 * c1y + D1 * c0y + f1y)
        if order == 2:
            vxf += ep**2 * (D0 * c2x + f2x + D1 * c1x)
            vyf += ep**2 * (D0 * c2y + f2y + D1 * c1y)
        phi = np.ones_like(vxf)
        phi[cn.Nx // 2, cn.Ny // 2] = 0
        t_map = skfmm.travel_time(phi, np.hypot(vxf, vyf))
        gx, gy = np.gradient(t_map, xr, yr)
        igx = RegularGridInterpolator((xr, yr), gx)
        igy = RegularGridInterpolator((xr, yr), gy)
        curr = np.array([x0, y0], float); xs, ys = [x0], [y0]
        for _ in range(n_steps):
            if not (xr[0] < curr[0] < xr[-1] and yr[0] < curr[1] < yr[-1]):
                break
            g = np.array([igx(curr)[0], igy(curr)[0]])
            mag = np.linalg.norm(g)
            if mag < 1e-8:
                break
            curr -= (g / mag) * step_size
            xs.append(curr[0]); ys.append(curr[1])
            if np.hypot(*curr) < r_stop:
                break
        return np.array(xs), np.array(ys)

    algo = FMM if path_algorithm == "FMM" else RK4

    if path[0]:
        print("Integrating order-comparison trajectories ...")
        x0r, y0r = starting_points[0]
        t4 = [algo(o, (x0r, y0r), 0.01, 5000, 0.1) for o in range(3)]
        for (tx, ty), col, lbl in [
            (t4[0], "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"),
            (t4[1], "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"),
            (t4[2], "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"),
        ]:
            traj_plot4.plot(tx, ty, "-", color=col, lw=1.5, label=lbl)
        traj_plot4.plot(x0r, y0r, "ko", ms=3)
        traj_plot4.legend(fontsize=6, loc="upper right", framealpha=0.7, edgecolor="none")

    if sum(path[1:]):
        print("Integrating multi-start trajectories ...")
        pt_colors = plt.cm.tab10(np.linspace(0, 0.9, len(starting_points)))
        trajs   = [[algo(o, sp, 0.01, 5000, 0.1) for sp in starting_points] for o in range(3)]
        tfields = [T0, T0 + ep*T1, T0 + ep*T1 + ep**2*T2]
        for order, (ax, pidx, tfield) in enumerate([
            (traj_plot1, 1, tfields[0]),
            (traj_plot2, 2, tfields[1]),
            (traj_plot3, 3, tfields[2]),
        ]):
            if path[pidx] and time_overlay:
                ax.imshow(tfield.T, cmap="YlOrRd", extent=EXTENT, origin="lower", alpha=0.95)
            if path[pidx]:
                for (x0i, y0i), col in zip(starting_points, pt_colors):
                    tx, ty = trajs[order][starting_points.index([x0i, y0i])]
                    ax.plot(tx, ty, "-", color=col, lw=0.8)
                    ax.plot(x0i, y0i, "o", color=col, ms=2.5)

    _style(traj_plot4, r"Order comparison  $(x_0^{ref}, y_0^{ref})$", left=True, bottom=True)
    _style(traj_plot1, r"$v_0$ only",                               bottom=True)
    _style(traj_plot2, r"$v_0+\varepsilon v_1$",                    bottom=True)
    _style(traj_plot3, r"$v$ to $\mathcal{O}(\varepsilon^2)$",     bottom=True)

    fig.suptitle(
        rf"Perturbative control navigation -- $\varepsilon={ep:.2g}$"
        rf"   grid ${cn.Nx}\times{cn.Ny}$   domain $[\!-\!{cn.Lx/2:.0f},{cn.Lx/2:.0f}]^2$",
        fontsize=11, fontweight="bold", y=0.975,
    )
    _draw_section_banners()

    my_path  = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(my_path, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, saveStr + ".pdf")
    print(f"Saving -> {out_path}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Preset loaders
# ---------------------------------------------------------------------------

def load_compare(Lx, Ly, N):
    """
    Non-conservative polar force with spatially varying mobility.

    Analytical polar derivatives are supplied for exact T2 (Eq. 1.48):
      f1R = cos(2*theta)       ->  d/dtheta f1R    = -2 sin(2*theta)
      f1theta = 1/(1+R)        ->  d/dR(R*f1theta) = 1/(1+R)^2
      D1 = 0.1*R*sin(2*theta)  ->  d/dtheta D1     = 0.2*R*cos(2*theta)
    """
    def f_R1(X, Y):
        return np.cos(2 * (np.arctan2(Y, X) % (2 * np.pi))) 

    def f_T1(X, Y):
        R = np.hypot(X, Y)
        return (1 + R**2)**(-1) 

    def f_R2(X, Y):
        R = np.hypot(X, Y); th = np.arctan2(Y, X) % (2 * np.pi)
        return np.exp(-R**2 / 10) * np.sin(2 * th) 

    def f_T2(X, Y):
        return np.zeros_like(X)

    def _D1(X, Y):
        R = np.hypot(X, Y); th = np.arctan2(Y, X) % (2 * np.pi)
        return 0.1 * R * np.sin(2 * th)  

    def _D2(X, Y):
        R = np.hypot(X, Y); th = np.arctan2(Y, X) % (2 * np.pi)
        return np.exp(-(R - 3)**2) * np.cos(2 * th) 

    def _dtheta_f1R(X, Y):
        return -2 * np.sin(2 * (np.arctan2(Y, X) % (2 * np.pi))) 

    def _dR_Rf1T(X, Y):
        R = np.hypot(X, Y)
        return (1.0 - R**2) / (1 + R**2)**2 

    def _dtheta_D1(X, Y):
        R = np.hypot(X, Y); th = np.arctan2(Y, X) % (2 * np.pi)
        return 0.2 * R * np.cos(2 * th)  

    f1 = Force(Lx, Ly, N, N, conservative=False, f_R=f_R1, f_theta=f_T1, cartesian=False,
               dtheta_fR_func=_dtheta_f1R, dR_RfT_func=_dR_Rf1T)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_R=f_R2, f_theta=f_T2, cartesian=False)
    D1 = Mobility(_D1, dtheta_func=_dtheta_D1)
    D2 = Mobility(_D2)
    return f1, f2, D1, D2, "compare"


def load_conservative1(Lx, Ly, N):
    def pot1(x, y):
        k = 2 * np.pi * (5 / Lx)
        return 4 * np.cos(k * x)**2 * np.cos(k * y)**2

    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=True, potential=pot1)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(lambda X, Y: np.zeros_like(X), dtheta_func=lambda X, Y: np.zeros_like(X))
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "conservative_peaks"


def load_conservative2(Lx, Ly, N):
    def pot1(x, y):
        k = 2 * np.pi * (5 / Lx)
        return -4 * np.cos(k * x)**2 * np.cos(k * y)**2

    def _zero(x, y):
        return np.zeros_like(x)

    f1 = Force(Lx, Ly, N, N, conservative=True, potential=pot1)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(lambda X, Y: np.zeros_like(X), dtheta_func=lambda X, Y: np.zeros_like(X))
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "conservative_wells"


def load_mob_only(Lx, Ly, N):
    def _zero(x, y):
        return np.zeros_like(x)

    # D1 = X = R cos(theta)  ->  d/dtheta D1 = -R sin(theta) = -Y
    def _D1(X, Y):
        return X.copy()

    def _dtheta_D1(X, Y):
        return -Y.copy()

    f1 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    f2 = Force(Lx, Ly, N, N, conservative=False, f_x=_zero, f_y=_zero, cartesian=True)
    D1 = Mobility(_D1, dtheta_func=_dtheta_D1)
    D2 = Mobility(lambda X, Y: np.zeros_like(X))
    return f1, f2, D1, D2, "only_mob"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(func):
    Lx, Ly = 10, 10
    N  = int(Lx / 0.1)
    D0 = 1.0
    r  = 0.9 * Lx / 2
    starting_points = [
        [ r,  r], [-r,  r], [-r, -r], [ r, -r],
        [0.0,  r], [-r, 0.0], [0.0, -r], [ r, 0.0],
    ]

    f1, f2, D1, D2, saveStr = func(Lx, Ly, N)
    cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0, D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.2)
    print("Computing T1 ...")
    cn.compute_T1()
    cn.compute_dT1()
    print("Computing T2 ...")
    cn.compute_T2()
    print("Computing dT2 ...")
    cn.compute_dT2()
    print("Plotting ...")
    plot_overview_figure(cn, starting_points, saveStr=saveStr, path_algorithm="RK4",
        path=[0,0,0,0], times=[1, 1, 1, 1],
        forces=[1, [1, 1], [1, 0]], potential=[0, 0],
        mobilities=[1, 1], contour=0)


def computation_time_test(Lx, Ly, N):
    D0 = 1.0
    chronos = []
    for dx in tqdm([0.1 + 0.005 * i for i in range(1, 41)]):
        N = int(Lx / dx)
        f1, f2, D1, D2, _ = load_conservative2(Lx, Ly, N)
        cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0, D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.15)
        t0 = time.time()
        cn.compute_T1()
        cn.compute_T2()
        cn.compute_dT2()
        cn.compute_control_force(0)
        cn.compute_control_force(1)
        cn.compute_control_force(2)
        chronos.append((dx, time.time() - t0))

    with open("computation_times.txt", "a") as fh:
        for dx, t in chronos:
            fh.write(f"dx={dx:.3f}: {t:.2f} seconds\n")
    print("\n".join(f"  dx={dx:.3f}: {t:.2f}s" for dx, t in chronos))
    plt.loglog([dx for dx, _ in chronos], [t for _, t in chronos], marker="o")
    plt.show()


if __name__ == "__main__":
    run(load_compare)
