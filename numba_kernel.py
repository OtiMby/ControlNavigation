"""
Numba-accelerated integration kernels for the perturbative control navigation solver.

Strategy
--------
Every kernel iterates pixel-by-pixel (prange over i, j) and accumulates
integrals in scalar running variables.  No (Nr, Nx, Ny) temporaries are
ever allocated; memory footprint is O(Nx · Ny) regardless of Nr.

Fields are sampled via bilinear interpolation on the pre-computed 2-D grids
supplied by control_nav.precompute_fields().

Conventions
-----------
* Simpson's rule requires Nr to be odd (even number of sub-intervals).
  control_nav.__init__ guarantees this.
* All accumulators use the trapezoid rule (1st-order, O(dr²) local error)
  which is unconditionally stable for the running integrals.
* L'Hôpital limits are applied analytically at r = 0 for every ratio g/r,
  g2/r, h/r, h2/r.
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bl(g, x0, dx, y0, dy, x, y):
    """
    Bilinear interpolation on a regular (Nx, Ny) grid.

    Parameters
    ----------
    g           : (Nx, Ny) float64 array — field values
    x0, dx      : x-origin and step
    y0, dy      : y-origin and step
    x, y        : query point (scalars)
    """
    nx, ny = g.shape
    ix = (x - x0) / dx
    iy = (y - y0) / dy
    i0 = max(0, min(int(ix), nx - 2))
    j0 = max(0, min(int(iy), ny - 2))
    fx = ix - i0;  fy = iy - j0
    return (g[i0,   j0  ] * (1.0 - fx) * (1.0 - fy)
          + g[i0+1, j0  ] * fx          * (1.0 - fy)
          + g[i0,   j0+1] * (1.0 - fx) * fy
          + g[i0+1, j0+1] * fx          * fy)


@njit(cache=True)
def _simp(buf, dr):
    """
    Composite Simpson's rule on Nr uniformly-spaced points with spacing dr.
    Nr must be odd (even number of sub-intervals).
    """
    Nr = buf.shape[0]
    s  = buf[0] + buf[Nr - 1]
    for k in range(1, Nr - 1):
        s += (4.0 if k & 1 else 2.0) * buf[k]
    return s * dr / 3.0


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def kernel_T1_noncons(X, Y, R, x0, dx, y0, dy, D1_g, f1R_g, D0, Nr):
    """
    T1 — non-conservative f1.

    Integrand : f1R(r, θ) − D1(r, θ)
    Result    : T1 = ∫₀^R [integrand] dr / D0²
    """
    Nx, Ny = X.shape
    out    = np.zeros((Nx, Ny))
    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0
            buf   = np.empty(Nr)
            for k in range(Nr):
                r  = k * dr
                xc = r * np.cos(th);  yc = r * np.sin(th)
                buf[k] = (_bl(f1R_g, x0, dx, y0, dy, xc, yc)
                        - _bl(D1_g,  x0, dx, y0, dy, xc, yc))
            out[i, j] = _simp(buf, dr) / D0**2
    return out


@njit(parallel=True, cache=True)
def kernel_T1_cons(X, Y, R, x0, dx, y0, dy, D1_g, phi_g, phi0, D0, Nr):
    """
    T1 — conservative f1.

    Integrand along path : −D1(r, θ)
    Pixel-level correction : [φ(X,Y) − φ(0,0)] / D0²
    """
    Nx, Ny = X.shape
    out    = np.zeros((Nx, Ny))
    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0
            buf   = np.empty(Nr)
            for k in range(Nr):
                r  = k * dr
                xc = r * np.cos(th);  yc = r * np.sin(th)
                buf[k] = -_bl(D1_g, x0, dx, y0, dy, xc, yc)
            out[i, j] = (_simp(buf, dr) + phi_g[i, j] - phi0) / D0**2
    return out


# ---------------------------------------------------------------------------
# dT1/dθ
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def kernel_dT1dTh_noncons(X, Y, R, x0, dx, y0, dy,
                           df1Rdth_g, dD1dth_g, D0, Nr):
    """
    dT1/dθ — non-conservative f1.

    Integrand : ∂θ f1R(r, θ) − ∂θ D1(r, θ)
    Result    : dT1/dθ = ∫₀^R [integrand] dr / D0²
    """
    Nx, Ny = X.shape
    out    = np.zeros((Nx, Ny))
    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0
            buf   = np.empty(Nr)
            for k in range(Nr):
                r  = k * dr
                xc = r * np.cos(th);  yc = r * np.sin(th)
                buf[k] = (_bl(df1Rdth_g, x0, dx, y0, dy, xc, yc)
                        - _bl(dD1dth_g,  x0, dx, y0, dy, xc, yc))
            out[i, j] = _simp(buf, dr) / D0**2
    return out


@njit(parallel=True, cache=True)
def kernel_dT1dTh_cons(X, Y, R, x0, dx, y0, dy,
                        dD1dth_g, dphi_g, dphi0, D0, Nr):
    """
    dT1/dθ — conservative f1.

    Integrand along path  : −∂θ D1(r, θ)
    Pixel-level correction: [∂θ φ(X,Y) − ∂θ φ(0,0)] / D0²
    """
    Nx, Ny = X.shape
    out    = np.zeros((Nx, Ny))
    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0
            buf   = np.empty(Nr)
            for k in range(Nr):
                r  = k * dr
                xc = r * np.cos(th);  yc = r * np.sin(th)
                buf[k] = -_bl(dD1dth_g, x0, dx, y0, dy, xc, yc)
            out[i, j] = (_simp(buf, dr) + dphi_g[i, j] - dphi0) / D0**2
    return out


# ---------------------------------------------------------------------------
# T2 + dT2/dθ — non-conservative  (Eq. 1.45)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def kernel_T2_dT2dTh_145(
        X, Y, R, x0, dx, y0, dy,
        D1_g, D2_g, f1R_g, f1T_g, f2R_g,
        df1Rdth_g, df1Tdth_g, dD1dth_g,
        d2f1Rdth2_g, d2D1dth2_g,
        df2Rdth_g, dD2dth_g,
        D0, Nr):
    """
    T2 and dT2/dθ in one pass — non-conservative f1 (Eq. 1.45).

    Running accumulators (trapezoid rule, updated each step k)
    ----------------------------------------------------------
    g   = ∫₀ʳ (∂θ f1R − ∂θ D1) / D0²  dr'      ratio  = g / r
    g2  = ∫₀ʳ (∂²θ f1R − ∂²θ D1) / D0²  dr'    dratio = g2 / r

    L'Hôpital at r = 0:   g/r  → (∂θ f1R − ∂θ D1)(0) / D0²
                           g2/r → (∂²θ f1R − ∂²θ D1)(0) / D0²

    dT2/dθ integrand (Leibniz rule, no FD in θ)
    --------------------------------------------
    ∂θ I₁₄₅ = (∂θ f2R − ∂θ D2) / D0
             + [2(f1R−D1)(∂θ f1R−∂θ D1) + f1T · ∂θ f1T] / D0²
             − C · dC
    where  C  = f1T/D0 − D0·ratio
           dC = ∂θ f1T / D0 − D0 · dratio
    """
    Nx, Ny = X.shape
    T2_out     = np.zeros((Nx, Ny))
    dT2dTh_out = np.zeros((Nx, Ny))

    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0

            buf_T2 = np.empty(Nr)
            buf_dT = np.empty(Nr)

            g  = 0.0;  g_prev  = 0.0
            g2 = 0.0;  g2_prev = 0.0

            for k in range(Nr):
                r   = k * dr
                r_s = r if r > 1e-12 else 1e-12
                xc  = r * np.cos(th);  yc = r * np.sin(th)

                # ── field evaluations ──────────────────────────────────────
                d1    = _bl(D1_g,         x0, dx, y0, dy, xc, yc)
                d2    = _bl(D2_g,         x0, dx, y0, dy, xc, yc)
                f1r   = _bl(f1R_g,        x0, dx, y0, dy, xc, yc)
                f1t   = _bl(f1T_g,        x0, dx, y0, dy, xc, yc)
                f2r   = _bl(f2R_g,        x0, dx, y0, dy, xc, yc)
                df1r  = _bl(df1Rdth_g,    x0, dx, y0, dy, xc, yc)
                df1t  = _bl(df1Tdth_g,    x0, dx, y0, dy, xc, yc)
                dd1   = _bl(dD1dth_g,     x0, dx, y0, dy, xc, yc)
                d2f1r = _bl(d2f1Rdth2_g,  x0, dx, y0, dy, xc, yc)
                d2d1  = _bl(d2D1dth2_g,   x0, dx, y0, dy, xc, yc)
                df2r  = _bl(df2Rdth_g,    x0, dx, y0, dy, xc, yc)
                dd2   = _bl(dD2dth_g,     x0, dx, y0, dy, xc, yc)

                # ── trapezoid update ───────────────────────────────────────
                cur_g  = (df1r  - dd1)  / D0**2
                cur_g2 = (d2f1r - d2d1) / D0**2

                if k == 0:
                    g  = 0.0;  ratio  = cur_g    # L'Hôpital
                    g2 = 0.0;  dratio = cur_g2
                else:
                    g  += 0.5 * (g_prev  + cur_g)  * dr;  ratio  = g  / r_s
                    g2 += 0.5 * (g2_prev + cur_g2) * dr;  dratio = g2 / r_s

                g_prev  = cur_g
                g2_prev = cur_g2

                # ── T2 integrand (Eq. 1.45) ───────────────────────────────
                C = f1t / D0 - D0 * ratio
                buf_T2[k] = ((f2r - d2) / D0
                           + ((f1r - d1)**2 + 0.5 * f1t**2) / D0**2
                           - 0.5 * C**2)

                # ── ∂θ T2 integrand — exact via Leibniz ───────────────────
                dC = df1t / D0 - D0 * dratio
                buf_dT[k] = ((df2r - dd2) / D0
                           + (2.0 * (f1r - d1) * (df1r - dd1)
                              + f1t * df1t) / D0**2
                           - C * dC)

            T2_out[i, j]     = _simp(buf_T2, dr) / D0
            dT2dTh_out[i, j] = _simp(buf_dT, dr) / D0

    return T2_out, dT2dTh_out


# ---------------------------------------------------------------------------
# T2 + dT2/dθ — conservative  (Eq. 1.51)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def kernel_T2_dT2dTh_151(
        X, Y, R, x0, dx, y0, dy,
        D1_g, D2_g, f1R_g, f1T_g, f2R_g,
        df1Rdth_g, df1Tdth_g, dD1dth_g, d2D1dth2_g,
        df2Rdth_g, dD2dth_g,
        D0, Nr):
    """
    T2 and dT2/dθ in one pass — conservative f1 (Eq. 1.51).

    Running accumulators
    --------------------
    h   = ∫₀ʳ ∂θ D1  dr'      h/r  → ∂θ D1(0)  at r = 0  (L'Hôpital)
    h2  = ∫₀ʳ ∂²θ D1 dr'      h2/r → ∂²θD1(0)  at r = 0

    T2 integrand (Eq. 1.51)
    -----------------------
    (f2R−D2)/D0 + [(f1R−D1)²+f1T²/2]/D0² − h²/(2·D0²·r²)

    ∂θ T2 integrand (Leibniz)
    -------------------------
    (∂θf2R−∂θD2)/D0 + [2(f1R−D1)(∂θf1R−∂θD1)+f1T·∂θf1T]/D0²
    − (h/r)·(h2/r)/D0²
    """
    Nx, Ny = X.shape
    T2_out     = np.zeros((Nx, Ny))
    dT2dTh_out = np.zeros((Nx, Ny))

    for i in prange(Nx):
        for j in range(Ny):
            r_end = R[i, j]
            th    = np.arctan2(Y[i, j], X[i, j])
            dr    = r_end / (Nr - 1) if r_end > 1e-12 else 0.0

            buf_T2 = np.empty(Nr)
            buf_dT = np.empty(Nr)

            h  = 0.0;  h_prev  = 0.0
            h2 = 0.0;  h2_prev = 0.0

            for k in range(Nr):
                r   = k * dr
                r_s = r if r > 1e-12 else 1e-12
                xc  = r * np.cos(th);  yc = r * np.sin(th)

                # ── field evaluations ──────────────────────────────────────
                d1   = _bl(D1_g,        x0, dx, y0, dy, xc, yc)
                d2   = _bl(D2_g,        x0, dx, y0, dy, xc, yc)
                f1r  = _bl(f1R_g,       x0, dx, y0, dy, xc, yc)
                f1t  = _bl(f1T_g,       x0, dx, y0, dy, xc, yc)
                f2r  = _bl(f2R_g,       x0, dx, y0, dy, xc, yc)
                df1r = _bl(df1Rdth_g,   x0, dx, y0, dy, xc, yc)
                df1t = _bl(df1Tdth_g,   x0, dx, y0, dy, xc, yc)
                dd1  = _bl(dD1dth_g,    x0, dx, y0, dy, xc, yc)
                d2d1 = _bl(d2D1dth2_g,  x0, dx, y0, dy, xc, yc)
                df2r = _bl(df2Rdth_g,   x0, dx, y0, dy, xc, yc)
                dd2  = _bl(dD2dth_g,    x0, dx, y0, dy, xc, yc)

                # ── trapezoid update ───────────────────────────────────────
                cur_h  = dd1
                cur_h2 = d2d1

                if k == 0:
                    h  = 0.0;  h_over_r  = cur_h   # L'Hôpital
                    h2 = 0.0;  h2_over_r = cur_h2
                else:
                    h  += 0.5 * (h_prev  + cur_h)  * dr;  h_over_r  = h  / r_s
                    h2 += 0.5 * (h2_prev + cur_h2) * dr;  h2_over_r = h2 / r_s

                h_prev  = cur_h
                h2_prev = cur_h2

                # ── T2 integrand (Eq. 1.51) ───────────────────────────────
                buf_T2[k] = ((f2r - d2) / D0
                           + ((f1r - d1)**2 + 0.5 * f1t**2) / D0**2
                           - h_over_r**2 / (2.0 * D0**2))

                # ── ∂θ T2 integrand — exact via Leibniz ───────────────────
                buf_dT[k] = ((df2r - dd2) / D0
                           + (2.0 * (f1r - d1) * (df1r - dd1)
                              + f1t * df1t) / D0**2
                           - h_over_r * h2_over_r / D0**2)

            T2_out[i, j]     = _simp(buf_T2, dr) / D0
            dT2dTh_out[i, j] = _simp(buf_dT, dr) / D0

    return T2_out, dT2dTh_out

# ---------------------------------------------------------------------------
# Parallel multi-path RK4 tracer
# ---------------------------------------------------------------------------
 
@njit(cache=True, inline='always')
def _bl0(g, x0, dx, y0, dy, x, y):
    """Bilinear sample with zero outside the grid (matches RGI fill_value=0)."""
    nx, ny = g.shape
    ix = (x - x0) / dx
    iy = (y - y0) / dy
    if ix < 0.0 or ix > nx - 1 or iy < 0.0 or iy > ny - 1:
        return 0.0
    i0 = int(ix)
    j0 = int(iy)
    if i0 > nx - 2:
        i0 = nx - 2
    if j0 > ny - 2:
        j0 = ny - 2
    fx = ix - i0
    fy = iy - j0
    return (g[i0,   j0  ] * (1.0 - fx) * (1.0 - fy)
          + g[i0+1, j0  ] * fx          * (1.0 - fy)
          + g[i0,   j0+1] * (1.0 - fx) * fy
          + g[i0+1, j0+1] * fx          * fy)
 
 
@njit(parallel=True, cache=True)
def kernel_trace_paths(vx, vy, x0, dx, y0, dy,
                       starts, dt, max_steps, r_stop, hx, hy):
    """
    RK4-integrate many trajectories of the drift field (vx, vy) in parallel.
 
    One prange iteration per start point; each writes its own rows, so there
    is no data race.  Same stop conditions as control_nav.path_RK4:
    leave the box |x|>hx or |y|>hy, or enter the disk r < r_stop.
 
    Returns
    -------
    xs, ys : (M, max_steps+1) float64   trajectory coordinates (padded)
    nstep  : (M,) int64                 number of valid points per path
    """
    M = starts.shape[0]
    xs    = np.empty((M, max_steps + 1))
    ys    = np.empty((M, max_steps + 1))
    nstep = np.empty(M, dtype=np.int64)
    rs2   = r_stop * r_stop
 
    for m in prange(M):
        x = starts[m, 0]
        y = starts[m, 1]
        xs[m, 0] = x
        ys[m, 0] = y
        cnt = 0
        for _ in range(max_steps):
            k1x = _bl0(vx, x0, dx, y0, dy, x, y)
            k1y = _bl0(vy, x0, dx, y0, dy, x, y)
            ax  = x + 0.5 * dt * k1x;  ay = y + 0.5 * dt * k1y
            k2x = _bl0(vx, x0, dx, y0, dy, ax, ay)
            k2y = _bl0(vy, x0, dx, y0, dy, ax, ay)
            bx  = x + 0.5 * dt * k2x;  by = y + 0.5 * dt * k2y
            k3x = _bl0(vx, x0, dx, y0, dy, bx, by)
            k3y = _bl0(vy, x0, dx, y0, dy, bx, by)
            cx  = x + dt * k3x;        cy = y + dt * k3y
            k4x = _bl0(vx, x0, dx, y0, dy, cx, cy)
            k4y = _bl0(vy, x0, dx, y0, dy, cx, cy)
 
            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            cnt += 1
            xs[m, cnt] = x
            ys[m, cnt] = y
            if abs(x) > hx or abs(y) > hy or (x * x + y * y) < rs2:
                break
        nstep[m] = cnt + 1   # including the start point
 
    return xs, ys, nstep
