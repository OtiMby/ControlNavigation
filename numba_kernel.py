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
            out[i, j] = (_simp(buf, dr) - (phi_g[i, j] - phi0)) / D0**2
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
            out[i, j] = (_simp(buf, dr) - (dphi_g[i, j] - dphi0)) / D0**2
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

            # Compute the integrand of the outer integral for Nr points in [0,R]
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
                # To compute the integrand of the outer integral at r(k), we can use the integrand at r(k-1), because it contains an integral [0,r(k)]
                # The inner integral is h(k)
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
    print(M)
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
        print(cnt)
        for _ in range(max_steps):
            if _ % 100 == 0:
                print(_)
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
 
            xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            yn = y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

            if x*xn + y *yn < 0:
                print(f"oscille a {int(x*x + y*y)}")
                break
            else:
                x, y = xn, yn
                cnt += 1
                xs[m, cnt] = x
                ys[m, cnt] = y
            if abs(x) > hx or abs(y) > hy or (x * x + y * y) < rs2:
                print((x * x + y * y))
                break
        nstep[m] = cnt + 1   # including the start point
 
    return xs, ys, nstep

@njit(parallel=True, cache=True)
def zermelo_kernel(Fx, Fy, D, theta_eq, Nt, dt, eps, Xh, Yh, Th):
    # --------- intégration rétrograde de zermelo ---------  -> Xh, Yh trajectoires
    def flow(x, y, t):
        """Flot optimal RÉTROGRADE d/dτ = -(flot avant), vectorisé sur les M caractéristiques.
        État (x, y, θ) :
            dx/dτ = -(F_x + D·cosθ)
            dy/dτ = -(F_y + D·sinθ)
            dθ/dτ = -func(θ, x, y, ε)
        avec D = D0 + εD1 + ε²D2,  F = εf1 + ε²f2.
        """
        V  = D(x,y)
        fx, fy = Fx(x,y), Fy(x,y)
        theta_dot = theta_eq(x,y)
        c, s = np.cos(t), np.sin(t)
        return -(fx + V*c), -(fy + V*s), -theta_dot(t, x, y, eps)

    for k in range(Nt):
        xk, yk, th = Xh[k], Yh[k], Th[k]
        a1x, a1y, a1t = flow(xk,              yk,              th)
        a2x, a2y, a2t = flow(xk + .5*dt*a1x,  yk + .5*dt*a1y,  th + .5*dt*a1t)
        a3x, a3y, a3t = flow(xk + .5*dt*a2x,  yk + .5*dt*a2y,  th + .5*dt*a2t)
        a4x, a4y, a4t = flow(xk +    dt*a3x,  yk +    dt*a3y,  th +    dt*a3t)
        Xh[k+1] = xk + (dt/6.)*(a1x + 2*a2x + 2*a3x + a4x)
        Yh[k+1] = yk + (dt/6.)*(a1y + 2*a2y + 2*a3y + a4y)
        Th[k+1] = th + (dt/6.)*(a1t + 2*a2t + 2*a3t + a4t)
    
    return Xh, Yh, Th

import itertools
import numpy as np
from numba import njit, prange


@njit(inline='always')
def _orient(ax, ay, bx, by, cx, cy):
    """Produit vectoriel (b-a) × (c-a) : orientation entre deux droites (AB) et (AC) ( pas les segments de la trajectoires)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _build_grid(B, ox, oy, inv, ncx, ncy):
    """
    Crée une grille dont les cellules contiennent les segments de la trajectoire ( B ) qui passent par cette cellule.
    B : trajectoire
    ox, oy : origine en x et y des cellules
    inv : taille d'une cellule
    ncx, ncy : nombre de cellule selon x et selon y

    La sortie est deux listes count et items
    items est une liste de (liste de segments). Chaque item de items correspond à une cellule
    count(i,j) contient l'indice de items ou se trouve la cellule (i,j)

    Les deux listes de sortie sont des tableaux 2D applatis en 1D
    
    """
    Mb = B.shape[0] - 1 # nombre de segments d'un chemin
    ncell = ncx * ncy # nombre de cellules 
    counts = np.zeros(ncell + 1, np.int64) # contient au debut le nombre de segments inclus pour chaque cellule. Tableau 2D applati en 1D
    for b in range(Mb): # pour chacun des segments b de B
        x1, y1, x2, y2 = B[b, 0], B[b, 1], B[b+1, 0], B[b+1, 1] # coordonées du segment b
        # toute cellule dont l'indice selon x se trouve entre cx0 et cx1 contiennent le segment b
        cx0 = int((min(x1, x2) - ox) * inv); cx1 = int((max(x1, x2) - ox) * inv) 
        # toute cellule dont l'indice selon y se trouve entre cy0 et cy1 contiennent le segment b
        cy0 = int((min(y1, y2) - oy) * inv); cy1 = int((max(y1, y2) - oy) * inv) 
        for cx in range(cx0, cx1 + 1): # pour chaque cellule contenant le segment b
            for cy in range(cy0, cy1 + 1):
                counts[cx * ncy + cy + 1] += 1 # comme c'est un tableau 2D applati l'indice (ix,iy) devient ix*nx + iy
    
    # somme cumulative, aprés la boucle counts contient l'indice de la cellule k dans liste items
    for k in range(ncell): 
        counts[k + 1] += counts[k]  
    
    # chaque élément de  items est une liste de segment correspondant à la cellule (i,j), tableau 2D applati en 1D
    items = np.empty(counts[ncell], np.int64) 
    cursor = counts[:ncell].copy()
    for b in range(Mb): # pour chaque segment on calcule les cellules concernées
        x1, y1, x2, y2 = B[b, 0], B[b, 1], B[b+1, 0], B[b+1, 1]
        cx0 = int((min(x1, x2) - ox) * inv); cx1 = int((max(x1, x2) - ox) * inv)
        cy0 = int((min(y1, y2) - oy) * inv); cy1 = int((max(y1, y2) - oy) * inv)
        for cx in range(cx0, cx1 + 1): # pour chaque cellule concernée on lui rajoute le segment k
            for cy in range(cy0, cy1 + 1):
                c = cx * ncy + cy
                items[cursor[c]] = b; cursor[c] += 1 
    return counts, items


@njit(parallel=True, cache=True, fastmath=True)
def _query(A, B, counts, items, ox, oy, inv, ncx, ncy, out_xy, offs, cnt, do_fill):
    """Pour chaque segment de A : ne teste que les segments de B des cellules voisines.
    Se fait en deux appels. Vu qu'on ne sait pas combien d'intersections on va avoir, on commence par faire un premier appel pour les compter puis 
    Un second pour remplir la liste des coordonées des intersections out_xy. Comme le calcul est parralélisé, 
    on ne peut pas créer la liste dynamiquement avec out_xy.append(intersection).
    Il faut initialiser correctement la liste out_xy et donner une place dans cette liste à chaque intersection 
    pour que les threads ne se marchent pas dessus
    """
    Ma = A.shape[0] - 1
    
    for a in prange(Ma):                      # parallélisé sur les cœurs
        # On commence par calculer les cellules concernées par le segment a de A
        x1, y1, x2, y2 = A[a, 0], A[a, 1], A[a+1, 0], A[a+1, 1]
        axmin = min(x1, x2); axmax = max(x1, x2)
        aymin = min(y1, y2); aymax = max(y1, y2)
        cx0 = max(0, int((axmin - ox) * inv)); cx1 = min(ncx - 1, int((axmax - ox) * inv))
        cy0 = max(0, int((aymin - oy) * inv)); cy1 = min(ncy - 1, int((aymax - oy) * inv))
        c = 0
        for cx in range(cx0, cx1 + 1): # pour chaque cellule c concernée par le segment a
            for cy in range(cy0, cy1 + 1):
                cell = cx * ncy + cy
                for idx in range(counts[cell], counts[cell + 1]): # pour chacun des indices des segments passant ( issus de B ) par  la cellule c
                    b = items[idx] # on récupere le segment
                    x3, y3, x4, y4 = B[b, 0], B[b, 1], B[b+1, 0], B[b+1, 1] # coordonées du segment en question
                    if axmax < min(x3, x4) or axmin > max(x3, x4): continue # si intersection impossible on passe au prochain segment
                    if aymax < min(y3, y4) or aymin > max(y3, y4): continue # pareil

                    # test d'orientation
                    d1 = _orient(x3, y3, x4, y4, x1, y1); d2 = _orient(x3, y3, x4, y4, x2, y2)
                    d3 = _orient(x1, y1, x2, y2, x3, y3); d4 = _orient(x1, y1, x2, y2, x4, y4)
                    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)): # si y'a intersection
                        # on récupère le lieu de l'intersection
                        t = d1 / (d1 - d2)
                        px = x1 + t * (x2 - x1)
                        py = y1 + t * (y2 - y1)
                        # dédup : on ne compte le croisement que dans SA cellule
                        if int((px - ox) * inv) != cx or int((py - oy) * inv) != cy: # si le lieu du croisement n'est pas dans la cellule c
                            continue
                        if do_fill: # si on est sur le remplissage
                            p = offs[a] + c; out_xy[p, 0] = px; out_xy[p, 1] = py # on remplit la place qui est allouée à l'interserction
                        c += 1
        if not do_fill: # si on est pas sur le remplissage, on calcule juste combien de place va falloir allouer et quelles sont ces places
            cnt[a] = c


def pair_intersections(A, B, target_per_cell=2.0):
    """Croisements entre 2 trajectoires (arrays (P,2)). Renvoie un array (K,2)."""
    pts = np.vstack((A, B)) # concatenne les deux trajectoires. (Na+Nb, 2)
    
    # permet de construire la boite englobante des deux trajectoires
    # ox, oy origines en x et y de la grand boite
    # w, h hauteur et largeur de la grande boite
    ox, oy = pts[:, 0].min(), pts[:, 1].min() 
    w = pts[:, 0].max() - ox + 1e-9; h = pts[:, 1].max() - oy + 1e-9 
    
    nseg = (A.shape[0] - 1) + (B.shape[0] - 1) # nombre de segments total à considerer
    
    # nombre de cellule selon le plus grand coté de la grande boite
    side = max(1, int(np.sqrt(max(1, nseg / target_per_cell)))) 
    
    inv = side / max(h,w) # inverse de la taille d'une cellule
    ncx = int(w * inv) + 1; ncy = int(h * inv) + 1 # nombres de cellule selon x, y
    counts, items = _build_grid(B, ox, oy, inv, ncx, ncy) # on constuit la grille
    Ma = A.shape[0] - 1
    cnt = np.zeros(Ma, np.int64)
    _query(A, B, counts, items, ox, oy, inv, ncx, ncy,
           np.empty((1, 2)), np.zeros(Ma, np.int64), cnt, False)   # passe 1 : compter pour savoir quelles sont les places de count allouée à chaque intersection
    offs = np.zeros(Ma, np.int64); np.cumsum(cnt[:-1], out=offs[1:]) # contient la place de chaque intersection dans la liste out_xy
    out_xy = np.empty((int(cnt.sum()), 2)) # contient les coordonées des intersections
    _query(A, B, counts, items, ox, oy, inv, ncx, ncy,
           out_xy, offs, cnt, True)                                # passe 2 : remplir  les espace alloué à chaque intersection
    return out_xy


def trajectory_intersections(trajectories, target_per_cell=2.0):
    """N trajectoires (chacune array (P,2)) -> liste de (i, j, (x, y))."""
    res = []
    for i, j in itertools.combinations(range(len(trajectories)), 2):
        pts = pair_intersections(np.asarray(trajectories[i], float),
                                 np.asarray(trajectories[j], float), target_per_cell)
        res.extend((i, j, (pts[k, 0], pts[k, 1])) for k in range(pts.shape[0]))
    return res
