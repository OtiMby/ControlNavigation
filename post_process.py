import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sympy as _sp
import warnings as _warnings
import json
import random as rd


from mobility_and_forces import Force, Mobility
from optimized_control_nav import control_nav
from scipy.integrate import simpson



from mobility_and_forces import Force, Mobility
from numba_kernel import *  # For parallel computing
from optimized_control_nav import control_nav

class simulation:
    def __init__(self, file: str):
            # Read controlnav object from file (usefull for post-processing)
            with open(file, "r") as f:
                data = json.load(f)
            self.Lx = data['L']
            self.Ly = data['L']
            self.Nx = data['N']
            self.Ny = data['N']
            self.Nr = int(np.hypot(self.Nx, self.Ny))
            self.conservative= data['conservative']
            self.f1 = Force(self.Lx, self.Ly, self.Nx, self.Ny, conservative=self.conservative, expr_phi=data['pot'])
            self.f2 = Force(self.Lx, self.Ly, self.Nx, self.Ny, conservative=False, expr_phi="0")
            self.D0 = data['D0']
            self.D1 = Mobility(expr=data['D1'],dtheta_func="0")
            self.D2 = Mobility(expr=data['D2'],dtheta_func="0")
            self.T0 = np.array(data['T0'])
            self.T1 = np.array(data['T1'])
            self.T2 = np.array(data['T2'])
            self.dT1dTh = np.array(data['dT1dTh'])
            self.dT2dTh = np.array(data['dT2dTh'])
            
            # Cartesian grid  (Nx, Ny)
            self.X, self.Y = np.mgrid[-self.Lx/2:self.Lx/2:self.Nx*1j, -self.Ly/2:self.Ly/2:self.Ny*1j]
            # Radial distance — guard against exact zero
            self.R  = np.where(np.hypot(self.X, self.Y) == 0, 1e-12,
                            np.hypot(self.X, self.Y))
            # Polar angle  (Nx, Ny)
            self._Th = np.arctan2(self.Y, self.X) % (2 * np.pi)
            # Unit radial / tangential vectors  (Nx, Ny)
            self._eRx =  np.cos(self._Th)
            self._eRy =  np.sin(self._Th)
            self._eTx = -np.sin(self._Th)
            self._eTy =  np.cos(self._Th)



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
        expansion remains valid. According to Tstar criteria
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
        Comparison on Tstar
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

    def compare_paths(self, order, theta_dot, n_theta, eps, dt=0.01, r_stop=0.0001, paths=False, random=False, thetas=[]):
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
        vx, vy = self.velocity_field(order, eps)

        vmax = np.hypot(vx, vy).max()
        if not random:
            tf = np.concatenate([np.linspace(-np.pi/2 + GAP,   np.pi/2 - GAP,   n_theta//2),
                                np.linspace( np.pi/2 + GAP, 3*np.pi/2 - GAP,   n_theta//2)])
        if random:
            tf = np.random.rand(n_theta) * 2*np.pi

        if len(thetas):
            tf = thetas.copy()

        M  = tf.size

        # --- caractéristiques exactes : rétrograde depuis la cible ----------
        Nt = int(self.Lx/dt)
        
        Xh, Yh = np.zeros((Nt + 1, M)), np.zeros((Nt + 1, M))
        th = tf.copy()                                  # cap AU but, pas de +pi

        # --------- intégration rétrograde de zermelo ---------  -> Xh, Yh trajectoires
        def flow(x, y, t):
            """Flot optimal RÉTROGRADE d/dτ = -(flot avant), vectorisé sur les M caractéristiques.
            État (x, y, θ) :
                dx/dτ = -(F_x + D·cosθ)
                dy/dτ = -(F_y + D·sinθ)
                dθ/dτ = -func(θ, x, y, ε)
            avec D = D0 + εD1 + ε²D2,  F = εf1 + ε²f2.
            """
            V  = self.D0 + eps*self.D1(x, y) + eps**2*self.D2(x, y)
            fx = eps*self.f1.fx(x, y) + eps**2*self.f2.fx(x, y)
            fy = eps*self.f1.fy(x, y) + eps**2*self.f2.fy(x, y)
            c, s = np.cos(t), np.sin(t)
            return -(fx + V*c), -(fy + V*s), -theta_dot(t, x, y, eps)

        for k in range(Nt):
            if k%100==0:
                print(k)
            xk, yk = Xh[k], Yh[k]
            a1x, a1y, a1t = flow(xk,              yk,              th)
            a2x, a2y, a2t = flow(xk + .5*dt*a1x,  yk + .5*dt*a1y,  th + .5*dt*a1t)
            a3x, a3y, a3t = flow(xk + .5*dt*a2x,  yk + .5*dt*a2y,  th + .5*dt*a2t)
            a4x, a4y, a4t = flow(xk +    dt*a3x,  yk +    dt*a3y,  th +    dt*a3t)
            Xh[k+1] = xk + (dt/6.)*(a1x + 2*a2x + 2*a3x + a4x)
            Yh[k+1] = yk + (dt/6.)*(a1y + 2*a2y + 2*a3y + a4y)
            th      = th + (dt/6.)*(a1t + 2*a2t + 2*a3t + a4t)

        outside = (np.abs(Xh) > box_x) | (np.abs(Yh) > box_y)
        too_far = (np.abs(Xh) > self.Lx) | (np.abs(Yh) > self.Ly)

        left    = outside.any(0) # 1D array where 1 if outside, 0 if not
        wrong = too_far.any(0).any()
        print(wrong)
        e       = np.where(left, outside.argmax(0), Nt) # 2D array where k_step ( number of steps to go outside
        print()
        if not left.all():
            _warnings.warn(
                f"compare_paths: {int((~left).sum())}/{M} caractéristiques "
                "n'ont pas atteint la boîte (nmax trop petit ou extrémale "
                "divergente) ; marquées NaN.", RuntimeWarning, stacklevel=2)
        sp = np.ascontiguousarray(
            np.column_stack([Xh[e, np.arange(M)], Yh[e, np.arange(M)]]),
            dtype=np.float64)
        
        print("zermelo computed")
        # --- RK4 perturbatif depuis les lancements vers la cible ------------  -> xs, ys -> trajectoires
        
        vx = np.ascontiguousarray(vx, dtype=np.float64)
        vy = np.ascontiguousarray(vy, dtype=np.float64)
        xr, yr = self.X[:, 0], self.Y[0, :]
        x0, dx = float(xr[0]), float(xr[1] - xr[0])
        y0, dy = float(yr[0]), float(yr[1] - yr[0])
        print("RK4")
        xs, ys, nstep = kernel_trace_paths(
            vx, vy, x0, dx, y0, dy, sp,
            float(dt), int(2*Nt), float(r_stop), self.Lx/2.0, self.Ly/2.0)

        spatial_err = []
        time_err = []
        ang_err = []

        cx, cy = self.compute_control_force(order, eps)
        kw     = dict(bounds_error=False, fill_value=0.)
        icx    = RegularGridInterpolator((xr, yr), cx, **kw)
        icy    = RegularGridInterpolator((xr, yr), cy, **kw)

        for k in range(M):
            
            #ang_err.append(np.abs(th[k] - np.atan2(icy(*sp[k]), icx(*sp[k]))))
            #z_area = -simpson(Yh[e,k], x=Xh[e, k])
            #num_area = simpson(ys[k], x=xs[k])
            #spatial_err.append(np.abs(z_area - num_area))
            if left[k] and nstep[k] >0:
                time_err.append((e[k] - nstep[k])*dt) # longueur d'une trajectoire
            else:
                time_err.append(np.nan)
        
        if paths:
            anpath = [(Xh[:e[k]+1, k], Yh[:e[k]+1, k]) for k in range(M)]
            rk4    = [(xs[k][:nstep[k]], ys[k][:nstep[k]]) for k in range(M)]
            return spatial_err, time_err, ang_err, [anpath, rk4]
        
        return spatial_err, time_err, ang_err

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

                

if __name__ == "__main__":
    import sys
    file = sys.argv[1]
    eps = float(sys.argv[2])

    cn = simulation(file=file)
    print("data loaded")
