import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsp
import sympy as sp
from scipy.interpolate import RegularGridInterpolator
import os


sp.init_printing(use_unicode=False)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.weight': 'light',
    'text.latex.preamble': r'''
    \usepackage{pgf}
    \usepackage{pgfplotstable}
    '''
    })
class mobolities():
    def D0(self):
        pass
    def D1(self,Ra,Th):
        pass
    def D2(self,Ra,Th):
        pass

class forces():
    def assign_X_Y(self,X,Y):
        self.X = X
        self.Y = Y

    def assign_Ra_Th(self, Ra, Th):
        self.Ra = Ra
        self.Th = Th

    def theta(self,x,y):
        th = np.arctan2(y,x) % (2*np.pi)
        return th 
 
    def radius(self,x,y):
        return np.sqrt(x**2 + y**2)


class times():
    def assign_X_Y(self,X,Y):
        self.X = X
        self.Y = Y
        self.Ra = self.radius(x=X, y=Y)
        self.Th = self.theta(x=X, y=Y)
        pass

    def assign_mobility(self, mob: mobolities):
        self.mob = mob

    def assign_force(self, force: forces):
        self.force = force

    def theta(self,x,y):
        th = np.arctan2(y,x) % (2*np.pi)
        return th 
 
    def radius(self,x,y):
        return np.sqrt(x**2 + y**2)

    def T0(self):
        return self.Ra/self.mob.D0()

    def T1(self):
        D0 = self.mob.D0()
        # On utilise "R_tilde" au lieu de "\\tilde{R}" pour éviter le SyntaxError
        self.sym_Ra, self.sym_RaT, self.sym_Th = sp.symbols("R R_tilde theta")
        sRa, sRaT, sTh = self.sym_Ra, self.sym_RaT, self.sym_Th
        
        self.sym_T1 = sp.integrate((self.force.f_R(sRaT,sTh) - self.mob.D1(sRaT,sTh))/(D0**2),(sRaT,0,sRa))
        self.num_T1 = sp.lambdify((sRa,sTh), self.sym_T1)
        return self.num_T1(self.Ra, self.Th) 

    def dT1dTh(self):
        # Correction : self au lieu de times
        if not hasattr(times, 'sym_T1'):
            self.T1()
            
        self.sym_dT1dTh = sp.diff(self.sym_T1, self.sym_Th)
        self.num_dT1dTh = sp.lambdify((self.sym_Ra, self.sym_Th), self.sym_dT1dTh)
        return self.num_dT1dTh(self.Ra, self.Th) 

    def T2(self):
        if not hasattr(times, 'sym_dT1dTh'):
            self.dT1dTh()
            
        dT1dTh = self.sym_dT1dTh.subs({self.sym_Ra: self.sym_RaT})
        
            
        D0 = self.mob.D0()
        sRa, sRaT, sTh = self.sym_Ra, self.sym_RaT, self.sym_Th

        
        expr1 = (self.force.f_R(sRaT,sTh) - self.mob.D1(sRaT,sTh))**2/D0**2 + 1/2 * (self.force.f_T(sRaT,sTh)/D0)**2 
        # integral in integral : eq 1.48
        sRaTT = sp.symbols("Rprime") # var to integrate (R'')
        RfTh = sRaTT * self.force.f_T(sRaTT, sTh)
        integ2 = sp.diff(RfTh, sRaTT)*0
        integ2 -= sp.diff(self.force.f_R(sRaTT, sTh), sTh) *0
        integ2 -= sp.diff(self.mob.D1(sRaTT, sTh), sTh)
        expr3 = sp.integrate(integ2, (sRaTT, 0, sRaT))
        expr3 = -expr3**2 / (2 * D0**2 * sRaT**2) 

        expr2 = -(self.mob.D2(sRaT,sTh)-self.force.f_R2(sRaT,sTh))/D0 
        #expr3 = -1/2*(self.force.f_T(sRaT,sTh)/D0 - D0/sRaT*dT1dTh)**2
        
        self.sym_T2 = sp.integrate(1/D0*(expr1 * 0 + expr2 * 0 + expr3), (sRaT, 0, sRa))
        self.num_T2 = sp.lambdify((sRa, sTh), self.sym_T2, modules=['scipy', 'numpy'])
        
        
        return self.num_T2(self.Ra, self.Th)

    def dT2dTh(self):
        if hasattr(times, 'sym_T2'):
            pass
        else:
            self.T2()
        self.sym_dT2dTh = sp.diff(self.sym_T2,self.sym_Th)
        #print("dT_2dTh = ", self.sym_dT2dTh)
        self.num_dT2dTh = sp.lambdify((self.sym_Ra,self.sym_Th),self.sym_dT2dTh)
        return self.num_dT2dTh(self.Ra, self.Th) 



class simple_control_problem():
    def __init__(self, Lx, Ly, Nx, Ny, mob: mobolities, force: forces, time: times):
        self.Lx = Lx
        self.Ly = Ly
        self.X, self.Y = np.mgrid[-Lx/2:+Lx/2:Nx*1j, 
                                  -Ly/2:+Ly/2:Ny*1j]
        self.Th = self.theta(self.X,self.Y)
        self.Ra = self.radius(self.X,self.Y)
        self.mob = mob
        self.force = force
        self.force.assign_Ra_Th(Ra=self.Ra, Th=self.Th)
        self.time = time
        self.time.assign_X_Y(X=self.X, Y=self.Y)
        self.time.assign_mobility(mob)
        self.time.assign_force(force)

    def eRa(self):
        eR_x = np.cos(self.Th)
        eR_y = np.sin(self.Th)
        return eR_x, eR_y

    def eTh(self):
        eT_x = -np.sin(self.Th)
        eT_y = np.cos(self.Th)
        return eT_x, eT_y

    def theta(self,x,y):
        th = np.arctan2(y,x) % (2*np.pi)
        return th 

    def radius(self,x,y):
        return np.sqrt(x**2 + y**2)


    def control_0th_order(self):
        c0_x, c0_y = self.eRa() 
        return -c0_x, -c0_y

    def control_1st_order(self):
        c1_x, c1_y = self.eTh() 
        R  = self.Ra
        D0 = self.mob.D0()
        dT1dTh = self.time.dT1dTh()
        expr = dT1dTh*D0/R
        return -c1_x*expr, -c1_y*expr

    def control_2nd_order(self):
        c2_ra_x, c2_ra_y = self.eRa() 
        c2_th_x, c2_th_y = self.eTh() 
        R  = self.Ra
        T  = self.Th
        D0 = self.mob.D0()
        dT1dTh = self.time.dT1dTh()
        dT2dTh = self.time.dT2dTh()
        expr_r = dT1dTh**2*D0**2/(2*R**2)

        expr_t = 1/R*(+dT1dTh*(self.force.f_R(R,T) - self.mob.D1(R,T)) \
                      -D0*dT2dTh)

        return c2_ra_x*(expr_r) + c2_th_x*(expr_t), \
               c2_ra_y*(expr_r) + c2_th_y*(expr_t)


def plot_overview_figure(cp1: simple_control_problem, saveStr="simple_dynamics_overview"):
    control_figure = plt.figure(figsize=(11,11))
    gs = gridsp.GridSpec(nrows=5, ncols=4,left=0.02, right=0.98, bottom=0.05,top=0.95,  hspace=0.2, wspace=-0.0)
    fR1_plot  = plt.subplot(gs[0,0])
    fT1_plot  = plt.subplot(gs[0,1])
    fR2_plot  = plt.subplot(gs[0,2])
    fT2_plot  = plt.subplot(gs[0,3])
    fTot_plot = plt.subplot(gs[1,0])

    m1_plot = plt.subplot(gs[1,2])
    m2_plot = plt.subplot(gs[1,3])

    tTot_plot = plt.subplot(gs[2,0])
    t0_plot = plt.subplot(gs[2,1])
    t1_plot = plt.subplot(gs[2,2])
    t2_plot = plt.subplot(gs[2,3])

    c0_plot = plt.subplot(gs[3,1])
    c1_plot = plt.subplot(gs[3,2])
    c2_plot = plt.subplot(gs[3,3])
    cTot_plot = plt.subplot(gs[3,0])

    traj_plot1 = plt.subplot(gs[4,1])
    traj_plot2 = plt.subplot(gs[4,2])
    traj_plot3 = plt.subplot(gs[4,3])
    traj_plot4 = plt.subplot(gs[4,0])


    text_plot = plt.subplot(gs[1,1])
    ax_list = [fR1_plot,fT1_plot,fR2_plot,fT2_plot,\
               fTot_plot,text_plot,m1_plot,m2_plot,\
               tTot_plot,t0_plot,t1_plot,t2_plot,\
               cTot_plot,c0_plot,c1_plot,c2_plot,\
              ]

    eps = 0.2

    fTot_plot.set_title(r"$\vec{F}(\vec{x})$")
    fR1_plot.set_title(r"$f_{1R}$")
    fR2_plot.set_title(r"$f_{2R}$")
    fT1_plot.set_title(r"$f_{1\theta}$")
    fT2_plot.set_title(r"$f_{2\theta}$")

    m1_plot.set_title(r"$D_1(\vec{x})$")
    m2_plot.set_title(r"$D_2(\vec{x})$")

    tTot_plot.set_title(r"$\mathcal{T}^*(\vec{x})$ to $\mathcal{O}(\varepsilon^2)$ with $\varepsilon=" + r"{:.1E}$".format(eps))
    t0_plot.set_title(r"$\mathcal{T}^*_0(\vec{x})$")
    t1_plot.set_title(r"$\mathcal{T}^*_1(\vec{x})$")
    t2_plot.set_title(r"$\mathcal{T}^*_2(\vec{x})$")

    c0_plot.set_title(r"$\hat{c}_0(\vec{x})$")
    c1_plot.set_title(r"$\hat{c}_1(\vec{x})$")
    c2_plot.set_title(r"$\hat{c}_2(\vec{x})$")
    cTot_plot.set_title(r"$\hat{c}(\vec{x})$ to $\mathcal{O}(\varepsilon^2)$")

    eR_x, eR_y = cp1.eRa()
    eT_x, eT_y = cp1.eTh()
    #f_1
    f_Rx,f_Tx = eR_x *cp1.force.f_R(cp1.Ra,cp1.Th), eT_x*cp1.force.f_T(cp1.Ra,cp1.Th)
    f_x = f_Rx + f_Tx 
    f_Ry,f_Ty = eR_y *cp1.force.f_R(cp1.Ra,cp1.Th), eT_y*cp1.force.f_T(cp1.Ra,cp1.Th)
    f_y = f_Ry + f_Ty

    fR1_plot.quiver(cp1.X, cp1.Y, f_Rx, f_Ry, color="green")
    fT1_plot.quiver(cp1.X, cp1.Y, f_Tx, f_Ty, color="green")

    #f_2
    f_2Rx,f_2Tx = eR_x *cp1.force.f_R2(cp1.Ra,cp1.Th), eT_x*cp1.force.f_T2(cp1.Ra,cp1.Th)
    f_2x = f_2Rx + f_2Tx 
    f_2Ry,f_2Ty = eR_y *cp1.force.f_R2(cp1.Ra,cp1.Th), eT_y*cp1.force.f_T2(cp1.Ra,cp1.Th)
    f_2y = f_2Ry + f_2Ty

    fR2_plot.quiver(cp1.X, cp1.Y, f_2Rx, f_2Ry, color="green")
    fT2_plot.quiver(cp1.X, cp1.Y, f_2Tx, f_2Ty, color="green")

    #f_tot
    fTot_plot.quiver(cp1.X, cp1.Y, eps*f_x + eps**2*f_2x , eps*f_y + eps**2*f_2y, color="black")
    #mobilities
    m1_plot.imshow(cp1.mob.D1(cp1.Ra,cp1.Th), cmap="Blues",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    m2_plot.imshow(cp1.mob.D2(cp1.Ra,cp1.Th), cmap="Blues",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    control_figure.colorbar(m2_plot.get_images()[0], ax=m2_plot, fraction=0.046, pad=0.04)

    T0, T1, T2 = cp1.time.T0(),cp1.time.T1(),cp1.time.T2()
    tTot = T0+eps*T1+eps**2*T2
    tTot_plot.imshow(tTot, cmap="Reds",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    t0_plot.imshow(T0, cmap="Reds",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    t1_plot.imshow(T1, cmap="Reds",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    control_figure.colorbar(t1_plot.get_images()[0], ax=t1_plot, fraction=0.046, pad=0.04, location='left')
    t2_plot.imshow(T2, cmap="Reds",extent=[-cp1.Lx/2,cp1.Lx/2,-cp1.Ly/2,cp1.Ly/2], origin="lower")
    control_figure.colorbar(t2_plot.get_images()[0], ax=t2_plot, fraction=0.046, pad=0.04)

    c0_plot.quiver(cp1.X, cp1.Y, *cp1.control_0th_order(), color="blue")
    c1_plot.quiver(cp1.X, cp1.Y, *cp1.control_1st_order(), color="red")
    c2_plot.quiver(cp1.X, cp1.Y, *cp1.control_2nd_order(), color="blue")


    text_plot.set_axis_off()
    try:
        text_plot.text(0.0,1.0,r"$\eta_1={}$".format(sp.latex(cp1.mob.mob_eta1(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
        text_plot.text(0.0,0.9,r"$A_1={}$".format(sp.latex(cp1.force.A1(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')

        text_plot.text(0.0,0.7,r"$f_{1R}="+r"{}$".format(sp.latex(cp1.force.f_R(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
        text_plot.text(0.0,0.6,r"$f_{1\theta}="+r"{}$".format(sp.latex(cp1.force.f_T(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
        text_plot.text(0.0,0.5,r"$f_{2R}="+r"{}$".format(sp.latex(cp1.force.f_R2(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
        text_plot.text(0.0,0.4,r"$f_{2\theta}="+r"{}$".format(sp.latex(cp1.force.f_T2(cp1.time.sym_Ra,cp1.time.sym_Th))),
                    fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
    except:
        pass
                    
    text_plot.text(0.0,0.2,r"$D_0={}$".format(sp.latex(cp1.mob.D0())),
                   fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
    text_plot.text(0.0,0.1,r"$D_1={}$".format(sp.latex(cp1.mob.D1(cp1.time.sym_Ra,cp1.time.sym_Th))),
                   fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')
    text_plot.text(0.0,0.0,r"$D_2={}$".format(sp.latex(cp1.mob.D2(cp1.time.sym_Ra,cp1.time.sym_Th))),
                   fontsize=10, transform=text_plot.transAxes, color="black", ha='left', va='top')


    #text_plot.text(0.1,0.4,r"$T_1={}$".format(sp.latex(cp1.time.sym_T1)))
    #text_plot.text(0.1,0.3,r"$T_2={}$".format(sp.latex(cp1.time.sym_T2)))

    cFull0_x, cFull0_y = cp1.control_0th_order()
    cFull1_x, cFull1_y = cp1.control_1st_order()
    cFull2_x, cFull2_y = cp1.control_2nd_order()
    cTot_plot.quiver(cp1.X, cp1.Y, cFull0_x + eps*cFull1_x + eps**2*cFull2_x,  cFull0_y + eps*cFull1_y + eps**2*cFull2_y, color="black")

    for ax in ax_list:
        ax.set_aspect(1)
        ax.set_xlim(-cp1.Lx/2,cp1.Lx/2)
        ax.set_ylim(-cp1.Ly/2,cp1.Ly/2)
        ax.set_xticks(np.linspace(-cp1.Lx/2,cp1.Lx/2,5))
        ax.set_yticks(np.linspace(-cp1.Ly/2,cp1.Ly/2,5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if ax!=text_plot:
            ax.plot([0],[0],"D",ms=2,color="black")

    for p in [cTot_plot, c0_plot, c1_plot, c2_plot]:
        p.set_xticklabels(np.linspace(-cp1.Lx/2,cp1.Lx/2,5),fontsize=8)

    for p in [fTot_plot, cTot_plot, m1_plot, fR1_plot]:
        p.set_yticklabels(np.linspace(-cp1.Ly/2,cp1.Ly/2,5),fontsize=8)


    # -----------------------------------------------------------------------
    # Trajectory integration under the control field (RK4 + bilinear interp)
    # -----------------------------------------------------------------------
 
    def _make_interp(vx_grid, vy_grid):
        """Build two RegularGridInterpolators, one for each component of the velocity field."""
        x1d = cp1.X[:, 0]   # shape (Nx,)
        y1d = cp1.Y[0, :]   # shape (Ny,)
        ivx = RegularGridInterpolator((x1d, y1d), vx_grid,bounds_error=False, fill_value=0.0)
        ivy = RegularGridInterpolator((x1d, y1d), vy_grid, bounds_error=False, fill_value=0.0)
        return ivx, ivy
 
    def RK4_integrate_traj(x0, y0, ivx, ivy, dt=0.05, n_steps=800, r_stop=0.15):
        xs, ys = [x0], [y0]
        x, y = x0, y0
        for _ in range(n_steps):
            def vel(xi, yi):
                pt = np.array([xi, yi])
                return float(ivx(pt)), float(ivy(pt))
            
            k1x, k1y = vel(x, y)
            k2x, k2y = vel(x + dt/2*k1x, y + dt/2*k1y)
            k3x, k3y = vel(x + dt/2*k2x, y + dt/2*k2y)
            k4x, k4y = vel(x + dt*k3x,   y + dt*k3y)
            x += dt/6*(k1x + 2*k2x + 2*k3x + k4x) # integrates vx
            y += dt/6*(k1y + 2*k2y + 2*k3y + k4y) # integrates vy
            xs.append(x); ys.append(y) # add position (x,y) to list to make a trajectory
            if (abs(x) > cp1.Lx/2 or abs(y) > cp1.Ly/2 # check if trajectory has left the domain
                    or np.hypot(x, y) < r_stop): # check if trajectory has reached the origin
                break
        return np.array(xs), np.array(ys)
 

    D0xy = cp1.mob.D0()
    D1xy = cp1.mob.D1(cp1.Ra,cp1.Th)
    D2xy = cp1.mob.D2(cp1.Ra,cp1.Th)

    # Pre-build interpolators for each order of approximation based on eq v = F + D*c. We solve dx/dt = f, f(x,y) = F + D*c w/ RK4
    # ivxk ivyk contains the values of the velocity field at each point (x,y) in the grid, for the k-th order approximation of the control
    # i.e ivx0([y,x]) = vx(x,y) at 0th order, ivx1([y,x]) = vx(x,y) at 1st order, etc. Same for ivy.

    ivx0, ivy0 = _make_interp(D0xy*cFull0_x,D0xy*cFull0_y) 
    ivx1, ivy1 = _make_interp(eps*f_x + D0xy*(cFull0_x + eps*cFull1_x) + eps*D1xy*cFull0_x, 
                              eps*f_y + D0xy*(cFull0_y + eps*cFull1_y) + eps*D1xy*cFull0_y)
    ivx2, ivy2 = _make_interp(eps*f_x + eps**2*f_2x + D0xy*(cFull0_x + eps*cFull1_x + eps**2*cFull2_x) + eps*D1xy*(cFull0_x + eps*cFull1_x) + eps**2*D2xy*cFull0_x,
                              eps*f_y + eps**2*f_2y + D0xy*(cFull0_y + eps*cFull1_y + eps**2*cFull2_y) + eps*D1xy*(cFull0_y + eps*cFull1_y) + eps**2*D2xy*cFull0_y)
 
    # Several starting points spread around the domain
    starting_points = [(4, 4), (-4, 4), (-4, -4), (4, -4),
                       (4, 0), (0, 4), (-4, 0), (0, -4)]
    pt_colors = plt.cm.tab10(np.linspace(0, 0.8, len(starting_points)))
 
    # traj_plot4: superpose the 3 orders from (-4,4) to compare corrections
    x0_ref, y0_ref = -4.0, 4.0
    traj_plot4.set_title(r"comparison of orders from $(x0ref,y0ref)$", fontsize=8)
    
    order_configs = [
        (ivx0, ivy0, 'royalblue',   r"$\mathcal{O}(\varepsilon^0)$"),
        (ivx1, ivy1, 'darkorange',  r"$\mathcal{O}(\varepsilon^1)$"),
        (ivx2, ivy2, 'forestgreen', r"$\mathcal{O}(\varepsilon^2)$"),
    ]

    for ivx, ivy, col, lbl in order_configs:
        tx, ty = RK4_integrate_traj(x0_ref, y0_ref, ivx, ivy)
        traj_plot4.plot(tx, ty, '-', color=col, lw=1.5, label=lbl)

    traj_plot4.legend(fontsize=6, loc='upper right')
    traj_plot4.plot(x0_ref, y0_ref, 'ko', ms=3)
    traj_plot4.plot(0, 0, 'D', ms=3, color='black')
    
    # traj_plot1, traj_plot2, traj_plot3: show the 3 orders separately for all starting points
    traj_configs = [
        (traj_plot1, ivx0, ivy0,
         r"${v}_0$ only"),
        (traj_plot2, ivx1, ivy1,
         r"${v}_0 + \varepsilon{v}_1$"),
        (traj_plot3, ivx2, ivy2,
         r"${v}$ to $\mathcal{O}(\varepsilon^2)$"),
    ]
 
    for ax, ivx, ivy, title in traj_configs:
        ax.set_title(title, fontsize=8)
        for (x0i, y0i), col in zip(starting_points, pt_colors):
            tx, ty = RK4_integrate_traj(x0i, y0i, ivx, ivy)
            ax.plot(tx, ty, '-', color=col, lw=1.0)
            ax.plot(x0i, y0i, 'o', color=col, ms=3)
        ax.plot(0, 0, 'D', ms=3, color='black')
 
    
 
    # Apply consistent axis formatting to all trajectory subplots
    traj_axes = [traj_plot1, traj_plot2, traj_plot3, traj_plot4]
    for ax in traj_axes:
        ax.set_aspect(1)
        ax.set_xlim(-cp1.Lx/2, cp1.Lx/2)
        ax.set_ylim(-cp1.Ly/2, cp1.Ly/2)
        ax.set_xticks(np.linspace(-cp1.Lx/2, cp1.Lx/2, 5))
        ax.set_yticks(np.linspace(-cp1.Ly/2, cp1.Ly/2, 5))
        ax.set_xticklabels(np.linspace(-cp1.Lx/2, cp1.Lx/2, 5), fontsize=8)
    traj_plot1.set_yticklabels(np.linspace(-cp1.Ly/2, cp1.Ly/2, 5), fontsize=8)
    for ax in [traj_plot2, traj_plot3, traj_plot4]:
        ax.set_yticklabels([])


    my_path = os.path.dirname(os.path.abspath(__file__))

    control_figure.savefig(my_path + "/Plots/" + saveStr + ".pdf", dpi=350)

###########################################
###########################################
###########################################

class simple_mobility(mobolities):
    def D0(self):
        return 1 
    def D1(self,Ra,Th):
        if isinstance(Ra,(sp.Basic,sp.MatrixBase)):
            return sp.exp(-((Ra-3)**2))*sp.sin(2*Th) 
        else:
            return np.exp(-((Ra-3)**2))*np.sin(2*Th)
        
    def D2(self,Ra,Th):
        if isinstance(Ra,(sp.Basic,sp.MatrixBase)):
            return sp.sin(Ra) 
        else:
            return np.sin(Ra) 

class simple_force(forces):
    #first order
    def f_R(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return  sp.cos(2*Th)
        else:
            return  np.cos(2*Th)
        
    def f_T(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return (1 + sp.exp(-Ra+2))**(-1)
        else:
            return (1 + np.exp(-Ra+2))**(-1)

    #second order
    def f_R2(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return  sp.sin(0)#sp.cos(2*Th) #+ sp.sin(2*Th)
        else:
            return  np.sin(0)#np.cos(2*Th) #+ np.sin(2*Th)

    def f_T2(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return sp.sin(0)#(1 + sp.exp(-Ra+2))**(-1) 
        else:
            return np.sin(0)#(1 + np.exp(-Ra+2))**(-1) 
    


class conservative_force(forces):
    def __init__(self, Lx, Ly):
        self.k = 2*np.pi * (5/Lx)
        self.k_sym = 2*sp.pi * sp.Rational(5, Lx)


    #first order
    def f_R(self, Ra, Th):
        

        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            X = Ra * sp.cos(Th)
            Y = Ra * sp.sin(Th)
            Potential = sp.cos(self.k_sym * X) * sp.cos(self.k_sym * Y)
            return -sp.diff(Potential, Ra)
        else:
            X = Ra * np.cos(Th)
            Y = Ra * np.sin(Th)
            Potential = np.cos(self.k * X) * np.cos(self.k * Y)
            dV_dR =  - (np.sin(self.k * X) * np.cos(self.k * Y)) * np.cos(Th) \
                - (np.cos(self.k * X) * np.sin(self.k * Y)) * np.sin(Th)
            return - self.k * dV_dR

    def f_T(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            X = Ra * sp.cos(Th)
            Y = Ra * sp.sin(Th)
            Potential = sp.cos(self.k_sym * X) * sp.cos(self.k_sym * Y)
            return -sp.diff(Potential, Th)
        else:
            X = Ra * np.cos(Th)
            Y = Ra * np.sin(Th)
            Potential = np.cos(self.k * X) * np.cos(self.k * Y)   # ← np.sin, pas sp.sin
            dV_dTh = - (np.cos(self.k * X) * np.sin(self.k * Y)) * (-Ra * np.sin(Th)) \
                - (np.sin(self.k * X) * np.cos(self.k * Y)) * ( Ra * np.cos(Th))
            return -self.k * dV_dTh  # composante angulaire (= Ra * f_theta_unitaire)
        
    #second order
    def f_R2(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return  0*Ra
        else:
            return  0*Ra

    def f_T2(self,Ra,Th):
        if isinstance(Th,(sp.Basic,sp.MatrixBase)):
            return 0*Ra
        else:
            return 0*Ra


class mobility2(mobolities):
    def D0(self):
        return 1 
    def D1(self,Ra,Th):
        if isinstance(Ra,(sp.Basic,sp.MatrixBase)):
            return sp.sin(2*Th) 
        else:
            return np.sin(2*Th) 
        
    def D2(self,Ra,Th):
        if isinstance(Ra,(sp.Basic,sp.MatrixBase)):
            return sp.sin(Ra) 
        else:
            return np.sin(Ra) 


class FullAsymmetricForce(forces):
    def __init__(self):
        super().__init__()

    # --- ORDRE 1 ---
    def f_R(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            return sp.cos(2*Th) 
        else:
            return np.cos(2*Th) 

    def f_T(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            return (1 + Ra**2)**(-1) 
        else:
            return (1 + Ra**2)**(-1) 

    # --- ORDRE 2 (Non nuls !) ---
    def f_R2(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            return sp.exp(-Ra**2/10) * sp.sin(2*Th) 
        else:
            return np.exp(-Ra**2/10) * np.sin(2*Th) 

    def f_T2(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            return sp.exp(-Ra**2/10) * sp.sin(2*0)
        else:
            return np.exp(-Ra**2) * np.sin(2*0)


class FullAsymmetricMobility(mobolities):
    def D0(self):
        return 1

    # --- ORDRE 1 ---
    def D1(self, Ra, Th):
        if isinstance(Th, (sp.Basic, sp.MatrixBase)):
            return sp.Rational(1, 10) * Ra * sp.sin(2*Th) 
        else:
            return 0.1 * Ra * np.sin(2*Th) 

    # --- ORDRE 2 (Non nul !) ---
    def D2(self, Ra, Th):
        if isinstance(Ra, (sp.Basic, sp.MatrixBase)):
            return sp.exp(-(Ra-3)**2) * sp.cos(2*Th) 
        else:
            return np.exp(-(Ra-3)**2) * np.cos(2*Th) 

##############################################################################

if __name__ == "__main__":
    sf = FullAsymmetricForce()
    sm = FullAsymmetricMobility()
    st = times()
    simple_cp2 = simple_control_problem(Lx=10, Ly=10, Nx=70, Ny=70, mob=sm, force=sf, time=st)
    plot_overview_figure(cp1=simple_cp2, saveStr="classic_ctl")
