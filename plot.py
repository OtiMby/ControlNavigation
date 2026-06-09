import time
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import json
from optimized_control_nav import (
    Force, 
    Mobility,
    control_nav,
    load_compare,
    load_conservative,
    load_mob_only,
    load_divless,
    load_cisaillement,
    load_multiples,
    load_divless_shifted, 
    load_divless_mob,
    load_crossed_shear
)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
import os 

# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------


def plot_overview_figure(
    cn,
    gp,
    eps,
    cuts,
    starting_points,
    compared_paths=None,
    saveStr="simple_dynamics_overview",
    Na = 70,
    lw = 1.5, ow=1.5,
    mlw = 0.7, mow=0.5,
    Npath=10
    ):
    X, Y   = cn.X, cn.Y
    EXTENT = [-cn.Lx / 2, cn.Lx / 2, -cn.Ly / 2, cn.Ly / 2]
    TICKS  = np.linspace(-cn.Lx / 2, cn.Lx / 2, 5)

    gshape = (len(gp), len(gp[0]))
    fig = plt.figure(figsize=(15, 22))
    gs  = GridSpec(
        gshape[0], gshape[1], figure=fig,
        height_ratios=[1 for _ in range(gshape[0])],
        left=0.07, right=0.97,
        bottom=0.03, top=0.94,
        hspace=0.55, wspace=0.38,
    )


    QSTRIDE = max(1, cn.Nx // Na)
    Xq, Yq  = X[::QSTRIDE, ::QSTRIDE], Y[::QSTRIDE, ::QSTRIDE]

    # Use cached unit vectors
    eRx, eRy = cn._eRx, cn._eRy
    eTx, eTy = cn._eTx, cn._eTy

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
        im = ax.imshow(data.T, cmap=cmap, extent=EXTENT, origin="lower")
        _add_colorbar(ax, im)

    def _decompose_force(force):
        fRv      = force.fR(X, Y)
        fTv      = force.fT(X, Y)
        fRx, fRy = fRv * eRx, fRv * eRy
        fTx, fTy = fTv * eTx, fTv * eTy
        return fRx, fRy, fTx, fTy, fRx + fTx, fRy + fTy

    def _make_cut(f, x):
        return np.array([[i, f(i)] for i in x])

    f1Rx, f1Ry, f1Tx, f1Ty, f1x, f1y = _decompose_force(cn.f1)
    f2Rx, f2Ry, f2Tx, f2Ty, f2x, f2y = _decompose_force(cn.f2)

    D1 = cn.D1(X, Y)
    D2 = cn.D2(X, Y)

    T0, T1, T2 = cn.T0, cn.T1, cn.T2
    tTot = T0 + eps * T1 + eps**2 * T2

    c0x, c0y = cn.compute_control_force(0, eps)
    c1x, c1y = cn.compute_control_force(1, eps)
    c2x, c2y = cn.compute_control_force(2, eps)

    vx, vy = cn.velocity_field(2, eps)

    algo = cn.path_RK4

    count = 0

    quant = [item for innerlist in gp for item in innerlist ]
    quant = {item for innerlist in quant for item in innerlist}

    def darken(color, f=0.55):
                    r, g, b = mcolors.to_rgb(color)
                    return (r * f, g * f, b * f)

    if "eps1" in quant or "eps2" in quant:
        constraining = cn.compute_eps()[1]
        res1D = cn.radial_eps_validity()

    if "p1" in quant:
        t1 = [algo(1, (x0r, y0r), eps, 0.001, 10000, 0.1) for (x0r, y0r) in starting_points]
    
    if "p2" in quant:
        t2 = [algo(2, (x0r, y0r), eps, 0.001, 10000, 0.1) for (x0r, y0r) in starting_points]
    
    if "p0" in quant:
        t0 = [algo(0, (x0r, y0r), eps, 0.001, 10000, 0.1) for (x0r, y0r) in starting_points]

    if "zp2" in quant:
        R = 0.7 * np.hypot(cn.Lx/2, cn.Ly/2)
        ths = np.linspace(0, np.pi/2, Npath)
        sp = [(-R*np.cos(th), R*np.sin(th)) for th in ths]
        vx, vy = cn.velocity_field(2, eps)
        tz = cn.trace_paths(vx, vy, sp)
    
    if "pc" in quant:
        t = []
        x0r, y0r = starting_points[0]
        if "p0" in quant:
            t.append([t0[0], "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        else:
            t.append([algo(0, (x0r, y0r), eps, 0.001, 10000, 0.1), "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        if "p1" in quant:
            t.append([t1[0], "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        else:
            t.append([algo(1, (x0r, y0r), eps, 0.001, 10000, 0.1), "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        if "p2" in quant:
            t.append([t2[0], "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
        else:
            t.append([algo(2, (x0r, y0r), eps, 0.001, 10000, 0.1), "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
    
    if "cp" in quant:
        apaths = compared_paths[0]
        numpaths = compared_paths[1]
        
    for i in range(gshape[0]):
        for j in range(gshape[1]):
            ax = fig.add_subplot(gs[i,j])

            if "F1R" in gp[i][j]:
                _quiver(ax,  f1Rx, f1Ry, "#2980b9")
                _style(ax,  r"$f_{1R}$",          left=True)
            
            if "F1T" in gp[i][j]:
                _quiver(ax,  f1Tx, f1Ty, "#2980b9")
                _style(ax,  r"$f_{1\theta}$")
            
            if "F2R" in gp[i][j]:
                _quiver(ax,  f2Rx, f2Ry, "#2980b9")
                _style(ax,  r"$f_{2R}$")

            if "F2T" in gp[i][j]:
                _quiver(ax,  f2Tx, f2Ty, "#2980b9")
                _style(ax,  r"$f_{2\theta}$")
    
            if "F1" in gp[i][j]:
                _quiver(ax, eps * f1x + eps**2 * f2x, eps * f1y + eps**2 * f2y, "#2c3e50")
                _style(ax, r"$\vec{F}_{tot}$", left=True, bottom=True)
            
            if "D1" in gp[i][j]:
                _ishow(ax, D1, "Blues")
                _style(ax,   r"$D_1(\vec{x})$", bottom=True)
            
            if "D2" in gp[i][j]:
                _ishow(ax, D2, "Blues")
                _style(ax,   r"$D_2(\vec{x})$",bottom=True)
            
            if "pot1" in gp[i][j]:
                _ishow(ax, cn.f1.potential(X, Y), "RdPu")
                _style(ax, r"$\phi_1(\vec{x})$",  bottom=True)
            
            if "pot2" in gp[i][j]:
                _ishow(ax, cn.f2.potential(X, Y), "RdPu")
                _style(ax, r"$\phi_2(\vec{x})$",  bottom=True)

            if "T" in gp[i][j]:
                _ishow(ax, tTot, "YlOrRd")
                _style(ax, rf"$\mathcal{{T}}^*$  ($\varepsilon={eps:.2g}$)",  bottom=True)
            
            if "Ttot-0" in gp[i][j]:
                _ishow(ax, tTot-T0, "YlOrRd")
                _style(ax, r"$\mathcal{{T}}^* - \mathcal{T}^*_0$", left=True, bottom=True)
            
            if "T1" in gp[i][j]:
                _ishow(ax, T1, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_1$", bottom=True)
            
            if "T2" in gp[i][j]:
                _ishow(ax, T2, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_2$", bottom=True)
            
            if "c1" in gp[i][j]:
                _quiver(ax, c1x, c1y, "#922b21")
                _style(ax, r"$\hat{c}_1$", left=True, bottom=True)
            
            if "c2" in gp[i][j]:
                _quiver(ax, c2x, c2y, "#1e8449")
                _style(ax, r"$\hat{c}_2$", left=True, bottom=True)
            
            if "v" in gp[i][j]:
                _quiver(ax, vx, vy, "#1e8449")
                _style(ax, r"$\vec{v}$", left=True, bottom=True)

            if "p1" in gp[i][j]:
                for tx, ty in t1:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v_0+\varepsilon v_1$",bottom=True)
               
            if "p2" in gp[i][j]:
                for tx, ty in t2:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v$ to $\mathcal{O}(\varepsilon^2)$",        bottom=True)
            
            if "p0" in gp[i][j]:
                for tx, ty in t0:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v_0$ only",bottom=True)

            if "pc" in gp[i][j]:
                for (tx, ty),col, lbl in t:
                    ax.plot(tx, ty, "-", color=col, lw=lw, label=lbl)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                    ax.legend(fontsize=6, loc="upper right", framealpha=0.7, edgecolor="none")

                _style(ax, r"Order comparison  $(x_0^{ref}, y_0^{ref})$", left=True, bottom=True)

            if "zp2" in gp[i][j]:
                for (tx, ty) in tz:
                    ax.plot(tx, ty, "-", lw=mlw)
                    ax.plot(tx[0], ty[0], "ko", ms=mow)
                    ax.legend(fontsize=6, loc="upper right", framealpha=0.7, edgecolor="none")
                _style(ax, r"multiple path in corner", left=True, bottom=True)
                ax.set_xlim(-cn.Lx/2, 0.5)
                ax.set_ylim(-0.5, cn.Ly/2)


            if "cut" in gp[i][j]: # make a cut on functions in 'func_map' on specified axis y=f(x)
                func_map = {
                    "T1":T1, "T2":T2, "T0":T0, 
                    "pot1" : cn.f1.potential, "pot2":cn.f2.potential,
                    "eT1":eps*T1, "eT2":eps**2*T2,}
                data = cuts[count]
                cut_f, funcs, title = data
                cut = _make_cut(cut_f, cn.X[:,0])
                for func in funcs:
                    if func in func_map.keys():
                        y = RegularGridInterpolator(([X[:,0], Y[0,:]]), func_map[func])(cut)
                        ax.loglog(np.abs(y[len(y)//2:]), label=func)
                        ax.grid()
                        ax.set_title(rf"cut at {title} of functions {' '.join(funcs)}")
                        ax.legend()
                        ax.set_xlabel('distance on the cut')
                count += 1

            if "eps1" in gp[i][j] or "eps2" in gp[i][j]: # plot radial eps validity in loglog
                R = np.linspace(0, np.hypot(cn.Lx/2,cn.Ly/2), cn.Nr)
                if "eps1" in gp[i][j]:
                    ax.loglog(R[1:], res1D[0], label=r"$\epsilon (R)$")
                    ax.set_title("total comparison")
                
                if "eps2" in gp[i][j]:
                    ax.loglog(R[1:], res1D[1], label=r"$\epsilon (R)$")
                    ax.set_title("first and 0-th order comparison")
                
                if "eps3" in gp[i][j]:
                    ax.loglog(R[1:], res1D[2], label=r"$\epsilon (R)$")
                    ax.set_title("most constraining eps")
                
                ax.loglog(R[1:], np.full_like(a=R[1:], fill_value=eps), label=r"current $\epsilon$")
                ax.grid()
                ax.legend()
                ax.set_xlabel('radius')
                ax.set_ylabel(r'smallest $\epsilon$ inside circle')
            
            if "area" in gp[i][j]: # show purple pixel where expansion is not valid according to cn.eps_validity
                eps_field_xy = np.empty_like(constraining)
                eps_field_xy[cn.order] = constraining
                eps_field = eps_field_xy.reshape(cn.X.shape)
                mask = np.ma.masked_where(eps_field >= eps, eps_field<=eps)
                ax.imshow(mask.T, extent=EXTENT, origin='lower')

            if "cp" in gp[i][j]: # plots anaylitical paths in dash lines and numerical paths in completes lines
                colors = plt.cm.tab10.colors   

                for k, ((xa, ya), (xn, yn)) in enumerate(zip(apaths, numpaths)):
                    c = colors[k % len(colors)]
                    ax.plot(xn, yn, "-",  lw=mlw, color=c, zorder=1)   # numerical
                    ax.plot(xa, ya, "--", lw=mlw, color=darken(c),zorder=2)   # exact   


    fig.suptitle(
        rf"Perturbative control navigation — $\varepsilon={eps:.2g}$"
        rf"   grid ${cn.Nx}\times{cn.Ny}$   domain $[\!-\!{cn.Lx/2:.0f},{cn.Lx/2:.0f}]^2$",
        fontsize=11, fontweight="bold", y=0.975,
    )

    # saving plots in Plots/"SaveStr"
    my_path  = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(my_path, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, saveStr + ".png")
    print(f"Saving -> {out_path}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def post_process(file, conservative,epsilon, plot_grid, cuts, params):
    with open(file, "r") as f:
        data = json.load(f)
    L = data['L']
    N = data['N']
    f1 = Force(L, L, N, N, conservative=conservative, expr_phi=data['pot'])
    f2 = Force(L, L, N, N, conservative=False, expr_phi="0")
    D1 = Mobility(expr="0",dtheta_func=lambda x, y: 0*x)
    D2 = Mobility(expr="0",dtheta_func=lambda x, y: 0*x)

    cn = control_nav(Lx=L, Ly=L, Nx=N, Ny=N, D0=1. ,D1=D1, D2=D2, f1=f1, f2=f2)
    cn.precompute_fields()
    cn.T1 = np.array(data['T1'])
    cn.T2 = np.array(data['T2'])
    cn.dT1dTh = np.array(data['dT1dTh'])
    cn.dT2dTh = np.array(data['dT2dTh'])
    print("data loaded")

    r  = 0.9 * L / 2
    starting_points = [ # starting points for plotting paths, first one is used for 'pc'
        [-r,  r], [ r,  r], [-r, -r], [ r, -r],
        [0.,  r], [-r, 0.], [0., -r], [ r, 0.],
    ]

    print("comparing paths")
    err, T, compared_paths = cn.compare_paths(2, cn.make_theta_eq , params['compare']['Np'], epsilon, Npaths=params['plot']['Npath'])
    print("plotting")
    plot_overview_figure(cn, plot_grid, epsilon, cuts, starting_points, compared_paths, saveStr=file, **params['plot'])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def computation_time_test(Lx, Ly, N):
    D0 = 1.0
    chronos = []
    for dx in tqdm([0.1 + 0.005 * i for i in range(1, 41)]):
        N = int(Lx / dx)
        f1, f2, D1, D2, _ = load_conservative(Lx, Ly, N)
        cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0,
                         D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.15)
        t0 = time.time()
        cn.compute_T1()
        cn.compute_dT1dTh()
        cn.compute_T2()
        cn.compute_dT2dTh()
        chronos.append((dx, time.time() - t0))

    with open("computation_times.txt", "a") as fh:
        for dx, t in chronos:
            fh.write(f"dx={dx:.3f}: {t:.2f} seconds\n")
    print("\n".join(f"  dx={dx:.3f}: {t:.2f}s" for dx, t in chronos))
    plt.loglog([dx for dx, _ in chronos], [t for _, t in chronos], marker="o")
    plt.xlabel("dx"); plt.ylabel("time (s)")
    plt.title("Computation time vs grid spacing")
    plt.show()


def run(inputs, plot_grid, cuts, params):
    Lx, Ly = params['computation']['Lx'], params['computation']['Ly']
    N  = int(Lx / params['computation']['dx'])
    D0 = params['computation']['D0']
    epsilon = params['computation']['eps']

    r  = 0.9 * Lx / 2
    starting_points = [ # starting points for plotting paths, first one is used for 'pc'
        [-r,  r], [ r,  r], [-r, -r], [ r, -r],
        [0.,  r], [-r, 0.], [0., -r], [ r, 0.],
    ]

    f1, f2, D1, D2, SaveStr = inputs(Lx, Ly, N)

    """
     eq is the equation in theta with the right forces. General is 
      dθdt=sinθ² ∂x​v+sinθcosθ(∂x​u−∂y​v)− cosθ² ∂y​u -> eq is a func of theta ( dtheta/dt )
      with u = f_x, v = f_y
    """

  
    cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0,D1=D1, D2=D2, f1=f1, f2=f2, epsilon=params['computation']["eps"])

    print("Computing T1 ...")
    cn.compute_T1()
    print("Computing dT1/dθ ...")
    cn.compute_dT1dTh()
    print("Computing T2 ...")
    cn.compute_T2()
    print("Computing dT2/dθ ...")
    cn.compute_dT2dTh()
    print("comparing numerical path with analytical...")
    
    eps = [0.1 - 0.01 * i for i in range(1)]
    eps.append(epsilon)
    eps.sort()
    errs = []
    Ts = []

    err, T, compared_paths = cn.compare_paths(2, cn.make_theta_eq , params['compare']['Np'], epsilon, Npaths=params['plot']['Npath'])

    for e in eps:
        err, T = cn.compare_paths(2, cn.make_theta_eq , params['compare']['Np'], e, Npaths=0)
        Ts.append(np.nanmean(T))
        errs.append(np.nanmean(err))
        print(fr"$\varepsilon$={round(e, 5)} -> erreur spatiale quad moyenne : {np.nanmean(err)}, erreur temporelle relative moyenne : {np.nanmean(T)}")
    
    print('Plotting...')
    plot_overview_figure(cn, plot_grid, epsilon, cuts, starting_points, compared_paths, saveStr=SaveStr+"2", **params['plot'])
    plt.plot(eps, Ts),
    plt.plot(eps, errs)
    plt.show()
    

if __name__ == "__main__":

    cuts=[ # create a cut at cuts[i][0] for func cuts[i][1] with title cuts[i][2]. Refer to 'func_map' to know functions name 
        [lambda x: 0, ['T2'], "y=0"], 
        [lambda x: 0, ['T1'], "y=0"], 
        [lambda x: 0.2, ['T2'], "y=0.2"]
    ]
    plot_tables = { # Create a grid plot with same shape as plot_tables[i], each cell of the table plots a specific func 
                    # Refer to logic table in plot function for names
                    # for paths -> pi : sum(order <= i) path, 
                    #              pc : all orders from same starting points (first point in list), 
                    #              zp2: 'Npath' different paths in corner
        "complete":[
            [["F1R"],   ["F1T"],        ["F2R"],    ["F2T"]],
            [["F1"],    ["pot1"],       ["D1"],     ["D2"]],
            [["T"],     ["Ttot-0"],     ["T1"],     ["T2"]],
            [["ctot"],  ["c1"],         ["c2"],     ["v"]],
            [["pc"],    ["p0"],         ["p1"],     ["p2"]],
            [["cut"],   ["cut"],        ["eps1"],   ["eps2"]],
        ],
        "1st":[
            [["F1R"],       ["F1T"],        ["F1"],         ["pot1"]],
            [["T"],         ["Ttot-0"],     ["T1"],         ["T2"]],
            [["c"],         ["c1"],         ["c2"],         ["v"]],
            [["cp", "F1"],    ["p1","T"],     ["p2", "F1"],    ["zp2", "F1"]],
            [["cut"],       ["cut"],        ["cut"],       ["eps2"]],
        ],
        "single":[[["F1", "zp2"]]]
    }

    params={ # Store all the params for computation, plotting and comparing
             # Lx, Ly : Size of grid, dx : size of pixel, D0, eps: constants
             # Na : Number of arrows, lw, ow: caracteristics of single path line (size), mlw, mow : caracteristics of multiple path line (size), Npath:Number of path line for 'zp2' plot
             # Np: Number of path to compare ( analytical vs Num)
        "computation": {"Lx":10., "Ly":10., "dx":0.01, "D0":1.0, "eps":0.005},
        "plot" : {"Na": 100, "lw":1, "ow":1, "mlw":1.1, "mow":1, "Npath": 10},
        "compare" : {"Np":400}
    }
    
    #run(inputs=load_cisaillement, plot_grid=plot_tables["1st"], cuts=cuts, params=params)
    post_process("post_process_test.json", True, 0.01, plot_grid=plot_tables["1st"], cuts=cuts, params=params)