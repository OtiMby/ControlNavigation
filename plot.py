import time
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
 
from optimized_control_nav import (
    control_nav,
    load_compare,
    load_conservative1,
    load_conservative2,
    load_mob_only,
    load_divless,
    load_divless2,
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
    cuts,
    starting_points,
    saveStr="simple_dynamics_overview",
    path_algorithm="FMM",
    Na = 70
    ):
    X, Y   = cn.X, cn.Y
    ep     = cn.epsilon
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
        return np.array([[f(i), i] for i in x])

    f1Rx, f1Ry, f1Tx, f1Ty, f1x, f1y = _decompose_force(cn.f1)
    f2Rx, f2Ry, f2Tx, f2Ty, f2x, f2y = _decompose_force(cn.f2)

    D1 = cn.D1(X, Y)
    D2 = cn.D2(X, Y)

    T0, T1, T2 = cn.T0, cn.T1, cn.T2
    tTot = T0 + ep * T1 + ep**2 * T2

    c0x, c0y = cn.c0x, cn.c0y
    c1x, c1y = cn.c1x, cn.c1y
    c2x, c2y = cn.c2x, cn.c2y   

    vx, vy = cn.velocity_field(2)

    algo = cn.path_FMM if path_algorithm == "FMM" else cn.path_RK4

    count = 0

    quant = [item for innerlist in gp for item in innerlist ]
    quant = {item for innerlist in quant for item in innerlist}

    if "eps1" in quant or "eps2" in quant:
        constraining = cn.compute_eps()[1]
        res1D = cn.radial_eps_validity()

    if "p1" in quant:
        t1 = [algo(1, (x0r, y0r), 0.01, 5000, 0.1) for (x0r, y0r) in starting_points]
    
    if "p2" in quant:
        t2 = [algo(2, (x0r, y0r), 0.01, 5000, 0.1) for (x0r, y0r) in starting_points]
    
    if "p0" in quant:
        t0 = [algo(0, (x0r, y0r), 0.01, 5000, 0.1) for (x0r, y0r) in starting_points]
    
    if "pc" in quant:
        t = []
        x0r, y0r = starting_points[0]
        if "p0" in quant:
            t.append([t0[0], "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        else:
            t.append([algo(0, (x0r, y0r), 0.01, 5000, 0.1), "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        if "p1" in quant:
            t.append([t1[0], "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        else:
            t.append([algo(1, (x0r, y0r), 0.01, 5000, 0.1), "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        if "p2" in quant:
            t.append([t2[0], "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
        else:
            t.append([algo(2, (x0r, y0r), 0.01, 5000, 0.1), "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
        
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
                _quiver(ax,  f2Rx, f1Ry, "#2980b9")
                _style(ax,  r"$f_{2R}$")

            if "F2T" in gp[i][j]:
                _quiver(ax,  f2Tx, f2Ty, "#2980b9")
                _style(ax,  r"$f_{2\theta}$")
    
            if "F1" in gp[i][j]:
                _quiver(ax, ep * f1x + ep**2 * f2x, ep * f1y + ep**2 * f2y, "#2c3e50")
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
                _style(ax, rf"$\mathcal{{T}}^*$  ($\varepsilon={ep:.2g}$)",  bottom=True)
            
            if "Ttot-0" in gp[i][j]:
                _ishow(ax, tTot-T0, "YlOrRd")
                _style(ax, r"$\mathcal{{T}}^* - \mathcal{T}^*_0$", left=True, bottom=True)
            
            if "T1" in gp[i][j]:
                _ishow(ax, T1, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_1$", bottom=True)
            
            if "T2" in gp[i][j]:
                _ishow(ax, T2, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_2$", bottom=True)
            
            if "c" in gp[i][j]:
                _quiver(ax, c0x + ep*c1x + ep**2*c2x,
                                c0y + ep*c1y + ep**2*c2y, "#1e8449")
                _style(ax, r"$\hat{c}$  to  $\mathcal{O}(\varepsilon^2)$", left=True, bottom=True)
            
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
                    ax.plot(tx, ty, "-", lw=1.5)
                    ax.plot(tx[0], ty[0], "ko", ms=3)
                _style(ax, r"$v_0+\varepsilon v_1$",bottom=True)
               
            if "p2" in gp[i][j]:
                for tx, ty in t2:
                    ax.plot(tx, ty, "-", lw=1.5)
                    ax.plot(tx[0], ty[0], "ko", ms=3)
                _style(ax, r"$v$ to $\mathcal{O}(\varepsilon^2)$",        bottom=True)
            
            if "p0" in gp[i][j]:
                for tx, ty in t0:
                    ax.plot(tx, ty, "-", lw=1.5)
                    ax.plot(tx[0], ty[0], "ko", ms=3)
                _style(ax, r"$v_0$ only",bottom=True)

            if "pc" in gp[i][j]:
                for (tx, ty),col, lbl in t:
                    ax.plot(tx, ty, "-", color=col, lw=1.5, label=lbl)
                    ax.plot(tx[0], ty[0], "ko", ms=3)
                    ax.legend(fontsize=6, loc="upper right", framealpha=0.7, edgecolor="none")
                _style(ax, r"Order comparison  $(x_0^{ref}, y_0^{ref})$", left=True, bottom=True)

            if "cut" in gp[i][j]:
                func_map = {
                    "T1":T1, "T2":T2, "T0":T0, 
                    "pot1" : cn.f1.potential, "pot2":cn.f2.potential,
                    "eT1":ep*T1, "eT2":ep**2*T2,}
                data = cuts[count]
                cut_f, funcs, title = data
                cut = _make_cut(cut_f, cn.X[:,0])
                for func in funcs:
                    if func in func_map.keys():
                        y = RegularGridInterpolator(([X[:,0], Y[0,:]]), func_map[func])(cut)
                        ax.plot(y, label=func)
                        ax.set_title(rf"cut at {title} of functions {' '.join(funcs)}")
                        ax.legend()
                        ax.set_xlabel('distance on the cut')
                count += 1

            if "eps1" in gp[i][j] or "eps2" in gp[i][j]:
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
                
                ax.loglog(R[1:], np.full_like(a=R[1:], fill_value=cn.epsilon), label=r"current $\epsilon$")
                ax.grid()
                ax.legend()
                ax.set_xlabel('radius')
                ax.set_ylabel(r'smallest $\epsilon$ inside circle')
            
            if "area" in gp[i][j]:
                eps_field_xy = np.empty_like(constraining)
                eps_field_xy[cn.order] = constraining
                eps_field = eps_field_xy.reshape(cn.X.shape)
                mask = np.ma.masked_where(eps_field >= ep, eps_field<=ep)
                ax.imshow(mask.T, extent=EXTENT, origin='lower')
    
    """peaks = np.zeros_like(tfield)
    #peaks = np.where(cn.f1.)
    np.ma.masked_where(peaks >= 0, peaks<=ep)
    ax.imshow(tfield.T, cmap="YlOrRd", extent=EXTENT,
                origin="lower", alpha=0.95)"""
        

    fig.suptitle(
        rf"Perturbative control navigation — $\varepsilon={ep:.2g}$"
        rf"   grid ${cn.Nx}\times{cn.Ny}$   domain $[\!-\!{cn.Lx/2:.0f},{cn.Lx/2:.0f}]^2$",
        fontsize=11, fontweight="bold", y=0.975,
    )

    my_path  = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(my_path, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, saveStr + ".png")
    print(f"Saving -> {out_path}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def computation_time_test(Lx, Ly, N):
    D0 = 1.0
    chronos = []
    for dx in tqdm([0.1 + 0.005 * i for i in range(1, 41)]):
        N = int(Lx / dx)
        f1, f2, D1, D2, _ = load_conservative2(Lx, Ly, N)
        cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0,
                         D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.15)
        t0 = time.time()
        cn.compute_T1()
        cn.compute_dT1dTh()
        cn.compute_T2()
        cn.compute_dT2dTh()
        cn.compute_all_control_forces()
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
    Lx, Ly = params['Lx'], params['Ly']
    N  = int(Lx / params['dx'])
    D0 = params['D0']

    r  = 0.9 * Lx / 2
    starting_points = [
        [-r,  r], [ r,  r], [-r, -r], [ r, -r],
        [0.,  r], [-r, 0.], [0., -r], [ r, 0.],
    ]

    f1, f2, D1, D2, saveStr = inputs(Lx, Ly, N)
    cn = control_nav(Lx=Lx, Ly=Ly, Nx=N, Ny=N, D0=D0,
                     D1=D1, D2=D2, f1=f1, f2=f2, epsilon=params["eps"])

    print("Computing T1 ...")
    cn.compute_T1()
    print("Computing dT1/dθ ...")
    cn.compute_dT1dTh()
    print("Computing T2 ...")
    cn.compute_T2()
    print("Computing dT2/dθ ...")
    cn.compute_dT2dTh()
    print("Computing control forces ...")
    cn.compute_all_control_forces()
    print("Plotting ...")
    
    plot_overview_figure(cn, plot_grid, cuts, starting_points, saveStr=saveStr, path_algorithm=params['algo'], Na=params["Na"])
    
if __name__ == "__main__":
    cuts=[
        [lambda x: 0, ['eT1', 'eT2'], "y=0"], 
        [lambda x: 0.75, ['T1'], "y=0.75"], 
    ]
    plot_tables = {
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
            [["pc","T"],    ["p1","T"],     ["p2", "T"],    ["p2", "F1"]],
            [["cut"],       ["cut"],        ["eps1"],       ["eps2"]],
        ]
    }
    params={
        "Lx":10, "Ly": 10, "dx":0.01, "D0":1.0, "algo":"RK4", "eps":0.03, "Na": 80
    }
    
    run(inputs=load_divless, plot_grid=plot_tables["1st"], cuts=cuts, params=params)