"""
Figure de synthèse (« overview ») pour le solveur de navigation perturbative.

Le tracé est piloté par une « table logique » : chaque cellule est une liste de
mots-clés (ex. ["T1"], ["cp", "F1"]) et la fonction plot_overview_figure dessine
le contenu correspondant. La structure principale (fonction unique + table
logique + point d'entrée __main__) est conservée.
"""

import sys
import os
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from post_process import simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from numba_kernel import trajectory_intersections

# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------


def plot_overview_figure(
    cn,
    gp,
    cuts,
    starting_points,
    saveStr="simple_dynamics_overview",
    Nradius=0,
    eps=0.01,
    Na=70,
    lw=1.5, ow=1.5,
    mlw=0.7, mow=0.5,
    Npath=10,
    dt=0.01,
    Ninter=200
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

    def _ishow(ax, data, cmap, min=None, max=None):
        im = ax.imshow(data.T, cmap=cmap, extent=EXTENT, origin="lower", vmax=max, vmin=min)
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

    # Ensemble de tous les mots-clés présents dans la table logique
    cells = [cell for row in gp for cell in row]
    quant = {key for cell in cells for key in cell}

    # ── Pré-calculs conditionnés par les mots-clés ───────────────────────────
    # eps maps / area : nécessitent compute_eps (et radial_eps_validity pour les
    # courbes eps*).
    if any(k in quant for k in ("eps1", "eps2", "eps3", "area")):
        constraining = cn.compute_eps()[1]
        if any(k in quant for k in ("eps1", "eps2", "eps3")):
            res1D = cn.radial_eps_validity()

    if "p1" in quant:
        t1 = [algo(1, (x0r, y0r), eps, 0.001, 100000, 0.1) for (x0r, y0r) in starting_points]

    if "p2" in quant:
        t2 = [algo(2, (x0r, y0r), eps, 0.001, 100000, 0.1) for (x0r, y0r) in starting_points]

    if "p0" in quant:
        t0 = [algo(0, (x0r, y0r), eps, 0.001, 100000, 0.1) for (x0r, y0r) in starting_points]

    if "pc" in quant: # plots multiples ordre starting from the same starting point.
        t = []
        x0r, y0r = starting_points[0]
        if "p0" in quant:
            t.append([t0[0], "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        else:
            t.append([algo(0, (x0r, y0r), eps, 0.001, 100000, 0.1), "#2980b9", r"$\mathcal{O}(\varepsilon^0)$"])
        if "p1" in quant:
            t.append([t1[0], "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        else:
            t.append([algo(1, (x0r, y0r), eps, 0.001, 100000, 0.1), "#e67e22", r"$\mathcal{O}(\varepsilon^1)$"])
        if "p2" in quant:
            t.append([t2[0], "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
        else:
            t.append([algo(2, (x0r, y0r), eps, 0.001, 100000, 0.1), "#27ae60", r"$\mathcal{O}(\varepsilon^2)$"])
        
    if "radius cp" in quant:
        errs, Ts = [], []
        radius   = np.linspace(1, 3 * cn.Ly / 8, Nradius)
        for R in radius:
            err, T, compared_path = cn.compare_paths(2, cn.make_theta_eq, 20, eps, Rmax=R) # pas a jour
            errs.append(np.mean(err))
            Ts.append(np.mean(T))

    # ── Boucle de tracé ──────────────────────────────────────────────────────
    count = 0   # indice courant dans `cuts` (incrémenté à chaque cellule "cut")
    for i in range(gshape[0]):
        for j in range(gshape[1]):
            ax = fig.add_subplot(gs[i, j])

            # Cellule vide : on masque l'axe et on passe à la suivante
            if not gp[i][j]:
                ax.axis("off")
                continue

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
                _style(ax,   r"$D_2(\vec{x})$", bottom=True)

            if "pot1" in gp[i][j]:
                _ishow(ax, cn.f1.potential(X, Y), "RdPu")
                _style(ax, r"$\phi_1(\vec{x})$",  bottom=True)

            if "pot2" in gp[i][j]:
                _ishow(ax, cn.f2.potential(X, Y), "RdPu")
                _style(ax, r"$\phi_2(\vec{x})$",  bottom=True)

            if "T" in gp[i][j]:
                Tmin = np.sqrt(2)*4.5*0.5 * 0.8
                Tmax = np.sqrt(2)*4.5*0.5 * 1.2
                
                _ishow(ax, tTot, "YlOrRd", min=Tmin, max=Tmax)
                _style(ax, rf"$\mathcal{{T}}^*$  ($\varepsilon={eps:.2g}$)",  bottom=True)

            if "Ttot-0" in gp[i][j]:
                _ishow(ax, tTot - T0, "YlOrRd")
                _style(ax, r"$\mathcal{T}^* - \mathcal{T}^*_0$", left=True, bottom=True)

            if "T1" in gp[i][j]:
                _ishow(ax, T1, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_1$", bottom=True)

            if "T2" in gp[i][j]:
                
                _ishow(ax, T2, "YlOrRd")
                _style(ax,   r"$\mathcal{T}^*_2$", bottom=True)

            # ── Contrôle optimal : ordre 0, ordres successifs, total ──────────
            if "c0" in gp[i][j]:
                _quiver(ax, c0x, c0y, "#34495e")
                _style(ax, r"$\hat{c}_0$", left=True, bottom=True)

            if "c1" in gp[i][j]:
                _quiver(ax, c1x, c1y, "#922b21")
                _style(ax, r"$\hat{c}_1$", left=True, bottom=True)

            if "c2" in gp[i][j]:
                _quiver(ax, c2x, c2y, "#1e8449")
                _style(ax, r"$\hat{c}_2$", left=True, bottom=True)

            if "c" in gp[i][j] or "ctot" in gp[i][j]:
                _quiver(ax, c2x, c2y, "#1e8449")
                _style(ax, r"$\hat{c}$ (total)", left=True, bottom=True)

            if "v" in gp[i][j]:
                _quiver(ax, vx, vy, "#1e8449")
                _style(ax, r"$\vec{v}$", left=True, bottom=True)

            if "p1" in gp[i][j]:
                for tx, ty in t1:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v_0+\varepsilon v_1$", bottom=True)

            if "p2" in gp[i][j]:
                for tx, ty in t2:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v$ to $\mathcal{O}(\varepsilon^2)$", bottom=True)

            if "p0" in gp[i][j]:
                for tx, ty in t0:
                    ax.plot(tx, ty, "-", lw=lw)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                _style(ax, r"$v_0$ only", bottom=True)

            if "pc" in gp[i][j]:
                for (tx, ty), col, lbl in t:
                    ax.plot(tx, ty, "-", color=col, lw=lw, label=lbl)
                    ax.plot(tx[0], ty[0], "ko", ms=ow)
                ax.legend(fontsize=6, loc="upper right", framealpha=0.7, edgecolor="none")
                _style(ax, r"Order comparison  $(x_0^{ref}, y_0^{ref})$", left=True, bottom=True)

            if "zp2" in gp[i][j]:

                R   = 0.7 * np.hypot(cn.Lx / 2, cn.Ly / 2)
                ths = np.linspace(0, np.pi / 2, Npath)
                sp  = [(-R * np.cos(th), R * np.sin(th)) for th in ths]
                paths = cn.trace_paths(vx, vy, sp, dt=dt, r_stop=0.01, steps=100000)

                for xs, ys in paths:
                    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                    if xs.size < 5:
                        continue

                    speed = np.hypot(np.gradient(xs), np.gradient(ys)) / dt
                    # on jette les extrémités : gradient peu fiable, et dernier pas tronqué près de r_stop
                    speed = speed[1:-1]; xs = xs[1:-1]; ys = ys[1:-1]

                    pts  = np.column_stack([xs, ys]).reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    seg_speed = 0.5 * (speed[:-1] + speed[1:])

                    # bornes ROBUSTES : les points trop rapides/lents n'influent plus sur l'échelle
                    lo, hi = np.percentile(seg_speed, [5, 95])
                    if hi <= lo:                                   # garde-fou : vitesse ~constante
                        lo, hi = seg_speed.min(), seg_speed.max() + 1e-12

                    norm = PowerNorm(gamma=8, vmin=lo, vmax=hi, clip=True)   # >hi -> rouge, sans étirer
                    lc = LineCollection(segs, cmap="YlOrRd", norm=norm)
                    lc.set_array(seg_speed)
                    lc.set_linewidth(mlw)
                    ax.add_collection(lc)

                ax.set_xlim(-cn.Lx/2, cn.Lx/10)
                ax.set_ylim(-cn.Ly/10, cn.Ly/2)

            if "cut" in gp[i][j]:  # coupe y=f(x) des fonctions de 'func_map'
                func_map = {
                    "T1": T1, "T2": T2, "T0": T0,
                    "pot1": cn.f1.potential(X, Y), "pot2": cn.f2.potential(X, Y),
                    "eT1": eps * T1, "eT2": eps**2 * T2,
                }
                cut_f, funcs, title = cuts[count]
                cut = _make_cut(cut_f, cn.X[:, 0])
                for func in funcs:
                    if func in func_map:
                        yc = RegularGridInterpolator((cn.X[:, 0], cn.Y[0, :]), func_map[func])(cut)

                        x_axis = cn.X[:, 0]                 # abscisse = coordonnée x de la coupe
                        i0     = 0              # x ≈ 0, début du tracé
                        #ifit   = np.searchsorted(x_axis, 1)  # premier indice avec x >= R ( ici R=1)

                        xcut = x_axis[i0:]
                        """xfit = x_axis[ifit:]
                        yfit = np.abs(yc[ifit:])

                        # loi de puissance a * x^b  (2 paramètres)
                        p, _ = curve_fit(lambda x, a, b: a * x**b, xfit, yfit, p0=[1.0, -3.0])
                        a, b = p
                        print(func, "-> a =", a, " b =", b)"""

                        ax.plot(xcut, np.abs(yc[i0:]), label=func)
                        #ax.plot(xfit, a * xfit**b, "--",
                                  #label=fr"{func} fit $\sim x^{{{b:.2f}}}$")
                        ax.grid()
                        ax.set_title(rf"cut at {title} of functions {' '.join(funcs)}")
                        ax.set_xlim(2, 4)
                        ax.legend()
                        ax.set_xlabel("Radius")
                count += 1

            if any(k in gp[i][j] for k in ("eps1", "eps2", "eps3")):  # eps validity (loglog)
                R = np.linspace(0, np.hypot(cn.Lx / 2, cn.Ly / 2), cn.Nr)
                if "eps1" in gp[i][j]:
                    ax.loglog(R[1:], res1D[0], label=r"$\epsilon (R)$")
                    ax.set_title("total comparison")

                if "eps2" in gp[i][j]:
                    ax.loglog(R[1:], res1D[1], label=r"$\epsilon (R)$")
                    ax.set_title("first and 0-th order comparison")

                if "eps3" in gp[i][j]:
                    ax.loglog(R[1:], res1D[2], label=r"$\epsilon (R)$")
                    ax.set_title("most constraining eps")

                ax.loglog(R[1:], np.full_like(R[1:], fill_value=eps), label=r"current $\epsilon$")
                ax.grid()
                ax.legend()
                ax.set_xlabel("radius")
                ax.set_ylabel(r"smallest $\epsilon$ inside circle")

            if "area" in gp[i][j]:  # pixels où le développement n'est PAS valide
                eps_field_xy = np.empty_like(constraining)
                eps_field_xy[cn.order] = constraining
                eps_field = eps_field_xy.reshape(cn.X.shape)
                mask = np.ma.masked_where(eps_field >= eps, eps_field <= eps)
                ax.imshow(mask.T, extent=EXTENT, origin="lower")

            if "radius cp" in gp[i][j]:  # erreur (num) et temps (exact) vs rayon
                ax.plot(radius, errs, "+", lw=0, zorder=1)   # numérique
                ax.plot(radius, Ts,   "o", lw=0, zorder=2)   # exact

            
            
            if "ze" in gp[i][j]:
                thetas = np.array([np.pi/4 - 1e-1, np.pi/4 + 1e-1])
                paths, sp, e   = cn.zermelo_paths(cn.make_theta_eq, eps, dt, thetas)

                for (tx, ty, th) in paths:
                    ax.plot(tx, ty, lw=mlw)

            if "cp" in gp[i][j]:  # caractéristiques exactes vs chemins numériques (axe R)
                
                thetas = np.array([np.pi/4 - 1e-1, np.pi/4 + 1e-1])
                err2, T_err2, ang_err2, compared_paths2   = cn.compare_paths(2, cn.make_theta_eq, Npath, eps, paths=True, thetas=thetas)
                err1, T_err1, ang_err1, compared_paths1 = cn.compare_paths(1, cn.make_theta_eq, Npath, eps,paths=True, thetas=thetas)
                apath    = compared_paths2[0][:,:,:1] # on enleve theta de zermelo paths
                numpath2 = compared_paths2[1] #order 2
                numpath1 = compared_paths1[1] #order 1

                info = [[apath, "viridis", None],[numpath2, "YlOrRd", T_err2], [numpath1, "Blues", T_err1]]
                for (paths, style,err) in info:
                    for k, (xs, ys) in enumerate(paths):

                        # crée une colormap en fonction de la vitesse en chaque point de la trajectoire
                        speed = np.hypot(np.gradient(xs), np.gradient(ys)) / dt
                        speed = speed[1:-1]; xs = xs[1:-1]; ys = ys[1:-1]

                        pts  = np.column_stack([xs, ys]).reshape(-1, 1, 2) # segments de trajectoires
                        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

                        seg_speed = 0.5 * (speed[:-1] + speed[1:])
                        lo, hi = np.percentile(seg_speed, [5, 95])
                        if hi <= lo:                                   # garde-fou : vitesse ~constante
                            lo, hi = seg_speed.min(), seg_speed.max() + 1e-12

                        norm = PowerNorm(gamma=4, vmin=lo, vmax=hi, clip=True)    
                        lc = LineCollection(segs, cmap=style, norm=norm)
                        lc.set_array(seg_speed)
                        lc.set_linewidth(mlw)
                        ax.add_collection(lc)
                        if err:
                            if err[k] > 0: # si erreur positive ( dans ce cas différence temporelle)
                                lc.set_alpha(0) # transparent
                            else:
                                lc.set_alpha(1)

            if "interp" in gp[i][j]:
                thetas = np.random.rand(100)*2*np.pi
                paths, sp, e   = cn.zermelo_paths(cn.make_theta_eq, eps, dt, thetas)
                trajs = []
                for (tx, ty, th) in paths:
                    ax.plot(tx, ty, lw=mlw)
                    trajs.append(np.array(list(zip(tx, ty))))                    

                xy_inter = trajectory_intersections(trajs, 2)
                print(len(xy_inter))
                for (i,j, (x,y)) in xy_inter:
                    ax.scatter(x, y, marker="o", s=10, c="red", zorder=5)

                    

    fig.suptitle(
        rf"Perturbative control navigation — $\varepsilon={eps:.2g}$"
        rf"   grid ${cn.Nx}\times{cn.Ny}$   domain $[\!-\!{cn.Lx/2:.0f},{cn.Lx/2:.0f}]^2$",
        fontsize=11, fontweight="bold", y=0.975,
    )

    # Sauvegarde dans Plots/"saveStr"
    my_path  = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(my_path, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, saveStr + f"_{eps}.png")
    print(f"Saving -> {out_path}")
    fig.savefig(out_path, dpi=900, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    # Fichier de données JSON (argv[1]) ; eps éventuel en argv[2]
    if len(sys.argv) > 1:
        file = f"datas/{sys.argv[1]}.json"  
    else:
        file = "datas/Conservative_peaks.json"

    cuts = [  # [fonction de coupe, [fonctions à tracer], titre] ; cf. 'func_map'
        [lambda x: np.sqrt(4.5**2/2 - x**2) if x**2<4.5**2/2 else 0,    ["T2"], "y=theta"],
        [lambda x: -x,    ["T2"], "y=-x"],
    ]

    plot_tables = {  # grilles de tracé ; chaque cellule est une liste de mots-clés
        # paths -> pi : chemin somme(ordre <= i) ;
        #          pc : tous les ordres depuis le même point de départ (1er de la liste) ;
        #          zp2: 'Npath' chemins différents dans le coin
        "complete": [
            [["F1R"],   ["F1T"],     ["F2R"],   ["F2T"]],
            [["F1"],    ["pot1"],    ["D1"],    ["D2"]],
            [["T"],     ["Ttot-0"],  ["T1"],    ["T2"]],
            [["ctot"],  ["c1"],      ["c2"],    ["v"]],
            [["pc"],    ["p0"],      ["p1"],    ["p2"]],
            [["cut"],   ["cut"],     ["eps1"],  ["eps2"]],
        ],
        "1st": [
            [["F1R"],        ["F1T"],     ["F1"],         ["pot1"]],
            [["T"],          ["Ttot-0"],  ["T1"],         ["T2"]],
            [["cp", "F1"],   ["p1", "T"], ["p2", "T"],   ["zp2", "F1"]],
        ],
        "single": [[["interp", "pot1"]]],
        "big grid": [
            [["F1R"],   ["F1T"],   ["F1"],         ["pot1"]],
            [["T1"],    ["T2"],    ["T", "area"],  []],
            [["cut"],   ["cut"],        ["cut"],        ["zp2", "F1"]],
            [["eps1"],  ["eps2"],  [""],  []],
        ],
        "cuts":[[["cut"],["cut"],["cut"]],
                [[], [],[],],
                [[], [],[],],]
    }

    params = {  # paramètres de calcul / tracé
        # eps : amplitude perturbative
        # Na  : nombre de flèches ; lw, ow : épaisseur/marqueur d'un chemin simple
        # mlw, mow : idem pour les chemins multiples ; Npath : nb de chemins 'zp2'
        # Nradius : nb de rayons pour 'radius cp' (>0 requis si 'radius cp' utilisé)
        "eps": 0.04, "Na": 80, "lw": 0.7, "ow": 0.7,
        "mlw": 0.7, "mow": 2, "Npath": 12, "Nradius": 10,
        "dt":0.001, "Ninter":200
    }
    if len(sys.argv) > 2:
        params["eps"] = float(sys.argv[2])

    cn = simulation(file=file)

    # Points de départ des chemins (utilisés par p0/p1/p2/pc), placés dans le coin (-x, +y)
    R0 = 0.7 * np.hypot(cn.Lx / 2, cn.Ly / 2)
    starting_points = [(-R0 * np.cos(a), R0 * np.sin(a))
                       for a in np.linspace(0.15, np.pi / 2 - 0.15, 5)]

    plot_overview_figure(
        cn,
        gp=plot_tables["single"],
        cuts=cuts,
        starting_points=starting_points,
        saveStr=f"div_less",
        **params   
    )