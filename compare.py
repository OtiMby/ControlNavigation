import Analytical_Methods.control as am
import Numerical_methods.control_nav as nm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsp
import numpy as np

if __name__ == "__main__":
    N = 120
    D0 = 1.0

    ### Analytical Methods ###
    sm = am.FullAsymmetricMobility()
    sf = am.FullAsymmetricForce()
    st = am.times()
    cpAM = am.simple_control_problem(Lx=10.0, Ly=10.0, Nx=N, Ny=N, mob=sm, force=sf, time=st)

    ### Numerical Methods ###
    f1, f2, D1, D2, saveStr = nm.load_compare(10.0,10.0,N)
    cpNM = nm.control_nav(Lx=10.0, Ly=10.0, Nx=N, Ny=N, D0=D0, D1=D1, D2=D2, f1=f1, f2=f2, epsilon=0.2)
    print('initialized control_nav instance, now computing T1...')
    cpNM.compute_T1()
    cpNM.compute_dT1()
    print(np.max(cpNM.T1 - cpAM.time.T1()))
    print('done computing T1, now computing T2...')
    cpNM.compute_T2()
    print('done computing T1 and T2')
    cpNM.compute_dT2()

    c0xn, c0yn = cpNM.compute_control_force(0)
    c1xn, c1yn = cpNM.compute_control_force(1)
    c2xn, c2yn = cpNM.compute_control_force(2)
    print(np.max(np.hypot(c2xn, c2yn)))

    i0 = np.argmin(np.abs(cpNM.X[0, :]))
    j0 = np.argmin(np.abs(cpNM.Y[:, 0]))

    c0xm, c0ym = cpAM.control_0th_order()
    c1xm, c1ym = cpAM.control_1st_order()
    c2xm, c2ym = cpAM.control_2nd_order()

    c1xn[j0, i0] = 0;  c1yn[j0, i0] = 0
    c2xn[j0, i0] = 0;  c2yn[j0, i0] = 0
    c1xm[j0, i0] = 0;  c1ym[j0, i0] = 0
    c2xm[j0, i0] = 0;  c2ym[j0, i0] = 0

    # -------------------------------------------------

    X = cpNM.X
    Y = cpNM.Y


    ## fig spec ##
    control_figure = plt.figure(figsize=(11,11))
    gs = gridsp.GridSpec(nrows=6, ncols=4, left=0.02, right=0.98, bottom=0.05,top=0.95,  hspace=0.15, wspace=-0.0)
    fR1_plot  = plt.subplot(gs[0,0])
    fT1_plot  = plt.subplot(gs[0,1])
    fR2_plot  = plt.subplot(gs[0,2])
    fT2_plot  = plt.subplot(gs[0,3])
    fTot_plot = plt.subplot(gs[1,0])

    m1_plot = plt.subplot(gs[1,2]) # plot mobility D1
    m2_plot = plt.subplot(gs[1,3]) # plot mobility D2

    tTot_plot = plt.subplot(gs[2,0]) 
    t0_plot = plt.subplot(gs[2,1])
    t1_plot = plt.subplot(gs[2,2])
    t2_plot = plt.subplot(gs[2,3])

    c0_plot = plt.subplot(gs[3,1])
    c1_plot = plt.subplot(gs[3,2])
    c2_plot = plt.subplot(gs[3,3])
    cTot_plot = plt.subplot(gs[3,0])

    text_plot = plt.subplot(gs[1,1])

    ax_list = [fR1_plot,fT1_plot,fR2_plot,fT2_plot,\
               fTot_plot,text_plot,m1_plot,m2_plot,\
               tTot_plot,t0_plot,t1_plot,t2_plot,\
               cTot_plot,c0_plot,c1_plot,c2_plot,\
              ]

    ϵ = cpNM.epsilon

    fTot_plot.set_title(r"$\vec{F}(\vec{x})$")
    fR1_plot.set_title(r"$f_{1R}$")
    fR2_plot.set_title(r"$f_{2R}$")
    fT1_plot.set_title(r"$f_{1\theta}$")
    fT2_plot.set_title(r"$f_{2\theta}$")

    m1_plot.set_title(r"$D_1(\vec{x})$")
    m2_plot.set_title(r"$D_2(\vec{x})$")

    tTot_plot.set_title(r"$\mathcal{T}^*(\vec{x})$ to $\mathcal{O}(\varepsilon^2)$ with $\varepsilon=" + r"{:.1E}$".format(ϵ))
    t0_plot.set_title(r"$\mathcal{T}^*_0(\vec{x})$")
    t1_plot.set_title(r"$\mathcal{T}^*_1(\vec{x})$")
    t2_plot.set_title(r"$\mathcal{T}^*_2(\vec{x})$")

    c0_plot.set_title(r"$\hat{c}_0(\vec{x})$")
    c1_plot.set_title(r"$\hat{c}_1(\vec{x})$")
    c2_plot.set_title(r"$\hat{c}_2(\vec{x})$")
    cTot_plot.set_title(r"$\hat{c}(\vec{x})$ to $\mathcal{O}(\varepsilon^2)$")

    ## plot forces ##
    
    eRx, eRy = cpNM.eRa(X, Y)
    eTx, eTy = cpNM.eTh(X, Y)

    # forces ordre 1
    if cpNM.f1.cartesian:
        f1R_xn = cpNM.f1.fx(X, Y) 
        f1R_yn = cpNM.f1.fy(X, Y)
        f1T_xn = cpNM.f1.fx(X, Y) 
        f1T_yn = -cpNM.f1.fy(X, Y)

    else:
        f1R = cpNM.f1._f_R(X, Y)
        f1T = cpNM.f1._f_theta(X, Y)
        f1R_xn, f1R_yn = f1R * eRx, f1R * eRy
        f1T_xn, f1T_yn = f1T * eTx, f1T * eTy
        f1_xn = f1R_xn + f1T_xn 
        f1_yn = f1R_yn + f1T_yn
    
    # forces ordre 2
    if cpNM.f2.cartesian:
        f2R_xn, f2R_yn = cpNM.f2.fx(X, Y), cpNM.f2.fy(X, Y)
        f2T_xn, f2T_yn = cpNM.f2.fx(X, Y), -cpNM.f2.fy(X, Y)
    else:
        f2R = cpNM.f2._f_R(X, Y)
        f2T = cpNM.f2._f_theta(X, Y)
        f2R_xn, f2R_yn = f2R * eRx, f2R * eRy
        f2T_xn, f2T_yn = f2T * eTx, f2T * eTy
        f2_xn = f2R_xn + f2T_xn
        f2_yn = f2R_yn + f2T_yn


    eR_x, eR_y = cpAM.eRa()
    eT_x, eT_y = cpAM.eTh()

    #f_1
    f_Rx,f_Tx = eR_x *cpAM.force.f_R(cpAM.Ra,cpAM.Th), eT_x*cpAM.force.f_T(cpAM.Ra,cpAM.Th)
    f1_x = f_Rx + f_Tx 
    f_Ry,f_Ty = eR_y *cpAM.force.f_R(cpAM.Ra,cpAM.Th), eT_y*cpAM.force.f_T(cpAM.Ra,cpAM.Th)
    f1_y = f_Ry + f_Ty

    #f_2
    f_2Rx,f_2Tx = eR_x *cpAM.force.f_R2(cpAM.Ra,cpAM.Th), eT_x*cpAM.force.f_T2(cpAM.Ra,cpAM.Th)
    f_2x = f_2Rx + f_2Tx 
    f_2Ry,f_2Ty = eR_y *cpAM.force.f_R2(cpAM.Ra,cpAM.Th), eT_y*cpAM.force.f_T2(cpAM.Ra,cpAM.Th)
    f_2y = f_2Ry + f_2Ty

    diff_f1x, diff_f1y = f1_x - f1_xn, f1_y - f1_yn
    diff_f2x, diff_f2y = f_2x - f2_xn, f_2y - f2_yn

    print(np.max(np.sqrt(diff_f1x**2 + diff_f1y**2)), np.max(np.sqrt(diff_f2x**2 + diff_f2y**2)))

    _nfR1 = np.where(np.hypot(f_Rx, f_Ry) > 0, np.hypot(f_Rx, f_Ry), np.nan)
    _nfT1 = np.where(np.hypot(f_Tx, f_Ty) > 0, np.hypot(f_Tx, f_Ty), np.nan)
    _nfR2 = np.where(np.hypot(f_2Rx, f_2Ry) > 0, np.hypot(f_2Rx, f_2Ry), np.nan)
    _nfT2 = np.where(np.hypot(f_2Tx, f_2Ty) > 0, np.hypot(f_2Tx, f_2Ty), np.nan)
    _nftot = np.where(np.hypot(f1_x + f_2x, f1_y + f_2y) > 0, np.hypot(f1_x + f_2x, f1_y + f_2y), np.nan)
    fR1_plot.quiver(X, Y, (f1R_xn - f_Rx) / _nfR1, (f1R_yn - f_Ry) / _nfR1, color="blue")
    fT1_plot.quiver(X, Y, (f1T_xn - f_Tx) / _nfT1, (f1T_yn - f_Ty) / _nfT1, color="blue")
    fR2_plot.quiver(X, Y, (f2R_xn - f_2Rx) / _nfR2, (f2R_yn - f_2Ry) / _nfR2, color="red")
    fT2_plot.quiver(X, Y, (f2T_xn - f_2Tx) / _nfT2, (f2T_yn - f_2Ty) / _nfT2, color="red")
    fTot_plot.quiver(X, Y, (diff_f1x + diff_f2x) / _nftot, (diff_f1y + diff_f2y) / _nftot, color="green")

    ## plot mobilities ##
    R = np.sqrt(X**2 + Y**2)
    Th = np.arctan2(Y, X) % (2*np.pi)
    _D1ref = cpAM.mob.D1(R, Th)
    _D2ref = cpAM.mob.D2(R, Th)
    D1_diff = (cpNM.D1(X, Y) - _D1ref) / np.where(_D1ref != 0, _D1ref, np.nan)
    D2_diff = (cpNM.D2(X, Y) - _D2ref) / np.where(_D2ref != 0, _D2ref, np.nan)

    m1_plot.imshow(D1_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")
    m2_plot.imshow(D2_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")

    ## plot times ##
    _T0ref = cpAM.time.T0()
    _T1ref = cpAM.time.T1()
    _T2ref = cpAM.time.T2()
    T0_abs = np.abs(cpNM.T0 - _T0ref)
    T0_av = np.abs(cpNM.T0 + _T0ref)

    T1_abs = np.abs(cpNM.T1 - _T1ref)
    T1_av = np.abs(cpNM.T1 + _T1ref)

    T2_abs = np.abs(cpNM.T2 - _T2ref)
    T2_av = np.abs(cpNM.T2 + _T2ref)

    Ttot_abs = T0_abs + ϵ * T1_abs + ϵ**2 * T2_abs
    Ttot_av =  T0_av + ϵ * T1_av + ϵ**2 * T2_av

    
    T0_diff = T0_abs/T0_av
    T1_diff = T1_abs/T0_av
    T2_diff = T2_abs/T0_av
    Ttot_diff = Ttot_abs/Ttot_av

    



    
    tTot_plot.imshow(Ttot_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")
    t0_plot.imshow(T0_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")
    t1_plot.imshow(T1_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")
    control_figure.colorbar(t1_plot.get_images()[0], ax=t1_plot, fraction=0.046, pad=0.04, location='left')
    t2_plot.imshow(T2_diff, extent=[-cpNM.Lx/2,cpNM.Lx/2,-cpNM.Ly/2,cpNM.Ly/2], origin="lower", cmap="viridis")
    control_figure.colorbar(t2_plot.get_images()[0], ax=t2_plot, fraction=0.046, pad=0.04, location='left')


    ## plot controls ##
    nc0 = np.hypot(c0xm, c0ym) + np.hypot(c0xn, c0yn)
    nc1 = np.hypot(c1xm, c1ym) + np.hypot(c1xn, c1yn)
    nc2 = np.hypot(c2xm, c2ym) + np.hypot(c2xn, c2yn)
    _nc0 = np.where(nc0 > 0.01 * np.nanmax(nc0), nc0, np.nan)
    _nc1 = np.where(nc1 > 0.01 * np.nanmax(nc1), nc1, np.nan)
    _nc2 = np.where(nc2 > 0.01 * np.nanmax(nc2), nc2, np.nan)
    _cTot_ref_x = c0xm + c0xn + ϵ * (c1xm + c1xn) + ϵ**2 * (c2xm + c2xn)
    _cTot_ref_y = c0ym + c0yn + ϵ * (c1ym + c1yn) + ϵ**2 * (c2ym + c2yn)
    _ncTot = np.where(np.hypot(_cTot_ref_x, _cTot_ref_y) > 0, np.hypot(_cTot_ref_x, _cTot_ref_y), np.nan)
    c0_diff_x = (c0xm - c0xn) / _nc0
    c0_diff_y = (c0ym - c0yn) / _nc0
    c1_diff_x = (c1xm - c1xn) / _nc1
    c1_diff_y = (c1ym - c1yn) / _nc1
    c2_diff_x = (c2xm - c2xn) / _nc2
    c2_diff_y = (c2ym - c2yn) / _nc2
    cTot_diff_x = (c0xm - c0xn + ϵ * (c1xm - c1xn) + ϵ**2 * (c2xm - c2xn)) / _ncTot
    cTot_diff_y = (c0ym - c0yn + ϵ * (c1ym - c1yn) + ϵ**2 * (c2ym - c2yn)) / _ncTot


    print(f"max rel error T2   : {np.nanmax(T2_diff)   * 100:.4f} % max T2 = {np.max(np.abs(T2_abs))}")
    print(f"max rel error T1   : {np.nanmax(T1_diff)   * 100:.4f} %")
    print(f"max rel error Ttot : {np.nanmax(Ttot_diff) * 100:.4f} %")
    print(f"max rel error |c1| : {np.nanmax(np.hypot(c1_diff_x, c1_diff_y)) * 100:.4f} %  (abs: {np.nanmax(np.hypot(c1_diff_x, c1_diff_y) * _nc1):.4e})")
    print(f"max rel error |c2| : {np.nanmax(np.hypot(c2_diff_x, c2_diff_y)) * 100:.4f} %  (abs: {np.nanmax(np.hypot(c2_diff_x, c2_diff_y) * _nc2):.4e})")
    print(f"max rel error |ctot|: {np.nanmax(np.hypot(cTot_diff_x, cTot_diff_y)) * 100:.4f} %")


    c0_plot.quiver(X, Y, c0_diff_x, c0_diff_y, color="blue")
    c1_plot.quiver(X, Y, c1_diff_x, c1_diff_y, color="orange")
    c2_plot.quiver(X, Y, c2_diff_x, c2_diff_y, color="red")
    cTot_plot.quiver(X, Y, cTot_diff_x, cTot_diff_y, color="green")

    ## text ##
    text_plot.text(0.5, 0.5, "Blue: 0th order\nOrange: 1st order\nRed: 2nd order\nGreen: Total", ha="center", va="center", fontsize=12)
    text_plot.axis("off")

    for ax in ax_list:
        ax.set_aspect(1)
        ax.set_xlim(-cpNM.Lx/2,cpNM.Lx/2)
        ax.set_ylim(-cpNM.Ly/2,cpNM.Ly/2)
        ax.set_xticks(np.linspace(-cpNM.Lx/2,cpNM.Lx/2,5))
        ax.set_yticks(np.linspace(-cpNM.Ly/2,cpNM.Ly/2,5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if ax!=text_plot:
            ax.plot([0],[0],"D",ms=1,color="black")

    for p in [cTot_plot, c0_plot, c1_plot, c2_plot]:
        p.set_xticklabels(np.linspace(-cpNM.Lx/2,cpNM.Lx/2,5),fontsize=8)

    for p in [fTot_plot, cTot_plot, m1_plot, fR1_plot]:
        p.set_yticklabels(np.linspace(-cpNM.Ly/2,cpNM.Ly/2,5),fontsize=8)
    
    plt.savefig("comparison_figure.png", dpi=300)
    plt.show()





