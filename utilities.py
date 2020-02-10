## @package utilities
## Defines various functions for smoothing, calculating
## fourier transforms, SVD, and so on.
from plot_attributes import *
from map_probes import \
    sp_name_dict, dead_probes, \
    imp_phis8, imp_phis32, midphi
from scipy.stats import linregress
from map_probes import \
    dead_probes
from scipy import trapz
from scipy.optimize import curve_fit

## Calculate the current centroid
# @param dict A psi-tet dictionary
# @returns rAvg Average radial centroid position
# @returns rStd radial centroid standard dev.
# @returns zAvg Average axial centroid position
# @returns zStd axial centroid standard dev.
def calcCentroid(psi_dict):
    time = psi_dict['sp_time']

    tol = 1e-1
    sp_names = psi_dict['sp_names']
    sp_Bpol = psi_dict['sp_Bpol']
    sp_Btor = psi_dict['sp_Btor']

    b_pol000 = []; rb_pol000 = []; zb_pol000 = [];
    b_pol045 = []; rb_pol045 = []; zb_pol045 = [];
    b_pol180 = []; rb_pol180 = []; zb_pol180 = [];
    b_pol225 = []; rb_pol225 = []; zb_pol225 = [];
    q = 1
    for i in range(np.shape(sp_names)[0]):
        name = str(sp_names[i])
        if (name in dead_probes) or name[5] == 'T':
            continue
        pos = sp_name_dict[name]
        if abs(pos[2]-0) < tol:
            b_pol000.append(sp_Bpol[q])
            rb_pol000.append(pos[0]*sp_Bpol[q])
            zb_pol000.append(pos[1]*sp_Bpol[q])
        if abs(pos[2]-45*pi/180.0) < tol:
            b_pol045.append(sp_Bpol[q])
            rb_pol045.append(pos[0]*sp_Bpol[q])
            zb_pol045.append(pos[1]*sp_Bpol[q])
        if abs(pos[2]-180*pi/180.0) < tol:
            b_pol180.append(sp_Bpol[q])
            rb_pol180.append(pos[0]*sp_Bpol[q])
            zb_pol180.append(pos[1]*sp_Bpol[q])
        if abs(pos[2]-225*pi/180.0) < tol:
            b_pol225.append(sp_Bpol[q])
            rb_pol225.append(pos[0]*sp_Bpol[q])
            zb_pol225.append(pos[1]*sp_Bpol[q])
        q = q + 1

    r_ma000 = np.sum(rb_pol000,0)/np.sum(b_pol000,0)
    r_ma045 = np.sum(rb_pol045,0)/np.sum(b_pol045,0)
    r_ma180 = np.sum(rb_pol180,0)/np.sum(b_pol180,0)
    r_ma225 = np.sum(rb_pol225,0)/np.sum(b_pol225,0)

    z_ma000 = np.sum(zb_pol000,0)/np.sum(b_pol000,0)
    z_ma045 = np.sum(zb_pol045,0)/np.sum(b_pol045,0)
    z_ma180 = np.sum(zb_pol180,0)/np.sum(b_pol180,0)
    z_ma225 = np.sum(zb_pol225,0)/np.sum(b_pol225,0)

    rAvg = np.ravel(np.mean([r_ma000, r_ma045, r_ma180, r_ma225],0))
    rStd = np.ravel(np.std([r_ma000, r_ma045, r_ma180, r_ma225],0))
    zAvg = np.ravel(np.mean([z_ma000, z_ma045, z_ma180, z_ma225],0))
    zStd = np.ravel(np.std([z_ma000, z_ma045, z_ma180, z_ma225],0))
    print(np.shape(rAvg),np.shape(rStd),np.shape(zAvg),np.shape(zStd))
    return rAvg,rStd,zAvg,zStd

## Python equivalent of the sihi_smooth function found in older
## matlab scripts. This does a boxcar average.
## Code simplified to only work for real-valued signals.
# @param y Signal in time
# @param time Time base associated with y
# @param f_1 Injector Frequency with which to apply the smoothing
# @returns x Smoothed signal in time
def sihi_smooth(y, time, f_1):
    injCyc = 1.0 / (1000.0 * f_1)
    Navg = 100
    Navg2 = int(Navg / 2.0)
    # make it 100 time points per injector cycle
    tint = np.linspace(time[0], time[len(time) - 1],
        int((time[len(time) - 1]-time[0]) / (injCyc / Navg)))
    yint = np.interp(tint, time, y)
    xint = np.zeros(len(tint))
    xint[0:Navg2] = np.mean(yint[0:Navg])

    for it in range(Navg2,len(tint) - Navg2):
        xint[it] = xint[it - 1] + \
            (yint[it + Navg2] - yint[it - Navg2]) / Navg

    #xint[0:Navg2] = xint[Navg2]
    xint[0:Navg2] = yint[0:Navg2]
    xint[len(tint) - Navg2:len(tint)] = \
        xint[len(tint) - Navg2 - 1]
    #xint[len(tint) - Navg2:len(tint)] = \
    #    yint[len(tint) - Navg2:len(tint)]
    x = np.interp(time, tint, xint)
    x[len(x) - 1] = x[len(x) - 2]
    x[np.asarray(np.isnan(x)).nonzero()] = 0
    return x

## Performs a SVD of the data in a psi-tet dictionary.
## Has dmd_flags to control which data is put into the matrix
## for the SVD.
# @param psi_dict A psi-tet dictionary
def SVD(psi_dict):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    data = np.vstack((psi_dict['curr01'],psi_dict['curr02']))
    if psi_dict['is_HITSI3'] == True:
        data = np.vstack((data,psi_dict['curr03']))
    #data = np.vstack((data,psi_dict['flux01']))
    #data = np.vstack((data,psi_dict['flux02']))
    data = np.vstack((data,psi_dict['sp_Bpol']))
    data = np.vstack((data,psi_dict['sp_Btor']))
    getshape = np.shape(data)[0]
    if psi_dict['use_IMP']:
        psi_dict['imp_Bpol'] = np.nan_to_num(psi_dict['imp_Bpol'])
        psi_dict['imp_Btor'] = np.nan_to_num(psi_dict['imp_Btor'])
        psi_dict['imp_Brad'] = np.nan_to_num(psi_dict['imp_Brad'])
        data = np.vstack((data,psi_dict['imp_Bpol']))
        shape1 = np.shape(psi_dict['imp_Bpol'])[0]
        shape2 = np.shape(psi_dict['imp_Btor'])[0]
        shape3 = np.shape(psi_dict['imp_Brad'])[0]
        imp_pol_indices = np.linspace(0,shape1,shape1, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Btor']))
        imp_tor_indices = np.linspace(shape1,shape2+shape1,shape2, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Brad']))
        imp_rad_indices = np.linspace(shape1+shape2, \
            shape3+shape2+shape1,shape3, \
            dtype = 'int')

    # correct injector currents
    if psi_dict['is_HITSI3'] == True:
        data[0:3,:] = data[0:3,:]*mu0
    else:
        data[0:2,:] = data[0:2,:]*mu0
    data = data[:,t0:tf]
    data_sub = data
    u,s,v = np.linalg.svd(data_sub)
    v = np.conj(np.transpose(v))
    psi_dict['SVD_data'] = data_sub
    psi_dict['SP_data'] = data
    psi_dict['U'] = u
    psi_dict['S'] = s
    psi_dict['V'] = v

## Plots the toroidal current for a shot
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_itor(psi_dict,j,color,filename):
    itor = psi_dict['tcurr']/1000.0
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['time']*1000.0
    plt.figure(75000,figsize=(figx, figy))
    plt.plot(time,abs(sihi_smooth(itor, \
        psi_dict['sp_time'],psi_dict['f_1'])), \
        color=color,linewidth=lw,label=filename)
    plt.grid(True)
    plt.xlim(0,0.6)
    ax = plt.gca()
    ax.set_yticks([0,15,30])
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([0,20,40,60,80,100])
    ax.set_xticks([0,0.3,0.6])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','15','30'])
        if 'low' in filename or 'medium' in filename or 'high' in filename:
            ax.set_yticklabels(['0','20','40','60','80','100'])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_current.png')
    plt.savefig(out_dir+'toroidal_current.eps')
    plt.savefig(out_dir+'toroidal_current.pdf')
    plt.savefig(out_dir+'toroidal_current.svg')

## Plots the BD chronos for a shot
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_chronos(psi_dict,j,color,filename):
    Vh = np.transpose(np.conj(psi_dict['V']))
    S = psi_dict['S']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(85000,figsize=(figx, figy))
    for i in range(3):
        plt.plot(time,S[i]*Vh[i,:]*1e4/S[0],color=color,linewidth=lw, alpha=alphas[i], \
            label='BOD mode '+str(i+1))
    if j == 1:
        plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper left')
    plt.grid(True)
    plt.ylim(-1500,1500)
    ax = plt.gca()
    ax.set_yticks([-1000,-500,0,500,1000])
    if j == 1 or j == 3:
        ax.set_yticklabels(['-1000','-500','0','500','1000'])
    else:
        ax.set_yticklabels([])
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD_chronos.png')
    plt.savefig(out_dir+'BD_chronos.eps')
    plt.savefig(out_dir+'BD_chronos.pdf')
    plt.savefig(out_dir+'BD_chronos.svg')
    plt.figure(95000,figsize=(figx, figy))
    plt.semilogy(range(1,len(S)+1),S/S[0],color=color,marker='o', \
        markersize=ms,markeredgecolor='k')
    plt.semilogy(range(1,len(S)+1),S/S[0],color=color)
    plt.grid(True)
    plt.ylim([1e-2,2e0])
    ax = plt.gca()
    ax.set_yticks([1e-2,1e-1,1e0])
    if j == 1 or j == 3:
        ax.set_yticklabels([r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    else:
        ax.set_yticklabels([])
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD.png')
    plt.savefig(out_dir+'BD.eps')
    plt.savefig(out_dir+'BD.pdf')
    plt.savefig(out_dir+'BD.svg')

## Plots the average electron and ion temperatures for a discharge
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_temperatures(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    # Plot Electron Temperature
    plt.figure(105000,figsize=(figx, figy))
    if 'te' in psi_dict.keys() and psi_dict['te'][10] != psi_dict['ti'][10]:
        plt.plot(time,psi_dict['te'][t0:tf],color=color, \
            linewidth=lw, alpha=1.0,label=filename+r' $T_e$')
    else:
        plt.plot(time,psi_dict['ti'][t0:tf],color=color, \
            linewidth=lw, alpha=1.0,label=filename+r' $T_e$')
    plt.grid(True)
    ax = plt.gca()
    ax.set_yticks([0,5,10,15])
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([0,20,40,60])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','5','10','15'])
        if 'low' in filename or 'medium' in filename or 'high' in filename:
            ax.set_yticklabels(['0','20','40','60'])
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0,0.3,0.6])
    plt.xlim(0,0.6)
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'Te.png')
    plt.savefig(out_dir+'Te.eps')
    plt.savefig(out_dir+'Te.pdf')
    plt.savefig(out_dir+'Te.svg')

    # Plot Ion Temperature
    plt.figure(105005,figsize=(figx, figy))
    plt.plot(time,psi_dict['ti'][t0:tf],color=color, \
        linewidth=lw, alpha=1.0,label=filename+r' $T$')
    plt.grid(True)
    ax = plt.gca()
    ax.set_yticks([0,10,20,30])
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([0,20,40,60])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','10','20','30'])
        if 'low' in filename or 'medium' in filename or 'high' in filename:
            ax.set_yticklabels(['0','20','40','60'])
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0,0.3,0.6])
    plt.xlim(0,0.6)
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'Ti.png')
    plt.savefig(out_dir+'Ti.eps')
    plt.savefig(out_dir+'Ti.pdf')
    plt.savefig(out_dir+'Ti.svg')

## Plots the line-averaged FIR density
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_nFIR(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(115000,figsize=(figx, figy))
    plt.plot(time,
        sihi_smooth(psi_dict['inter_n'][t0:tf], \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1'])/1e19, \
        color=color, \
        linewidth=lw, alpha=1.0)
    plt.grid(True)
    plt.ylim(0.0,1.3)
    #plt.ylim(-0.5,0.5)
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        plt.ylim(0,8.0)
    ax = plt.gca()
    ax.set_yticks([0.0,0.6,1.2])
    #ax.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([0,1,2,4,6,8])
    plt.xlim(0,0.6)
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','0.6','1.2'])
        #ax.set_yticklabels(['-0.4','-0.2','0','0.2','0.4'])
        if 'low' in filename or 'medium' in filename or 'high' in filename:
            ax.set_yticklabels(['0','1.0','2.0','4.0','6.0','8.0'])
    else:
        ax.set_yticklabels([])
    #ax.set_yticklabels(['-0.4','-0.2','0','0.2','0.4'])
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticklabels(['0','1.0','2.0','4.0','6.0','8.0'])
    ax.set_xticks([0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'nFIR.png')
    plt.savefig(out_dir+'nFIR.eps')
    plt.savefig(out_dir+'nFIR.pdf')
    plt.savefig(out_dir+'nFIR.svg')

## Plots the volume-averaged density
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_navg(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    # Plot Chord-Averaged Density from Interferometry
    plt.figure(155000,figsize=(figx, figy))
    plt.plot(time,
        psi_dict['ne'][t0:tf], \
        color=color, \
        linewidth=lw, alpha=1.0,label=filename)
    plt.grid(True)
    plt.ylim(5e18,7e19)
    ax = plt.gca()
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([1e19,2e19,3e19,4e19,5e19,7e19])
    else:
        ax.set_yticks([0.4e19,0.6e19,1.2e19])
    plt.xlim(0,0.6)
    if j == 1 or j == 3:
        if 'low' in filename or 'medium' in filename or 'high' in filename:
            ax.set_yticklabels(['1.0','2.0','3.0','4.0','5.0','6.0'])
        else:
            ax.set_yticklabels(['0.4','0.6','1.2'])
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'navg.png')
    plt.savefig(out_dir+'navg.eps')
    plt.savefig(out_dir+'navg.pdf')
    plt.savefig(out_dir+'navg.svg')

## Plots the centroid avg and std
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param color Color of the line
# @param filename Name of the file
def plot_centroid(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    r,sigr,z,sigz = calcCentroid(psi_dict)
    r = r*100
    r = sihi_smooth(r,psi_dict['sp_time'],psi_dict['f_1'])
    sigr = sigr*100
    sigr = sihi_smooth(sigr,psi_dict['sp_time'],psi_dict['f_1'])
    z = z*100
    z = sihi_smooth(z,psi_dict['sp_time'],psi_dict['f_1'])
    sigz = sigz*100
    sigz = sihi_smooth(sigz,psi_dict['sp_time'],psi_dict['f_1'])
    # Plot R
    plt.figure(125000,figsize=(figx, figy))
    plt.plot(time,r[t0:tf],color=color, \
        linewidth=lw, alpha=1.0,label=filename)
    plt.grid(True)
    ax = plt.gca()
    ax.set_yticks([25,30])
    if j == 1 or j == 3:
        ax.set_yticklabels(['25','30'])
    else:
        ax.set_yticklabels([])
    plt.xlim(0.3,0.6)
    plt.ylim(24,30)
    ax.set_xticks([0.3,0.6])
    ax.set_xticklabels(['0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'centroidavg.png')
    plt.savefig(out_dir+'centroidavg.eps')
    plt.savefig(out_dir+'centroidavg.pdf')
    plt.savefig(out_dir+'centroidavg.svg')
    # Plot sigmaR
    plt.figure(135000,figsize=(figx, figy))
    plt.plot(time,sigr[t0:tf],color=color, \
        linewidth=lw, alpha=1.0, label=filename)
    plt.grid(True)
    ax = plt.gca()
    plt.ylim(0,10)
    ax.set_yticks([0,5,10])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','5','10'])
    else:
        ax.set_yticklabels([])
    plt.xlim(0.3,0.6)
    ax.set_xticks([0.3,0.6])
    ax.set_xticklabels(['0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'centroidstd.png')
    plt.savefig(out_dir+'centroidstd.eps')
    plt.savefig(out_dir+'centroidstd.pdf')
    plt.savefig(out_dir+'centroidstd.svg')
    # plot Z
    plt.figure(125005,figsize=(figx, figy))
    plt.plot(time,z[t0:tf],color=color, \
        linewidth=lw, alpha=1.0,label=filename)
    plt.grid(True)
    ax = plt.gca()
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        ax.set_yticks([-10,-5,0,5,10])
        if j == 1 or j == 3:
            ax.set_yticklabels(['-10','-5','0','5','10'])
        else:
            ax.set_yticklabels([])
        plt.ylim(-11,11)
    else:
        ax.set_yticks([-5,0,5])
        if j == 1 or j == 3:
            ax.set_yticklabels(['-5','0','5'])
        else:
            ax.set_yticklabels([])
        plt.ylim(-6,6)
    plt.xlim(0.3,0.6)
    ax.set_xticks([0.3,0.6])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'zcentroidavg.png')
    plt.savefig(out_dir+'zcentroidavg.eps')
    plt.savefig(out_dir+'zcentroidavg.pdf')
    plt.savefig(out_dir+'zcentroidavg.svg')
    # Plot sigmaZ
    plt.figure(135005,figsize=(figx, figy))
    plt.plot(time,sigz[t0:tf],color=color, \
        linewidth=lw, alpha=1.0, label=filename)
    plt.grid(True)
    ax = plt.gca()
    if 'low' in filename or 'medium' in filename or 'high' in filename:
        plt.ylim(0,12)
        ax.set_yticks([0,4,8,12])
        if j == 1 or j == 3:
            ax.set_yticklabels(['0','4','8','12'])
        else:
            ax.set_yticklabels([])
    else:
        plt.ylim(0,8)
        ax.set_yticks([0,4,8])
        if j == 1 or j == 3:
            ax.set_yticklabels(['0','4','8'])
        else:
            ax.set_yticklabels([])
    plt.xlim(0.3,0.6)
    ax.set_xticks([0.3,0.6])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'zcentroidstd.png')
    plt.savefig(out_dir+'zcentroidstd.eps')
    plt.savefig(out_dir+'zcentroidstd.pdf')
    plt.savefig(out_dir+'zcentroidstd.svg')

## Plots the power balance from psi-tet
# @param psi_dict A psi-tet dictionary
# @param j Index of the subplot
# @param filename Name of the file
def plot_power_balance(psi_dict,j,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(135000+j,figsize=(figx, figy))
    ## Electron heat flux to the wall
    ewall = 0
    ## Ion heat flux to the wall
    iwall = 0
    #plt.subplot(2,2,1)
    plt.plot(time,sihi_smooth(psi_dict['visc'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='r', \
        linewidth=lw, alpha=1.0, label='Viscous heating')
    plt.plot(time,sihi_smooth(psi_dict['ohmic'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='b', \
        linewidth=lw, alpha=1.0, label='Ohmic heating')
    plt.plot(time,psi_dict['fpow'][t0:tf]/1e6, \
        color='m', \
        linewidth=lw, alpha=1.0,label='field power')
    plt.plot(time,psi_dict['therm'][t0:tf]/1e6, \
        color='m', \
        linewidth=lw, alpha=0.65,label='thermal power')
    plt.plot(time,psi_dict['ppow'][t0:tf]/1e6, \
        color='m', \
        linewidth=lw, alpha=0.5,label='kinetic power')
    plt.plot(time,sihi_smooth(psi_dict['e_adv'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='g', \
        linewidth=lw, alpha=0.6,label='electron advection')
    plt.plot(time,sihi_smooth(psi_dict['i_adv'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='g', \
        linewidth=lw, alpha=1.0,label='ion advection')
    plt.plot(time,sihi_smooth(psi_dict['econd'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='y', \
        linewidth=lw, alpha=0.6,label='electron conduction')
    plt.plot(time,sihi_smooth(psi_dict['icond'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='y', \
        linewidth=lw, alpha=1.0,label='ion conduction')
    plt.plot(time,sihi_smooth(psi_dict['equil'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='c', \
        linewidth=lw, alpha=1.0,label='Collisional heating')
    plt.plot(time,sihi_smooth(psi_dict['inj_power'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='k', \
        linewidth=lw, alpha=1.0,label='Injector Power')
    plt.plot(time,sihi_smooth(-psi_dict['ewall'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='pink', \
        linewidth=lw, alpha=0.6,label='Electron heat flux at the wall')
    plt.plot(time,sihi_smooth(-psi_dict['iwall'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='pink', \
        linewidth=lw, alpha=1.0,label='Ion heat flux at the wall')
    plt.legend(edgecolor='k',facecolor='white', \
        framealpha=1.0,fontsize=ms,loc='center right',ncol=2)
    plt.grid(True)
    plt.yscale('symlog',linthreshy=1e-3)
    plt.ylim(-1e1,1e1)
    ax = plt.gca()
    ax.set_yticks([-1,-1e-1,-1e-2,-1e-3,0,1e-3,1e-2,1e-1,1])
    ax.set_yticklabels([r'$-10^{0}$','',r'$-10^{-2}$','','0', \
        '',r'$10^{-2}$','',r'$10^{0}$'])
    plt.xlim(0,0.6)
    ax.set_xticks([0.0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'power_balance'+str(int(psi_dict['f_1']))+'.png')
    plt.savefig(out_dir+'power_balance'+str(int(psi_dict['f_1']))+'.eps')
    plt.savefig(out_dir+'power_balance'+str(int(psi_dict['f_1']))+'.pdf')
    plt.savefig(out_dir+'power_balance'+str(int(psi_dict['f_1']))+'.svg')

## Plots all the heating terms
# @param psi_dict A psi-tet dictionary
# @param color Color of the line
# @param filename Name of the file
# @param directory Name of the directory where the file resides
def plot_all_heat_flows(psi_dict,color,filename,directory):
    power_dir = 'power_plots/'
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    q = 0
    power = np.loadtxt(directory+filename+str(int(psi_dict['f_1']))+'_powers.hist')
    time = power[:,0]
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time1 = (np.abs(time - psi_dict['sp_time'][t0])).argmin()
    time2 = (np.abs(time - psi_dict['sp_time'][tf])).argmin()
    print(time1,time2,time[time1],time[time2])
    time = time[time1:time2]*1e3
    power[:,1:10] = power[:,1:10]/1e6
    power[:,16:18] = power[:,16:18]/1e6
    power[:,19:22] = power[:,19:22]/1e6
    strlist = [\
    'ohmic', \
    'visc', \
    'icond', \
    'econd', \
    'iadv', \
    'eadv1', \
    'eadv2', \
    'nadv', \
    'eke', \
    'ike', \
    'me', \
    'therm', \
    'pe', \
    'pi', \
    'beta', \
    'qdiff', \
    'qdiffwall', \
    'divv', \
    'coll', \
    'compress1', \
    'compress2', \
    'vi', \
    've', \
    'ch', \
    'mach', \
    'alfmach', \
    'che', \
    'vxb']
    for i in range(1,29):
        plt.figure(145000+q,figsize=(figx, figy))
        plt.grid(True)
        ax = plt.gca()
        plt.xlim(0,0.6)
        if strlist[i-1] == 'beta':
            plt.ylim(0, 100)
            # if statement here dealing with the fact that the 'beta'
            # that is outputted by PSI-Tet is not the 'confinement beta'
            if "10eV" in filename:
                pwall = 0.75*1.6*10*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            elif "1eV" in filename:
                pwall = 0.75*1.6*1*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            elif "low" in filename:
                pwall = 1.0*1.6*3*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            elif "medium" in filename:
                pwall = 2.58*1.6*3*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            elif "high" in filename:
                pwall = 5.16*1.6*3*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            else:
                pwall = 0.75*1.6*3*2
                fac = (power[time1,12])*2.0/3.0/pwall
                beta = ((power[time1:time2,12])*2.0/3.0/fac-pwall)/power[time1:time2,11]
            plt.plot(time, sihi_smooth(beta*100, \
                time*1e-3,psi_dict['f_1']), \
                color=color, linewidth=lw+1, alpha=1.0,label=str(psi_dict['f_1'][0]))
        else:
            plt.plot(time, sihi_smooth(power[time1:time2,i], \
                time*1e-3,psi_dict['f_1']), \
                color=color, linewidth=lw+1, alpha=1.0,label=str(psi_dict['f_1'][0]))
            plt.ylim(1.2*min(power[time1:time2,i]), \
                1.2*max(power[time1:time2,i]))
        ax.set_xticks([0.0,0.3,0.6])
        ax.set_xticklabels(['0','0.3','0.6'])
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.savefig(out_dir+power_dir+strlist[i-1]+'.png')
        plt.savefig(out_dir+power_dir+strlist[i-1]+'.eps')
        plt.savefig(out_dir+power_dir+strlist[i-1]+'.pdf')
        plt.savefig(out_dir+power_dir+strlist[i-1]+'.svg')
        q = q+1
    icond = power[time1:time2,3]
    econd = power[time1:time2,4]
    eke = power[time1:time2,9]
    ike = power[time1:time2,10]
    me = power[time1:time2,11]
    therm = power[time1:time2,12]
    compress1 = power[time1:time2,20]
    compress2 = power[time1:time2,21]
    compress = compress1+compress2
    dsize = len(power[time1:time2,0])-2
    dTE = np.zeros(dsize)
    dKE = np.zeros(dsize)
    dME = np.zeros(dsize)
    for i in range(1,dsize):
      dKE[i] = ((eke[i]+ike[i])-(eke[i-1]+ike[i-1]))/(time[i]-time[i-1])*1e-3
      dTE[i] = (therm[i]-therm[i-1])/(time[i]-time[i-1])*1e-3
      dME[i] = (me[i]-me[i-1])/(time[i]-time[i-1])*1e-3
    # Plot Power Balance
    plt.figure(2317423,figsize=(figx,figy))
    plt.plot(time[1:dsize+1], \
        abs(sihi_smooth(dKE+dTE+dME-icond[1:dsize+1]-econd[1:dsize+1], \
        time[1:dsize+1]*1e-3,psi_dict['f_1'])), \
        color=color,linewidth=lw,label=r'$P_{inj}$ from power balance')
    plt.grid(True)
    ax = plt.gca()
    plt.xlim(0,0.6)
    ax.set_xticks([0.0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+power_dir+'inj.png')
    plt.savefig(out_dir+power_dir+'inj.eps')
    plt.savefig(out_dir+power_dir+'inj.pdf')
    plt.savefig(out_dir+power_dir+'inj.svg')

    # Plot injector impedance
    iinjx = np.interp(time,psi_dict['sp_time'][t0:tf]*1e3,psi_dict['curr01'][t0:tf])
    iinjy = np.interp(time,psi_dict['sp_time'][t0:tf]*1e3,psi_dict['curr02'][t0:tf])
    plt.figure(2317476,figsize=(figx,figy))
    plt.plot(time[1:dsize+1],abs(sihi_smooth(dKE+dTE+dME-icond[1:dsize+1]-econd[1:dsize+1], \
        time[1:dsize+1]*1e-3,psi_dict['f_1']))*1e3/ \
        np.sqrt(iinjx[1:dsize+1]**2+iinjy[1:dsize+1]**2), \
        color=color,linewidth=lw,label=r'$Z_{inj}$ from power balance')
    ydata = abs(sihi_smooth(dKE+dTE+dME-icond[1:dsize+1]-econd[1:dsize+1], \
        time[1:dsize+1]*1e-3,psi_dict['f_1']))*1e3/ \
        np.sqrt(iinjx[1:dsize+1]**2+iinjy[1:dsize+1]**2)
    itor = abs(np.interp(time, \
        psi_dict['sp_time'][t0:tf]*1e3,psi_dict['tcurr'][t0:tf]))
    nden = np.interp(time, \
        psi_dict['sp_time'][t0:tf]*1e3,psi_dict['inter_n'][t0:tf])
    itor_nden = 2*itor[int(dsize/2):dsize+1]/ \
        (8*pi*0.25**3*1.6*1e-19*nden[int(dsize/2):dsize+1])
    ydata = ydata[int(dsize/2)-1:dsize]
    # Fit a two-parameter linear model for Zinj
    popt, pcov = curve_fit(Zinj_model,itor_nden,ydata)
    perr = np.sqrt(np.diag(pcov))
    print(popt,perr)
    plt.plot(time[int(dsize/2):dsize+1], \
        Zinj_model(itor_nden,popt[0],popt[1]), \
        color='k',linewidth=lw)
    plt.grid(True)
    ax = plt.gca()
    plt.xlim(0,0.6)
    ax.set_xticks([0.0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+power_dir+'Zinj.png')
    plt.savefig(out_dir+power_dir+'Zinj.eps')
    plt.savefig(out_dir+power_dir+'Zinj.pdf')
    plt.savefig(out_dir+power_dir+'Zinj.svg')
    # Plot the C2 fitting coefficient as a function of frequency
    plt.figure(2317477,figsize=(figx,figy))
    plt.errorbar(psi_dict['f_1'],popt[1]*14.5/psi_dict['f_1'],yerr=perr[1],color=color, \
        marker='o',markersize=ms+6,markeredgecolor='k',elinewidth=lw)
    plt.grid(True)
    ax = plt.gca()
    plt.ylim(0,3)
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+power_dir+'C2.png')
    plt.savefig(out_dir+power_dir+'C2.eps')
    plt.savefig(out_dir+power_dir+'C2.pdf')
    plt.savefig(out_dir+power_dir+'C2.svg')
    # Plot the smoothed Compressional Heat
    plt.figure(2317424,figsize=(figx,figy))
    plt.plot(time,sihi_smooth(compress, \
        time*1e-3,psi_dict['f_1']),color=color,linewidth=lw)
    plt.grid(True)
    ax = plt.gca()
    plt.xlim(0,0.6)
    ax.set_xticks([0.0,0.3,0.6])
    ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+power_dir+'compress.png')
    plt.savefig(out_dir+power_dir+'compress.eps')
    plt.savefig(out_dir+power_dir+'compress.pdf')
    plt.savefig(out_dir+power_dir+'compress.svg')

## Plots all the powers together
# @param psi_dict A psi-tet dictionary
# @param filename Name of the file
# @param directory Name of the directory where the file resides
def plot_powers(psi_dict,j,filename,directory):
    inj_freq = psi_dict['f_1']
    power = np.loadtxt(directory+filename+str(int(psi_dict['f_1']))+'_powers.hist')
    time = power[:,0]
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time1 = (np.abs(time - psi_dict['sp_time'][t0])).argmin()
    time2 = (np.abs(time - psi_dict['sp_time'][tf])).argmin()
    print(time1,time2,time[time1],time[time2])
    time = time[time1:time2]
    power[:,1:10] = power[:,1:10]/1e6
    power[:,16:18] = power[:,16:18]/1e6
    power[:,19:22] = power[:,19:22]/1e6
    ohmic = power[time1:time2,1]
    visc  = power[time1:time2,2]
    icond = power[time1:time2,3]
    econd = power[time1:time2,4]
    iadv  = power[time1:time2,5]
    eadv1 = power[time1:time2,6]
    eadv2 = power[time1:time2,7]
    nadv  = power[time1:time2,8]
    eke   = power[time1:time2,9]
    ike   = power[time1:time2,10]
    me    = power[time1:time2,11]
    therm = power[time1:time2,12]
    pelec    = power[time1:time2,13]
    pion    = power[time1:time2,14]
    beta  = power[time1:time2,15]
    qdiff = power[time1:time2,16]
    qdiffwall = power[time1:time2,17]
    divv  = power[time1:time2,18]
    coll  = power[time1:time2,19]
    compress1 = power[time1:time2,20]
    compress2 = power[time1:time2,21]
    compress = compress1+compress2
    vi = power[time1:time2,22]
    ve = power[time1:time2,23]
    ch = power[time1:time2,24]
    mach = power[time1:time2,25]
    alfmach = power[time1:time2,26]
    che = power[time1:time2,27]
    vxb = power[time1:time2,28]

    dsize = len(power[time1:time2,0])-2
    dTE = np.zeros(dsize)
    dKE = np.zeros(dsize)
    dME = np.zeros(dsize)
    for i in range(1,dsize):
      dKE[i] = ((eke[i]+ike[i])-(eke[i-1]+ike[i-1]))/(time[i]-time[i-1])
      dTE[i] = (therm[i]-therm[i-1])/(time[i]-time[i-1])
      dME[i] = (me[i]-me[i-1])/(time[i]-time[i-1])
    dKE = dKE/1e6
    dTE = dTE/1e6
    dME = dME/1e6
    time = time*1e3
    # plot the thermal flows
    plt.figure(145000,figsize=(figx,figy))
    plt.plot(time,nadv,'b',linewidth=lw,label=r'$|k_b(T_i+T_e)\vec{u}\cdot\nabla n|$')
    plt.plot(time,iadv,'g',linewidth=lw,label=r'$|n\vec{u}\cdot\nabla k_bT_i|$')
    plt.plot(time,eadv1,'g',linewidth=lw,label=r'$|n\vec{u}\cdot\nabla k_bT_e|$')
    plt.plot(time,eadv2,'m',linewidth=lw,label=r'$|\vec{J}\cdot\nabla k_bT_e|$')
    plt.plot(time,ohmic,'r',linewidth=lw,label=r'$|\eta J^2|$')
    plt.plot(time,compress,'y',linewidth=lw,label=r'$|\frac{\gamma nk_b}{\gamma-1}(T_i+T_e)\nabla\cdot \vec{u}|$')
    plt.plot(time[1:dsize+1],dTE,'gray',linewidth=lw,label=r'$|\frac{d}{dt}\frac{nk_b (T_i+T_e)}{\gamma-1} |$')
    plt.plot(time,visc,'hotpink',linewidth=lw,label=r'$|\nu (\nabla \vec{u})^T:\hat{W}|$')
    plt.plot(time,qdiffwall,'c',linewidth=lw,label=r'$|\frac{k_b}{\gamma-1}\int_V (T_i+T_e)D\nabla^2n|$')
    plt.plot(time,icond+econd,'orange',linewidth=lw,label=r'$|\int_\Omega (\vec{q}_i+\vec{q}_e)\cdot \vec{d\Omega}|$')
    plt.yscale('symlog',linthreshy=1e-2)
    #if j == 1:
    #    plt.legend(loc='upper right',framealpha=1.0,fontsize=ls,ncol=2)
    plt.ylim(-10,10)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    # plot the thermal flows
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance_symlog.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance_symlog.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance_symlog.eps')
    plt.figure(155000,figsize=(figx,figy))
    plt.plot(time,abs(nadv),'b',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V(T_i+T_e)\vec{u}\cdot\nabla n dV|$')
    plt.plot(time,abs(iadv),'g',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V n\vec{u}\cdot\nabla T_i dV|$')
    plt.plot(time,abs(eadv1),'burlywood',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V n\vec{u}\cdot\nabla T_e dV|$')
    plt.plot(time,abs(eadv2),'m',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V \vec{J}\cdot\nabla T_e dV|$')
    plt.plot(time,abs(ohmic),'r',linewidth=lw,label=r'$|\int_V \eta J^2 dV|$')
    plt.plot(time,abs(compress),'y',linewidth=lw,label=r'$\frac{\gamma k_b}{\gamma-1}|\int_V n(T_i+T_e)\nabla\cdot \vec{u} dV|$')
    plt.plot(time[1:dsize+1],abs(dTE),'gray',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V \frac{d}{dt} n(T_e+T_i) dV|$')
    plt.plot(time,abs(visc),'hotpink',linewidth=lw,label=r'$|\int_V \nu (\nabla \vec{u})^T:\hat{W} dV|$')
    plt.plot(time,abs(qdiff+qdiffwall),'c',linewidth=lw,label=r'$\frac{k_b}{\gamma-1}|\int_V (T_i+T_e)D\nabla^2n dV|$')
    plt.plot(time,abs(icond+econd),'orange',linewidth=lw,label=r'$|\int_\Omega (\vec{q}_i+\vec{q}_e)\cdot \vec{d\Omega}|$')
    plt.yscale('log')
    #if j == 1:
    #    plt.legend(loc='lower right',framealpha=1.0,fontsize=ls,ncol=2)
    plt.ylim(1e-8,1e0)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'heat_balance.eps')
    # plot the power balance
    plt.figure(165000,figsize=(figx,figy))
    plt.plot(time[1:dsize+1],abs(sihi_smooth(dKE,time[1:dsize+1],psi_dict['f_1'])),'r',linewidth=lw,label=r'$\frac{d}{dt} KE$')
    plt.plot(time[1:dsize+1],abs(sihi_smooth(dTE,time[1:dsize+1],psi_dict['f_1'])),'b',linewidth=lw,label=r'$\frac{d}{dt} TE$')
    plt.plot(time[1:dsize+1],abs(sihi_smooth(dME,time[1:dsize+1],psi_dict['f_1'])),'g',linewidth=lw,label=r'$\frac{d}{dt} ME$')
    plt.plot(time,abs(sihi_smooth(icond,time,psi_dict['f_1'])),'m',linewidth=lw,label=r'$\int_\Omega \vec{q}_i\cdot \vec{d\Omega}$')
    plt.plot(time,abs(sihi_smooth(econd,time,psi_dict['f_1'])),'orange',linewidth=lw,label=r'$\int_\Omega \vec{q}_e\cdot \vec{d\Omega}$')
    plt.plot(time[1:dsize+1],abs(dKE+dTE+dME-icond[1:dsize+1]-econd[1:dsize+1]),'c',linewidth=lw,label=r'$P_{inj}$ from power balance')
    plt.yscale('log')
    #if j == 1:
    #    plt.legend(loc='upper right',framealpha=1.0,fontsize=ls,ncol=2)
    plt.ylim(1e-3,1e1)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'power_balance.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'power_balance.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'power_balance.eps')
    # plot the total energy
    plt.figure(175000,figsize=(figx,figy))
    plt.plot(time,ike+eke,'r',linewidth=lw,label=r'Kinetic Energy')
    plt.plot(time,therm,'b',linewidth=lw,label=r'Thermal Energy')
    plt.plot(time,me,'g',linewidth=lw,label=r'Magnetic Energy')
    plt.plot(time,ike+eke+me+therm,'c',linewidth=lw,label=r'Total Energy')
    plt.plot(time,-che,'k',linewidth=lw,label=r'$H_C$')
    plt.yscale('log')
    #if j == 1:
    #    plt.legend(loc='upper left',framealpha=1.0,fontsize=ls)
    plt.ylim(1e-2,1e2)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'energies.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'energies.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'energies.eps')
    # plot the pressures and beta
    plt.figure(185000,figsize=(figx,figy))
    plt.plot(time,me,'r',linewidth=lw,label=r'$B^2/2\mu_0$')
    plt.plot(time,pion,'b',linewidth=lw,label=r'$P_i$')
    plt.plot(time,pelec,'g',linewidth=lw,label=r'$P_e$')
    plt.plot(time,pion+pelec,'m',linewidth=lw,label=r'$P = P_i + P_e$')
    plt.plot(time,beta*100,'orange',linewidth=lw,label=r'$\beta$ (%)')
    plt.yscale('log')
    if j == 1:
        plt.legend(loc='upper right',framealpha=1.0,fontsize=ls)
    plt.ylim(1e0,1e3)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'pressures.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'pressures.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'pressures.eps')
    #
    # plot turbulence things
    plt.figure(195000,figsize=(figx,figy))
    plt.plot(time,vi,'r',linewidth=lw,label=r'$V_i$')
    plt.plot(time,ve,'b',linewidth=lw,label=r'$V_e$')
    plt.plot(time,abs(ch),'c',linewidth=lw,label=r'$V \cdot B$')
    plt.plot(time,abs(vxb),'g',linewidth=lw,label=r'$V\times B$')
    plt.plot(time,mach,'m',linewidth=lw,label=r'$M$')
    plt.plot(time,alfmach,'orange',linewidth=lw,label=r'$M_A$')
    plt.yscale('log')
    if j == 1:
        plt.legend(loc='upper right',framealpha=1.0,fontsize=ls)
    plt.ylim(1e-3,1e5)
    ax = plt.gca()
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    if j == 2 or j == 4:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'turbulence.svg')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'turbulence.pdf')
    plt.savefig(out_dir+filename+str(int(psi_dict['f_1']))+'turbulence.eps')

## Plots all the powers together
# @param itor_nden proportional to the toroidal current divided by the density
# @param C1 First fitting coefficient of the model, not used currently
# @param C2 Second fitting coefficient of the model
# @returns Z The model fit for the injector impedance
def Zinj_model(itor_nden,C1,C2):
    Zinj = mu0*0.5*(0.0*itor_nden+2*C2*pi*14500)
    return Zinj
