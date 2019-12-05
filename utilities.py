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

## Calculate the current centroid
# @param dict A psi-tet dictionary
# @returns rAvg Average radial centroid position
# @returns rStd radial centroid standard dev.
# @returns zAvg Average axial centroid position
# @returns zStd axial centroid standard dev.
def calcCentroid(dict):
    time = dict['sp_time']

    tol = 1e-1
    sp_names = dict['sp_names']
    sp_Bpol = dict['sp_Bpol']
    sp_Btor = dict['sp_Btor']

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
# @param dict A psi-tet dictionary
def SVD(dict):
    t0 = dict['t0']
    tf = dict['tf']
    data = np.vstack((dict['curr01'],dict['curr02']))
    if dict['is_HITSI3'] == True:
        data = np.vstack((data,dict['curr03']))
    #data = np.vstack((data,dict['flux01']))
    #data = np.vstack((data,dict['flux02']))
    data = np.vstack((data,dict['sp_Bpol']))
    data = np.vstack((data,dict['sp_Btor']))
    getshape = np.shape(data)[0]
    if dict['use_IMP']:
        dict['imp_Bpol'] = np.nan_to_num(dict['imp_Bpol'])[::10,:]
        dict['imp_Btor'] = np.nan_to_num(dict['imp_Btor'])[::10,:]
        dict['imp_Brad'] = np.nan_to_num(dict['imp_Brad'])[::10,:]
        dict['imp_Bpol'] = dict['imp_Bpol'][::1]
        dict['imp_Btor'] = dict['imp_Btor'][::1]
        dict['imp_Brad'] = dict['imp_Brad'][::1]
        data = np.vstack((data,dict['imp_Bpol']))
        shape1 = np.shape(dict['imp_Bpol'])[0]
        shape2 = np.shape(dict['imp_Btor'])[0]
        shape3 = np.shape(dict['imp_Brad'])[0]
        imp_pol_indices = np.linspace(0,shape1,shape1, \
            dtype = 'int')
        data = np.vstack((data,dict['imp_Btor']))
        imp_tor_indices = np.linspace(shape1,shape2+shape1,shape2, \
            dtype = 'int')
        data = np.vstack((data,dict['imp_Brad']))
        imp_rad_indices = np.linspace(shape1+shape2, \
            shape3+shape2+shape1,shape3, \
            dtype = 'int')

    # correct injector currents
    if dict['is_HITSI3'] == True:
        data[0:3,:] = data[0:3,:]*mu0
    else:
        data[0:2,:] = data[0:2,:]*mu0
    data = data[:,t0:tf]
    data_sub = data #subtract_linear_trend(dict,data)
    u,s,v = np.linalg.svd(data_sub)
    v = np.conj(np.transpose(v))
    dict['SVD_data'] = data_sub
    dict['SP_data'] = data
    dict['U'] = u
    dict['S'] = s
    dict['V'] = v

## Identifies and subtracts a linear trend from each
## of the time signals contained in
## the SVD data associated with the dictionary 'dict'. This is
## to help DMD algorithms, since DMD does not deal well with
## non-exponential growth.
# @param dict A dictionary with SVD data
# @param data The SVD data matrix
# @returns data_subtracted The SVD data matrix
#  with the linear trend subtracted off
def subtract_linear_trend(dict,data):
    state_size = np.shape(data)[0]
    tsize = np.shape(data)[1]
    t0 = dict['t0']
    tf = dict['tf']
    time = dict['sp_time'][t0:tf]
    dt = dict['sp_time'][1] - dict['sp_time'][0]
    data_subtracted = np.zeros((state_size,tsize))
    for i in range(state_size):
        slope, intercept, r_value, p_value, std_err = linregress(time,data[i,:])
        data_subtracted[i,:] = data[i,:] - (slope*time+intercept)
        if i == 10:
            plt.figure()
            plt.plot(time,data_subtracted[i,:],'g')
            plt.plot(time,slope*time+intercept,'b')
            plt.plot(time,data[i,:],'r')
            plt.savefig(out_dir+'linear_trend_test.png')
    return data_subtracted

## Computes the toroidal mode spectrum using the
## surface midplane gap probes
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_sp(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset+size_bpol-32: \
            offset+size_bpol:2,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset+size_bpol-32: \
            offset+size_bpol:2,:]

    tsize = len(t_vec)
    phi = midphi
    nmax = 7
    amps = fourier_calc(nmax,tsize,Bfield_anom,phi)
    plt.figure(50000,figsize=(figx, figy))
    for m in range(nmax+1):
        plt.plot(t_vec*1000, \
            amps[m,:],label='n = '+str(m), \
            linewidth=lw)
            #plt.yscale('log')
    plt.legend(fontsize=ls,loc='upper right',ncol=2)
    plt.axvline(x=23.34,color='k')
    plt.xlabel('Time (ms)', fontsize=fs)
    plt.ylabel(r'$\delta B$', fontsize=fs)
    plt.title('Surface Probes', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_amps_sp.png')
    dict['toroidal_amps'] = amps
    plt.figure(60000,figsize=(figx, figy))
    plt.title('Surface Probes', fontsize=fs)
    plt.bar(range(nmax+1),amps[:,0]*1e4,color='r',edgecolor='k')
    plt.xlabel('Toroidal Mode',fontsize=fs)
    plt.ylabel('B (G)',fontsize=fs)
    ax = plt.gca()
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.png')

## Computes the toroidal mode spectrum using
## a set of 8 or 32 IMPs
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_imp(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    size_imp_bpol = np.shape(dict['imp_Bpol'])[0]
    size_imp_btor = np.shape(dict['imp_Btor'])[0]
    size_imp_brad = np.shape(dict['imp_Brad'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset+size_bpol+size_btor: \
            offset+size_bpol+size_btor+size_imp_bpol,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset+size_bpol+size_btor: \
            offset+size_bpol+size_btor+size_imp_bpol,:]

    print('sihi smooth freq = ',f_1)
    tsize = len(t_vec)
    num_IMPs = dict['num_IMPs']
    phis = np.zeros(160*num_IMPs)
    if num_IMPs == 8:
        imp_phis = imp_phis8
        nmax = 3
    elif num_IMPs == 32:
        imp_phis = imp_phis32
        nmax = 10
    else:
        print('Invalid number for the number of IMPs')
        exit()
    for i in range(num_IMPs):
        phis[i*160:(i+1)*160] = np.ones(160)*imp_phis[i]
    # subsample as needed
    phis = phis[::10]
    phis = phis[:len(phis)]
    amps = np.zeros((nmax+1,16,tsize))
    plt.figure(figsize=(figx+2, figy+2))
    for k in range(16):
        amps[:,k,:] = fourier_calc(nmax,tsize,Bfield_anom[k::16,:],phis[k::16])
        plt.subplot(4,4,(k+1))
        amax = np.max(np.max(amps[:,k,:]))
        for m in range(nmax+1):
            plt.plot(t_vec*1000, \
                amps[m,k,:]/amax, \
                label='n = '+str(m))
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts-6)
        ax.tick_params(axis='both', which='minor', labelsize=ts-6)
    plt.savefig(out_dir+'toroidal_amps_imp.png')

    plt.figure(170000,figsize=(figx, figy))
    avg_amps = np.mean(abs(amps),axis=1)
    for m in range(nmax+1):
        plt.plot(t_vec*1000, \
            avg_amps[m,:],label='n = '+str(m), \
            linewidth=lw)
    plt.xlabel('Time (ms)', fontsize=fs)
    plt.title('Average of IMPs', fontsize=fs)
    plt.ylabel(r'$\delta B$', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_imp.png')
    dict['toroidal_amps'] = avg_amps
    plt.figure(180000,figsize=(figx, figy))
    plt.title('Average of IMP Probes', fontsize=fs)
    plt.bar(range(nmax+1),avg_amps[:,0]*1e4,color='r',edgecolor='k')
    plt.xlabel('Toroidal Mode',fontsize=fs)
    plt.ylabel('B (G)',fontsize=fs)
    ax = plt.gca()
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.png')

## Computes the poloidal mode spectrum for each
## of the four poloidal slices of the surface probes
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def poloidal_modes(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset:offset+size_bpol,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset:offset+size_bpol,:]
    tsize = len(t_vec)
    # Find the poloidal gap probes
    k1 = 0
    k2 = 0
    j = 0
    B = np.zeros((16,tsize))
    theta = np.zeros(16)
    temp_B = np.zeros((64,tsize))
    temp_theta = np.zeros(16)
    for key in sp_name_dict.keys():
        if key in dead_probes:
            if key[5] == 'P':
                k2 = k2 + 1
            continue
        if key[5] == 'P' and \
            key[2:5] != 'L05' and key[2:5] != 'L06':
            temp_B[k2, :] = Bfield_anom[j, :]
        if key[5:9] == 'P225' and \
            key[2:5] != 'L05' and key[2:5] != 'L06':
            temp_theta[k1] = sp_name_dict[key][3]
            k1 = k1 + 1
        if key[5] == 'P':
            j = j + 1
            k2 = k2 + 1
    phi_str = [r'$0^o$',r'$45^o$', \
        r'$180^0$',r'$225^o$']
    nmax = 7
    for i in range(4):
        plt.figure(80000,figsize=(figx, figy))
        B = temp_B[i::4,:]
        inds = ~np.all(B == 0, axis=1)
        B = B[inds]
        theta = temp_theta[np.where(inds)]
        amps = fourier_calc(nmax,tsize,B,theta)
        # can normalize by Bwall here
        # b1 = np.sqrt(int.sp.B_L04T000**2 + int.sp.B_L04P000**2)
        # b2 = np.sqrt(int.sp.B_L04T045**2 + int.sp.B_L04P045**2)
        # b3 = np.sqrt(int.sp.B_L04T180**2 + int.sp.B_L04P180**2)
        # b4 = np.sqrt(int.sp.B_L04T225**2 + int.sp.B_L04P225**2)
        # b0 = sihi_smooth((b1+b2+b3+b4)/4.0,t_vec,f_1)
        plt.subplot(2,2,i+1)
        for m in range(nmax+1):
            plt.plot(t_vec*1000, \
            amps[m,:],label='m = '+str(m),
            linewidth=lw)
        plt.title(r'$\phi$ = '+phi_str[i],fontsize=fs)
        if i == 0 or i == 2:
            plt.ylabel(r'$\delta B$', fontsize=fs)
        if i >= 2:
            plt.xlabel('Time (ms)',fontsize=fs)
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.savefig(out_dir+'poloidal_amps.png')
        dict['poloidal_amps'] = amps
        plt.figure(70000,figsize=(figx, figy))
        plt.subplot(2,2,i+1)
        plt.title(r'$\phi$ = '+phi_str[i],fontsize=fs)
        plt.bar(range(nmax+1),amps[:,0]*1e4,color='r',edgecolor='k')
        if i == 0 or i == 2:
            plt.ylabel('B (G)', fontsize=fs)
        if i >= 2:
            plt.xlabel('Poloidal Mode',fontsize=fs)
        ax = plt.gca()
        ax.set_xticks([0,1,2,3,4,5,6,7])
        ax.set_xticklabels(['0','1','2','3','4','5','6','7'])
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.png')

## Performs the fourier calculation based on Wrobel 2011
# @param nmax The toroidal/poloidal number resolution
# @param tsize The number of time snapshots
# @param b Magnetic field signals of a toroidal/poloidal set of probes
# @param phi Toroidal/poloidal angles associated with the probes
# @returns amps The toroidal/poloidal mode amplitudes
def fourier_calc(nmax,tsize,b,phi):
    # Set up mode calculation- code adapted from JSW
    minvar = 1e-10 # minimum field variance for calcs.
    amps = np.zeros((nmax+1,tsize))
    phases = np.zeros((nmax+1,tsize))
    vardata = np.zeros(np.shape(b)) + minvar
        # Calculate highest nmax possible
    nprobes = np.shape(b)[0]
    mcoeff = np.zeros((2*nmax + 1,2*nmax + 1))
    for nn in range(nmax+1):
        for m in range(nmax+1):
            mcoeff[m, nn] = \
                sum(np.cos(m*phi) * np.cos(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax, nn] = \
                sum(np.sin(m*phi) * np.cos(nn*phi))

    for nn in range(1,nmax+1):
        for m in range(nmax+1):
            mcoeff[m,nn+nmax] = \
                sum(np.cos(m*phi) * np.sin(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax,nn+nmax] = \
                sum(np.sin(m*phi) * np.sin(nn*phi))

    asnbs    = np.zeros(2*nmax + 1)
    varasnbs = np.zeros(2*nmax + 1)
    rhs      = np.zeros(2*nmax + 1)
    veca     = np.zeros((nmax+1,tsize))
    vecb     = np.zeros((nmax,tsize))
    for m in range(tsize):
        bflds = b[:, m]
        varbflds = vardata[:, m]
        for nn in range(nmax+1):
            rhs[nn] = sum(bflds*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(bflds*np.sin(nn*phi))
        asnbs,g1,g2,g3 = np.linalg.lstsq(mcoeff,rhs)
        for nn in range(nmax+1):
            rhs[nn] = sum(np.sqrt(varbflds)*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(np.sqrt(varbflds)*np.sin(nn*phi))
        veca[0:nmax+1, m] = asnbs[0:nmax+1]
        vecb[0:nmax, m] = asnbs[nmax+1:2*nmax+1]
    amps[0,:] = veca[0, :]
    phases[0,:] = 0.0 * veca[0, :]

    for m in range(nmax):
        amps[m+1,:] = \
            np.sqrt(veca[m+1, :]**2 + vecb[m, :]**2)
        phases[m+1,:] = np.arctan2(vecb[m, :], veca[m+1, :])
    return amps

## Plots the toroidal current for a shot
# @param psi_dict A psi-tet dictionary
def plot_itor(psi_dict,j,color,filename):
    itor = psi_dict['tcurr']/1000.0
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['time']*1000.0
    plt.figure(75000,figsize=(figx, figy))
    plt.plot(time,abs(itor),color=color,linewidth=lw,label=filename+r' $I_{tor}$', \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()])
    #plt.plot(time,np.sqrt((psi_dict['curr01']/1000.0)**2+(psi_dict['curr02']/1000.0)**2), \
    #    color=color,alpha=0.6,linewidth=lw,label=r'$\sqrt{(I^{inj}_x)^2+(I^{inj}_y)^2}$', \
    #    path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'),pe.Normal()])

    #plt.plot(time,dict['curr02']/1000.0,'k',alpha=0.5,linewidth=lw,label=r'$I_y$')
    #if j == 1:
    #    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper left')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
    #plt.ylabel(r'$I_{tor}$ (kA)', fontsize=fs)
    plt.grid(True)
    plt.xlim(0,0.6)
    ax = plt.gca()
    ax.set_yticks([0,10,20,30])
    ax.set_xticks([0,0.3,0.6])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','10','20','30'])
    else:
        ax.set_yticklabels([])
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_current.png')
    plt.savefig(out_dir+'toroidal_current.eps')
    plt.savefig(out_dir+'toroidal_current.pdf')
    plt.savefig(out_dir+'toroidal_current.svg')

## Plots the BD chronos for a shot
# @param psi_dict A psi-tet dictionary
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
            path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
            pe.Normal()],label='BOD mode '+str(i+1))
    if j == 1:
        plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper left')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
    #h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}V_{ki}^*$', fontsize=fs)
    #h.set_rotation(0)
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
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Mode Number k', fontsize=fs)
    #h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}$', fontsize=fs)
    #h.set_rotation(0)
    plt.grid(True)
    plt.ylim([1e-2,2e0])
    ax = plt.gca()
    ax.set_yticks([1e-2,1e-1,1e0])
    if j == 1 or j == 3:
        ax.set_yticklabels([r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    else:
        ax.set_yticklabels([])
    plt.xlim([0,20])
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD.png')
    plt.savefig(out_dir+'BD.eps')
    plt.savefig(out_dir+'BD.pdf')
    plt.savefig(out_dir+'BD.svg')

## Plots the average electron and ion temperatures for a shot
# @param psi_dict A psi-tet dictionary
def plot_temperatures(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(105000,figsize=(figx, figy))
    if 'te' in psi_dict.keys() and psi_dict['te'][10] != psi_dict['ti'][10]:
        plt.plot(time,psi_dict['te'][t0:tf],color=color, \
            linewidth=lw, alpha=1.0,linestyle='--', \
            path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
            pe.Normal()],label=filename+r' $T_e$')
        plt.plot(time,psi_dict['ti'][t0:tf],color=color, \
            linewidth=lw, alpha=1.0, \
            path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
            pe.Normal()],label=filename+r' $T_i$')
    else:
        plt.plot(time,psi_dict['ti'][t0:tf],color=color, \
            linewidth=lw, alpha=1.0, \
            path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
            pe.Normal()],label=filename+r' $T$')
    #if j == 1:
    #    plt.legend(edgecolor='k',facecolor='white', \
    #        framealpha=1.0,fontsize=ls,loc='upper left')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
    #h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}V_{ki}^*$', fontsize=fs)
    #h.set_rotation(0)
    plt.grid(True)
    #plt.ylim(-1500,1500)
    ax = plt.gca()
    ax.set_yticks([0,10,20,30])
    if j == 1 or j == 3:
        ax.set_yticklabels(['0','10','20','30'])
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0,0.3,0.6])
    plt.xlim(0,0.6)
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'temperatures.png')
    plt.savefig(out_dir+'temperatures.eps')
    plt.savefig(out_dir+'temperatures.pdf')
    plt.savefig(out_dir+'temperatures.svg')

## Plots the line-averaged FIR density
# @param psi_dict A psi-tet dictionary
def plot_nFIR(psi_dict,j,color,filename):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(115000,figsize=(figx, figy))
    plt.plot(time,sihi_smooth(psi_dict['inter_n'][t0:tf], \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color=color, \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label=filename)
    #if j == 1:
    #    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper right')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
    #h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}V_{ki}^*$', fontsize=fs)
    #h.set_rotation(0)
    plt.grid(True)
    plt.ylim(4e18,1.3e19)
    #plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks([0.4e19,0.8e19,1.2e19])
    plt.xlim(0,0.6)
    if j == 1 or j == 3:
        ax.set_yticklabels(['0.4','0.8','1.2'])
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0,0.3,0.6])
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['0','0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'nFIR.png')
    plt.savefig(out_dir+'nFIR.eps')
    plt.savefig(out_dir+'nFIR.pdf')
    plt.savefig(out_dir+'nFIR.svg')

## Plots the centroid avg and std
# @param psi_dict A psi-tet dictionary
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
    plt.figure(125000,figsize=(figx, figy))
    plt.plot(time,r[t0:tf],color=color, \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label=filename)
    #if j == 1:
    #    plt.legend(edgecolor='k',facecolor='white',
    #        framealpha=1.0,fontsize=ls,loc='upper right')
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
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'centroidavg.png')
    plt.savefig(out_dir+'centroidavg.eps')
    plt.savefig(out_dir+'centroidavg.pdf')
    plt.savefig(out_dir+'centroidavg.svg')
    plt.figure(135000,figsize=(figx, figy))
    plt.plot(time,sigr[t0:tf],color=color, \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label=filename)
    #if j == 1:
    #    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper right')
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
    if j == 1 or j == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['0.3','0.6'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'centroidstd.png')
    plt.savefig(out_dir+'centroidstd.eps')
    plt.savefig(out_dir+'centroidstd.pdf')
    plt.savefig(out_dir+'centroidstd.svg')

## Plots the power balance from psi-tet
# @param psi_dict A psi-tet dictionary
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
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Viscous heating')
    plt.plot(time,sihi_smooth(psi_dict['ohmic'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='b', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Ohmic heating')
    #plt.legend(edgecolor='k',facecolor='white',fontsize=ms,loc='upper right')
    #ax = plt.gca()
    #ax.tick_params(axis='both', which='major', labelsize=ts)
    #ax.tick_params(axis='both', which='minor', labelsize=ts)
    #plt.grid(True)
    #plt.subplot(2,2,2)
    plt.plot(time,sihi_smooth(abs(psi_dict['fpow'][t0:tf])/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='m', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='field power')
    plt.plot(time,sihi_smooth(abs(psi_dict['therm'][t0:tf])/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='m', \
        linewidth=lw, alpha=0.65, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='thermal power')
    plt.plot(time,sihi_smooth(abs(psi_dict['ppow'][t0:tf])/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='m', \
        linewidth=lw, alpha=0.5, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='kinetic power')
    #plt.legend(edgecolor='k',facecolor='white',fontsize=ms,loc='upper right')
    #ax = plt.gca()
    #ax.tick_params(axis='both', which='major', labelsize=ts)
    #ax.tick_params(axis='both', which='minor', labelsize=ts)
    #plt.grid(True)
    #plt.subplot(2,2,3)
    plt.plot(time,sihi_smooth(psi_dict['e_adv'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='g', \
        linewidth=lw, alpha=0.6, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='electron advection')
    plt.plot(time,sihi_smooth(psi_dict['i_adv'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='g', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='ion advection')
    #plt.legend(edgecolor='k',facecolor='white',fontsize=ms,loc='upper right')
    plt.plot(time,sihi_smooth(psi_dict['econd'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='y', \
        linewidth=lw, alpha=0.6, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='electron conduction')
    plt.plot(time,sihi_smooth(psi_dict['icond'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='y', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='ion conduction')
    plt.plot(time,sihi_smooth(psi_dict['equil'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='c', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Collisional heating')
    #plt.legend(edgecolor='k',facecolor='white',fontsize=ms,loc='upper right')
    #ax = plt.gca()
    #ax.tick_params(axis='both', which='major', labelsize=ts)
    #ax.tick_params(axis='both', which='minor', labelsize=ts)
    #plt.grid(True)
    #plt.subplot(2,2,4)
    plt.plot(time,sihi_smooth(psi_dict['inj_power'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='k', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Injector Power')
    plt.plot(time,sihi_smooth(-psi_dict['ewall'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='pink', \
        linewidth=lw, alpha=0.6, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Electron heat flux at the wall')
    plt.plot(time,sihi_smooth(-psi_dict['iwall'][t0:tf]/1e6, \
        psi_dict['sp_time'][t0:tf],psi_dict['f_1']),color='pink', \
        linewidth=lw, alpha=1.0, \
        path_effects=[pe.Stroke(linewidth=lw+2,foreground='k'), \
        pe.Normal()],label='Ion heat flux at the wall')
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

def plot_individual_heat_flows(psi_dict,color):
    power_dir = 'power_plots/'
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    q = 0
    for i in ['visc','ohmic','fpow','ppow','econd','icond', \
        'e_adv','i_adv','ewall','iwall','equil','inj_power','therm']:
        plt.figure(145000+q,figsize=(figx, figy))
        plt.plot(time,
            sihi_smooth(abs(psi_dict[i][t0:tf]/1e6), \
            psi_dict['sp_time'][t0:tf],psi_dict['f_1']), \
            color=color, linewidth=lw+2, alpha=1.0, \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()],label=str(psi_dict['f_1'][0]))
        #plt.legend(edgecolor='k',facecolor='white', \
        #    framealpha=1.0,fontsize=ls)
        plt.grid(True)
        ax = plt.gca()
        plt.xlim(0,0.6)
        ax.set_xticks([0.0,0.3,0.6])
        ax.set_xticklabels(['0','0.3','0.6'])
        ax.tick_params(axis='both', which='major', labelsize=ts+6)
        ax.tick_params(axis='both', which='minor', labelsize=ts+6)
        plt.savefig(out_dir+power_dir+i+'.png')
        plt.savefig(out_dir+power_dir+i+'.eps')
        plt.savefig(out_dir+power_dir+i+'.pdf')
        plt.savefig(out_dir+power_dir+i+'.svg')
        q = q+1

## This function is not yet functional. This will at some point
# compute the IDS chord-averaging and analysis
# @param psi_dict A psi-tet dictionary
def IDS(psi_dict):
    # ids locations
    nchords = 108
    npts = 201
    pos = np.transpose(np.loadtxt('ids_locations.txt'))
    pos = np.reshape(pos,(3,nchords,npts))
    Distances = np.zeros((nchords, npts))
    Directions = np.zeros((3,nchords))
    ImpactParameters = np.zeros(nchords)
    for i in range(nchords):
        Distances[i,1] = np.sqrt((pos[0,i,0]-pos[0,i,1])**2 + \
            (pos[1,i,0]-pos[1,i,1])**2)
        for j in range(1,npts-1):
            Distances[i,j] = np.sqrt((pos[0,i,j-1]-pos[0,i,j+1])**2+ \
                (pos[1,i,j-1]-pos[1,i,j+1])**2)
        Distances[i,npts] = np.sqrt((pos[0,i,npts-2]-pos[0,i,npts-1])**2 \
            +(pos[1,i,npts-2]-pos[1,i,npts-1])**2)
#        Directions[:,i] = [x1(i)-x0(i),y1(i)-y0(i),0]/norm([x1(i)-x0(i),y1(i)-y0(i),0]);
#        ImpactParameters[i] = ImpactParameter(x0(i),y0(i),x1(i),y1(i))
        Distances[i,:] = Distances[i,:]/2.0
    tsize = np.shape(psi_dict['ids_n'])[1]
    VAvgPerChord = np.zeros((nchords,tsize))
    TAvgPerChord = np.zeros((nchords,tsize))
    for i in range(nchords):
        for j in range(tsize):
            numV = 0
            numT = 0
            denom = 0
            for k in range(npts):
                numV = numV + psi_dict['ids_n'][i,k,j]^2* \
                    psi_dict['ids_n'][:,i,k,j]*Directions[:,i]* \
                    Distances[i,k]
                denom = denom + psi_dict['ids_n'][i,k,j]^2* \
                    Distances[i,k]
                numT = numT + psi_dict['ids_n'][i,k,j]^2* \
                    psi_dict['ids_T'][i,k,j]*Distances[i,k]
            if denom != 0:
                VAvgPerChord[i,j] = numV/denom
                TAvgPerChord[i,j] = numT/denom

## Calculates the impact parameters of IDS_Coords
# @param IDS_Coords
def IDS_impacts(IDS_Coords):
    s = np.shape(IDS_Coord)[2]
    for i in range(s):
        x = IDS_Coord[1,:,i]
        y = IDS_Coord[2,:,i]
        m[i] = ( y[1] - y[0] ) / ( x[1] - x[0] )
        b[i] = y[1] - m[i]*x[1]
        x0[i] = -m[i]*b[i] / (1+m[i]^2)
        # y0[i] = m[i]*x0[i] + b[i]
        y0[i] = b[i] / (m[i]^2 + 1)
        z = x0[i] + np.sqrt(-1)*y0[i]
        thet[i] = np.angle(z)
        # numpy sign and matlab sign may be different!
        impact[i] = np.sqrt(x0[i]**2 + y0[i]**2) * np.sign(thet[i])
        impact[i] = np.sqrt(x0[i]**2 + y0[i]**2)
