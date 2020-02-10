## @package createShotData
# Contains all the routines necessary for converting
# experimental data in a MDSplus tree into a psi-tet
# dictionary
import MDSplus
from psitet import psiObject
from map_probes import sp_name_dict, dead_probes
from scipy.io import savemat
from dataclasses import asdict
import numpy as np

## Reads in experimental shot information
# from a MDSplus tree and reformats into a
# psi-tet dictionary. Works for HIT-SI and HIT-SI3
# @param shotname The name of the experimental discharge (string)
def createShotData(shotname):
    if(len(shotname) == 6):
        treetype = 'hitsi'
    elif(len(shotname) == 9):
        treetype = 'hitsi3'
    else:
        print('Shotname does not match any known tree')
        exit()
    shot = int(shotname)
    corr_probe_shot = 190604001 # first shot with correct R_CAB in tree
    if shot > 190604001:
        corr_probe = 0 #Don't correct
    elif shot < 181129001:
        corr_probe = 1 # Need to correct for cable resistance
        print('pre-Nov. 29th, 2018 data, cable resistance correction WILL be added');
    else:
        corr_probe = 2 #Need to correct data that was double corrected.
        print('shot between Nov. 29th, 2018 and Jun. 4th, 2019: probe resistance overcorrection WILL be fixed');

    p = asdict(psiObject())

    t0 = 0
    tf = 4.5e-3
    dt = 1.995001e-6
    tsize = int(tf/dt)+1
    time = np.linspace(t0,tf,tsize)

    probe = ['01', '02', '03', '04', '05', '06', \
        '07', '08', '09', '10', '11', '12', \
        '13', '14', '15', '16', '17']
    array = 'M'

    tree = MDSplus.Tree(treetype, \
        np.asarray(shotname).astype(int), \
        'READONLY')
    p['time'] = time
    p['freq'] = tree.getNode('\\SIHI_FREQ').data()
    if treetype == 'hitsi':
        p['curr01'] = gen_data_in(tree,'i_inj_x',time)
        p['curr02'] = gen_data_in(tree,'i_inj_y',time)
        p['flux01'] = gen_data_in(tree,'psi_inj_x',time)
        p['flux02'] = gen_data_in(tree,'psi_inj_y',time)
        p['v_divfx01'] = gen_data_in(tree,'v_divfx',time)
        p['v_divfy02'] = gen_data_in(tree,'v_divfy',time)
        p['volt01'] = gen_data_in(tree,'v_inj_x',time)
        p['volt02'] = gen_data_in(tree,'v_inj_y',time)
        p['tcurr'] = gen_data_in(tree,'i_tor_spaavg',time)
        p['is_HITSI3'] = False
    elif treetype == 'hitsi3':
        p['curr01'] = gen_data_in(tree,'i_inj_a',time)
        p['curr02'] = gen_data_in(tree,'i_inj_b',time)
        p['curr03'] = gen_data_in(tree,'i_inj_c',time)
        p['flux01'] = gen_data_in(tree,'psi_inj_a',time)
        p['flux02'] = gen_data_in(tree,'psi_inj_b',time)
        p['flux03'] = gen_data_in(tree,'psi_inj_c',time)
        p['volt01'] = gen_data_in(tree,'v_inj_a',time)
        p['volt02'] = gen_data_in(tree,'v_inj_b',time)
        p['volt03'] = gen_data_in(tree,'v_inj_c',time)
        p['tcurr'] = gen_data_in(tree,'i_tor_spaavg',time)
        p['is_HITSI3'] = True
    else:
        print("not a valid Tree")

    try:
        p['inter_n'] = gen_data_in(tree,'n_avg_s1',time)
        pass
    except MDSplus.TreeNODATA:
        print('no FIR signal')
        pass

    cb_field, ins_depth = get_imp(tree,time,probe,array)

    p['modes_mag'],p['modes_phase'] = get_modes(tree,time)

    p['sp_Bpol'],p['sp_Btor'],p['sp_names'],p['sp_B'] = \
        get_sp(tree,time)

    p['imp_Brad'],p['imp_Bpol'],p['imp_Btor'] = \
        imp_correction(tree,shot,corr_probe, \
            corr_probe_shot,probe,array,cb_field)

    # write everything to file
    filename = 'exppsi_'+shotname+'.mat'
    savemat(filename,p)

## Gets the IMP signals
# @param tree A MDSplus tree object
# @param time The surface probe time base
# @param probe List of probe names
# @param array Not sure, but related to
# which IMP signals to read
# @returns cb_field Uncorrected IMP magnetic field vector
# @returns ins_depth Insert depth of the IMP
def get_imp(tree,time,probe,array):
    tsize = len(time)
    shift = True
    dafi = 'dafi_cf'
    shot = tree.shot

    N = len(probe)

    cb_field = np.zeros((3, N, tsize))
    ins_depth = np.zeros(N)

    # calibration factors based on dafi impedence.
    # these values good from shot 121973
    pol_scale = [1.02,1.016,1.014,1.018,1.013,1.022, \
        1.027,1.024,1.025,1.022,1.02,1.028,1.029,1.022, \
        1.022,1.024,1.021]
    #there is no first toroidal probe
    tor_scale = [0,1.025,1.012,1.014,1.008,1.022, \
        1.024,1.024,1.023,1.012,1.019,1.02,1.022, \
        1.019,1.021,1.026,1.025]

    # insert IMP stuff here
    nodeflags = [True, True, True]
    for j in range(N):
        pnode = 'B_IMP_M_P' +probe[j] +':b_winding'
        tnode = 'B_IMP_M_T' +probe[j] +':b_winding'
        rnode = 'B_IMP_M_R' +probe[j]
        Pnode = tree.getNode('\\'+pnode)
        Tnode = tree.getNode('\\'+tnode)
        Rnode = tree.getNode('\\'+rnode)

        try:
            rsig = Rnode.data()
            pass
        except MDSplus.TreeNODATA:
            nodeflags[0] = False
            cbw_rad = np.zeros(len(time))*np.nan
            pass
        try:
            psig = Pnode.data()
            pass
        except MDSplus.TreeNODATA:
            nodeflags[1] = False
            cbw_pol = np.zeros(len(time))*np.nan
            pass
        try:
            tsig = Tnode.data()
            pass
        except MDSplus.TreeNODATA:
            nodeflags[2] = False
            cbw_tor = np.zeros(len(time))*np.nan
            pass

        if nodeflags[0] == True:
            dtr = tree.tdiExecute('samplinginterval(\\'+rnode+')')
            tminr = tree.tdiExecute('minval(dim_of(\\'+rnode+'))')
            tlengthr = len(tree.tdiExecute('dim_of(\\'+rnode+')').data())
            trad = tminr + dtr*np.linspace(0,tlengthr,tlengthr)
            # shifting time base here for digi differences
            if shift:
                trad = imp_time_shift(trad, shot, array, \
                    probe[j], 'R')
            bw_rad = tree.tdiExecute( \
                '''slanted_baseline2(sub_baseline_string("\\\\''' +rnode+r'"))')

            cbw_rad = np.interp(time,trad,bw_rad)

        if nodeflags[1] == True:
            dtp = tree.tdiExecute('samplinginterval(\\'+pnode+')')
            tminp = tree.tdiExecute('minval(dim_of(\\'+pnode+'))')
            tlengthp = len(tree.tdiExecute('dim_of(\\'+pnode+')').data())
            tpol = tminr + dtp*np.linspace(0,tlengthp,tlengthp)
            if shift:
                tpol = imp_time_shift(tpol, shot, array, \
                    probe[j], 'P')
            bw_pol = tree.tdiExecute( \
                '''slanted_baseline2(sub_baseline_string("\\\\''' +pnode+r'"))')

            cbw_pol = np.interp(time,tpol,bw_pol)
            if dafi == 'dafi_cf':
                if shot >= 121973:
                    cbw_pol = cbw_pol*pol_scale[j]

        if nodeflags[2] == True:
            dtt = tree.tdiExecute('samplinginterval(\\'+tnode+')')
            tmint = tree.tdiExecute('minval(dim_of(\\'+tnode+'))')
            tlengtht = len(tree.tdiExecute('dim_of(\\'+tnode+')').data())
            ttor = tmint + dtr*np.linspace(0,tlengtht,tlengtht)
            # shifting time base here for digi differences
            if shift:
                ttor = imp_time_shift(ttor, shot, array, \
                    probe[j], 'T')
            bw_tor = tree.tdiExecute( \
                '''slanted_baseline2(sub_baseline_string("\\\\''' +tnode+r'"))')

            cbw_tor = np.interp(time,ttor,bw_tor)
            if dafi == 'dafi_cf':
                if shot >= 121973:
                    cbw_tor = cbw_tor*tor_scale[j]

        cb_field[0,j,:] = cbw_rad
        if array == 'M':
            if j == 1:
                # there is no rot ang for the 1st probe
                # b/c there is no toroidal probe
                cb_field[1,j,:] = cbw_pol
                cb_field[2,j,:] = cbw_tor
            elif j > 1:
                rotnode = 'B_IMP_'+array+'_T'+ \
                    probe[j]+':ROT_ANG'
                rot_ang = tree.getNode('\\'+rotnode).data()
                cb_field[1,j,:] = cbw_pol*np.cos(rot_ang) - \
                    cbw_tor*np.sin(rot_ang)
                cb_field[2,j,:] = cbw_tor*np.cos(rot_ang) + \
                    cbw_pol*np.sin(rot_ang)
        else:
            rotnode = 'B_IMP_'+array+'_T'+probe[j]+':ROT_ANG'
            try:
                rot_ang = tree.getNode('\\'+rotnode).data()
                cb_field[1,j,:] = cbw_pol*np.cos(rot_ang) - \
                    cbw_tor*np.sin(rot_ang)
                cb_field[2,j,:] = cbw_tor*np.cos(rot_ang) + \
                    cbw_pol*np.sin(rot_ang)
                pass
            except MDSplus.TreeNODATA:
                cb_field[1,j,:] = cbw_pol
                cb_field[2,j,:] = cbw_tor
                pass
        r_string1 = r'\B_IMP_M_R'+probe[j]+ \
            r':R:R_CAL_FACT'
        r_string2 = r'\b_imp_ins_d'
        ins_depth[j] = tree.getNode(r_string1).data() - \
            tree.getNode(r_string2).data()

    #imp_Brad = cb_field[0,:,:]
    #imp_Bpol = cb_field[1,:,:]
    #imp_Btor = cb_field[2,:,:]
    # 'r' correction
    if shot >= 150122011 and shot < 151112006:
        ins_depth = ins_depth + 0.076

    return cb_field,ins_depth

## Gets the surface probe signals
# @param tree A MDSplus tree object
# @param time The surface probe time base
# @returns sp_Bpol Poloidal surface probe signals
# @returns sp_Btor Toroidal surface probe signals
# @returns sp_names surface probe names
def get_sp(tree,time):
    jp = 0
    sp_Bpol = []
    sp_Btor = []
    sp_names = []
    sp_B = []
    for node in sp_name_dict.keys():
        if node in dead_probes:
            continue
        sp_names.append(node)
        #node = 'b' + node[1:]
        if node[5] == 'P':
            sp_Bpol.append(gen_data_in(tree,node,time))
        else:
            sp_Btor.append(gen_data_in(tree,node,time))
        sp_B.append(gen_data_in(tree,node,time))
    return sp_Bpol,sp_Btor,sp_names,sp_B

## Shifts the time base of the probe signals due to time
# base differences between digitizers
# @param time The surface probe time base
# @param shot Shot number (integer)
# @param array Not sure, but related to
# which IMP signals to read
# @param probe List of probe names
# @param dir magnetic field direction, R, P, T
# @returns tout The new time base for the IMP
def imp_time_shift(time,shot,array,probe,dir):

    shift612 = 5e-6
    shift2412 = 2.5e-6
    tout = time

    if shot < 117860:
        tout = tin

    elif shot >= 117860 and shot <= 118389:
        if array == 'M':
            if probe == '06':
                if dir == 'R':
                    tout = time - shift2412
        elif array == 'B':
            if probe == '02':
                tout = time - shift612
            elif probe == '03':
                tout = time - shift612
            elif probe == '04':
                tout = time - shift2412
            elif probe == '05':
                tout = time - shift612
            elif probe == '06':
                tout = time - shift612

    elif shot > 118389 and shot <= 121973:
        if array == 'M':
            if probe == '08':
                if dir == 'R':
                    tout = time - shift2412
            elif probe == '10':
                tout = time - shift612
            elif probe == '12':
                tout = time - shift612
            elif probe == '14':
                tout = time - shift612
            elif probe == '17':
                tout = time - shift612
        elif array == 'B':
            if probe == '08':
                tout = time - shift2412

    elif shot > 121973 and shot <= 127542:
        if array == 'M':
            if probe == '06':
                if dir == 'P':
                    tout = time - shift2412
                elif dir == 'T':
                    tout = time - shift2412
            elif probe == '15':
                if dir == 'P':
                    tout = time - shift2412
                elif dir == 'T':
                    tout = time - shift2412
            if dir == 'R':
                tout = time - shift612

    # changed the digitization rate before these shots so the time shift is
    # smaller (only the 612's are off by as much as a usec)
    elif shot > 127542:
        tout = time

    return tout

## Helper function for interpolating onto surface
# probe time base
# @param tree A MDSplus tree object
# @node Name of a valid MDSplus tree node (string)
# @param time The surface probe time base
# @returns x The signal corresponding to the node in the Tree,
# interpolated to the surface probe timebase
def gen_data_in(tree,node,time):
    x = tree.getNode('\\'+node).data()
    dt = tree.tdiExecute('samplinginterval(\\'+node+')')
    tmin = tree.tdiExecute('minval(dim_of(\\'+node+'))')
    tlength = len(x)
    t = tmin + dt*np.linspace(0,tlength,tlength)
    x = np.interp(time,t,x)
    return x

## Helper function getting the fourier modes from
# the diagnostic gap probes
# @param tree A MDSplus tree object
# @param time The surface probe time base
# @returns modes_mag Magnitude of the fourier modes
# @returns modes_phase Phases of the fourier modes
def get_modes(tree,time):
    tsize = len(time)
    modes_mag = np.zeros((4,7,tsize))
    modes_phase = np.zeros((4,7,tsize))
    for i in range(7):
        node = 'b_t_l05_n'+str(i)+':magnitude'
        modes_mag[0,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_t_l06_n'+str(i)+':magnitude'
        modes_mag[1,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_p_l05_n'+str(i)+':magnitude'
        modes_mag[2,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_p_l06_n'+str(i)+':magnitude'
        modes_mag[3,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_t_l05_n'+str(i)+':phase'
        modes_phase[0,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_t_l06_n'+str(i)+':phase'
        modes_phase[1,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_p_l05_n'+str(i)+':phase'
        modes_phase[2,i,:] = \
            gen_data_in(tree,node,time)
        node = 'b_p_l06_n'+str(i)+':phase'
        modes_phase[3,i,:] = \
            gen_data_in(tree,node,time)
    return modes_mag,modes_phase

## This function accounts for IMP corrections beyond
# the simple DAFI corrections.
# This is based on imp_in.m from ACH on June 4 2019
# @param tree A MDSplus tree object
# @param shot Shot number (integer)
# @param corr_probe The surface probe time base
# @param corr_probe_shot Shot name of the hitsi3 reference
# with the correct IMP values
# @param probe List of probe names
# @param array Not sure, but related to
# which IMP signals to read
# @param cb_field Uncorrected IMP magnetic field vector
# @returns imp_Brad Corrected radial IMP signals
# @returns imp_Bpol Corrected poloidal IMP signals
# @returns imp_Btor Corrected toroidal IMP signals
def imp_correction(tree,shot,corr_probe,corr_probe_shot,probe,array,cb_field):
    dir = ['R','P','T']
    N = len(probe)
    if corr_probe==1:
        tree.close()
        new_tree = MDSplus.Tree('hitsi3',corr_probe_shot,'READONLY')
        R_P = np.zeros((3,N))
        for n in range(3):
            for m in range(N):
                nodeR_P = 'B_IMP_'+array+'_'+dir[n] \
                    +probe[m]+':R_CAB' # probe resistance
                try:
                    R_P = new_tree.getNode('\\'+nodeR_P).data()
                    pass
                except MDSplus.TreeNODATA:
                    R_P = 8
                    print('No R_CAB data for node: '+ \
                        nodeR_P+', assuming R_P = 8')
                    pass

                R_T = 50.0 # DAFI variations from 50 Ohms already accounted for
                calFac = (R_T + R_P) / R_T
                cb_field[n,m,:] = calFac * cb_field[n,m,:]
    elif corr_probe == 2:
        incorrectR_P = np.zeros((3, N))
        for n in range(3):
            for m in range(N):
                nodeR_P = 'B_IMP_'+array+'_'+ \
                    dir[n]+probe[m]+':R_P'
                try:
                    R_P = tree.getNode('\\'+nodeR_P).data()
                    pass
                except MDSplus.TreeNODATA:
                    R_P = 8
                    print('No R_CAB data for node: '+ \
                        nodeR_P+', assuming R_P = 8')
                    pass
                incorrectR_P[n,m] = R_P

        tree.close()
        new_tree = MDSplus.Tree('hitsi3',corr_probe_shot,'READONLY')
        R_P = np.zeros((3, N))
        for n in range(3):
            for m in range(N):
                nodeR_CAB = 'B_IMP_'+array+'_'+ \
                    dir[n]+probe[m]+':R_CAB'
                try:
                    R_CAB = new_tree.getNode('\\'+nodeR_CAB).data()
                    pass
                except MDSplus.TreeNODATA:
                    R_CAB = 4
                    print('No R_CAB data for node: '+ \
                        nodeR_CAB+', assuming R_P = 4')
                    pass
                R_P[n,m] = R_CAB

        R_T = 50.0
        calFac = (R_P + R_T) / (incorrectR_P + R_T)
        for n in range(np.shape(cb_field)[2]):
            cb_field[:,:,n] = calFac * cb_field[:,:,n]

    imp_Brad = cb_field[0,:,:]
    imp_Bpol = cb_field[1,:,:]
    imp_Btor = cb_field[2,:,:]

    return imp_Brad, imp_Bpol, imp_Btor

