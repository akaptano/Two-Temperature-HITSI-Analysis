## @package calcCentroid
# Calculates the current centroid
from plot_attributes import *
from map_probes import \
    dead_probes, map_name2position

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
