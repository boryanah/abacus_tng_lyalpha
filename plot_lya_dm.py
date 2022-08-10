import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter as gf


save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
fp_dm = "fp"
want_cic = True

z = 2.3
zs = [2.6, 2.4, 2.3]
snaps = [28, 29, 30]
snap_dic = {}
for i in range(len(zs)):
    snap_dic[zs[i]] = snaps[i]


def get_2d_hist(DM_file, sigma, deltaF_file=None, F_file=None):
    """Get 2d hist for deltaF vs rho_DM"""
    bins = [np.linspace(-1, 3, 200), np.linspace(-.4, .2, 75)]
    if deltaF_file is not None :
        #deltaF = np.transpose(h5py.File(deltaF_file,'r')['map'][:], (1,0,2))
        deltaF = h5py.File(deltaF_file,'r')['map'][:]
        deltaF = np.ravel(gf(deltaF, sigma, mode='wrap'))
    if F_file is not None:
        F = h5py.File(F_file,'r')['map'][:]
        deltaF = F/np.mean(F) - 1
        deltaF = np.ravel(gf(deltaF, sigma, mode='wrap'))
    
    DM = np.ravel(gf(h5py.File(DM_file,'r')['DM/dens'][:], 
                     sigma , mode='wrap'))
    print('deltaF:', deltaF.shape)
    print('DM:', DM.shape)
    h,_, _ = np.histogram2d( DM-1, deltaF, bins=[bins[0],bins[1]], density=True)
    
    return bins, h

def deltaF_rhoDM_contours(fig, ax, sigma, DM_file, ls='solid', color='C0', deltaF_file=None, F_file=None, label='c'):
    bins, h = get_2d_hist(DM_file=DM_file, sigma=sigma, deltaF_file=deltaF_file, F_file=F_file)

    xmbin = [(bins[0][b]+bins[0][b+1])/2 for b in range(len(bins[0])-1)]
    ymbin = [(bins[1][b]+bins[1][b+1])/2 for b in range(len(bins[1])-1)]
    X, Y = np.meshgrid(xmbin,ymbin)
    
    cs = ax.contour(X.T, Y.T, h, levels=[0.02, 0.68, 0.98], linestyles=ls, colors=color)

    cs.collections[1].set_label(label)
    ax.set_xlim(bins[0][0]-0.05, bins[0][-1]+0.05)
    ax.set_ylim(bins[1][0]-0.05, bins[1][-1]+0.05)
    ax.set_ylabel(r'$\mathrm{\delta^{sm}_{F}}$')
    ax.set_xlabel(r'$(\rho_{\rm DM}/<\rho_{\rm DM}>)^{sm} - 1$')
    ax.grid(True)
    ax.legend(loc='upper right')
    
fig, ax= plt.subplots(1,1)
deltaF_rhoDM_contours(fig, ax, sigma=4, DM_file=os.path.join(save_dir,'DM_Density_field/TNG_DM_z2.4.hdf5'),
                      deltaF_file=os.path.join(save_dir,'noiseless_maps/map_TNG_true_1.0_z2.4.hdf5'),
                      label='Hydro')

"""
deltaF_rhoDM_contours(fig, ax, sigma=4,DM_file=os.path.join(save_dir,'DM_Density_field/TNG_DM_z2.4.hdf5'),
                      F_file=os.path.join(data_dir,'FGPA/FGPA_z2.4_1.0.hdf5'), ls='--',
                      color='C1', label='FGPA')
"""

ax.set_title('MF=MF(z=2.45)')
fig.tight_layout()
fig.savefig('dm_lya.png')

quit()
f = h5py.File(save_dir+f"DM_Density_field/TNG_DM_z{z:.1f}.hdf5")
print(list(f.keys()))
dens = f['DM']['dens'][:]
print(dens.shape)
#print(list(f['Gas'].keys()))
print(dens.min(), dens.max(), np.mean(dens))
#dens = f['Gas'][:]
plt.imshow(dens[:, :, 20])
plt.colorbar()
plt.savefig("their_density.png")
plt.close()

if want_cic:
    density = np.load(save_dir+f"density_cic_ngrid_205_snap_{snap_dic}_{fp_dm}.npy")
else:
    density = np.load(save_dir+f"density_ngrid_205_snap_{snap_dic}_{fp_dm}.npy")
density /= np.mean(density)
print(density.shape)
plt.imshow(density[:, :, 20])
plt.colorbar()
plt.savefig(f"my_density_{fp_dm}.png")
plt.close()

diff = (density-dens)/np.std(dens)
plt.imshow(diff[:, :, 20])
plt.colorbar()
plt.savefig(f"diff_{fp_dm}.png")
plt.close()

print(diff, diff.max(), diff.min())


