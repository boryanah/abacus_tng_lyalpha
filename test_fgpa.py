import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf

from compute_power import compute_pk1d, compute_pk3d
from tools import tau2deltaF
from nbodykit.lab import cosmology

# colors
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

def plot_skewer(tau, color, ls, label, skewer_id, Lbox=205.):
    if len(np.shape(tau)) == 2:
        tau = tau.reshape(int(np.sqrt(tau.shape[0])), int(np.sqrt(tau.shape[0])), tau.shape[1])
    bins = np.linspace(0., Lbox, tau.shape[2] + 1)
    binc = (bins[1:]+bins[:-1])*.5
    i = skewer_id
    j = i
    plt.plot(binc, tau[i, j, :], color=color, ls=ls, label=label)

# information about simulation
ngrid = int(sys.argv[1]) #410 #205 #820
Lbox = 205. # cMpc/h
fp_dm = 'fp'
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]
res = sys.argv[2] # "0.05" "0.25" "0.5" "1.0"
redshift = 2.44

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"

# load deltaF
deltaF_file = os.path.join(save_dir, f'noiseless_maps/deltaF_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
deltaF = np.load(deltaF_file)

# load tau (real)
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_real_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
tau = np.load(tau_file)
print("mean, std of tau real = ", np.mean(tau), np.std(tau))

# select one skewer and plot
plt.figure(1, figsize=(11, 10))
sum = 0
plot_skewer(tau, color=colors[sum%len(colors)], ls='--', label='tau no RSD', skewer_id=150)

# load tau (redshift)
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_redshift_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
tau = np.load(tau_file)
print("mean, std of tau redshift = ", np.mean(tau), np.std(tau))

# select one skewer and plot
plot_skewer(tau, color=colors[sum%len(colors)], ls='-', label='tau w RSD', skewer_id=150)
sum += 1

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)

# plot power spectrum
plt.figure(2, figsize=(9, 7))
plt.plot(k_hMpc, p1d_hMpc*k_hMpc/np.pi, label='FGPA')

n_k_bins = 20
n_mu_bins = 16
mu_want = [0., 0.33, 0.66, 1.]

k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)

no_nans = 0
for i in range(n_k_bins):
    if not np.any(np.isnan(mu[i])): no_nans = i
int_mu = []
for i in range(len(mu_want)):
    int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))
print(k_hMpc[:, int_mu[3]])


cosmo = cosmology.Planck15
P_L = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

plt.figure(3, figsize=(9, 7))
plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]), color='violet')
plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc)[:, int_mu[1]]/P_L.__call__(k_hMpc[:, int_mu[1]]), color='cyan')
plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc)[:, int_mu[2]]/P_L.__call__(k_hMpc[:, int_mu[2]]), color='yellow')
plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]), color='red')

print((p3d_hMpc)[:, int_mu[0]])
print((p3d_hMpc)[:, int_mu[3]])

# load Ly alpha from Mahdi
if res == "1.0":
    deltaF_file = os.path.join(save_dir, 'noiseless_maps/map_TNG_true_1.0_z2.4.hdf5')
    deltaF = h5py.File(deltaF_file,'r')['map'][:]
elif res == "0.5":
    deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.5_z2.4.hdf5')
    deltaF = h5py.File(deltaF_file,'r')['map'][:]
elif res == "0.25":
    deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.25_z2.5_820_voxels.hdf5')
    deltaF = h5py.File(deltaF_file,'r')['map'][:]
elif res == "0.05":
    tau_file = os.path.join(save_dir,'spectra_z2.4/spectra_TNG_true_1.0_z2.4.hdf5')
    f = h5py.File(tau_file, 'r')
    tau = gf(f['tau/H/1/1215'][:], 1, mode='wrap')
    print(tau.shape)
    plt.figure(1)
    plot_skewer(tau, color=colors[sum%len(colors)], ls='-', label='tau high', skewer_id=150)
    plt.legend()
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("r [Mpc/h]")
    plt.ylabel("tau(r)")
    plt.xlim([0., Lbox])
    #plt.ylim([np.sqrt(1.e-3), np.sqrt(.1)])
    plt.ylim([-0.5, 50.])
    plt.savefig(f"tau_fgpa_rsd_high_ngrid{ngrid}.png")
    print("mean, std of tau high = ", np.mean(tau), np.std(tau))
    redshift = f['Header'].attrs['redshift']
    deltaF = tau2deltaF(tau, redshift, mean_F=None)

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)

# plot power spectrum
plt.figure(2)
plt.plot(k_hMpc, p1d_hMpc*k_hMpc/np.pi, label='TNG')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("k P(k)/pi")
plt.xlim([0.03, 20.])
plt.ylim([1.e-3, .1])
plt.savefig(f"power_fgpa_rsd_high_ngrid{ngrid}.png")
plt.close()

k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)
print((p3d_hMpc)[:, int_mu[0]])
print((p3d_hMpc)[:, int_mu[3]])

no_nans = 0
for i in range(n_k_bins):
    if not np.any(np.isnan(mu[i])): no_nans = i
int_mu = []
for i in range(len(mu_want)):
    int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))

plt.figure(3)
plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]), color='violet', ls='--')
plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc)[:, int_mu[1]]/P_L.__call__(k_hMpc[:, int_mu[1]]), color='cyan', ls='--')
plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc)[:, int_mu[2]]/P_L.__call__(k_hMpc[:, int_mu[2]]), color='yellow', ls='--')
plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]), color='red', ls='--')
plt.plot([], [], ls='--', label='TNG')
plt.plot([], [], ls='-', label='FGPA')
plt.legend()
#plt.ylim([0., 1.])
plt.ylim([0., 0.4])
plt.xlim([0.03, 20.])
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("P(k)/P_L(k)")
plt.savefig(f"power3d_fgpa_rsd_high_ngrid{ngrid}.png")
plt.close()
