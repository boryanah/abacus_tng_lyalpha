import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf

import h5py
import numpy as np

from compute_power import compute_fk1d, compute_pk1d, power_mahdi
from tools import tau2deltaF
from fake_spectra import fluxstatistics as fs

from astropy.cosmology import Planck15 as cosmo

#def sinc(k, R):
#    return np.sin(0.5*k*R)/(0.5*k*R)

# colors
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

# directory with lyman alpha skewers
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"

# load high-resolution deltaF
tau_file = os.path.join(save_dir,'spectra_z2.4/spectra_TNG_true_1.0_z2.4.hdf5')
f = h5py.File(tau_file, 'r')
tau_hd = gf(f['tau/H/1/1215'][:], 1, mode='wrap')
ngrid_hd = tau_hd.shape[-1] # high-resolution
print(tau_hd[20, 20])
redshift = f['Header'].attrs['redshift']
deltaF_hd = tau2deltaF(tau_hd, redshift, mean_F=None)
if len(np.shape(deltaF_hd)) == 2:
    deltaF_hd = deltaF_hd.reshape(int(np.sqrt(deltaF_hd.shape[0])), int(np.sqrt(deltaF_hd.shape[0])), deltaF_hd.shape[1])
print(tau_hd[20, 20])

# low-res maps
res = sys.argv[1] # "0.25" "0.5" "1.0"
if res == "1.0":
    deltaF_file = os.path.join(save_dir, 'noiseless_maps/map_TNG_true_1.0_z2.4.hdf5')
elif res == "0.5":
    deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.5_z2.4.hdf5')
elif res == "0.25":
    deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.25_z2.5_820_voxels.hdf5')
f = h5py.File(deltaF_file, 'r')
#deltaF = f['map'][:] # differs by 5%
#assert len(np.shape(deltaF)) == 3
tau = -np.log(f['flux'][:])
print(tau[20, 20, 20])
deltaF = tau2deltaF(tau, redshift, mean_F=None)
print(tau[20, 20, 20])
ngrid = tau.shape[-1] # low-resolution

# compute fourier modes 
L_hMpc = 205. # cMpc/h
k_hMpc, fk1d_hMpc = compute_fk1d(deltaF, L_hMpc)
k_hd_hMpc, fk1d_hd_hMpc = compute_fk1d(deltaF_hd, L_hMpc)

# figure 1: skewers; figure 2: fourier
plt.figure(1, figsize=(9, 7))
plt.figure(2, figsize=(9, 7))

# define bins for plotting skewer
bins_hd = np.linspace(0., L_hMpc, deltaF_hd.shape[2] + 1)
bins = np.linspace(0., L_hMpc, deltaF.shape[2] + 1)
binc_hd = (bins_hd[1:] + bins_hd[:-1])*.5
binc = (bins[1:]+bins[:-1])*.5

assert deltaF.shape[0] == deltaF.shape[1] 
assert deltaF_hd.shape[0] == deltaF_hd.shape[1]
n_skew = 1
#i_choice = (np.linspace(0, deltaF.shape[0]-1, n_skew)).astype(int)
i_choice = [150]
j_choice = i_choice
sum = 0
for i in i_choice:
    for j in j_choice:
        plt.figure(1)
        plt.plot(binc, deltaF[i, j, :], color=colors[sum%len(colors)], ls='-', label=f'low-resolution, {res}')
        plt.plot(binc_hd, deltaF_hd[i, j, :], color=colors[sum%len(colors)], ls='--', label='high-resolution, 0.05')

        plt.figure(2)
        plt.plot(k_hMpc, fk1d_hMpc[i, j, :]*np.sqrt(k_hMpc/np.pi), color=colors[sum%len(colors)], ls='-', label=f'low-resolution, {res}')
        plt.plot(k_hd_hMpc, fk1d_hd_hMpc[i, j, :]*np.sqrt(k_hd_hMpc/np.pi), color=colors[sum%len(colors)], ls='--', label='high-resolution, 0.05')

        plt.figure(3)
        plt.plot(k_hMpc, np.interp(k_hMpc, k_hd_hMpc, fk1d_hd_hMpc[i, j, :])/fk1d_hMpc[i, j, :], color=colors[sum%len(colors)], ls='-', label=f'high-to-low, {res}')
        sum += 1
plt.figure(1)
plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel("r [Mpc/h]")
plt.ylabel("deltaF(r)")
plt.xlim([0., L_hMpc])
#plt.ylim([np.sqrt(1.e-3), np.sqrt(.1)])
plt.savefig(f"flux_real_1d_res{res}.png")
plt.close()


plt.figure(2)
plt.legend()
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("FFT[deltaF(r)]")
#plt.xlim([0.03, 20.])
#plt.ylim([np.sqrt(1.e-3), np.sqrt(.1)])
plt.savefig(f"flux_fourier_1d_res{res}.png")
plt.close()

plt.figure(3)
R = L_hMpc/ngrid # Mpc/h
R_hd = L_hMpc/ngrid_hd # Mpc/h
window = np.sinc(k_hMpc*R/(2.*np.pi))
window_hd = np.sinc(k_hd_hMpc*R_hd/(2.*np.pi))
#sigma = R/(2*np.sqrt(2*np.log(2)))
#window = np.exp(-0.5 * (k * sigma)**2) * np.sinc(k * R/2/np.pi)
plt.plot(k_hMpc, 1./window, 'k--', label='1/(sinc(kR/2))')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("Ratio skewer")
plt.xlim([0.01, 10.])
plt.ylim([0.95, 1.1])
plt.savefig(f"flux_fourier_1d_ratio_high_low_res{res}.png")
plt.close()

k_hMpc, pk1d_hMpc = compute_pk1d(deltaF, L_hMpc)
pk1d_hMpc /= window**2
k_hd_hMpc, pk1d_hd_hMpc = compute_pk1d(deltaF_hd, L_hMpc)
pk1d_hd_hMpc /= window_hd**2
#k_hMpc, pk1d_hMpc = power_mahdi(tau, redshift, L_hMpc, cosmo)
#k_hd_hMpc, pk1d_hd_hMpc = power_mahdi(tau_hd, redshift, L_hMpc, cosmo)

sum = 0
plt.figure(4)
plt.plot(k_hMpc, pk1d_hMpc*(k_hMpc/np.pi), color=colors[sum%len(colors)], ls='-', label=f'low-resolution, {res}')
plt.plot(k_hd_hMpc, pk1d_hd_hMpc*(k_hd_hMpc/np.pi), color=colors[sum%len(colors)], ls='--', label='high-resolution, 0.05')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("k P(k)/pi")
plt.xlim([0.01, 10.])
plt.ylim([1.e-3, .1])
plt.savefig(f"power_1d_high_low_res{res}.png")
plt.close()

plt.figure(5)
plt.plot(k_hMpc, np.interp(k_hMpc, k_hd_hMpc, pk1d_hd_hMpc)/pk1d_hMpc, color=colors[sum%len(colors)], ls='-', label=f'high-to-low ({res})')
#plt.plot(k_hMpc, 1./window**2, 'k--', label='1/(sinc(kR/2))^2')
plt.legend()
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("k [h/Mpc]")
plt.ylabel("Ratio 1D")
plt.xlim([0.01, 10.])
plt.ylim([0.95, 1.1])
plt.savefig(f"power_1d_ratio_high_low_res{res}.png")
plt.close()
