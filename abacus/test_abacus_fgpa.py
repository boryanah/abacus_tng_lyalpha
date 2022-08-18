import os
import gc
import sys
sys.path.append("..")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import asdf
from tools import tau2deltaF_mine
from compute_power import compute_pk1d, compute_pk3d
from nbodykit.lab import cosmology

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
save_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name

# simulation directory
Lbox = 2000. # Mpc/h
ngrid = 8000
ngrid_yz = 200
part_type = "AB"
paint_type = 'TSC' # 'CIC' # 'TSC'
z = 2.5

# load optical depth
tau = asdf.open(f'{save_dir}/fgpa/z{z:.3f}/maps_{paint_type}_Ndim{ngrid_yz:d}_{ngrid:d}_part{part_type}.asdf')['data']['OpticalDepth']
deltaF = tau2deltaF_mine(tau, z, mean_F=None)

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
np.savez(f"power1d_abacus_fgpa_ngrid{ngrid:d}.npz", p1d_hMpc=p1d_hMpc, k_hMpc=k_hMpc)
print("1d power = ", p1d_hMpc)

# plot power spectrum
plt.figure(2, figsize=(9, 7))
plt.plot(k_hMpc, p1d_hMpc*k_hMpc/np.pi, label='FGPA')

# define bins
n_k_bins = 20
n_mu_bins = 16
mu_want = [0., 0.33, 0.66, 1.]
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)
np.savez(f"power3d_abacus_fgpa_ngrid{ngrid:d}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)

no_nans = 0
for i in range(n_k_bins):
    if not np.any(np.isnan(mu[i])): no_nans = i
int_mu = []
for i in range(len(mu_want)):
    int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))
print(p3d_hMpc[:, int_mu[3]])

cosmo = cosmology.Planck15
P_L = cosmology.LinearPower(cosmo, z, transfer='EisensteinHu')

plt.figure(3, figsize=(9, 7))
plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]), color='violet')
plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc)[:, int_mu[1]]/P_L.__call__(k_hMpc[:, int_mu[1]]), color='cyan')
plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc)[:, int_mu[2]]/P_L.__call__(k_hMpc[:, int_mu[2]]), color='yellow')
plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]), color='red')
