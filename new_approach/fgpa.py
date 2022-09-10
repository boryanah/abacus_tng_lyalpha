# question: when we run out of RSD bs, what do we do w stuff that's too close? (not possible) too far (can't observe?)
# question should I match mean flux in real or redshift space
# check redshift formula
# figure out bias and beta (sherwood?)

#from dataclasses import field
import os
import gc
import sys
sys.path.append("..")

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tools import rsd_tau, density2tau, tau2deltaF
from compute_power import compute_pk1d, compute_pk3d

# sim info
ngrid = int(sys.argv[1]) #820 #410 #205
paste = sys.argv[2] # 'CIC' 'TSC'
sim_name = "TNG300"
fp_dm = 'fp'
want_rsd = True
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]

# load snaps and zs
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    Lbox = 205.
    N_part = 2500.**3.
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
    Lbox = 500.
snaps = snaps.astype(int)

# create a dictionary
snap2z_dict = {}
for i in range(len(zs)):
    snap2z_dict[snaps[i]] = zs[i]
rsd_str = "_rsd" if want_rsd else ""
paste_str = f"_{paste}" if paste == "TSC" else ""
redshift = snap2z_dict[snapshot]

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
data_dir = save_dir+"DM_Density_field/"

# load tau field
if paste == "CIC":
    tau = np.load(data_dir+f"tau{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
else:
    tau = np.load(data_dir+f"tau{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")

# TESTING!!!!!!!!!
tau /= (N_part/ngrid**3)

# convert to deltaF by rescaling
deltaF = tau2deltaF(tau, redshift, mean_F=None)
np.save(save_dir+f'noiseless_maps/deltaF{paste_str}_new_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy', deltaF)

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
np.savez(f"power1d_new_fgpa{paste_str}_ngrid{ngrid:d}.npz", p1d_hMpc=p1d_hMpc, k_hMpc=k_hMpc)

# params for the 3D power
n_k_bins = 20
n_mu_bins = 16
mu_want = [0., 0.33, 0.66, 1.]

# compute and save 3d power
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)
np.savez(f"power3d_new_fgpa{paste_str}_ngrid{ngrid:d}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)

