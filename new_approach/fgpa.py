# test the 1+delta idea
# voigt profile implement

#from dataclasses import field
import os
import gc
import sys
sys.path.append("..")

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tools import rsd_tau, density2tau, tau2deltaF, tau2deltaF_mine
from compute_power import compute_pk1d, compute_pk3d

# sim info
ngrid = int(sys.argv[1]) #820 #410 #205
paste = sys.argv[2] # 'CIC' 'TSC'
sim_name = "TNG300-3_DM"
fp_dm = 'fp'
want_rsd = True
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]

if "TNG300" in sim_name:
    Lbox = 205. # cMpc/h
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    if "-3" in sim_name:
        n_total = 625**3
    elif "-1" in sim_name:
        n_total = 2500**3
elif "MNTG" in sim_name:
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)

if "DM" in sim_name: 
    fp_dm = "dm"
else:
    fp_dm = "fp"

# create a dictionary
snap2z_dict = {}
for i in range(len(zs)):
    snap2z_dict[snaps[i]] = zs[i]
rsd_str = "_rsd" if want_rsd else ""
paste_str = f"_{paste}" if paste == "TSC" else ""
if ngrid == 675: fp_dm = 'dm'
redshift = snap2z_dict[snapshot]

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
data_dir = save_dir+"DM_Density_field/"

# load tau field
if paste == "CIC":
    tau = np.load(data_dir+f"tau{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
else:
    tau = np.load(data_dir+f"tau{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")

# normalize (doesn't matter cause of rescale)
tau /= (n_total/ngrid**3)

# convert to deltaF by rescaling
deltaF = tau2deltaF(tau, redshift, mean_F=None)
#deltaF = tau2deltaF_mine(tau, redshift, mean_F=None) # same
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

