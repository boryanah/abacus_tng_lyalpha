import os
import gc
import sys

import numpy as np
import h5py

ngrid = int(sys.argv[1]) # 410 820 205
paste = sys.argv[2] # "TSC" # "CIC"
if paste == "TSC":
    from tools import numba_tsc_3D
elif paste == "CIC": 
    from tools import numba_cic_3D

# path to simulation
#basePaths = ['/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/', '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG_DM/output/']
basePaths = ['/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/', '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n625TNG_DM/output/']

want_rsd = False 
rsd_str = "_rsd" if want_rsd else ""
if want_rsd:
    from astropy.cosmology import FlatLambdaCDM
    h = 0.6774
    cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

z_ints =  [2.44]# [2.58, 2.44, 2.32]
zs = [2.58, 2.44, 2.32]
snaps = [28, 29, 30]
z_dict = {}
for i in range(len(zs)):
    key = f"{zs[i]:.3f}"
    z_dict[key] = snaps[i]

data_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/DM_Density_field/"

# sim info
sim_name = "TNG300-3_DM"

if "TNG300" in sim_name:
    Lbox = 205000. # ckpc/h
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    if "-3" in sim_name:
        n_chunks = [600, 4]
        n_total = 625**3
    elif "-1" in sim_name:
        n_chunks = [600, 75]
        n_total = 2500**3
elif "MNTG" in sim_name:
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)

if "DM" in sim_name: 
    fp_dm = "dm"
    n_chunk = n_chunks[1]
    basePath = basePaths[1]
else:
    fp_dm = "fp"
    n_chunk = n_chunks[0]
    basePath = basePaths[0]

for z_int in z_ints:

    print("z_int, fp_dm", z_int, fp_dm)
    print("redshift = ", z_int, fp_dm)
    snapshot = z_dict[f"{z_int:.3f}"]
    z_exact = zs[np.where(snapshot == snaps)[0]]
    print("z exact = ", z_exact)

    if want_rsd:
        H_z = cosmo.H(z_exact).value
        print("H(z) = ", H_z)

    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    vfield = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    for i in range(n_chunk):
        print("chunk = ", i)#, end='\r')

        # read positions of DM particles
        fn = basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5'
        pos = h5py.File(fn, 'r')['PartType1']['Coordinates'][:]
        vel1d = h5py.File(fn, 'r')['PartType1']['Velocities'][:, 2]/np.sqrt(1.+z_exact)
        if want_rsd:
            pos[:, 2] += vel1d*(1+z_exact)/H_z * h * 1000. # kpc/h

        if paste == "TSC":
            numba_tsc_3D(pos, density, Lbox)#, nthread=30)
            numba_tsc_3D(pos, vfield, Lbox, weights=vel1d)
        elif paste == "CIC":
            numba_cic_3D(pos, density, Lbox)
            numba_cic_3D(pos, vfield, Lbox, weights=vel1d)
        #print("sum of density entries = ", density.sum())

        del pos, vel1d

    if paste == "CIC":
        np.save(data_dir+f"density{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", density)
        np.save(data_dir+f"vfield{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", vfield)
    else:
        np.save(data_dir+f"density{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", density)
        np.save(data_dir+f"vfield{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", vfield)
    del density, vfield
    gc.collect()
