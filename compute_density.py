import os
import gc

import numpy as np
import h5py

#paste = "TSC"
paste = "CIC"
if paste == "TSC":
    from tools import numba_tsc_3D
elif paste == "CIC": 
    from tools import numba_cic_3D

# path to simulation
basePaths = ['/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/', '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG_DM/output/']
n_chunks = [600, 75]

want_rsd = True
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
n_total = 2500**3
ngrid = 410 #820 #205 #512
Lbox = 205000. # ckpc/h
sim_name = "TNG300"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)


for z_int in z_ints:
    for j, fp_dm in enumerate(['fp', 'dm']):
        if fp_dm == "dm": continue

        print("z_int, fp_dm", z_int, fp_dm)
        n_chunk = n_chunks[j]
        basePath = basePaths[j]
        print("redshift = ", z_int, fp_dm)
        snapshot = z_dict[f"{z_int:.3f}"]
        z_exact = zs[np.where(snapshot == snaps)[0]]
        print("z exact = ", z_exact)

        if want_rsd:
            H_z = cosmo.H(z_exact).value
            print("H(z) = ", H_z)

        # TESTING
        """
        if paste == "CIC":
            density = np.load(data_dir+f"density_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
        else:
            density = np.load(data_dir+f"density_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
        """
        density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        vfield = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        for i in range(n_chunk): #2*n_chunk//3):#, n_chunk): #(n_chunk//2, n_chunk): #(n_chunk//2): # TESTING!!!!!!!!!!!!!!!!!
            print("chunk = ", i)#, end='\r')

            # read positions of DM particles
            fn = basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5'
            pos = h5py.File(fn, 'r')['PartType1']['Coordinates'][:]
            vel1d = h5py.File(fn, 'r')['PartType1']['Velocities'][:, 2]/np.sqrt(1.+z_exact)
            if want_rsd:
                pos[:, 2] += vel1d*(1+z_exact)/H_z * h * 1000. # kpc/h
                #del vel1d
                
            #n_file = pos.shape[0]
            #print("number of particles in chunk = ", n_file)
            if paste == "TSC":
                numba_tsc_3D(pos, density, Lbox)#, nthread=30)
                numba_tsc_3D(pos, vfield, Lbox, weights=vel1d)
            elif paste == "CIC":
                numba_cic_3D(pos, density, Lbox)
                numba_cic_3D(pos, vfield, Lbox, weights=vel1d)
            #print("obtained tsc")
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
