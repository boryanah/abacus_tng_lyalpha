# todo: try different density measurements and then try also to do it on a rolling basis (for abacus)
import os
import gc
import sys
sys.path.append("../")

import numpy as np
import h5py

# choices
ngrid = int(sys.argv[1]) # 410 820 205
paste = sys.argv[2] # "TSC" # "CIC"
z_ints =  [2.44]# [2.58, 2.44, 2.32]
sim_name = "TNG300"
want_rsd = True
#gamma = 1.46
gamma = 1.66
dens_exp = (2.-0.7*(gamma - 1.))

if paste == "TSC":
    from tools import numba_tsc_3D
elif paste == "CIC": 
    from tools import numba_cic_3D

# path to simulation
basePaths = ['/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/', '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG_DM/output/']
n_chunks = [600, 75]

# do you want to implement RSD
rsd_str = "_rsd" if want_rsd else ""
paste_str = f"_{paste}" if paste == "TSC" else ""
if want_rsd:
    from astropy.cosmology import FlatLambdaCDM
    h = 0.6774
    cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

# what is the corresponding snapshot
zs = [2.58, 2.44, 2.32]
snaps = [28, 29, 30]
z_dict = {}
for i in range(len(zs)):
    key = f"{zs[i]:.3f}"
    z_dict[key] = snaps[i]

# where did we save the DM field
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
data_dir = save_dir+"DM_Density_field/"

# sim info
n_total = 2500**3
Lbox = 205000. # ckpc/h
cell_size = Lbox/ngrid # ckpc/h
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)

for z_int in z_ints:
    for j, fp_dm in enumerate(['fp', 'dm']):
        if fp_dm == "dm": continue

        # currently we're at
        print("z_int, fp_dm", z_int, fp_dm)
        n_chunk = n_chunks[j]
        basePath = basePaths[j]
        print("redshift = ", z_int, fp_dm)
        snapshot = z_dict[f"{z_int:.3f}"]
        z_exact = zs[np.where(snapshot == snaps)[0]]
        print("z exact = ", z_exact)

        # load DM density maps in real space
        if paste == "CIC":
            density = np.load(data_dir+f"density_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
        else:
            density = np.load(data_dir+f"density_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")

        # density is number of particles per cell
        mean_density = np.mean(density)
        print("mean density = ", mean_density, n_total/ngrid**3)
        density /= mean_density # 1 + delta
        density_noise = np.load(save_dir + f'noiseless_maps/density_noise{paste_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
        one_plus_delta_lognorm = density_noise / density
        one_plus_delta_lognorm[density == 0.] = 0.
        density_product = density**(dens_exp-1.)*one_plus_delta_lognorm**dens_exp
        del density, one_plus_delta_lognorm

        # hubble at redshift
        if want_rsd:
            H_z = cosmo.H(z_exact).value
            print("H(z) = ", H_z)

        # initialize field
        tau = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        for i in range(n_chunk):
            print("chunk = ", i, n_chunk)#, end='\r')

            # read positions of DM particles
            fn = basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5'
            pos = h5py.File(fn, 'r')['PartType1']['Coordinates'][:] # ckpc/h
            pos_ijk = (pos / cell_size).astype(int) # NGP cause lazy (could interpret but that's slow too) #GroupShear = interpn(cells, mark, GroupPos, bounds_error=False, fill_value=None); cells = (np.arange(ngrid)+0.5, np.arange(ngrid)+0.5, np.arange(ngrid)+0.5)
            pos_ijk %= ngrid 
            weight = (density_product)[pos_ijk[:, 0], pos_ijk[:, 1], pos_ijk[:, 2]]
            vel1d = h5py.File(fn, 'r')['PartType1']['Velocities'][:, 2]/np.sqrt(1.+z_exact)

            # perturb position
            if want_rsd:
                pos[:, 2] += vel1d*(1+z_exact)/H_z * h * 1000. # kpc/h
                
            # add to the gridded densities
            if paste == "TSC":
                numba_tsc_3D(pos, tau, Lbox, weights=weight)
            elif paste == "CIC":
                numba_cic_3D(pos, tau, Lbox, weights=weight)
            del pos, pos_ijk, weight, vel1d
            
        # save the field
        if paste == "CIC":
            np.save(data_dir+f"tau{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", tau)
        else:
            np.save(data_dir+f"tau{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy", tau)
        del tau, density_product
        gc.collect()
