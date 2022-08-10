import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
fp_dm = "fp"
#fp_dm = "dm"
#want_cic = False
want_cic = True
want_rsd = True
rsd_str = "_rsd" if want_rsd else ""

z = 2.3
zs = [2.6, 2.4, 2.3]
snaps = [28, 29, 30]
snap_dic = {}
for i in range(len(zs)):
    snap_dic[zs[i]] = snaps[i]

f = h5py.File(save_dir+f"DM_Density_field/TNG_DM_z{z:.1f}.hdf5")
dens = f['DM']['dens'][:]
#print(list(f.keys()))
print("min dens, max dens, mean dens = ", dens.min(), dens.max(), np.mean(dens))
#dens = f['Gas'][:]
print("theirs = ", dens[:2, :2, :2])

plt.imshow(dens[:, :, 20], vmin = dens.min(), vmax=dens.max())
#plt.imshow(dens[:, :, 20], vmin = dens.min(), vmax=dens.max())
plt.colorbar()
plt.savefig(f"their_density_cic_z{z:.1f}_dm.png")
plt.close()

if want_cic:
    density = np.load(save_dir+f"DM_Density_field/density{rsd_str}_cic_ngrid_205_snap_{snap_dic[z]}_{fp_dm}.npy")
else:
    density = np.load(save_dir+f"DM_Density_field/density{rsd_str}_ngrid_205_snap_{snap_dic[z]}_{fp_dm}.npy")
density /= np.mean(density)
print("mine = ", density[:2, :2, :2])
plt.imshow(density[:, :, 20], vmin = dens.min(), vmax=dens.max())
plt.colorbar()
print("mine min max mean dens = ", density.min(), density.max(), np.mean(density))
if want_cic:
    plt.savefig(f"my_density{rsd_str}_cic_z{z:.1f}_{fp_dm}.png")
else:
    plt.savefig(f"my_density{rsd_str}_tsc_z{z:.1f}_{fp_dm}.png")
plt.close()

#diff = (density-dens)/np.std(dens)
diff = (density-dens)/dens
plt.imshow(diff[:, :, 20], vmin=-10, vmax=10)
plt.colorbar()
if want_cic:
    plt.savefig(f"difference{rsd_str}_cic_z{z:.1f}_{fp_dm}.png")
else:
    plt.savefig(f"difference{rsd_str}_tsc_z{z:.1f}_{fp_dm}.png")
plt.close()

print("median, mean, max, min diff = ", np.median(diff), np.mean(diff), diff.max(), diff.min())


