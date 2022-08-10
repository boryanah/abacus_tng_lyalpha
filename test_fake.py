import numpy as np
import h5py
from fake_spectra import fluxstatistics as fs

save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
f = h5py.File(save_dir + '/spectra_z2.4/spectra_TNG_true_1.0_z2.4.hdf5','r')
z = f['Header'].attrs['redshift'][()]
print("actual redshift = ", z)
#z = 2.45
z = 2.4
tau =  f['tau/H/1/1215'][:]

# The boxsize in km/s
#vmax = cosmo.H(z).value*boxszie/((1+z)*cosmo.h)

mean_flux_desired = np.exp(-1.330e-3 * (1. + z)**4.094)
print("desired mean = ", mean_flux_desired)

scale = fs.mean_flux(tau, mean_flux_desired)
print("scale of bird = ", scale)

print("mahdi mean = ", np.mean(np.exp(-scale*tau)))

scale = 0.80826046
print("scale of buba = ", scale)
print("buba mean = ", np.mean(np.exp(-scale*tau)))
