# question: when we run out of RSD bs, what do we do w stuff that's too close? (not possible) too far (can't observe?)
# question should I match mean flux in real or redshift space
# check redshift formula
# figure out bias and beta (sherwood?)

#from dataclasses import field
import os
import gc
import sys

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tools import rsd_tau, density2tau, tau2deltaF, rsd_tau_Voigt

from pmesh.pm import ParticleMesh
from nbodykit.mockmaker import gaussian_real_fields
from compute_power import compute_pk1d, compute_pk3d

def get_noise(k, n=0.732, k1=0.0341):
    return 1./(1 + (k/k1)**n)

def Pk_extra(k):
    return get_noise(k)

# choices
ngrid = int(sys.argv[1]) #820 #410 #205 # 675 (TNG300-3 DM)
paste = sys.argv[2] # 'CIC' 'TSC'
want_rsd = False
fp_dm = 'fp'
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]

# sim info
Lbox = 205. # cMpc/h
h = 0.6774
sim_name = "TNG300"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)
snap2z_dict = {}
for i in range(len(zs)):
    snap2z_dict[snaps[i]] = zs[i]

# processing
if ngrid == 675: fp_dm = 'dm'
rsd_str = "_rsd" if want_rsd else ""
paste_str = f"_{paste}" if paste == "TSC" else ""
redshift = snap2z_dict[snapshot]
cell_size = Lbox/ngrid
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
E_z = H_z/h # 100 times E_z
print("H(z) = ", H_z)

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"

# load DM density maps
data_dir = save_dir+"DM_Density_field/"
if paste == "CIC":
    density = np.load(data_dir+f"density{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
    vfield = np.load(data_dir+f"vfield{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
else:
    density = np.load(data_dir+f"density{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
    vfield = np.load(data_dir+f"vfield{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")

# density is number of particles per cell; vfield is same thing but weighted by the 1d velocity
vfield[density != 0.] /= density[density != 0.] # km/s, peculiar
density /= np.mean(density)

# needs higher resolution along the line of sight (could do it for each line of sight)
print("before the PM stuff")

# generate noise in 3D (should be done in 1D)
"""
# create empty mesh, generate white noise (variance unity), multiply by P(k)**0.5 
pm = ParticleMesh(Nmesh=[ngrid, ngrid, ngrid], BoxSize=Lbox)
delta_extra, _ = gaussian_real_fields(pm, Pk_extra, seed=42)
delta_extra = np.asarray(delta_extra)
print(delta_extra.dtype, delta_extra.shape)
"""

# generate noise in 1D (slow via nbodykit)
"""
delta_extra = np.zeros((ngrid, ngrid, ngrid))
for i in range(ngrid):
    for j in range(ngrid):
        # create empty mesh, generate white noise (variance unity), multiply by P(k)**0.5 
        pm = ParticleMesh(Nmesh=[ngrid, 1], BoxSize=Lbox)
        delta1d_extra, _ = gaussian_real_fields(pm, Pk_extra, seed=42+j*ngrid+i)
        delta1d_extra = np.asarray(delta1d_extra)
        #print(delta1d_extra.dtype, delta1d_extra.shape)
        delta_extra[i, j, :] = delta1d_extra[:, 0]
del pm; gc.collect()
"""

"""
def generate_noise_rfft(Ndim_x, Ndim_y, Ndim_z, cell_size):
    # noise parameters
    if Ndim_z % 2 == 0:
        Ndim_rfft = int(Ndim_z/2+1)
    else:
        print("not a good idea to be odd")
        Ndim_rfft = int((Ndim_z+1)/2)
    k_hMpc = np.fft.rfftfreq(Ndim_z)*2*np.pi/cell_size
    P_hMpc = Pk_extra(k_hMpc)

    # generate random Fourier modes
    gen = np.random.RandomState(42)
    modes = np.empty([Ndim_x, Ndim_y, Ndim_rfft], dtype=np.complex64)
    modes[:].real = np.reshape(gen.normal(size=Ndim_x*Ndim_y*Ndim_rfft), [Ndim_x, Ndim_y, Ndim_rfft])
    modes[:].imag = np.reshape(gen.normal(size=Ndim_x*Ndim_y*Ndim_rfft), [Ndim_x, Ndim_y, Ndim_rfft])

    # normalize to desired power (and enforce real for i=0, i=Ndim_rfft-1)
    modes[:, :, 0] = modes[:, :, 0].real * np.sqrt(P_hMpc[0])
    modes[:, :, -1] = modes[:, :, -1].real * np.sqrt(P_hMpc[-1])
    modes[:, :, 1:-1] *= np.sqrt(0.5*P_hMpc[1:-1])

    # inverse FFT to get (normalized) delta field
    delta_extra = np.fft.irfft(modes, n=Ndim_z)# * np.sqrt(Ndim_z/cell_size)

    # to ensure unit variance
    delta_extra /= np.std(delta_extra)
    return delta_extra

# generate noise in 1D (faster)
delta_extra = generate_noise_rfft(ngrid, ngrid, ngrid, cell_size)

# to ensure proper normalization (should be done in nbodykit)
#delta_extra /= Lbox**1.5

# to ensure unit variance
delta_extra /= np.std(delta_extra)
sigma_extra = np.std(delta_extra)
mean_extra = np.mean(delta_extra)

# save tau
print("saving density")
dens_file = os.path.join(save_dir, f'noiseless_maps/density_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
print("before = ", density.mean())

print("mean, std of extra field = ", mean_extra, sigma_extra)
delta_lognorm = np.exp(delta_extra - sigma_extra**2/2.) - 1.
print("mean, std of LN field = ", delta_lognorm.mean(), delta_lognorm.std())
density *= (1 + delta_lognorm)
del delta_lognorm, delta_extra; gc.collect()
print("after = ", density.mean())

# save tau
print("saving density noise")
dens_file = os.path.join(save_dir, f'noiseless_maps/density_noise{paste_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(dens_file, density)

# generate tau
tau_0 = 1.
gamma = 1.46
power = (2.-0.7*(gamma - 1.))
#power = 1.6 
tau = density2tau(density, tau_0, power)
#del density; gc.collect()
"""

# save tau
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_real{paste_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
#np.save(tau_file, tau)
tau = np.load(tau_file)

# we have ngrid^3 cells
bins = np.linspace(0., ngrid, ngrid+1) # should be 0 1 2 ... 205
binc = (bins[1:] + bins[:-1]) * .5
binc *= cell_size # cMpc/h
print(tau.shape, vfield.shape, binc[0], binc[-2:], Lbox, tau.mean(), vfield.mean())

# apply RSD to tau
#tau = rsd_tau(tau, vfield, binc, E_z, redshift, Lbox)
tau = rsd_tau_Voigt(tau, density, vfield, binc, E_z, redshift, Lbox)
del density; gc.collect()
print("rsded")

# save tau
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_redshift{paste_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')

# convert to deltaF by rescaling
deltaF = tau2deltaF(tau, redshift, mean_F=None)

# save deltaF
deltaF_file = os.path.join(save_dir, f'noiseless_maps/deltaF{paste_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
#deltaF = np.load(deltaF_file)
np.save(deltaF_file, deltaF)
print("saved")

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
np.savez(f"power1d_fgpa{paste_str}_ngrid{ngrid:d}.npz", p1d_hMpc=p1d_hMpc, k_hMpc=k_hMpc)

# params for the 3D power
n_k_bins = 20
n_mu_bins = 16

# compute and save 3d power
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)
np.savez(f"power3d_fgpa{paste_str}_ngrid{ngrid:d}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)
