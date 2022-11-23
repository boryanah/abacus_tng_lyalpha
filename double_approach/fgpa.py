#from dataclasses import field
import os
import gc
import time
import sys
sys.path.append("../")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf
from scipy.optimize import minimize

import h5py
from astropy.cosmology import FlatLambdaCDM

from tools import rsd_tau, tau2deltaF, get_mean_flux
from compute_power import compute_pk1d, compute_pk3d

# miscellaneous constants
gamma = 1.46

def get_k_hMpc_real(Ndim, L_hMpc):
    """ get the fourier modes """
    # get frequencies from numpy.fft
    k_hMpc = np.fft.rfftfreq(Ndim)
    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    k_hMpc *= (2.0*np.pi) * Ndim / L_hMpc
    return k_hMpc

def get_mean_std(F, L_hMpc):
    """ given flux, compute mean and variance by downsampling"""
    # match dimensions
    Ndim_los = F.shape[2]
    Ndim_tr = F.shape[0]
    assert F.shape[1] == Ndim_tr
    #print("dimensions = ", Ndim_tr, Ndim_los)

    # filter out small scales
    k_rfft = get_k_hMpc_real(Ndim_los, L_hMpc)
    choice = np.abs(k_rfft) > 1.
    Ndim_down = 31
    #Ndim_down = Ndim_tr # TESTING
    step = Ndim_tr//Ndim_down
    Ndim_down = Ndim_tr//step
    i_down = np.arange(Ndim_tr)[::step]
    j_down = np.arange(Ndim_tr)[::step]
    F_down = np.zeros((len(i_down), len(j_down), Ndim_los), dtype=np.float32)
    #sum = 0
    for i, i_ind_down in enumerate(i_down):
        for j, j_ind_down in enumerate(j_down):
            #print(i_ind_down, j_ind_down)
            f = F[i_ind_down, j_ind_down, :]
            f_fft = np.fft.rfft(f)
            #sum += (f_fft[0].real/Ndim_los)
            f_fft[choice] = 0.
            F_down[i, j, :] = np.fft.irfft(f_fft, n=Ndim_los)
    #print("mech mean = ", sum/(len(i_down)*len(j_down))) # matches
    std = np.std(F_down)
    mean = np.mean(F_down)
    return mean, std


# user choices
ngrid = int(sys.argv[1]) #820 #410 #205 #675, 625 (TNG300-3 DM)
paste = sys.argv[2] # 'CIC' 'TSC'
want_rsd = False # initial density field in real or rsd space (real is physical)
fp_dm = 'fp'
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]
want_voigt = False
voigt_str = "_voigt" if want_voigt else ""

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

# useful parameters
if ngrid == 675 or ngrid == 625: fp_dm = 'dm'
rsd_str = "_rsd" if want_rsd else ""
paste_str = f"_{paste}" if paste == "TSC" else ""
redshift = snap2z_dict[snapshot]
cell_size = Lbox/ngrid
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
E_z = H_z/h # 100 times E_z
print("H(z) = ", H_z)
print("redshift = ", redshift)

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"

# initialize figures
fig, ax = plt.subplots(figsize=(11, 10), ncols=1, nrows=1)
n_bins = 1001

# for plotting stuff
def plot_pdf(F, n_bins, label):
    bins = np.linspace(0., 1., n_bins)
    binc = (bins[1:]+bins[:-1])*.5
    hist, edges = np.histogram(F, density=True, bins=bins)
    ax.plot(binc, hist, label=label)
    return hist, binc

# load TNG forest (true) data
#res = "0.05"
res = "0.25"
fn = f"TNG_mean_std_res{res}.npy"
fn_pdf = f"TNG_hist_res{res}.npz"
if os.path.exists(fn) and os.path.exists(fn_pdf):
    # load mean and std
    mean, std = np.load(fn)
    
    # load pdf
    data = np.load(fn_pdf)
    hist, binc = data['hist'], data['binc']
    ax.plot(binc, hist, label=f"TNG, {res} res")
else:
    # load flux, optical depth and delta flux
    if res == "0.25":
        deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.25_z2.5_820_voxels.hdf5')
        deltaF = h5py.File(deltaF_file, 'r')['map'][:]
        mean_F = get_mean_flux(redshift)
        F = (1.+deltaF)*mean_F
        print("min F = ", np.min(F))
        tau = -np.log(F)
        print("min tau = ", np.min(tau[~np.isnan(tau)]))
        tau[F <= 0.] = np.min(tau[~np.isnan(tau)])
    elif res == "0.05":
        tau_file = os.path.join(save_dir,'spectra_z2.4/spectra_TNG_true_1.0_z2.4.hdf5')
        f = h5py.File(tau_file, 'r')
        tau = gf(f['tau/H/1/1215'][:], 1, mode='wrap')
        redshift = f['Header'].attrs['redshift']
        print("redshift in file = ", redshift)
        deltaF, F = tau2deltaF(tau, redshift, mean_F=None, return_flux=True)

    # compute mean and std of flux
    mean, std = get_mean_std(F, Lbox)
    np.save(fn, np.array([mean, std]))
    print("mean, std = ", mean, std)

    # compute pdf
    hist, binc = plot_pdf(F, n_bins, f"TNG, {res} res")
    np.savez(fn, hist=hist, binc=binc)
    del tau, deltaF, F; gc.collect()

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

def generate_noise_rfft(Ndim_x, Ndim_y, Ndim_z, cell_size):
    """ generate Gaussian random noise noise (checked against nbodykit) """

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

    # to ensure proper normalization
    #delta_extra /= Lbox**0.5

    # to ensure unit variance
    delta_extra /= np.std(delta_extra)
    return delta_extra

# load/save gaussian small-scale noise field
dextra_file = os.path.join(save_dir, f'noiseless_maps/delta_extra_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
if os.path.exists(dextra_file):
    delta_extra = np.load(dextra_file)
else:
    # generate noise in 1D
    delta_extra = generate_noise_rfft(ngrid, ngrid, ngrid, cell_size)

    # mean and variance
    sigma_extra = np.std(delta_extra)
    mean_extra = np.mean(delta_extra)
    print("mean, std of extra field = ", mean_extra, sigma_extra)
    np.save(dextra_file, delta_extra)

# midpoint of each cell along LOS
rsd_bins = np.linspace(0., ngrid, ngrid+1)
rsd_binc = (rsd_bins[1:] + rsd_bins[:-1]) * .5
rsd_binc *= cell_size # cMpc/h

def down(field):
    """ downsample fields"""
    ngrid_down = 31
    #ngrid_down = ngrid # TESTING
    step = ngrid//ngrid_down
    #ngrid_down = ngrid//step
    #i_down = np.arange(ngrid)[::step]
    #j_down = np.arange(ngrid)[::step]
    #field = field[i_down, j_down, :]
    field = field[::step, ::step, :]
    print("downsampled to = ", field.shape)
    return field

# downsample fields for faster computation
delta_extra_down = down(delta_extra)
density_down = down(density)
vfield_down = down(vfield)

def get_F(x, density, vfield, delta_extra, Lbox, gamma):
    """ compute flux given the fields """
    # read current guess
    scale, sigma_extra = x
    print("scale, sigma_extra = ", scale, sigma_extra)

    # let's change the variance
    delta_extra_new = delta_extra*sigma_extra
    #print("std of extra field = ", np.std(delta_extra_new))
    
    # create lognormal field
    delta_lognorm = np.exp(delta_extra_new - sigma_extra**2/2.) - 1.
    #print("mean, std of lognormal field = ", delta_lognorm.mean(), delta_lognorm.std())

    # add to density
    density_new = density*(1 + delta_lognorm)
    del delta_lognorm, delta_extra_new; gc.collect()
    #print("mean, std of combined density = ", density_new.mean(), density_new.std())

    # generate tau in real and rsd space
    tau_0 = scale
    power = (2.-0.7*(gamma - 1.)) # 1.6 
    tau = tau_0*density_new**power
    tau = rsd_tau(tau, vfield, rsd_binc, E_z, redshift, Lbox)
    F = np.exp(-tau)
    return F

def get_chi2(x, density, vfield, delta_extra, Lbox, gamma, mean, std):
    """ compute chi2 for the given parameters """
    # compute chi2
    F = get_F(x, density, vfield, delta_extra, Lbox, gamma)
    mean_new, std_new = get_mean_std(F, Lbox)
    chi2 = (mean_new-mean)**2 + (std_new-std)**2
    print("chi2 = ", chi2)
    return chi2

# minimize unless you know the parameters already
given = True
if given:
    x = np.array([0.43580632, 1.14421267])
else:
    x0 = np.array([1., 1.])
    res = minimize(get_chi2, x0, args=(density_down, vfield_down, delta_extra_down, Lbox, gamma, mean, std,), method='powell')#'BFGS')
    x = res['x']

# for these parameters, compute F and deltaF
F = get_F(x, density, vfield, delta_extra, Lbox, gamma)
deltaF = F/mean - 1.

# sanity check that minimization worked
mean_bf, std_bf = get_mean_std(F, Lbox)
print("mean and std of F = ", mean_bf, std_bf, mean, std)

# plot the PDF
hist, binc = plot_pdf(F.flatten(), n_bins, f"FGPA")
ax.legend()
ax.set_ylabel('PDF')
ax.set_xlabel('Flux')
ax.set_yscale('log')
plt.savefig("pdf_best.png")
plt.show()

# save the pdf
fn_pdf = f"FGPA_hist_best.npz"
np.savez(fn_pdf, hist=hist, binc=binc)

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
np.savez(f"power1d_double_fgpa{paste_str}{voigt_str}_ngrid{ngrid:d}.npz", p1d_hMpc=p1d_hMpc, k_hMpc=k_hMpc)

# params for the 3D power
n_k_bins = 20
n_mu_bins = 16

# compute and save 3d power
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)
np.savez(f"power3d_double_fgpa{paste_str}{voigt_str}_ngrid{ngrid:d}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)
