import os
import gc
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf

import h5py
import numpy as np
import scipy.stats as spt

from nbodykit.lab import cosmology
from tools import tau2deltaF, get_mean_flux
from fake_spectra import fluxstatistics as fs

def func(a, tau, target):
    res = np.abs(np.mean(np.exp(-a*tau))-target)
    print(res, a)
    return res

def power_mahdi(tau, redshift, L_hMpc, cosmo, mean_flux=None):

    if len(np.shape(tau)) == 3:
        # compute raw P3D power (no binning)
        (n_xy, n_xy, n_z) = np.shape(tau)
        nspec = n_xy**2
    elif len(np.shape(tau)) == 2:
        (nspec, n_z) = np.shape(tau)
        n_xy = int(np.sqrt(nspec))
    tau = tau.reshape(nspec, n_z)

    vmax = cosmo.H(redshift).value*L_hMpc/((1+redshift)*cosmo.h)
    # spectral resolution
    spec_res = vmax/tau.shape[1]
    print(tau.shape[1])
    if mean_flux is None:
        mean_flux = get_mean_flux(redshift)
    k_hres, pk_hres = fs.flux_power(tau, vmax=vmax, spec_res=spec_res, mean_flux_desired=mean_flux)
    # s/km to h/cMpc
    k_hres *= cosmo.H(redshift).value*cosmo.h/(1+redshift)
    pk_hres /= cosmo.H(redshift).value*cosmo.h/(1+redshift)
    return k_hres, pk_hres


def _measure_raw_p3d(delta_flux):
    """Get Fourier modes and compute raw P3D"""

    # normalize FFT with number cells
    norm_fac = 1.0 / delta_flux.size

    # get Fourier modes from skewers grid
    fourier_modes = np.fft.fftn(delta_flux) * norm_fac
    print('got Fourier modes')

    # get raw power
    return (np.abs(fourier_modes) ** 2).flatten()

def k_mu_box(L_hMpc, n_xy, n_z):
    # cell width in (x,y) directions (Mpc/h)
    d_xy = L_hMpc / n_xy
    k_xy = np.fft.fftfreq(n_xy, d=d_xy) * 2. * np.pi

    # cell width in z direction (Mpc/h)
    d_z = L_hMpc / n_z
    k_z = np.fft.fftfreq(n_z, d=d_z) * 2. * np.pi

    # h/Mpc
    x = k_xy[:, np.newaxis, np.newaxis]
    y = k_xy[np.newaxis, :, np.newaxis]
    z = k_z[np.newaxis, np.newaxis, :]
    k_box = np.sqrt(x**2 + y**2 + z**2)

    # construct mu in two steps, without NaN warnings
    mu_box = z/np.ones_like(k_box)
    mu_box[k_box > 0.] /= k_box[k_box > 0.]
    mu_box[k_box == 0.] = 0.#np.nan
    return k_box, mu_box

def compute_pk3d(delta_F, L_hMpc, n_k_bins, n_mu_bins, k_hMpc_max=20.):
    """Actually measure P3D from skewers grid (in h/Mpc units)"""

    if len(np.shape(delta_F)) == 3:
        # compute raw P3D power (no binning)
        (n_xy, n_xy, n_z) = np.shape(delta_F)
    elif len(np.shape(delta_F)) == 2:
        (nspec, n_z) = np.shape(delta_F)
        n_xy = int(np.sqrt(nspec))
    
    delta_F = delta_F.reshape(n_xy, n_xy, n_z)
    raw_p3d = _measure_raw_p3d(delta_F)
    #raw_p3d = raw_p3d.reshape(n_xy, n_xy, n_z)

    # this stores *all* Fourier wavenumbers in the box (no binning)
    k_box, mu_box = k_mu_box(L_hMpc, n_xy, n_z)
    k_box = k_box.flatten()
    mu_box = mu_box.flatten()

    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_hMpc_max)

    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999*np.min(k_box[k_box > 0.]))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)

    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
    
    # get rid of k=0, mu=0 mode
    k_box = k_box[1:]
    mu_box = mu_box[1:]
    raw_p3d = raw_p3d[1:]

    # compute bin averages
    binned_p3d = spt.binned_statistic_2d(k_box, mu_box, raw_p3d, statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned power')
    binned_counts = spt.binned_statistic_2d(k_box,mu_box,raw_p3d,
                                            statistic='count', bins=[k_bin_edges,mu_bin_edges])[0]
    print('got bin counts')
    
    # free memory in raw_p3d
    del raw_p3d
    gc.collect()

    # compute binned values for (k,mu)
    binned_k = spt.binned_statistic_2d(k_box, mu_box, k_box,
                                       statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned k')
    binned_mu = spt.binned_statistic_2d(k_box, mu_box, mu_box,
                                        statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned mu')

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    k_hMpc = binned_k
    mu = binned_mu
    counts = binned_counts

    return k_hMpc, mu, p3d_hMpc, counts

def compute_pk3d_fourier(dF_fft, kperp_hMpc, klos_hMpc, nperp, nlos, L_hMpc, n_k_bins, n_mu_bins, k_hMpc_max=20.):
    """Actually measure P3D from skewers grid (in h/Mpc units)"""

    # automize
    dF_fft /= nperp**2*nlos # not sure because need to think but I think so
    raw_p3d = (np.abs(dF_fft)**2).flatten()

    # h/Mpc
    kx = kperp_hMpc[:, np.newaxis, np.newaxis]
    ky = kperp_hMpc[np.newaxis, :, np.newaxis]
    kz = klos_hMpc[np.newaxis, np.newaxis, :]
    k_box = np.sqrt(kx**2 + ky**2 + kz**2)

    # construct mu in two steps, without NaN warnings
    mu_box = kz/np.ones_like(k_box)
    mu_box[k_box > 0.] /= k_box[k_box > 0.]
    mu_box[k_box == 0.] = 0.

    # flatten arrays
    k_box = k_box.flatten()
    mu_box = mu_box.flatten()

    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_hMpc_max)

    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999*np.min(k_box[k_box > 0.]))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)

    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    # get rid of k=0, mu=0 mode
    k_box = k_box[1:]
    mu_box = mu_box[1:]
    raw_p3d = raw_p3d[1:]    

    # compute bin averages
    binned_p3d = spt.binned_statistic_2d(k_box, mu_box, raw_p3d, statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned power')
    binned_counts = spt.binned_statistic_2d(k_box,mu_box,raw_p3d,
                                            statistic='count', bins=[k_bin_edges,mu_bin_edges])[0]
    print('got bin counts')

    # compute binned values for (k,mu)
    binned_k = spt.binned_statistic_2d(k_box, mu_box, k_box,
                                       statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned k')
    binned_mu = spt.binned_statistic_2d(k_box, mu_box, mu_box,
                                        statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned mu')

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    k_hMpc = binned_k
    mu = binned_mu
    counts = binned_counts

    return k_hMpc, mu, p3d_hMpc, counts

def compute_fk1d(delta_F, L_hMpc):
    # get dimensions of array
    if len(np.shape(delta_F)) == 3:
        (nx, ny, npix) = np.shape(delta_F)
        nspec = nx*ny
    elif len(np.shape(delta_F)) == 2:
        (nspec, npix) = np.shape(delta_F)
        nx = int(np.sqrt(nspec))
        ny = nx
    delta_F = delta_F.reshape(nspec, npix)
    nk = npix//2 + 1 if npix%2 == 0 else (npix + 1)//2

    # get 1D Fourier modes for each skewer
    f1d_hMpc = np.fft.rfft(delta_F, axis=1)
    f1d_hMpc *= np.sqrt(L_hMpc) / npix # in units of sqrt power

    # get frequencies from numpy.fft
    kp = np.fft.rfftfreq(npix)

    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    kp_hMpc = kp * (2.0*np.pi) * npix / L_hMpc    

    # reshape
    f1d_hMpc = f1d_hMpc.reshape(nx, ny, nk)
    f1d_hMpc = np.abs(f1d_hMpc)

    return kp_hMpc, f1d_hMpc

def compute_pk1d(delta_F, L_hMpc):
    # get dimensions of array
    if len(np.shape(delta_F)) == 3:
        (nx, ny, npix) = np.shape(delta_F)
        nspec = nx*ny
    elif len(np.shape(delta_F)) == 2:
        (nspec, npix) = np.shape(delta_F)
    delta_F = delta_F.reshape(nspec, npix)

    # get 1D Fourier modes for each skewer
    fourier = np.fft.rfft(delta_F, axis=1)
    print("fouriered")

    # compute amplitude of Fourier modes
    power_skewer = np.abs(fourier)**2

    # compute mean of power in all spectra
    mean_power = np.sum(power_skewer, axis=0)/nspec
    assert np.shape(mean_power) == (npix//2+1,), 'wrong dimensions in p1d'
    print("computed mean power")

    # normalize power spectrum using cosmology convention
    # white noise power should be P=sigma^2*dx
    p1d_hMpc = mean_power * L_hMpc / npix**2

    # get frequencies from numpy.fft
    kp = np.fft.rfftfreq(npix)

    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    kp_hMpc = kp * (2.0*np.pi) * npix / L_hMpc    

    return kp_hMpc, p1d_hMpc

def compute_pk1d_fourier(dF_fft, kperp_hMpc, klos_hMpc, nperp, nlos, L_hMpc):
    #nperp = len(klos_hMpc) # TESTING
    #nlos = len(klos_hMpc) # TESTING

    # get dimensions of array
    if len(np.shape(dF_fft)) == 3:
        (nx, ny, npix) = np.shape(dF_fft)
        nspec = nx*ny
    elif len(np.shape(dF_fft)) == 2:
        (nspec, npix) = np.shape(dF_fft)
    dF_fft = dF_fft.reshape(nspec, npix)
    dF_fft /= nperp # me

    # compute amplitude of Fourier modes
    power_skewer = np.abs(dF_fft)**2

    # compute mean of power in all spectra
    mean_power = np.sum(power_skewer, axis=0)/nspec # og
    #mean_power = power_skewer[:, 0]/nspec
    #mean_power = np.sum(power_skewer, axis=0)/205**2 # me
    print(np.shape(mean_power))
    assert np.shape(mean_power) == (npix,), 'wrong dimensions in p1d'
    print("computed mean power")

    # normalize power spectrum using cosmology convention
    # white noise power should be P=sigma^2*dx
    #p1d_hMpc = mean_power * L_hMpc / npix**2 # og
    p1d_hMpc = mean_power * L_hMpc / nlos**2 # me
    p1d_hMpc /= (nlos/npix)**2. # me tva uj baca
    #p1d_hMpc *= (nlos/npix)**2.

    # p/205^2 (65/205)^2 /205^2

    # get frequencies from numpy.fft
    #klos_hMpc = np.fft.fftfreq(npix)

    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    #klos_hMpc *= (2.0*np.pi) * npix / L_hMpc    
    #klos_hMpc *= (nlos/npix)

    return klos_hMpc, p1d_hMpc

def compute_pk1d_fourier_old(dF_fft, kperp_hMpc, klos_hMpc, nperp, nlos, L_hMpc):

    # set minimum k to make sure we cover fundamental mode
    n_k_bins = 15# len(klos_hMpc)
    lnk_min = np.log(0.9999*np.min(klos_hMpc[klos_hMpc > 0.]))
    lnk_max = np.log(np.max(klos_hMpc))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)

    # automize
    dF_fft /= nperp**2*nlos # not sure because need to think but I think so
    raw_p3d = (np.abs(dF_fft)**2)

    # h/Mpc
    kx = kperp_hMpc[:, np.newaxis, np.newaxis]
    ky = kperp_hMpc[np.newaxis, :, np.newaxis]
    kz = klos_hMpc[np.newaxis, np.newaxis, :]
    k_box = np.sqrt(kx**2 + ky**2 + kz**2)

    # construct mu in two steps, without NaN warnings
    mu_box = kz/np.ones_like(k_box)
    mu_box[k_box > 0.] /= k_box[k_box > 0.]
    mu_box[k_box == 0.] = 0.

    # flatten arrays
    k_box = k_box.flatten()
    mu_box = mu_box.flatten()
    raw_p3d = raw_p3d.flatten()

    # discard zeroth mode
    k_box = k_box[1:]
    mu_box = mu_box[1:]
    raw_p3d = raw_p3d[1:]

    # select the parallel modes
    mask = mu_box == 1.
    print("wait what about now", np.sum(mask))

    
    if True:
        #print(raw_p3d[mask], k_box[mask], klos_hMpc, raw_p3d[mask].shape, klos_hMpc[klos_hMpc > 0.].shape); quit()
        return  k_box[mask], raw_p3d[mask]
    

    binned_p1d, _ = np.histogram(k_box[mask], bins=k_bin_edges, weights=raw_p3d[mask])
    counts_p1d, _ = np.histogram(k_box[mask], bins=k_bin_edges)
    binned_p1d[counts_p1d > 0.] /= counts_p1d[counts_p1d > 0.]

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p1d_hMpc = binned_p1d #* L_hMpc**3.
    klos_hMpc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    return klos_hMpc, p1d_hMpc

def compute_pk1d_fourier_new(dF_fft, kperp_hMpc, klos_hMpc, nperp, nlos, L_hMpc):

    # set minimum k to make sure we cover fundamental mode
    n_k_bins = 20# len(klos_hMpc)
    lnk_min = np.log(0.9999*np.min(klos_hMpc[klos_hMpc > 0.]))
    lnk_max = np.log(np.max(klos_hMpc))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)
    k_bin_cents = (k_bin_edges[1:]+k_bin_edges[:-1])*.5

    # define mu-binning
    n_mu_bins = 16
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    # automize
    dF_fft /= nperp**2*nlos # not sure because need to think but I think so
    raw_p3d = (np.abs(dF_fft)**2)

    # h/Mpc
    kx = kperp_hMpc[:, np.newaxis, np.newaxis]
    ky = kperp_hMpc[np.newaxis, :, np.newaxis]
    kz = klos_hMpc[np.newaxis, np.newaxis, :]
    k_box = np.sqrt(kx**2 + ky**2 + kz**2)
    k_perp = np.sqrt(kx**2 + ky**2 + kz**2*0.)
    k_par = np.sqrt(kx**2*0. + ky**2*0. + kz**2)

    # construct mu in two steps, without NaN warnings
    mu_box = kz/np.ones_like(k_box)
    mu_box[k_box > 0.] /= k_box[k_box > 0.]
    mu_box[k_box == 0.] = 0.

    # flatten arrays
    k_box = k_box.flatten()
    k_perp = k_perp.flatten()
    k_par = k_par.flatten()
    mu_box = mu_box.flatten()
    raw_p3d = raw_p3d.flatten()

    # discard zeroth mode
    k_box = k_box[1:]
    k_perp = k_perp[1:]
    k_par = k_par[1:]
    mu_box = mu_box[1:]
    raw_p3d = raw_p3d[1:]

    # compute bin averages
    binned_p3d = spt.binned_statistic_2d(k_box, mu_box, raw_p3d, statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned power')
    binned_counts = spt.binned_statistic_2d(k_box,mu_box,raw_p3d,
                                            statistic='count', bins=[k_bin_edges,mu_bin_edges])[0]
    print('got bin counts')

    # compute binned values for (k,mu)
    binned_k = spt.binned_statistic_2d(k_box, mu_box, k_box,
                                       statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned k')
    binned_mu = spt.binned_statistic_2d(k_box, mu_box, mu_box,
                                        statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print('got binned mu')

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    k_hMpc = binned_k
    mu = binned_mu
    counts = binned_counts
    kpar = mu*k_hMpc
    kper = np.sqrt(1.-mu**2.)*k_hMpc

    # sega brat tr smetnem integrala
    from scipy import interpolate
    from scipy.integrate import quad

    f = interpolate.interp2d(kpar.flatten()[counts.flatten() > 0.], kper.flatten()[counts.flatten() > 0.], p3d_hMpc.flatten()[counts.flatten() > 0.], kind='cubic', bounds_error=False, fill_value=0.)

    def integrand(kper, kpar):
        return f(kpar, kper)*kper/(2.*np.pi) # dkper

    p1d_hMpc = np.zeros(len(k_bin_cents))
    for i in range(len(k_bin_cents)):
        p1d_hMpc[i] = quad(integrand, 0, np.inf, args=(k_bin_cents[i]))[0]        
    klos_hMpc = k_bin_cents
    print(p1d_hMpc, k_bin_cents)

    return klos_hMpc, p1d_hMpc

def main():
    save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
    ##redshift = 2.4
    redshift = 2.45
    print(redshift)
    L_hMpc = 205. # cMpc/h
    res = sys.argv[1] # "0.05" "0.25" "0.5" "1.0"
    if res == "1.0":
        deltaF_file = os.path.join(save_dir, 'noiseless_maps/map_TNG_true_1.0_z2.4.hdf5')
        deltaF = h5py.File(deltaF_file,'r')['map'][:]
    elif res == "0.5":
        deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.5_z2.4.hdf5')
        deltaF = h5py.File(deltaF_file,'r')['map'][:]
    elif res == "0.25":
        deltaF_file = os.path.join(save_dir, 'map_TNG_true_0.25_z2.5_820_voxels.hdf5')
        deltaF = h5py.File(deltaF_file,'r')['map'][:]
    elif res == "0.05":
        tau_file = os.path.join(save_dir,'spectra_z2.4/spectra_TNG_true_1.0_z2.4.hdf5')
        f = h5py.File(tau_file, 'r')
        tau = gf(f['tau/H/1/1215'][:], 1, mode='wrap')
        print(tau.shape)
        deltaF = tau2deltaF(tau, redshift, mean_F=None)
        del tau

    k_hMpc, p1d_hMpc = compute_pk1d(deltaF, L_hMpc)

    plt.figure(figsize=(9, 7))
    plt.plot(k_hMpc, p1d_hMpc*k_hMpc/np.pi)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("k P(k)/pi")
    plt.xlim([0.03, 20.])
    plt.ylim([1.e-3, .1])
    plt.savefig(f"power_1d_res{res}.png")
    plt.close()

    n_k_bins = 20
    n_mu_bins = 16
    mu_want = [0., 0.33, 0.66, 1.]
    k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, L_hMpc, n_k_bins, n_mu_bins)
    #print("mu bins = ", mu)
    #print("k bins = ", k_hMpc)

    no_nans = 0
    for i in range(n_k_bins):
        if not np.any(np.isnan(mu[i])): no_nans = i
    int_mu = []
    for i in range(len(mu_want)):
        int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))
    print("mu of interest = ", int_mu, mu[no_nans, int_mu])
    cosmo = cosmology.Planck15
    P_L = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    #P_L = P_L.__call__(k_hMpc[:, 3])

    print("ratio = ", (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]))

    plt.figure(figsize=(9, 7))
    power = 3.
    print("k3 Pk = ", (p3d_hMpc*k_hMpc**power/(2.*np.pi**2))[:, int_mu[0]])
    plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc*k_hMpc**power/(2.*np.pi**2))[:, int_mu[0]], color='violet')
    plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc*k_hMpc**power/(2.*np.pi**2))[:, int_mu[1]], color='cyan')
    plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc*k_hMpc**power/(2.*np.pi**2))[:, int_mu[2]], color='yellow')
    plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc*k_hMpc**power/(2.*np.pi**2))[:, int_mu[3]], color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("k^3 P(k)/2 pi^2")
    plt.xlim([0.03, 20.])
    plt.ylim([1.e-4, 1.])
    plt.savefig(f"power_3d_res{res}.png")
    plt.close()

    plt.figure(figsize=(9, 7))
    plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]), color='violet')
    plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc)[:, int_mu[1]]/P_L.__call__(k_hMpc[:, int_mu[1]]), color='cyan')
    plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc)[:, int_mu[2]]/P_L.__call__(k_hMpc[:, int_mu[2]]), color='yellow')
    plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]), color='red')
    plt.ylim([0., 0.4])
    plt.xlim([0.03, 20.])
    print("linear = ", P_L.__call__(k_hMpc[:, 3]))
    print("3d power = ", (p3d_hMpc)[:, int_mu[0]])
    print("weird stuff = ", (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]))
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P(k)/P_L(k)")
    plt.savefig(f"power_ratio_3d_res{res}.png")
    plt.close()
