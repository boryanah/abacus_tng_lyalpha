# this is a script for getting beta and bias
import sys
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbodykit.lab import cosmology
from scipy.optimize import minimize

from compute_power import compute_pk3d

def get_kaiser(P_lin, beta, b, mu):
    return b**2.*(1+beta*mu**2.)**2.*P_lin

def get_bias(b, P_kmu0, P_lin):
    if b < 0.: return np.inf
    chi2 = np.sum((P_kmu0/(b**2*P_lin) - 1.)**2.)
    print("chi2 bias = ", chi2)
    return chi2

def get_beta(beta, b, P_kmu1, P_lin):
    chi2 = np.sum((P_kmu1/(b**2*(1+beta)**2.*P_lin) - 1.)**2)
    print("chi2 beta = ", chi2)
    return chi2 

save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"
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

redshift = 2.44
L_hMpc = 205.
n_k_bins = 20
n_mu_bins = 16
mu_want = [0., 0.33, 0.66, 1.]
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, L_hMpc, n_k_bins, n_mu_bins)

no_nans = 0
for i in range(n_k_bins):
    if not np.any(np.isnan(mu[i])): no_nans = i
int_mu = []
for i in range(len(mu_want)):
    int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))
print("mu of interest = ", int_mu, mu[no_nans, int_mu])
cosmo = cosmology.Planck15
P_L = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

k_max = 0.2 # h/Mpc

# take mu = 0 
ks = k_hMpc[:, int_mu[0]]
ks = ks[ks < k_max]
P_lin = P_L.__call__(ks)
P_kmu0 = np.interp(ks, k_hMpc[:, int_mu[0]], p3d_hMpc[:, int_mu[0]])
print(P_kmu0/P_lin)

opt = {}
opt['maxiter'] = 1000
opt['disp'] = True
opt['maxfun'] = 5000
b0 = 1.
res = minimize(get_bias, b0, args=(P_kmu0, P_lin), tol=1.e-3, options=opt)
b = res['x']
print("b = ", b)

# take mu = 1 
ks = k_hMpc[:, int_mu[3]]
ks = ks[ks < k_max]
P_lin = P_L.__call__(ks)
P_kmu1 = np.interp(ks, k_hMpc[:, int_mu[3]], p3d_hMpc[:, int_mu[3]])
P_kmu1[np.isnan(P_kmu1)] = 0.
print(P_kmu1/P_lin)

opt = {}
opt['maxiter'] = 1000
opt['disp'] = True
opt['maxfun'] = 5000
beta0 = 1.
res = minimize(get_beta, beta0, args=(b, P_kmu1, P_lin), options=opt)# tol=1.e-5
beta = res['x']
print("beta = ", beta)
