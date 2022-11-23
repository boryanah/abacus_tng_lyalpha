import time
from Corrfunc.theory import DDsmu
import numpy as np

def get_corr(x1, y1, z1, w1, rpbins, nbins_mu, lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None, w2=None):

    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1
    
    # single precision mode
    # to do: make this native 
    rpbins = rpbins.astype(np.float32)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    w1 = w1.astype(np.float32)
    #pos1 = np.array([x1, y1, z1]).T % lbox
    lbox = np.float32(lbox)

    #nbins_mu = 40
    if autocorr == 1: 
        results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, weights1=w1, weight_type='pair_product', periodic = True, boxsize = lbox)#, max_cells_per_dim = num_cells)
        DD_counts = results['weightavg']#['npairs']
    else:
        x2 = x2.astype(np.float32)
        y2 = y2.astype(np.float32)
        z2 = z2.astype(np.float32)
        results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, weights1=w1, X2 = x2, Y2 = y2, Z2 = z2, weights2=w2,
            periodic = True, boxsize = lbox)#, max_cells_per_dim = num_cells)
        DD_counts = results['weightavg']#['npairs']
    DD_counts = DD_counts.reshape((len(rpbins) - 1, nbins_mu))

    #mu_bins = np.linspace(0, 1, nbins_mu+1)
    #RR_counts = 2*np.pi/3*(rpbins[1:, None]**3 - rpbins[:-1, None]**3)*(mu_bins[None, 1:] - mu_bins[None, :-1]) / lbox**3 * ND1 * ND2 * 2
    #xi_s_mu = DD_counts/RR_counts - 1

    xi_s_mu = DD_counts
    
    return xi_s_mu

# specify directory
sim_name = "AbacusSummit_base_c000_ph000"
save_dir = f"/global/cscratch1/sd/boryanah/Ly_alpha/{sim_name}/"

# params
lbox = 500. #cMpc/h
ngrid = 1728
cell_size = lbox/ngrid

# midpoint of each cell along LOS
rsd_bins = np.linspace(0., ngrid, ngrid+1)
rsd_binc = (rsd_bins[1:] + rsd_bins[:-1]) * .5
rsd_binc *= cell_size # cMpc/h
#rsd_binc = (np.arange(ngrid)+0.5)*cell_size

# rsdbins
down = 20 #5
x1, y1, z1 = np.meshgrid(rsd_binc[::down], rsd_binc[::down], rsd_binc[::down])

# load deltaF
fn = f"tmp_deltaF_yz_losz_ngrid{ngrid:d}.npy"
dF = np.load(save_dir+fn).astype(np.float32)

# downsample
dF = dF[::down, ::down, ::down]
#dF = np.transpose(dF, (1, 2, 0)) # wait, no, losz?

# transpose so that rsd along third axis
x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
w1 = dF.flatten()

# plot params
#rpbins = np.linspace(0, 200, 51)
rpbins = np.linspace(0, 150, 31)
rpbinc = (rpbins[1:] + rpbins[:-1])*.5
nbins_mu = 4
Nthread = 16
t = time.time()
xi_s_mu = get_corr(x1, y1, z1, w1, rpbins, nbins_mu, lbox, Nthread, num_cells=20, x2=None, y2=None, z2=None, w2=None)
np.savez("autocorr.npz", xi_s_mu=xi_s_mu, rpbinc=rpbinc)
print(time.time()-t)
