# rightwrap in case of rsd and the middle one being actually too far away from anything close by
import numpy as np
import numba
#from nbodykit.lab import ArrayCatalog, FieldMesh
#from nbodykit.base.mesh import MeshFilter
from scipy.optimize import minimize
from fake_spectra import fluxstatistics as fs
import asdf
#from fast_cksum.cksum_io import CksumWriter

@numba.vectorize
def rightwrap(x, L):
    if x >= L:
        return x - L
    return x

@numba.njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@numba.jit(nopython=True, nogil=True)
def numba_tsc_3D(positions, density, boxsize, weights=np.empty(0)):
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the tsc weights for each dimension
        wx = .75 - dx**2
        wxm1 = .5*(.5 + dx)**2 # og not 1.5 cause wrt to adjacent cell
        wxp1 = .5*(.5 - dx)**2
        wy = .75 - dy**2
        wym1 = .5*(.5 + dy)**2
        wyp1 = .5*(.5 - dy)**2
        if threeD:
            wz = .75 - dz**2
            wzm1 = .5*(.5 + dz)**2
            wzp1 = .5*(.5 - dz)**2
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W
        
        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W

@numba.jit(nopython=True, nogil=True)
def numba_tsc_irregular_3D(positions, density, boxsize, weights=np.empty(0)):
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize[0])*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize[1])*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize[2])*gz # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the tsc weights for each dimension
        wx = .75 - dx**2
        wxm1 = .5*(.5 + dx)**2
        wxp1 = .5*(.5 - dx)**2
        wy = .75 - dy**2
        wym1 = .5*(.5 + dy)**2
        wyp1 = .5*(.5 - dy)**2
        if threeD:
            wz = .75 - dz**2
            wzm1 = .5*(.5 + dz)**2
            wzp1 = .5*(.5 - dz)**2
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W
        
        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W

def compress_asdf(asdf_fn, table, header):
    """
    Given the file name of the asdf file, the table and the header, compress the table info and save as `asdf_fn' 
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # set compression options here
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, CksumWriter(str(asdf_fn)) as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp, all_array_compression='blsc', compression_kwargs=compression_kwargs)

def save_asdf(filename, table, header):
    """
    Save light cone catalog
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]
        
    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    output_file.write_to(filename)
    output_file.close()
            
@numba.jit(nopython=True, nogil=True)
def numba_cic_3D(positions, density, boxsize, weights=np.empty(0)):
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the tsc weights for each dimension
        wx = 1. - np.abs(dx)
        if dx > 0.: # on the right of the center ( < )
            wxm1 = dx
            wxp1 = 0.
        else: # on the left of the center
            wxp1 = -dx
            wxm1 = 0.
        wy = 1. - np.abs(dy)
        if dy > 0.:
            wym1 = dy 
            wyp1 = 0.
        else:
            wyp1 = -dy
            wym1 = 0.
        if threeD:
            wz = 1. - np.abs(dz)
            if dz > 0.:
                wzm1 = dz
                wzp1 = 0.
            else:
                wzp1 = -dz
                wzm1 = 0.
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W
        
        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W


@numba.jit(nopython=True, nogil=True)
def numba_cic_1D(positions, density, boxsize, weights=np.empty(0)):
    gx = np.uint32(len(density))
    zero = np.uint32(0)
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n]/boxsize)*gx # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        
        # calculate distance to cell center
        dx = ix - px
        
        # find the tsc weights for each dimension
        wx = 1. - np.abs(dx)
        if dx > 0.: # on the right of the center ( < )
            wxm1 = dx
            wxp1 = 0.
        else: # on the left of the center
            wxp1 = -dx
            wxm1 = 0.

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix, gx)
        ixp1 = rightwrap(ix + 1, gx)
        #if ixm1 >= zero or ixm1 < gx: density[ixm1] += wxm1 *W
        #if ixw  >= zero or ixw  < gx: density[ixw ] += wx *W
        #if ixp1 >= zero or ixp1 < gx: density[ixp1] += wxp1 *W
        density[ixm1] += wxm1 *W
        density[ixw ] += wx *W
        density[ixp1] += wxp1 *W

@numba.jit(nopython=True, nogil=True) 
def rsd_tau(tau, vfield, binc, E_z, redshift, Lbox):
    dtype = np.float32
    ngrid = tau.shape[2]
    ngrid_tr = tau.shape[0]
    assert ngrid_tr == tau.shape[1]
    tau1d = np.zeros(ngrid, dtype=dtype)
    pos1d = np.zeros(ngrid, dtype=dtype)
    tau1d_new = np.zeros(ngrid, dtype=dtype)

    # loop over each skewer
    for i in range(ngrid_tr):
        for j in range(ngrid_tr):        
            tau1d[:] = tau[i, j, :]
            #print("max min vel in km/s", vfield[i, j, :].min(), vfield[i, j, :].max())
            extra = vfield[i, j, :]*(1+redshift)/(E_z) # cMpc/h
            #print("adding cmpc/h = ", extra.min(), extra.max())
            pos1d[:] = binc + extra
            numba_cic_1D(pos1d, tau1d_new, Lbox, weights=tau1d)
            #print(np.sum(tau1d), np.sum(tau1d_new))
            tau[i, j, :] = tau1d_new
            tau1d_new *= 0.
    return tau

def func(a, tau, target):
    res = np.abs(np.mean(np.exp(-a*tau))-target)
    print(res, a)
    return res

def density2tau(density, tau_0, power=1.6):
    tau = tau_0 * (density)**power
    return tau

def get_mean_flux(redshift):
    mean_F = np.exp(-1.330e-3 * (1 + redshift)**4.094)
    return mean_F

def tau2deltaF_mine(tau, redshift, mean_F=None):
    if mean_F is None:
        mean_F = get_mean_flux(redshift)
    else:
        assert mean_F > 0.

    # compute flux for given tau
    F = np.exp(-tau)
    print("mean flux before = ", F.mean())
    
    # taylor expansion for initial guess
    a0 = ( np.mean(F) - mean_F ) / np.mean(F * tau) + 1.
    opt = {'maxiter': 3, 'disp': True, 'maxfun': 3}
    skip = 1
    if skip:
        #0.8261691877586387
        #a = 0.80826046 # 0.05
        a = 0.50113696 # FGPA Abacus 2.5 8000, 200
        #a = 0.22076345 # FGPA
        print("skipping the minimization!!!")
    else:
        res = minimize(func, a0, args=(tau, mean_F), tol=1.e-4, options=opt)
        a = res['x']
    F = np.exp(-a*tau)
    print("mean flux after = ", F.mean(), mean_F)
    deltaF = F/np.mean(F) - 1.
    return deltaF

def tau2deltaF(tau, redshift, mean_F=None):
    if mean_F is None:
        mean_F = np.exp(-1.330e-3 * (1 + redshift)**4.094)
    else:
        assert mean_F > 0.
        #mean_F = 0.8192
    scale = fs.mean_flux(tau, mean_F)
    deltaF = np.exp(-scale*tau)/mean_F - 1.
    return deltaF


@numba.jit(nopython=True, nogil=True) 
# this is kinda like NGP
def numba_pseudo_cic_3D(positions, density, boxsize, weights=np.empty(0)): # missing the neighbors
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the cic weights for each dimension
        wx = 1. - np.abs(dx)
        wy = 1. - np.abs(dy)
        if threeD:
            wz = 1. - np.abs(dz)
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixw  = rightwrap(ix    , gx)
        iyw  = rightwrap(iy    , gy)
        if threeD:
            izw  = rightwrap(iz    , gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
