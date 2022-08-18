import gc
import os
import sys
sys.path.append("..")

import numpy as np
import asdf
#from astropy.io import ascii

#from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from bitpacked import unpack_rvint
from tools import compress_asdf

# redshifts
#zs = [2.0, 2.5, 3.0]
zs = [2.5]

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
save_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name

# simulation directory
sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name
Lbox = 2000. # Mpc/h
n_chunks = 34 # in x direction
Ndim_yz = 8000 # in y and z
chunk_size = Lbox/n_chunks # chunk size is actually 2000/(1701/50)
cell_size = Lbox/Ndim_yz
part_type = "AB"
subsample_types = ['B', 'A']
paint_type = 'TSC' # 'CIC' # 'TSC'
if paint_type == 'TSC':
    from tools import numba_tsc_irregular_3D as numba_map_3D
elif paint_type == 'CIC':
    from tools import numba_cic_irregular_3D as numba_map_3D
else:
    print("Not Implemented"); quit()
map_types = ['density', 'velocity'] # 'both'

# 35 chunk edges, 8001 cell edges 
#chunk_edges = np.linspace(0., Lbox, n_chunks+1) # equivalent
chunk_edges = np.arange(n_chunks+1) * chunk_size
cell_edges = np.arange(Ndim_yz+1)*cell_size

# I think the chunks aren't exact (left comes early and right expands beyond)
"""
# offsets to subtract (based on the cell edges) and cell indices for each chunk
offsets = np.zeros(n_chunks)
prev_ind = 0
cell_inds_chunk = {}
for i_chunk in range(n_chunks):
    fin_ind = np.argmax(cell_edges/chunk_edges[i_chunk+1] > 1.)
    if i_chunk == 0:
        prev_ind = 0
    else:
        prev_ind = np.argmax(cell_edges/chunk_edges[i_chunk] > 1.) - 1 # this also works
    if fin_ind == 0: fin_ind = Ndim_yz

    cell_inds_chunk[i_chunk] = np.arange(prev_ind, fin_ind)
    offsets[i_chunk] = prev_ind * cell_size
    #prev_ind = fin_ind - 1 # og
print(cell_inds_chunk)
"""

# for each redshift
for i in range(len(zs)):
    # this redshift
    z = zs[i]
    print("redshift = ", z)

    # create new directories
    os.makedirs((save_dir+f'/z{z:.3f}/'), exist_ok=True)

    # loop over each chunk
    for i_chunk in range(n_chunks):

        # for each map type
        for map_type in map_types:

            #if i_chunk != 0: # 0 1 31 32 33 next
            #    continue

            #Ndim_x = len(cell_inds_chunk[i_chunk])
            #offset = offsets[i_chunk]
            #boxsize = np.array([Ndim_x, Ndim_yz, Ndim_yz], dtype=np.float32)*cell_size
            
            # initiate density of box
            #density = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)
            #velocity = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)

            # loop over A and B subsample
            for type_AB in subsample_types:
                # halo and field particles
                fn_halo = sim_dir+f'/halos/z{z:.3f}/halo_rv_{type_AB}/halo_rv_{type_AB}_{i_chunk:03d}.asdf'
                fn_field = sim_dir+f'/halos/z{z:.3f}/field_rv_{type_AB}/field_rv_{type_AB}_{i_chunk:03d}.asdf'

                # write out the halo (L0+L1) matter particles
                halo_data = (asdf.open(fn_halo)['data'])['rvint']
                if map_type == 'both' or map_type == 'velocity':
                    pos_halo, vel_halo = unpack_rvint(halo_data, Lbox, float_dtype=np.float32, velout=None)
                    vel_halo = vel_halo[:, 0]
                elif map_type == 'density':
                    pos_halo, _ = unpack_rvint(halo_data, Lbox, float_dtype=np.float32, velout=False)
                pos_halo += Lbox/2.
                x_halo = pos_halo[:, 0]
                print("pos_halo[:, 0].min(), pos_halo[:, 0].max() = ", x_halo.min(), x_halo.max())

                # executes once for each chunk to set up global offset, Ndim_x and the 3D arrays
                if type_AB == subsample_types[0]:
                    if i_chunk == n_chunks - 1:
                        choice = x_halo < Lbox/2.
                        edge_min = x_halo[~choice].min()
                        edge_max = x_halo[choice].max()
                    else:
                        edge_min = x_halo.min() # not used
                        edge_max = x_halo.max()

                    fin_ind = np.argmax(cell_edges/edge_max > 1.)
                    if fin_ind == 0: fin_ind = Ndim_yz
                    if i_chunk == 0:
                        prev_ind = 0
                    else:
                        #prev_ind = np.argmax(cell_edges/chunk_edges[i_chunk] > 1.) - 1
                        prev_ind = np.argmax(cell_edges/edge_min > 1.) - 1 

                    if i_chunk == n_chunks - 1:
                        cell_inds_chunk = np.hstack((np.arange(prev_ind, Ndim_yz), np.arange(fin_ind)))

                        # get offset
                        offset = cell_edges[prev_ind] # left edge of the leftmost cell constraint 
                        #onset = cell_edges[fin_ind + 1] # right edge of the rightmost (wrapped) cell constraint
                        onset = Lbox - offset # right edge of the midpoint cell constraint         
                        pos_halo[~choice, 0] -= offset # let's say 1800 is offset and then 1800 becomes 0; 2000 becomes 200 and next dude should follow
                        pos_halo[choice, 0] += onset # for 0; 200; for 40, 240

                    else:
                        cell_inds_chunk = np.arange(prev_ind, fin_ind)

                        # get offset
                        offset = cell_edges[prev_ind] # prev_ind * cell_size
                        pos_halo[:, 0] -= offset

                    # get boxsize
                    Ndim_x = len(cell_inds_chunk) 
                    boxsize = np.array([Ndim_x, Ndim_yz, Ndim_yz], dtype=np.float32)*cell_size

                    # initialize fields
                    if map_type == 'density':
                        density = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)
                    elif map_type == 'velocity':
                        velocity = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)
                    elif map_type == 'both':
                        density = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)
                        velocity = np.zeros((Ndim_x, Ndim_yz, Ndim_yz), dtype=np.float32)
                    print("i_chunk, offset, pos_halo[:, 0].min(), pos_halo[:, 0].max(), chunk_size, Ndim_x = ", i_chunk, offset, pos_halo[:, 0].min(), pos_halo[:, 0].max(), chunk_size, Ndim_x)

                else:

                    if i_chunk == n_chunks - 1:
                        choice = x_halo < Lbox/2.
                        pos_halo[~choice, 0] -= offset # you should subtract the very left edge of the left cell constraint
                        pos_halo[choice, 0] += onset # you should add the very right edge of the right cell constraint
                    else:
                        pos_halo[:, 0] -= offset


                # add to density
                if map_type == 'density':
                    numba_map_3D(pos_halo, density, boxsize=boxsize)
                elif map_type == 'velocity':
                    numba_map_3D(pos_halo, velocity, boxsize=boxsize, weights=vel_halo)
                    del vel_halo
                elif map_type == 'both':
                    numba_map_3D(pos_halo, density, boxsize=boxsize)
                    numba_map_3D(pos_halo, velocity, boxsize=boxsize, weights=vel_halo)
                    del vel_halo
                del halo_data, pos_halo
                gc.collect()


                # write out the field matter particles
                field_data = (asdf.open(fn_field)['data'])['rvint']
                if map_type == 'both' or map_type == 'velocity':
                    pos_field, vel_field = unpack_rvint(field_data, Lbox, float_dtype=np.float32, velout=None)
                    vel_field = vel_field[:, 0]
                elif map_type == 'density':
                    pos_field, _ = unpack_rvint(field_data, Lbox, float_dtype=np.float32, velout=False)
                pos_field += Lbox/2.
                x_field = pos_field[:, 0]
                print("pos_field = ", x_field.min(), x_field.max())

                choice = x_field < Lbox/2.
                if i_chunk == n_chunks - 1:
                    pos_field[~choice, 0] -= offset # you should subtract the very left edge of the left cell constraint
                    pos_field[choice, 0] += onset # you should add the very right edge of the right cell constraint
                else:
                    pos_field[:, 0] -= offset


                # add to density
                if map_type == 'density':
                    numba_map_3D(pos_field, density, boxsize=boxsize)
                elif map_type == 'velocity':
                    numba_map_3D(pos_field, velocity, boxsize=boxsize, weights=vel_field)
                    del vel_field
                elif map_type == 'both':
                    numba_map_3D(pos_field, density, boxsize=boxsize)
                    numba_map_3D(pos_field, velocity, boxsize=boxsize, weights=vel_field)
                    del vel_field
                del field_data, pos_field
                gc.collect()



            header = {}
            header['Redshift'] = z
            header['Simulation'] = sim_name
            header['ChunkNumber'] = i_chunk
            header['ChunkIndex'] = cell_inds_chunk
            header['PaintMode'] = paint_type
            header['ParticleType'] = part_type
            table = {}
            if map_type == 'density':
                table['Density'] = density
                compress_asdf(save_dir+f'/z{z:.3f}/density_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del density
            elif map_type == 'velocity':
                table['Velocity'] = velocity
                compress_asdf(save_dir+f'/z{z:.3f}/velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del velocity
            elif map_type == 'both':
                table['Density'] = density
                table['Velocity'] = velocity
                compress_asdf(save_dir+f'/z{z:.3f}/density_velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del density, velocity
            del table
            gc.collect() 

