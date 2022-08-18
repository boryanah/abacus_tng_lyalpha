import gc
import os
import sys
sys.path.append("..")

import numpy as np
import asdf

from abacusnbody.data.read_abacus import read_asdf
from tools import compress_asdf as write_asdf
#from tools import save_asdf as write_asdf

# redshifts
#zs = [2.0, 2.5, 3.0]
zs = [2.5]

# sample directories
sim_name = "AbacusSummit_base_c000_ph000"
#sim_name = "AbacusSummit_base_c000_ph006"
save_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name

# simulation directory
#sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name
sim_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name
Lbox = 2000. # Mpc/h
n_chunks = 170 # in x direction
Ndim_yz = 8000 # in y and z
chunk_size = Lbox/n_chunks
#chunk_size = Lbox/(1701/10.)
cell_size = Lbox/Ndim_yz
part_type = "slices"
paint_type = 'TSC' # 'CIC' # 'TSC'
if paint_type == 'TSC':
    from tools import numba_tsc_irregular_3D as numba_map_3D
elif paint_type == 'CIC':
    from tools import numba_cic_irregular_3D as numba_map_3D
else:
    print("Not Implemented"); quit()
map_types = ['both'] # ['density', 'velocity']

# chunk edges and cell edges 
chunk_edges = np.arange(n_chunks+1) * chunk_size
cell_edges = np.arange(Ndim_yz+1)*cell_size

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

            # load halo particles
            fn_halo = sim_dir+f'/slices/z{z:.3f}/L0_pack9/slab{i_chunk:03d}.L0.pack9.asdf'
            if map_type == 'both' or map_type == 'velocity':
                table = read_asdf(fn_halo, load=['pos', 'vel'])
                pos_halo = table['pos']
                vel_halo = table['vel']
                vel_halo = vel_halo[:, 0]
            elif map_type == 'density':
                table = read_asdf(fn_halo, load=['pos'])
                pos_halo = table['pos']
            pos_halo += Lbox/2.
            x_halo = pos_halo[:, 0]
            print("pos_halo[:, 0].min(), pos_halo[:, 0].max() = ", x_halo.min(), x_halo.max())

            # executes once for each chunk to set up global offset, Ndim_x and the 3D arrays
            if i_chunk == n_chunks - 1:
                choice = x_halo < Lbox/2.
                edge_min = x_halo[~choice].min()
                edge_max = x_halo[choice].max()
            else:
                edge_min = x_halo.min()
                edge_max = x_halo.max()

            fin_ind = np.argmax(cell_edges/edge_max > 1.)
            #fin_ind = np.argmax(cell_edges/chunk_edges[i_chunk + 1] > 1.)
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
                onset = Lbox - offset # right edge of the midpoint cell constraint         
                pos_halo[~choice, 0] -= offset 
                pos_halo[choice, 0] += onset 
            else:
                cell_inds_chunk = np.arange(prev_ind, fin_ind)

                # get offset
                offset = cell_edges[prev_ind]
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

            res = Ndim_x * cell_size - pos_halo[:, 0].max()
            assert (res < cell_size) and (res >= 0.)
            
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
            del table, pos_halo
            gc.collect()

            # load field particles
            fn_field = sim_dir+f'/slices/z{z:.3f}/field_pack9/slab{i_chunk:03d}.field.pack9.asdf'
            if map_type == 'both' or map_type == 'velocity':
                table = read_asdf(fn_field, load=['pos', 'vel'])
                pos_field = table['pos']
                vel_field = table['vel']
                vel_field = vel_field[:, 0]
            elif map_type == 'density':
                table = read_asdf(fn_field, load=['pos'])
                pos_field = table['pos']
            pos_field += Lbox/2.
            x_field = pos_field[:, 0]
            print("pos_field[:, 0].min(), pos_field[:, 0].max() = ", x_field.min(), x_field.max())

            choice = x_field < Lbox/2.
            if i_chunk == n_chunks - 1:
                pos_field[~choice, 0] -= offset # you should subtract the very left edge of the left cell constraint
                pos_field[choice, 0] += onset # you should add the very right edge of the right cell constraint
            else:
                pos_field[:, 0] -= offset

            res = Ndim_x * cell_size - pos_field[:, 0].max()
            assert (res < cell_size) and (res >= 0.)
                
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
            del table, pos_field
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
                write_asdf(save_dir+f'/z{z:.3f}/density_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del density
            elif map_type == 'velocity':
                table['Velocity'] = velocity
                write_asdf(save_dir+f'/z{z:.3f}/velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del velocity
            elif map_type == 'both':
                table['Density'] = density
                table['Velocity'] = velocity
                write_asdf(save_dir+f'/z{z:.3f}/density_velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf', table, header)
                del density, velocity
            del table
            gc.collect() 
