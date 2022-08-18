# could invert the direction of x y and z into y, z and x and add that to the header (should probably google)
import gc
import sys
sys.path.append("..")

import numpy as np
import asdf

from tools import compress_asdf

# sample directories
save_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name

# simulation directory
Lbox = 2000. # Mpc/h
Ndim_yz = 8000 # in y and z
cell_size = Lbox/Ndim_yz
#part_type = "AB"
part_type = "slices"
if part_type == "slices":
    sim_name = "AbacusSummit_base_c000_ph000"
    n_chunks = 170 # in x direction
elif part_type == "AB":
    sim_name = "AbacusSummit_base_c000_ph006"
    n_chunks = 34 # in x direction
paint_type = 'TSC' # 'CIC' # 'TSC'
z = 2.5

# distance between each skewer
dist_skewer = 10. # Mpc/h

# select which indices in the y and z direction to keep
Ndim_down_x = 8000
Ndim_down_yz = np.int(np.round(Lbox/dist_skewer))
cell_step_yz = np.int(np.round(dist_skewer/cell_size))
inds_yz = np.arange(0, Ndim_yz, cell_step_yz)
print("this should be every 40 until 7999 [0, 40, 80 ... 7999] = ", inds_yz)

# initialize downsampled density
density_field = np.zeros((Ndim_down_x, Ndim_down_yz, Ndim_down_yz))
velocity_field = np.zeros((Ndim_down_x, Ndim_down_yz, Ndim_down_yz))

# loop over each chunk
for i_chunk in range(n_chunks):
    print("i_chunk = ", i_chunk)

    if part_type == "AB":
        # load density and velocity
        f = asdf.open(save_dir+f'/z{z:.3f}/velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf')
        cell_inds_chunk = f['header']['ChunkIndex']
        velocity = f['data']['Velocity']
        assert len(cell_inds_chunk) == velocity.shape[0]
        print("loaded velocity")
        
        # add to the final arrays
        #velocity_field[cell_inds_chunk, :, :] += velocity[:, inds_yz, inds_yz] # doesn't work
        velocity_field[cell_inds_chunk, :, :] += velocity[:, ::cell_step_yz, ::cell_step_yz]
        f.close()
        del velocity
        gc.collect()
        print("done with velocity")

        f = asdf.open(save_dir+f'/z{z:.3f}/density_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf')
        cell_inds_chunk = f['header']['ChunkIndex']
        density = f['data']['Density']
        assert len(cell_inds_chunk) == density.shape[0]
        print("loaded density")

        # add to the final arrays
        density_field[cell_inds_chunk, :, :] += density[:, ::cell_step_yz, ::cell_step_yz]
        del density
        gc.collect()
        f.close()
        print("done with density"
        )
    else:

        # load density and velocity
        f = asdf.open(save_dir+f'/z{z:.3f}/density_velocity_{paint_type}_Ndim{Ndim_yz:d}_part{part_type}_nchunk{i_chunk:d}.asdf')
        cell_inds_chunk = f['header']['ChunkIndex']
        velocity = f['data']['Velocity']
        density = f['data']['Density']
        assert len(cell_inds_chunk) == velocity.shape[0]
        print("loaded velocity")
        
        # add to the final arrays
        density_field[cell_inds_chunk, :, :] += density[:, ::cell_step_yz, ::cell_step_yz]
        velocity_field[cell_inds_chunk, :, :] += velocity[:, ::cell_step_yz, ::cell_step_yz] # cause above doesn't work
        f.close()
        del velocity, density
        gc.collect()
        print("done with velocity")
        
# save the downsampled arrays
header = {}
header['Redshift'] = z
header['Simulation'] = sim_name
header['PaintMode'] = paint_type
header['ParticleType'] = part_type
header['DirectionsOrder'] = 'XYZ'
table = {}
table['Density'] = density_field
table['Velocity'] = velocity_field
compress_asdf(save_dir+f'/z{z:.3f}/density_velocity_{paint_type}_Ndim{Ndim_down_yz:d}_{Ndim_down_x:d}_part{part_type}.asdf', table, header)
