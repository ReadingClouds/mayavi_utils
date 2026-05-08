import xarray as xr
import numpy as np

from pathlib import Path
from monc_utils.io.datain import (get_data_on_grid, get_full_dim_name)

#import os
#os.environ['QT_API'] = 'pyqt5'

import logging
logging.basicConfig(level=logging.INFO)


from mayavi_utils import (gal_trans_vel_round,
                          gen_figure, 
                          set_light,
                          current_lights,
                          gen_volume, 
                          gen_contour3d,
                          trajectory_points,
                          get_colormap_as_array,
                          add_colour_table_to_volume,
                          anim,
                          gen_anim_stills,
                          )
from mayavi import mlab

from load_data import load_data

root_path = "F:/traj_data/r_50m_60s_Monsoon3/"

data_path = Path(root_path)

trajectories_path = data_path / "trajectories/"

output_path = data_path / 'images/'

output_path.mkdir(exist_ok=True)

#%% Options to plot
qcl_on = True
qcl_as_contour = True
qt_on = False
traj_on = True

real_time_animate = True

#%% Input field data
file_prefix = "diagnostics_3d_ts_"
ref_file = "diagnostics_ts_"

selector = "*"

dataset = load_data(data_path, file_prefix, ref_file, selector)

qcl = get_data_on_grid(dataset, 'q_cloud_liquid_mass')

print(f'{qcl=}')

qt = get_data_on_grid(dataset, 'q_total')

print(f'{qt=}')


dim_map = {c:get_full_dim_name(qcl, c) for c in 'xyz'}    

case = "cloud"
minim = "hybrid_fixed_point_iterator"
interp_order = 5
expt = "ref"

trajectory_path = trajectories_path / f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
trajectory_data_path = trajectories_path /  f"{file_prefix}data_{case}_{interp_order}_{minim}_{expt}.nc"

ds_traj = xr.open_dataset(trajectory_path)

print(ds_traj)

ds_traj_data = xr.open_dataset(trajectory_data_path)
print(ds_traj_data)

ds_all = xr.merge((ds_traj, ds_traj_data))
print(ds_all)

timestep = ds_all.attrs["trajectory timestep"]

ntimes = ds_all.time.size

delt = ds_all.time.values[1] - ds_all.time.values[0]

print(f"Time step = {delt}")

var = ["x", "y", "z", "q_cloud_liquid_mass"]

ds = ds_all[var]

select = None
select = [5, 6, 8, 12, 13]

if select is not None:
    ds = ds.sel(trajectory_number=(np.isin(ds_traj.object_label, select)))
# or  ds = ds.where(ds_ref.object_label.isin(select), drop=True)

ds.load()

# Subset field data to just range of trajectory data.
sl = slice(ds.time_index.values[0], ds.time_index.values[-1]+1, None)
qcl = qcl.isel(time=sl)
qt = qt.isel(time=sl)

# Plot at trajectory ref_time.
ref_time = ds.ref_time
print(ref_time)

ds_ref = ds.sel(time=ref_time)
print(ds_ref)

fig =  gen_figure()

current_lights(fig)

# These are the defaults anyway
set_light(fig, 0, elevation=45.0, azimuth=45.0, intensity=1.0, activate=True)
set_light(fig, 1, elevation=-30.0, azimuth=-60.0, intensity=0.6, activate=True)
set_light(fig, 2,  elevation=-30.0, azimuth=60.0, intensity=0.5, activate=False)
#set_light(fig, 3, elevation=0, azimuth=-60, intensity=1.0, activate=True)
#set_light(fig, 4, elevation=20, azimuth=0, intensity=1.0, activate=True)


updateable_objects = []


if qcl_on:
    qcl_sel = qcl.sel(time=ref_time, drop=False)
    if qcl_as_contour:
        display_obj_list = gen_contour3d(
                        qcl_sel, [5.0E-5],
                        fig, 
                        dim_map=dim_map, 
                        color=(1,1,1),
                        trans=0.5,
                        add_axes=True,
                        add_base_plane=True,
                        add_timelab=True,
                        axis_args={'nticks':8})

        [qcl_contour] = [d for d in display_obj_list if d['type']== 'contour3d']
        qcl_contour['source'] = qcl        
        updateable_objects.append(qcl_contour)
    
    else:
        display_obj_list = gen_volume(
                        qcl_sel,
                        fig, 
                        dim_map=dim_map, 
                        color=(1,1,1),
                        trans=0.1,
                        add_axes=True,
                        add_base_plane=True,
                        add_timelab=True,
                        axis_args={'nticks':8})

        [qcl_volume] = [d for d in display_obj_list if d['type']== 'volume']
        qcl_volume['source'] = qcl
        updateable_objects.append(qcl_volume)

if qt_on:
    qtp = qt.sel(time=ref_time, drop=False)
    qtp.attrs = qt.attrs
    print(f'{qtp=}')

    if qcl_on:
        display_obj_list_qt = gen_volume(qtp, fig, 
                              dim_map=dim_map, 
                              color=(1,0,0),
                              trans=0.003,
                              add_axes=False,
                              add_base_plane=False,
                              add_timelab=False,
                             )
                             
                                                     
    else:
        display_obj_list_qt = gen_volume(
                            qtp, fig, 
                            dim_map=dim_map, 
                            # scale=1E3, 
                            color=(1,1,1),
                            trans=0.005,
                            add_axes=True,
                            add_base_plane=True,
                            add_timelab=True,
                            axis_args={'nticks':8})

    [qt_volume] = [d for d in display_obj_list_qt if d['type']== 'volume']
    qt_volume['source'] = qt
    updateable_objects.append(qt_volume)
    
if traj_on:
    traj = ds_ref

    x = traj['x'].values
    y = traj['y'].values
    z = traj['z'].values
    
    # Scalar field for point size 
    # Point size is given by 
    # scalar_size_min + (scalar_size - scalar_size_range[0] / scalar_size_range[1].
    # This is then scaled by scale_factor.
    # If scalar_size_range=None, scalar_size_range is given by min and max of scalar_size.
    # If scalar_size_range=[value, None], scalar_size_range[1] is given by max of scalar_size.
    # If scalar_size_range=[None, value], scalar_size_range[0] is given by min of scalar_size.
    
 
    s = traj["q_cloud_liquid_mass"].values

    # Scalar field for point color
    iobj = traj.object_label.values

    sc = iobj / np.max(iobj) * 255 
    
    display_obj_list_tr = trajectory_points(x, y, z, 
                      scalar_size=s, 
                      scalar_color=sc, 
                      name = 'Cloud_Trajectories', 
                      scale_factor=ds.attrs['dx'],
                      scalar_size_min=0.2, 
                      scalar_size_range=[0.0,1E-3])

    [trajectories] =[d for d in display_obj_list_tr if d['type']== 'trajectories']
    trajectories['source_xyz'] = ds[['x','y','z']]
    trajectories['source_scalar_size'] = ds["q_cloud_liquid_mass"]
    trajectories['source_scalar_color'] = None
    updateable_objects.append(trajectories)
    
ur, nx_roll = gal_trans_vel_round(-7.5, 60.0, qt.attrs['dx'])
vr, ny_roll = gal_trans_vel_round(-1., 60.0, qt.attrs['dy'])

frame_vel = {'ur':ur, 'vr':vr, 'nx_roll':nx_roll, 'ny_roll':ny_roll, 'delt':60}


if real_time_animate:

    anim(updateable_objects, frame_vel, fig)
    
else:

    figname = 'BOMEX_Smag_cloud_vol_traj'

    gen_anim_stills(updateable_objects, frame_vel, fig, output_path, figname, type=None)


mlab.show()




    
