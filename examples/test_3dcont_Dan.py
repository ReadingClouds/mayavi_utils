import xarray as xr

from pathlib import Path
from monc_utils.io.datain import (get_data_on_grid, get_full_dim_name)

from mayavi_utils import (gen_figure, 
                          set_light,
                          current_lights,
                          gen_volume, 
                          gen_contour3d,
                          get_colormap_as_array,
                          add_colour_table_to_volume,
                          )
from mayavi import mlab

from load_data import load_data
root_path = "F:/Dan/"

data_path = Path(root_path)

#%% Options to plot
qcl_on = True
qcl_as_contour = False
qt_on = False

#%% Input field data
file_prefix = "diagnostics_006p25m_3d_ts_"
ref_file = "diagnostics_006p25m_ts_"

selector = "*"


dataset = load_data(data_path, file_prefix, ref_file, selector)
    
qcl = get_data_on_grid(dataset, 'q_cloud_liquid_mass').astype('float32')

print(f'{qcl=}')

#qt = get_data_on_grid(dataset, 'q_total').astype('float32')

#print(f'{qt=}')

#dims = {c:c for c in 'xyz'}
#print(dims)

dim_map = {c:get_full_dim_name(qcl, c) for c in 'xyz'}    
#print(dims)

#print(qcl.dims)

fig =  gen_figure()

current_lights(fig)

# These are the defaults anyway
set_light(fig, 0, elevation=45.0, azimuth=45.0, intensity=1.0, activate=True)
set_light(fig, 1, elevation=-30.0, azimuth=-60.0, intensity=0.6, activate=True)
set_light(fig, 2,  elevation=-30.0, azimuth=60.0, intensity=0.5, activate=False)
#set_light(fig, 3, elevation=0, azimuth=-60, intensity=1.0, activate=True)
#set_light(fig, 4, elevation=20, azimuth=0, intensity=1.0, activate=True)


updateable_objects = []

ref_time = qcl.time.values[0]

if qcl_on:
    if qcl_as_contour:
        display_obj_list = gen_contour3d(
                        qcl.sel(time=ref_time, drop=False), [0.00005],
#                        qcl.isel(time=i), [0.00005],
                        fig, 
                        dim_map=dim_map, 
                        color=(1,1,1),
                        trans=1.0,
                        add_axes=True,
                        add_base_plane=True,
                        add_timelab=True,
                        axis_args={'nticks':8})

        [qcl_contour] = [d for d in display_obj_list if d['type']== 'contour3d']
        qcl_contour['source'] = qcl        
        updateable_objects.append(qcl_contour)
    
    else:
        display_obj_list = gen_volume(
                        qcl.sel(time=ref_time, drop=False),
                        fig, 
                        dim_map=dim_map, 
                        color=(1,1,1),
                        trans=0.5,
                        add_axes=True,
                        add_base_plane=True,
                        add_timelab=True,
                        axis_args={'nticks':8})

        [qcl_volume] = [d for d in display_obj_list if d['type']== 'volume']
        qcl_volume['source'] = qcl
        updateable_objects.append(qcl_volume)




#rgbsg = get_colormap_as_array("Greys")
#qcl_volume = add_colour_table_to_volume(qcl_volume, rgbsg, alpha_max=0.1)
if qt_on:
    qtp = qt.isel(time=0)

    print(f'{qtp=}')

    qtp.attrs = qt.attrs

    print(f'{qtp=}')

if qcl_on and qt_on:
    display_obj_list_qt = gen_volume(qtp, fig, 
                          dim_map=dim_map, 
                          color=(1,0,0),
                          trans=0.003,
                          add_axes=False,
                          add_base_plane=False,
                          add_timelab=False,
                         )
                         
                                                 
elif qt_on:
    display_obj_list_qt = gen_volume(
                        qtp, fig, 
                        dim_map=dim_map, 
                        color=(1,1,1),
                        trans=0.005,
                        add_axes=True,
                        add_base_plane=True,
                        add_timelab=True,
                        axis_args={'nticks':8})




mlab.show()




    
