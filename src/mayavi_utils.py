import numpy as np

from mayavi import mlab
from mayavi.core.lut_manager import LUTManager 
from tvtk.util import ctf
from tvtk.pyface.light_manager import CameraLight

from math import log10,ceil
import time

import xarray as xr


def gal_trans(x, y, u, v, delt, Lx, Ly):

    x = (x - u * delt) % Lx
    y = (y - v * delt) % Ly
    
    return x, y
    
def gal_trans_vel_round(spd, delta_t, delta_s):
    
    n = spd * delta_t / delta_s
    n_rounded = float(round(n))
    spd_rounded = spd *n_rounded / n
   
    return spd_rounded, int(-n_rounded)
    
def size_scale(scalar_size, scalar_size_min, scalar_size_range):
    s = (scalar_size - scalar_size_range[0] ) / scalar_size_range[1]
    
    s = scalar_size_min + np.clip(s,0,None)

    return s


def field_grid(field, dim_map=None):
    if dim_map is None:
        dim_map = {c:c for c in 'xyz'}
        
    x = field.coords[dim_map['x']].values
    y = field.coords[dim_map['y']].values
    z = field.coords[dim_map['z']].values
    
    dz = z[-1] - z[-2]
    
    X, Y, Z  = np.meshgrid(x, y, z, indexing='ij', copy=False)
        
    Lx = field.attrs['Lx']
    Ly = field.attrs['Ly']
    Lz = z.max() + dz / 2

    extent = [0, Lx, 0, Ly, 0, Lz]   
        
    return extent, X, Y, Z

def gen_figure(size=(1500,1500), bgcolor=(0.5, 0.5, 1.0)):
    fig = mlab.figure(size=size, bgcolor=bgcolor, )# fgcolor=(0., 0., 0.), 
    
    fig.scene._lift()
    return fig
    
def gen_axes(extent, nticks=10, label_format='%4.0f'):
    axes = mlab.axes(extent=extent)
    axes.axes.fly_mode = 'none'
    axes.axes.number_of_labels = nticks+1
    axes.axes.label_format = label_format
    axes.axes.font_factor = 0.5

    mlab.outline(extent=extent)
    
    axes_obj = {'type':'axes',               
                'object':axes,
                'extent': extent,
               }
        
    return axes_obj
    
def gen_baseplane(extent, color=(0.0,0.3,0.7)):
    xp, yp = np.mgrid[extent[0]:extent[1]:2j, extent[2]:extent[3]:2j]
    zp = np.zeros_like(xp)

    base_plane = mlab.surf(xp, yp, zp, color=color)
    base_plane_obj = {'type':'base_plane',
                      'object':base_plane,
                      'extent': extent,
                     }
    return base_plane_obj


def gen_volume(field, fig, 
               dim_map=None, 
               color=(1,1,1),
               trans=None,
               add_axes=False, 
               add_base_plane=False, 
               add_timelab=False,
               axis_args=None
               ):
                   
    if dim_map is None:
        dim_map = {c:c for c in 'xyz'}

    display_obj_list = []        
       
    extent, X, Y, Z = field_grid(field, dim_map=dim_map)
    
    data = field.values
    
    dmin = data.min()
    dmax = data.max()
    vmin = dmin
    
    if trans is None:
        vmax = dmax
    else:
        vmax = dmax / trans
     
    scalar_f = mlab.pipeline.scalar_field(X, Y, Z, data)

    volume = mlab.pipeline.volume(scalar_f, 
                                  figure=fig, 
                                  color=color, 
                                  vmin=vmin, vmax=vmax)
        
    if add_timelab:
        dtime = field.time.item()
        timelab = mlab.text3d(0, 0, 1.05 * extent[5], 
                              f"{dtime:6.0f}", 
                              name="Time_label", 
                              scale = extent[1]/50)
    else:
        timelab = None
        
    volume_obj = {'type':'volume',
                  'object':volume,
                  'timelab':timelab,
                  'extent': extent,
                  'dim_map':dim_map,
                 }
                 
    display_obj_list.append(volume_obj)

    if add_axes:      
        if axis_args is None: axis_args={}
        axes_obj = gen_axes(extent, **axis_args)
        display_obj_list.append(axes_obj)

      
    if add_base_plane:
        base_plane_obj = gen_baseplane(extent, color=(0.0,0.3,0.7))
        display_obj_list.append(base_plane_obj)              

    return display_obj_list
    
def gen_contour3d(field, contours, fig, 
               dim_map=None, 
               color=(1,1,1),
               trans=None,
               add_axes=False, 
               add_base_plane=False, 
               add_timelab=False,
               axis_args=None
               ):
                   
    if dim_map is None:
        dim_map = {c:c for c in 'xyz'}
        
    display_obj_list = []        
        
    extent, X, Y, Z = field_grid(field, dim_map=dim_map)
    
    data = field.values
    
    dmin = data.min()
    dmax = data.max()
    vmin = dmin
    
    if trans is None:
        vmax = dmax
    else:
        vmax = dmax / trans

    contour = mlab.contour3d(X, Y, Z, data, 
                             contours=contours,
                             figure=fig, 
                             color=color, 
                             opacity=trans)

    if add_timelab:
        dtime = field.time.item()
        timelab = mlab.text3d(0, 0, 1.05 * extent[5], 
                              f"{dtime:6.0f}", 
                              name="Time_label", 
                              scale = extent[1]/50)
    else:
        timelab = None
        
    contour_obj = {'type':'contour3d',
                  'object':contour,
                  'timelab':timelab,
                  'extent': extent,
                  'dim_map':dim_map,
                 }
                 
    display_obj_list.append(contour_obj)
    
    print(f'{extent=}')
   
    if add_axes:  
        if axis_args is None: axis_args={}
        axes_obj = gen_axes(extent, **axis_args)
        display_obj_list.append(axes_obj)
      
    if add_base_plane:
        base_plane_obj = gen_baseplane(extent, color=(0.0,0.3,0.7))
        display_obj_list.append(base_plane_obj)              

    return display_obj_list
    
def trajectory_points(x, y, z, 
                      scalar_size, 
                      scalar_color, 
                      name, 
                      scale_factor,
                      scalar_size_min=0.1,
                      scalar_size_range=None):
                          
    display_obj_list = []        

    if scalar_size_range is None:
        scalar_size_range = [scalar_size.min(), scalar_size.max()]
        
    if scalar_size_range[0] is None:
        scalar_size_range[0] = scalar_size.min()

    if scalar_size_range[1] is None:
        scalar_size_range[1] = scalar_size.max()  
                    
    s = size_scale(scalar_size, scalar_size_min, scalar_size_range)              
                
    pts = mlab.quiver3d(x, y, z, 
                        s, s, s, 
                        scalars = scalar_color, 
                        scale_factor=scale_factor, 
                        name=name,  
                        mode = "sphere")

    pts.glyph.color_mode = "color_by_scalar"
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    # pts.glyph.glyph.clamping = False

    # pts.actor.property.representation = "surface"
    
    trajectory_points_obj = {'type':'trajectories',
                             'object':pts,
                             'scalar_size_min':scalar_size_min,
                             'scalar_size_range':scalar_size_range,
                            }
    
    display_obj_list.append(trajectory_points_obj)
    
    return display_obj_list

        
def get_colormap_as_array(colormap_name = None, num_colors=None):
    
    # Get a mayavi colormap as a numpy array
    # "gray"  #"blue-red" anything on https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html#adding-color-or-size-variations 
    if colormap_name is None: colormap_name =  "Greys"    
    if num_colors is None: num_colors = 256
    
    lut_mngr = LUTManager(number_of_colors=num_colors, lut_mode=colormap_name)
    
    colormap = lut_mngr.lut.table.to_array().astype(float) / (num_colors - 1)  # ~ (num_colors, 4) ∈ (0.0, 1.0)**4
    rgbs = colormap[:, :3]  # retrieve rgb channels
    #print(f"{colormap=}")
    
    return rgbs

def add_colour_table_to_volume(volume, rgbs, alpha_max=None):
    
    if alpha_max is None:
        alpha_max = 1.0

    # Customize ColorTransferFunction (ctf) and OpacityTransferFunction (otf)
    # NOTE 1: otf is to be generated from the "alpha" channel of ctf internally
    # NOTE 2: Those functions can be registered separately. If you want to do that refer to the 
    # src of load_ctfs at https://github.com/enthought/mayavi/blob/master/tvtk/util/ctf.py#L65
    volume_ctf = ctf.save_ctfs(volume._volume_property)
    #volume_ctf["range"] = [0.0, 1.0]
    #volume_ctf["rgb"] = [[i / (len(rgbs) - 1), *rgb] for i, rgb in enumerate(rgbs)]
    volume_ctf["rgb"] = [[i / (len(rgbs) - 1), 1,1,1] for i, rgb in enumerate(rgbs)]
    volume_ctf["alpha"] = [[0.0, 0.0],  # alpha has to be 0.0 at 0.0 == scalar
                           [1.0, alpha_max]]  # if 0.3 == scalar alpha ch is set at 0.3,
    # Register the new ctf to the volume module
    ctf.load_ctfs(volume_ctf, volume._volume_property)
    # Notify the volume of the registration of new ctf and otf
    volume.update_ctf = True
    
    return volume

def update_display_object(display_obj, time_index, time, frame_str, frame_vel):
    
    match display_obj['type']:
        case 'volume' | 'contour3d':
            
            field = display_obj['source']
            dim_map = display_obj['dim_map']
            
            Nx = field.coords[dim_map['x']].size
            Ny = field.coords[dim_map['y']].size
            field_now = field.sel(time=time, drop=False)
            field_now
            
            # print(f'{field_now.time.values=}')

            field_now = field_now.roll(
                        {dim_map['x']:(frame_vel['nx_roll']*time_index)%Nx, 
                         dim_map['y']:(frame_vel['ny_roll']*time_index)%Ny}).compute()
            
            obj = display_obj['object']
            obj.mlab_source.scalars = field_now.values
            
            timelab = display_obj.get('timelab', None)
            if timelab is not None:
        
                dtime = field_now.time.item()
#                 print(f"{time_index=} {dtime=}")
    
                timelab.text = f"{frame_str.format(n=time_index)} {dtime:6.0f}"
                
        case 'trajectories':
            
            traj = display_obj['source_xyz']
            Lx = traj.attrs['Lx']
            Ly = traj.attrs['Ly']
            
            ref_time = traj.ref_time
            ref_time_index = traj.ref_time_index
            
            if time in traj.time.values: 
                
                scalar_size_range = display_obj['scalar_size_range']
                scalar_size_min = display_obj['scalar_size_min']

                traj = traj.sel(time=time)
                x = traj['x'].values
                y = traj['y'].values
                z = traj['z'].values

                x, y = gal_trans(x, y, frame_vel['ur'], frame_vel['vr'], time_index * frame_vel['delt'], Lx, Ly)
                scalar_size = display_obj['source_scalar_size'].sel(time=time).values
              
                s = size_scale(scalar_size, scalar_size_min, scalar_size_range)              

            else:
                x=y=z=s=0
            pts = display_obj['object']
            ms = pts.mlab_source
            ms.trait_set(x=x, y=y, z=z, u=s, v=s, w=s)
       
    return display_obj

@mlab.animate(delay=100, ui=True, support_movie=True) 
def anim(olist, frame_vel, fig):
    times = olist[0]['source'].time.values
    ntimes = times.size
    frame_fm = f':0{int(ceil(log10(ntimes)))}d'
    frame_str = '{n' + frame_fm + '}'
    i=0
    while True:
        i = (i + 1) % ntimes
 
        fig.scene.disable_render = True
        for display_obj in olist: 
            
            display_obj = update_display_object(display_obj, i, times[i], frame_str,  frame_vel)             

#        camera_light0.azimuth = (50.0 +i*100) %360 -180
        
        fig.scene.disable_render = False
        yield

def gen_anim_stills(olist, frame_vel, fig, path, name, type=None ):
    
    if type is None : type = 'png'
    times = olist[0]['source'].time.values
    ntimes = times.size
    frame_fm = f':0{int(ceil(log10(ntimes)))}d'
    frame_str = '{n' + frame_fm + '}'
    
    for i in range(ntimes):
        
        fig.scene.disable_render = True
        for display_obj in olist:

            display_obj = update_display_object(display_obj, i, times[i], frame_str,  frame_vel)             

        fig.scene.disable_render = False
        mlab.draw(fig)
        
        figname = f'{name}_{frame_str.format(n=i)}.{type}'
        
        mlab.savefig(str(path / figname), figure=fig)
        print(f'Saved {path / figname}')
        
    return

def new_light(fig, light_number):
    
    number_of_lights = fig.scene.light_manager.number_of_lights
    
    print(f'Creating new light {light_number=} {number_of_lights=}')
    camera_light = CameraLight(fig.scene)
    fig.scene.light_manager.lights[(light_number-1):(light_number-1)] = [camera_light]
    fig.scene.light_manager.number_of_lights = light_number + 1
    print(f'Created new light {number_of_lights=}')
    return

def set_light(fig, light_number, elevation=None, azimuth=None, intensity=None, activate=False):
    
    number_of_lights = fig.scene.light_manager.number_of_lights
    print(f'{light_number=} {number_of_lights=}')
    
    if light_number < 0 : raise ValueError(f'{light_number=} must be non-negative.')
    if light_number > 7 : raise ValueError(f'{light_number=} <=7')
    
    if light_number > (number_of_lights -1):
        new_light(fig, light_number)
    
    camera_light = fig.scene.light_manager.lights[light_number]
 
    if not elevation is None: camera_light.elevation = np.clip(elevation, -90, 90)
    if not azimuth is None: camera_light.azimuth = np.clip(azimuth, -180, 180)
    if not intensity is None: camera_light.intensity = np.clip(intensity, 0, 1.0)
 
    camera_light.activate = activate
    return

def current_lights(fig):
    
    for i in range(fig.scene.light_manager.number_of_lights):
        c = fig.scene.light_manager.lights[i]
        print(f'{i} {c.elevation} {c.azimuth} {c.intensity} {c.activate}')
        
    return
