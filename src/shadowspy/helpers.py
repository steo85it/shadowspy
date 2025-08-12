import logging
import os
import datetime
import xarray as xr
import rioxarray
import pandas as pd
# from line_profiler_pycharm import profile
from rasterio._io import Resampling
from tqdm import tqdm
import geopandas as gpd
import numpy as np
from pathlib import Path
import time

from shadowspy.flux_util import get_Fsun
from shadowspy.image_util import read_img_properties
from shadowspy.render_dem import irradiance_at_date, render_match_image, render_at_date
from mesh_operations.mesh_utils import import_mesh
from mesh_operations.mesh_tools import crop_mesh

RAYTRACING_BACKEND = 'embree' # 'cgal' #
RAYTRACING_BACKEND = RAYTRACING_BACKEND.lower()
if RAYTRACING_BACKEND == 'cgal':
    from shadowspy.shape import CgalTrimeshShapeModel as MyTrimeshShapeModel, get_centroids
elif RAYTRACING_BACKEND == 'embree':
    try:
        import embree
    except:
        logging.error("* You need to add embree_vars to the PATH to use embree")
        exit()
    from shadowspy.shape import EmbreeTrimeshShapeModel as MyTrimeshShapeModel
else:
    raise ValueError('RAYTRACING_BACKEND should be one of: "cgal", "embree"')

def setup_directories(opt):
    # prepare dirs
    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(f"{opt.outdir}{opt.siteid}/", exist_ok=True)
    os.makedirs(opt.tmpdir, exist_ok=True)

def setup_geometry(*,
    meshes,         # dict with 'stereo' & 'cart' paths
    dem_path,       # string path to your DEM GeoTIFF
    dem_mask,       # either GeoDataFrame or None
    crs,            # the DEM/mesh CRS (e.g. an EPSG string)
    scatter=False,  # whether you’re using the “scatter” branch
    ffmat_path=None,
    Vst_path=None,
    basemesh_path=None
):
    """
    One‐time geometry setup.  Returns a dict with:
      - dem_path          (str)
      - meshes            (same dict you passed in)
      - basemesh_path     (str or None)
      - P_st, N_st        np.ndarray centroids & normals of stereo mesh
      - P,   N            same for cart mesh
      - shape_model       MyTrimeshShapeModel for cart mesh
      - shape_model_st    for stereo mesh (or scatter‐based override)
      - basemesh          MyTrimeshShapeModel or None
    """
    start = time.time()
    # — 1) maybe crop your meshes once —
    if isinstance(dem_mask, gpd.GeoDataFrame):
        dem_mask = dem_mask.to_crs(crs)
        meshes_dir = Path(meshes['stereo']).parent
        cropped = {
            'stereo': meshes_dir / "cropped_st.vtk",
            'cart':   meshes_dir / "cropped.vtk",
        }
        crop_mesh(dem_mask, meshes, mask=dem_mask, meshes_cropped=cropped)
        stereo_file = str(cropped['stereo'])
        cart_file   = str(cropped['cart'])
    else:
        stereo_file = meshes['stereo']
        cart_file   = meshes['cart']

    print(f"first import: {time.time()-start}s")
    start = time.time()

    # — 2) import meshes once —
    V_st, F_st, N_st, P_st = import_mesh(
        stereo_file, get_normals=True, get_centroids=True
    )
    V, F, N, P = import_mesh(
        cart_file, get_normals=True, get_centroids=True
    )

    print(f"second import: {time.time()-start}s")
    start = time.time()

    # — 3) build your shape_model(s) —
    if scatter and ffmat_path and Vst_path:
        from flux.compressed_form_factors import CompressedFormFactorMatrix
        FF = CompressedFormFactorMatrix.from_file(ffmat_path)
        shape_model = FF.shape_model

        Vst_arr = np.load(Vst_path)
        Vst_arr = np.hstack([Vst_arr, np.ones((len(Vst_arr),1))])
        N_st2   = get_surface_normals(Vst_arr, shape_model.F)
        N_st2[N_st2[:,2]>0] *= -1
        shape_model_st = MyTrimeshShapeModel(Vst_arr, shape_model.F, N_st2)
    else:
        shape_model    = MyTrimeshShapeModel(V.astype(np.float64), F,    N)
        shape_model_st = MyTrimeshShapeModel(V_st.astype(np.float64), F_st, N_st)

    print(f"third import: {time.time()-start}s")
    start = time.time()

    # — 4) optional basemesh —
    if basemesh_path:
        V_ds, F_ds, N_ds, P_ds = import_mesh(
            basemesh_path, get_normals=True, get_centroids=True
        )
        basemesh = MyTrimeshShapeModel(V_ds, F_ds, N_ds)
    else:
        basemesh = None

    print(f"fourth import: {time.time()-start}s")

    # — 5) package everything up —
    return {
        'dem_path':       dem_path,
        'meshes':         meshes,
        'basemesh_path':  basemesh_path,
        'P_st':           P_st,
        'N_st':           N_st,
        'P':              P,
        'N':              N,
        'shape_model':    shape_model,
        'shape_model_st': shape_model_st,
        'basemesh':       basemesh,
    }

#
# # --- then in your main: ---
#
# # 0) before the loop
# geom = setup_geometry(meshes, dem_mask, crs,
#                       scatter=scatter,
#                       ffmat_path=ffmat_path,
#                       Vst_path=Vst_path,
#                       basemesh_path=basemesh_path)
#
# for epoch in epochs:
#     date_illum_str = epoch.strftime("%Y%m%d")
#     print(f"Rendering at {date_illum_str}")
#
#     # everything you need is in `geom`
#     render_at_date( date_illum_str,
#                     P_st=geom['P_st'],
#                     N_st=geom['N_st'],
#                     P=geom['P'],
#                     N=geom['N'],
#                     shape_model=geom['shape_model'],
#                     shape_model_st=geom['shape_model_st'],
#                     basemesh=geom['basemesh'],
#                     # … plus any per‐epoch parameters …
#                   )


# in helpers.py

def process_data_list(data_list,
                      static,            # ← the dict from setup_geometry
                      dynamic_common,    # ← the lighter dict you just built
                      use_azi_ele,
                      use_image_times,
                      opt):
    dsi_epo_path_dict = {}
    dem = xr.open_dataarray(static['dem_path'])

    for data in tqdm(data_list, total=len(data_list)):
        # 2a) prepare the per‐epoch bits
        common_args, func_args = prepare_processing(
            use_azi_ele, use_image_times, data, dynamic_common, opt
        )

        # 2b) merge everything:
        # full_args = {
        #     **static,      # includes P, N, shape_models, dem_path, etc.
        #     **common_args, # includes date strings, azi_ele, img_name, etc.
        #     **func_args,   # anything else you need
        # }
        full_args = {
            **static,  # all your one‐time geometry & models
            **dynamic_common,  # your small common params
            **func_args,  # the per‐epoch bits (azi/ele, epo_in, img_name, etc.)
        }

        # print(full_args)
        # exit()

        # 2c) pick your renderer or irradiance function
        if opt.irradiance_only:
            dsi, date_illum_str = irradiance_at_date(**full_args)
            key, value = dump_processing_results(dsi, dem, func_args, opt)
        else:
            if use_image_times:
                dsi_path = render_match_image(**full_args)
                key, value = func_args['epo_in'], dsi_path
            else:
                dsi, date_illum_str = render_at_date(**full_args)
                key, value = dump_processing_results(dsi, dem, func_args, opt)

        dsi_epo_path_dict[key] = value

    return dsi_epo_path_dict


#@profile
# def process_data_list(data_list, common_args, use_azi_ele, use_image_times, opt):
#     dsi_epo_path_dict = {}
#     dem = xr.open_dataarray(common_args['dem_path'])
#
#     for data in tqdm(data_list, total=len(data_list)):
#         common_args, func_args = prepare_processing(use_azi_ele, use_image_times, data, common_args, opt)
#         full_args = {**common_args, **func_args}
#
#         try:
#             epostr = f"{func_args['azi_ele_deg'][0]}_{func_args['azi_ele_deg'][1]}"
#         except:
#             epostr = datetime.datetime.strptime(func_args['epo_in'], '%Y-%m-%d %H:%M:%S.%f')
#             epostr = epostr.strftime('%y%m%d%H%M%S')
#
#         # if os.path.exists(f"{opt.outdir}{full_args['img_name']}_{epostr}.tif"):
#         #     print(f"- {opt.outdir}{full_args['img_name']}_{epostr}.tif already processed. Skip.")
#         #     continue
#
#         if opt.irradiance_only:
#             dsi, date_illum_str = irradiance_at_date(**full_args)
#             key, value = dump_processing_results(dsi, dem, func_args, opt)
#             dsi_epo_path_dict[key] = value
#         else:
#             if use_image_times:
#                 dsi_path = render_match_image(**full_args)
#                 dsi_epo_path_dict[func_args['epo_in']] = dsi_path
#             else:
#                 dsi, date_illum_str = render_at_date(**full_args)
#                 key, value = dump_processing_results(dsi, dem, func_args, opt)
#                 dsi_epo_path_dict[key] = value
#
#     return dsi_epo_path_dict


def prepare_processing(use_azi_ele, use_image_times, data, common_args, opt):
    if use_azi_ele:
        # For azimuth-elevation inputs
        func_args = {'azi_ele_deg': data, 'epo_in': '2000-01-01 00:00:00.0'}
    elif use_image_times:
        func_args = {'pdir': opt.root, 'img_name': data[0], 'epo_utc': data[1], 'epo_in': data[1], 'meas_path': data[2]}
    else:
        func_args = {'epo_utc': data, 'epo_in': data}

    if opt.flux_path not in [None, 'None']:
        Fsun = get_Fsun(opt.flux_path, func_args['epo_in'], wavelength=opt.wavelength)
    else:
        Fsun = opt.Fsun

    common_args['inc_flux'] = Fsun

    if opt.ffmat_path not in [None, 'None']:
        common_args['ffmat_path'] = opt.ffmat_path
    if opt.Vst_path not in [None, 'None']:
        common_args['Vst_path'] = opt.Vst_path

    return common_args, func_args

#@profile
def dump_processing_results(dsi, dem, func_args, opt):
    # get illum epoch string
    try:
        epostr = f"{func_args['azi_ele_deg'][0]}_{func_args['azi_ele_deg'][1]}"
    except:
        epostr = datetime.datetime.strptime(func_args['epo_in'], '%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')

    # define useful quantities
    outpath = f"{opt.outdir}{opt.siteid}/{opt.siteid}_{epostr}.tif"

    # save each output to raster to save memory
    dsi.rio.write_crs(dem.rio.crs, inplace=True)
    dsi = dsi.assign_coords(time=func_args['epo_in'])
    dsi = dsi.expand_dims(dim="time")
    dsi = dsi.rio.reproject_match(dem, resampling=Resampling.nearest) # cubic_spline)
    dsi.flux.rio.to_raster(outpath, compress='zstd')

    from matplotlib import pyplot as plt
    dsi.flux.plot(robust=True)
    plt.show()
    # dsi.flux.plot(vmin=0, vmax=0.1)
    # plt.show()

    return epostr, outpath
