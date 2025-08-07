import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import rioxarray
# from line_profiler_pycharm import profile
from shapely.geometry import box
import logging

from mesh_operations import mesh_generation
from mesh_operations.helpers import prepare_inner_outer_mesh

#@profile
def prepare_dem_mesh(dem_path, tmpdir, siteid, opt):
    dem = xr.open_dataarray(dem_path)
    ext = opt.mesh_ext

    if opt.bbox_roi:
        minx, miny, maxx, maxy = opt.bbox_roi
        bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
        clipped_dem_path = f"{tmpdir}clipped_dem_{siteid}_{bbox_str}.tif"
        dem.rio.clip([box(minx, miny, maxx, maxy)]).rio.to_raster(clipped_dem_path) # 91% of time spent here (when gen_mesh not called)
        dem_path = clipped_dem_path
        logging.info(f"Clipped {opt.dem_path} to {bbox_str} and saved to {dem_path}")

    meshpath = generate_mesh(dem_path, tmpdir, siteid, ext, opt)
    inner_mesh, outer_mesh = generate_outer_mesh(dem_path, meshpath, tmpdir, ext, opt)

    return inner_mesh, outer_mesh, dem_path


def generate_mesh(dem_path, tmpdir, siteid, ext, opt):
    meshpath = tmpdir + dem_path.split('/')[-1].split('.')[0]
    mesh_generation.make(opt.base_resolution, [1], dem_path, out_path=f"{tmpdir}{siteid}_",
                         mesh_ext=ext, rescale_fact=1e-3, lonlat0=opt.lonlat0_stereo)
    shutil.move(f"{tmpdir}{siteid}_b{opt.base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{tmpdir}{siteid}_b{opt.base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    return meshpath


def generate_outer_mesh(dem_path, meshpath, tmpdir, ext, opt):
    # prepare full mesh (inner + outer)
    fartopo_path = opt.fartopo_path if opt.fartopo_path not in ['None', 'same_dem'] \
        else opt.dem_path if opt.fartopo_path == 'same_dem' \
        else None
    extres = {float(k): int(v) for k, v in opt.extres.items()}
    max_extension = float(opt.max_extension)

    if fartopo_path is not None:
        len_inner_faces_path = f'{tmpdir}len_inner_faces.txt'
        if os.path.exists(len_inner_faces_path):
            logging.info(f"- Reading existing stacked mesh file")
            last_ext = max({ext: res for ext, res in extres.items() if ext < max_extension}.keys())
            len_inner_faces = pd.read_csv(len_inner_faces_path, header=None).values[0][0]
            inner_mesh_path = meshpath
            outer_mesh_path = f"{tmpdir}LDEM_{int(last_ext)}M_outer"
        else:
            len_inner_faces, inner_mesh_path, outer_mesh_path = prepare_inner_outer_mesh(dem_path, fartopo_path, extres,
                                                                                         max_extension, opt.Rb, tmpdir,
                                                                                         meshpath, ext)
            with open(len_inner_faces_path, 'w') as f:
                f.write('%d' % len_inner_faces)
        # set full path with ext
        outer_mesh_path = outer_mesh_path + opt.mesh_ext if outer_mesh_path is not None else None
    else:
        inner_mesh_path = meshpath
        outer_mesh_path = None

    return inner_mesh_path, outer_mesh_path
