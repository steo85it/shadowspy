import logging
import time
import xarray as xr
import pandas as pd

# from examples.download_kernels import download_kernels
from config import ShSpOpt
from shadowspy.data_handling import fetch_and_process_data
from shadowspy.dem_processing import prepare_dem_mesh
from shadowspy.helpers import setup_directories, process_data_list
from shadowspy.raster_products import basic_raster_stats
from shadowspy.utilities import run_log
# from line_profiler_pycharm import profile

# @profile
def main_pipeline(opt):

    start_glb = time.time()
    
    # download kernels
    # if opt.download_kernels:
    #     download_kernels()

    # prepare useful dirs
    setup_directories(opt)

    # prepare mesh of the input dem
    start = time.time()
    logging.info(f"- Computing trimesh for {opt.dem_path}...")
    inner_mesh_path, outer_mesh_path, dem_path = prepare_dem_mesh(opt.dem_path, opt.tmpdir, opt.siteid, opt)
    logging.info(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # Determine the mode and prepare data list
    data_list, use_azi_ele, use_image_times = fetch_and_process_data(opt)
    logging.info(f"- Illuminating input DEM at {data_list}.")

    # Common arguments for both cases
    common_args = {
        'meshes': {'stereo': f"{inner_mesh_path}_st{opt.mesh_ext}", 'cart': f"{inner_mesh_path}{opt.mesh_ext}"},
        'basemesh_path': outer_mesh_path,
        'path_to_furnsh': f"{opt.indir}simple.furnsh",
        'point': opt.point_source,
        'scatter': opt.scatter,
        'extsource_coord': opt.extsource_coord,
        'source': opt.source,
        'dem_path': dem_path,
    }

    # actually compute irradiance at each element of data_list
    dsi_epo_path_dict = process_data_list(data_list, common_args, use_azi_ele, use_image_times, opt)

    # prepare mean, sum, max stats rasters
    if not use_azi_ele:
        dem = xr.open_dataarray(common_args['dem_path'])
        basic_raster_stats(dsi_epo_path_dict, opt.time_step_hours, crs=dem.rio.crs, outdir=opt.outdir, siteid=opt.siteid)

        # set up logs
        run_log(Fsun=opt.Fsun, Rb=opt.Rb, base_resolution=opt.base_resolution, siteid=opt.siteid, dem_path=dem_path, outdir=opt.outdir,
                start_time=opt.start_time, end_time=opt.end_time, time_step_hours=opt.time_step_hours,
                runtime_sec=round(time.time() - start_glb, 2), logpath=f"{opt.outdir}illum_stats_{opt.siteid}_{int(time.time())}.json")

    logging.info(f"Completed in {round(time.time() - start_glb, 2)} seconds.")

if __name__ == '__main__':

    opt = ShSpOpt()
    opt.setup_config()
    main_pipeline(opt)
