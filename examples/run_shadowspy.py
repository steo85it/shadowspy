import logging
import time
import xarray as xr
import pandas as pd

# from examples.download_kernels import download_kernels
from config import ShSpOpt
from shadowspy.data_handling import fetch_and_process_data
from shadowspy.dem_processing import prepare_dem_mesh
from shadowspy.helpers import setup_directories, process_data_list, setup_geometry, prepare_processing
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

    common_args = {
        'meshes': {'stereo': f"{inner_mesh_path}_st{opt.mesh_ext}", 'cart': f"{inner_mesh_path}{opt.mesh_ext}"},
        'basemesh_path': outer_mesh_path,
        'path_to_furnsh': f"{opt.indir}simple.furnsh",
        'point': opt.point_source,
        'scatter': opt.scatter,
        'extsource_coord': opt.extsource_coord,
        'source': opt.source,
        'observer': opt.observer,
        'frame': opt.frame,
        'dem_path': dem_path,
    }

    common_args, func_args = prepare_processing(use_azi_ele, use_image_times, data_list, common_args, opt)

    start = time.time()
    print("Starting setup_geometry...")
    # 0) Build your static geometry exactly once
    static = setup_geometry(
        meshes={'stereo': f"{inner_mesh_path}_st{opt.mesh_ext}",
                'cart': f"{inner_mesh_path}{opt.mesh_ext}"},
        dem_path=dem_path,
        dem_mask=None,  # GeoDataFrame or None
        crs=None,  # same CRS string
        scatter=opt.scatter,
        ffmat_path=getattr(opt, 'ffmat_path', None),
        Vst_path=getattr(opt, 'Vst_path', None),
        basemesh_path=outer_mesh_path,
    )
    print(f"Done with setup_geometry after {round(time.time()-start, 0)}. Starting process_data_list... ")
    start = time.time()
    # 1) Build the slim “dynamic common args”
    dynamic_common = {
        'path_to_furnsh': f"{opt.indir}simple.furnsh",
        'point': opt.point_source,
        'extsource_coord': opt.extsource_coord,
        'source': opt.source,
        'observer': opt.observer,
        'frame': opt.frame,
    }

    # 2) Change process_data_list to accept `static` and `dynamic_common`
    dsi_epo_path_dict = process_data_list(
        data_list,
        static=static,
        dynamic_common=dynamic_common,
        use_azi_ele=use_azi_ele,
        use_image_times=use_image_times,
        opt=opt
    )
    print(f"Done with process_data_list after {round(time.time()-start, 0)}. Starting basic_raster_stats... ")

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
