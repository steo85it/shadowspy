import logging
import os
import datetime
import xarray as xr
import rioxarray
import pandas as pd
# from line_profiler_pycharm import profile
from rasterio._io import Resampling
from tqdm import tqdm

from shadowspy.flux_util import get_Fsun
from shadowspy.image_util import read_img_properties
from shadowspy.render_dem import irradiance_at_date, render_match_image, render_at_date


def setup_directories(opt):
    # prepare dirs
    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(f"{opt.outdir}{opt.siteid}/", exist_ok=True)
    os.makedirs(opt.tmpdir, exist_ok=True)

#@profile
def process_data_list(data_list, common_args, use_azi_ele, use_image_times, opt):
    dsi_epo_path_dict = {}
    dem = xr.open_dataarray(common_args['dem_path'])

    for data in tqdm(data_list, total=len(data_list)):
        common_args, func_args = prepare_processing(use_azi_ele, use_image_times, data, common_args, opt)
        full_args = {**common_args, **func_args}

        try:
            epostr = f"{func_args['azi_ele_deg'][0]}_{func_args['azi_ele_deg'][1]}"
        except:
            epostr = datetime.datetime.strptime(func_args['epo_in'], '%Y-%m-%d %H:%M:%S.%f')
            epostr = epostr.strftime('%y%m%d%H%M%S')

        # if os.path.exists(f"{opt.outdir}{full_args['img_name']}_{epostr}.tif"):
        #     print(f"- {opt.outdir}{full_args['img_name']}_{epostr}.tif already processed. Skip.")
        #     continue

        if opt.irradiance_only:
            dsi, date_illum_str = irradiance_at_date(**full_args)
            key, value = dump_processing_results(dsi, dem, func_args, opt)
            dsi_epo_path_dict[key] = value
        else:
            if use_image_times:
                dsi_path = render_match_image(**full_args)
                dsi_epo_path_dict[func_args['epo_in']] = dsi_path
            else:
                dsi, date_illum_str = render_at_date(**full_args)
                key, value = dump_processing_results(dsi, dem, func_args, opt)
                dsi_epo_path_dict[key] = value

    return dsi_epo_path_dict


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
    dsi.flux.rio.to_raster(outpath, compression='zstd')

    # from matplotlib import pyplot as plt
    # dsi.flux.plot(robust=True)
    # plt.show()
    # dsi.flux.plot(vmin=0, vmax=0.1)
    # plt.show()

    return epostr, outpath
