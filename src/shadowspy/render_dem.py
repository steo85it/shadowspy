import logging
import os
import time

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from mesh_operations.mesh_utils import remove_degenerate_faces

# from line_profiler_pycharm import profile

from shadowspy.coord_tools import cart2sph, azimuth_elevation_to_cartesian, map_projection_to_azimuth
from mesh_operations.mesh_utils import import_mesh
from mesh_operations.mesh_tools import crop_mesh
from shadowspy.spice_util import get_sourcevec
import xarray as xr
from rasterio.enums import Resampling

from shadowspy.photometry import mmpf_mh_boyd2017lpsc
from shadowspy.math_util import angle_btw
from shadowspy.flux_util import get_Fsun
from shadowspy.shape import get_centroids as get_cents, get_surface_normals
from mesh_operations.plotting import rasterize_with_raytracing


def plot3d(mesh_path, var_to_plot, center='P'):

    import pyvista as pv
    if center == 'P':
        grid = pv.read(f"{mesh_path}")
        grid.cell_data[''] = np.nan
        grid.cell_data[''][:] = 0
        grid.cell_data[''][:] = var_to_plot
        grid.plot(show_scalar_bar=True, show_axes=True, cpos='xy')
    elif center == 'V':
        mesh = pv.read(f"{mesh_path}")
        pl = pv.Plotter()
        pl.add_mesh(mesh, show_edges=False)
        surf_points = mesh.extract_surface().points
        pl.add_points(surf_points, scalars=var_to_plot,
                      point_size=10)
        pl.show(cpos='xy')


def extended_source(sun_vecs, extsource_coord):
    import csv

    extsun_ = []
    with open(extsource_coord) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for row in csvReader:
            extsun_.append([x for x in row if x != ''])
    extsun_ = np.vstack(extsun_)
    # print(f"# Sun is an extended source (see {extsun_coord})...")

    sun_veccs = np.repeat(sun_vecs, extsun_.shape[0], axis=0)
    Zs = np.array([0, 0, 1])
    Us = sun_veccs / np.linalg.norm(sun_veccs, axis=1)[:, np.newaxis]
    Vs = np.cross(Zs, Us)
    Ws = np.cross(Us, Vs)
    Rs = 695700.  # Sun radius, km

    extsun_tiled = np.tile(extsun_, (sun_vecs.shape[0], 1))
    return sun_veccs + Vs * extsun_tiled[:, 0][:, np.newaxis] * Rs + Ws * extsun_tiled[:, 1][:, np.newaxis] * Rs

#@profile
def get_flux_at_date(shape_model, utc0, path_to_furnsh, albedo1=0.1, source='SUN',  point=True, scatter=False,
                     inc_flux=1361., frame='MOON_ME', observer='MOON',
                     center='P', basemesh=None, return_irradiance=False, azi_ele_deg=None, extsource_coord=None,
                     crs=None, ffmat_path=None):

    if center == 'V':
        C = shape_model.V
        N = shape_model.VN
    elif center == 'P':
        C = shape_model.P
        N = shape_model.N
    else:
        logging.error("*** center should be set to V or P. Exit.")
        exit()

    if azi_ele_deg == None:
        point_source_vecs = get_sourcevec(utc0=utc0, stepet=1, et_linspace=np.linspace(0, 1, 1),
                                   path_to_furnsh=path_to_furnsh,
                                   target=source, frame=frame, observer=observer)#*1e3
    else:
        # getting float lat/lon to pass
        latitude_deg, longitude_deg = np.rad2deg(np.vstack(cart2sph(np.mean(C, axis=0)))[1:])
        latitude_deg = latitude_deg[0]
        longitude_deg = longitude_deg[0]

        logging.warning("- Using source_distance = 1.5e8 km ~ 1AU. Adapt for other bodies.")

        # convert direction to local azimuth to retrieve consistent sun direction
        if crs is not None:
            azimuth_deg = map_projection_to_azimuth(latitude_deg, longitude_deg, direction_or_angle=azi_ele_deg[0], proj_wkt=crs)
            azi_angle_ = int(azimuth_deg)
        else:
            logging.warning("- No crs provided. Using input azimuth.")
            azi_angle_ = azi_ele_deg[0]

        point_source_vecs = azimuth_elevation_to_cartesian(azimuth_deg=azi_angle_, elevation_deg=azi_ele_deg[1],
                                                           distance=1.5e8,
                                                           observer_lat=latitude_deg, observer_lon=longitude_deg,
                                                           observer_alt=0)

    if point:
        # if point Sun
        source_vecs = point_source_vecs
        sourcedir = source_vecs / np.linalg.norm(source_vecs)
    else:
        if extsource_coord == None:
            logging.error(f"* Requested extended source, but set extsun_coord to None.")
            exit()
        
        if source == 'SUN':
            source_vecs = extended_source(point_source_vecs,
                                          extsource_coord=extsource_coord) #f"examples/aux/coordflux_100pts_outline33_centerlast_R1_F1_stdlimbdark.txt")
        elif source == 'EARTH':
            source_vecs = extended_source(point_source_vecs,
                                          extsource_coord=extsource_coord) #f"examples/aux/Source2D_Earth_100pts_centerlast.inc")
        else:
            logging.error(f"* Only SUN and EARTH set as possible light-sources. User asked for {source}. Exit.")
            exit()

        sourcedir = source_vecs / np.linalg.norm(source_vecs, axis=1)[:, np.newaxis]

    E = shape_model.get_direct_irradiance(inc_flux, sourcedir, basemesh=basemesh, center=center)

    if scatter:
        FF_path = "/panfs/ccds02/nobackup/people/sberton2/habnich/shadowspy_dev/examples/out/DG1/FF_0.05_3.0_1e-2.bin"
        if azi_ele_deg is not None:
            utc0 = np.array(['2026 AUG 01 00:00:00.00'])
        D = sourcedir
        Qrefl, QIR = irradiance_with_scattered_flux(ffmat_path, D, E)
        E = E + Qrefl + QIR

    if return_irradiance:
        print("shape E out", E.shape)
        return E

    # # get Moon centered cartesian coordinates of the Sun at date and correct to hr faces centers
    faces_to_sun = point_source_vecs - C

    # compute incidence angles for visible faces (redundant, many faces are the same, but shouldn't be an issue)
    incidence_angle1 = angle_btw(faces_to_sun, N)
    # compute emission angles to fictious camera above the scene (obs set to zenith)
    emission_angle1 = angle_btw(C, N)
    # compute phase angles
    phase_angle1 = angle_btw(C, faces_to_sun)

    # # get photometry of first bounce
    photom1 = mmpf_mh_boyd2017lpsc(phase=phase_angle1, emission=emission_angle1, incidence=incidence_angle1)

    # # compute radiance out of scatterer
    return E * albedo1 * photom1 * np.pi / inc_flux

def irradiance_with_scattered_flux(FF_path, D, E0):
    from flux.compressed_form_factors import CompressedFormFactorMatrix
    from scipy.constants import Stefan_Boltzmann as sigmaSB
    from flux.thermal import PccThermalModel1D
    from flux.model import update_incoming_radiances

    # if full FF is passed and scattered flux is requested, then...
    z = 0 #np.linspace(0, 3e-3, 31)
    FF = CompressedFormFactorMatrix.from_file(FF_path)
    shape_model = FF.shape_model

    rho=0.11
    emiss=0.95
    Fgeotherm = 0.005
    ti = 120
    rhoc = 9.6e5

    # get the previous fluxes and surface temperature
    Qrefl_prev = 0
    QIR_prev = 0
    Q0 = (1 - rho) * E0 + Fgeotherm
    T0 = 110
    Tsurf_prev = (Q0 / (sigmaSB * emiss)) ** 0.25

    emiss = 0 # PccThermalModel1D(z, T0, ti, rhoc, emiss, Fgeotherm, Q0, Tsurf_prev, bcond='Q')

    # get the next reflected and infrared fluxes
    Qrefl, QIR = update_incoming_radiances(
        FF, E0, rho, emiss, Qrefl=Qrefl_prev,
        QIR=QIR_prev, Tsurf=Tsurf_prev)

    return Qrefl, QIR

def render_at_date(
        # — all the per‐epoch args —
        P_st,
        # N_st,
        # P,
        # N,
        # dem_path,
        # meshes,
        shape_model,
        shape_model_st,
        basemesh,
        path_to_furnsh,
        source,
        observer,
        frame,
        inc_flux,
        extsource_coord,
        center='P',
        return_irradiance=False,
        epo_utc=None,
        azi_ele_deg=None,
        point=True,
        scatter=False,
        crs=None,
        show=False,
        # — catch everything else —
        **kwargs
    ):
    """
    Render terrain at one epoch, using prebuilt shape_models & geometry.
    """

    # ─── 1) format your date strings ────────────────────────────────
    if azi_ele_deg is None:
        dt = datetime.strptime(epo_utc.strip(), '%Y-%m-%d %H:%M:%S.%f')
        date_illum_str   = dt.strftime('%Y%m%d%H%M%S')
        date_illum_spice = dt.strftime('%Y %m %d %H:%M:%S')
    else:
        date_illum_str   = None
        date_illum_spice = None

    if center == 'V':
        assert "V_st" in kwargs, "V_st must be provided if center='V'."
        V_st = kwargs["V_st"]

    # ─── 2) compute flux at this epoch ─────────────────────────────
    print(f"Computing flux at {date_illum_str}")
    flux_at_obs = get_flux_at_date(
        shape_model=shape_model,
        utc0=date_illum_spice,
        path_to_furnsh=path_to_furnsh,
        source=source,
        observer=observer,
        frame=frame,
        inc_flux=inc_flux,
        center=center,
        point=point,
        scatter=scatter,
        basemesh=basemesh,
        return_irradiance=return_irradiance,
        azi_ele_deg=azi_ele_deg,
        extsource_coord=extsource_coord,
        crs=crs
    )

    # ─── 3) rasterize ───────────────────────────────────────────────
    if scatter:
        # uses shape_model_st under the hood
        ds = rasterize_with_raytracing({'flux': flux_at_obs}, shape_model_st)

    else:
        # build a tiny DataFrame of (x,y,flux)
        coords = V_st if center=='V' else P_st
        coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
        # flux = np.ascontiguousarray(np.asarray(flux_at_obs, dtype=np.float64))

        df = pd.DataFrame({
            'x': coords[:,0],
            'y': coords[:,1],
            'flux': flux_at_obs
        })

        # drop duplicates
        df = df[~df.duplicated(subset=['x','y'])]
        df = df.set_index(['y','x'], verify_integrity=True)

        ds = df.to_xarray()

    # assign CRS & convert to km
    if crs is not None:
        ds.rio.write_crs(crs, inplace=True)
    ds = ds.assign_coords(x = ds.x*1e3, y = ds.y*1e3)

    # fill holes by linear interpolation
    dsi = ds.interpolate_na(dim="x").interpolate_na(dim="y")

    return dsi, date_illum_str


def irradiance_at_date(**kwargs):
    """
    Get terrain irradiance at epoch
    :param inc_flux:
    :param basemesh_path:
    :param show:
    :param point:
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh:
    :param epo_utc:
    :param outdir:
    :param center: str, can take values P or V
    :param crs: str
    :param dem_mask: GeoDataFrame, polygon of region to crop
    :param return_irradiance: bool, must be True
    :return:
    """

    return render_at_date(return_irradiance=True, **kwargs)


def render_match_image(pdir, meshes, path_to_furnsh, img_name, epo_utc,
                       meas_path, outdir=None, center='P', basemesh_path=None, point=True, scatter=False,
                       **kwargs):
    """
    Render input terrain at epoch and match observed flux to input image
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh: str
    :param img_name: str
    :param epo_utc: str
    :param meas_path: str
    :param outdir:
    :param center:
    :return: str
    """
    print(f"- Processing {img_name} and clipping to {meas_path}...")

    # define processing dirs
    if outdir == None:
        outdir = f"{pdir}out/"
    os.makedirs(outdir, exist_ok=True)

    if not basemesh_path is None:
        logging.warning("* Outer mesh not yet implemented for render_match_image. Setting back to None.")
        basemesh_path = None
    
    # interpolate to NAC nodes
    meas = xr.open_dataarray(meas_path)
    meas = meas.where(meas >= 0)

    # raster outer shape to polygon
    ds = (meas.coarsen(x=1, boundary="trim").mean(skipna=True).
          coarsen(y=1, boundary="trim").mean(skipna=True))
    df = ds.to_dataframe().reset_index()
    # remove nans even for weird shapes
    df = df.loc[df.band_data > 0, ['x', 'y']] * 1e-3
    meas_outer_poly = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.x, df.y))
    meas_outer_poly = meas_outer_poly.dropna(axis=0).dissolve().convex_hull
    meas_outer_poly = gpd.GeoDataFrame({'geometry': meas_outer_poly.buffer(0.05, join_style=2)})
    meas_outer_poly.set_crs(meas.rio.crs, inplace=True) # both crs should be in km, to be consistent with Sun...

    # get full rendering at date
    dsi, date_illum_str = render_at_date(meshes, path_to_furnsh, epo_utc, center=center, crs=meas.rio.crs,
                                         dem_mask=meas_outer_poly, basemesh_path=basemesh_path, point=point,
                                         scatter=False)

    # interp to measured image coordinates
    rendering = dsi.rio.reproject_match(meas, Resampling=Resampling.bilinear,
                                        nodata=np.nan)

    # TODO make the threshold an adjustable parameter
    mask = meas.sel({'band': 1}) > 0.005

    # # apply "exposure factor" (median of ratio) to rendered image
    exposure_factor = rendering.flux.values / meas.sel({'band': 1}).values
    # plt.imshow(np.log10(exposure_factor), vmax=1, vmin=-1)
    # plt.colorbar()
    # plt.show()

    exposure_factor = exposure_factor[mask]
    # print(np.min(exposure_factor), np.max(exposure_factor), np.mean(exposure_factor),
    #       np.median(exposure_factor), np.std(exposure_factor))
    #
    #fig, ax = plt.subplots()
    #ax.hist(rendering.flux.values.ravel(), bins=100, range=[0.001,0.05], label='rendering')
    #ax.hist(meas.sel({'band': 1}).values.ravel(), bins=100, range=[0.001,0.05], label='NAC')
    #plt.legend()
    #plt.show()

    exposure_factor = np.nanmedian(exposure_factor) #.ravel()[~np.isnan(exposure_factor.ravel())])
    
    if exposure_factor > 0:
        rendering /= exposure_factor
    else:  # weird cases
        max_ratio = rendering.max() / meas.sel({'band': 1}).max()
        rendering /= max_ratio
        print(f"# Exposure=={exposure_factor}: possible issue or mainly shadowed image (?). "
              f"Normalizing with max_ratio={max_ratio.flux.values}. Exit.")
        exit()
        
    # save simulated image to raster
    outraster = f"{outdir}{img_name}_{date_illum_str}.tif"
    rendering.transpose('y', 'x').rio.to_raster(outraster, compress='zstd')

    print(f"- Flux for {img_name} saved to {outraster} (xy resolution = {rendering.rio.resolution()}mpp). "
          f"Normalized by {exposure_factor}.")

    # after writing:
    try:
        rendering.close()
    except Exception:
        pass
    try:
        meas.close()
    except Exception:
        pass

    return outraster
