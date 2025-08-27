import pandas as pd
import xarray as xr
# import datetime
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def _load_flux_table(filin):
    df = pd.read_csv(filin, sep=r"\s+", comment=";", skiprows=142, header=None)
    df.columns = ["Wavelength (nm)","Irr Mar 25-29","Irr Mar 30-Apr 4","Irr Apr 10-16","Source"]
    df = df.loc[:, ["Wavelength (nm)", "Irr Apr 10-16"]].set_index("Wavelength (nm)")
    # store as numpy for cheap math later
    wl = df.index.to_numpy()
    vals = df.iloc[:,0].to_numpy(dtype=np.float64)
    step = float(np.nanmean(np.diff(wl)))
    return wl, vals, step

def get_Fsun(filin, epoch, wavelength=None):
    wl, vals, step = _load_flux_table(filin)
    if wavelength is None:
        return float((vals * step).sum())
    w = np.atleast_1d(wavelength)
    m = (wl >= w.min()) & (wl <= w.max()) if w.size > 1 else (wl == w.item())
    return float((vals[m] * step).sum())

# OLD STUFF BELOW
def get_Fsun3(filin, epoch, wavelength=None,  ):

    df = pd.read_csv(filin, sep='\s+', comment=';', skiprows=142, header=None)
    df.columns = ["Wavelength (nm)",  "Irradiance Mar 25-29",  "Irradiance Mar 30-Apr 4",
                  "Irradiance Apr 10-16", "Data Source"]

    df = df.loc[:, ["Wavelength (nm)", "Irradiance Apr 10-16"]]
    df = df.set_index('Wavelength (nm)')

    step = np.nanmean(df.index.diff()).round(2)

    if isinstance(wavelength, list):
        df = df.loc[wavelength[0]: wavelength[-1]]
    else:
        df = df.loc[wavelength-step: wavelength+step]

    return (df*step).sum().values

def get_Fsun2(filin, epoch, wavelength=None,  ):

    ds = xr.open_dataset(filin)
    # avoid issue with pre-1670 epochs
    ds_red = ds.isel(time=slice(100, 1000))  # reduce ds to useful epochs

    # convert cftimes to datetime
    datetimeindex = ds_red.indexes['time'].to_datetimeindex()
    ds_red['time'] = datetimeindex
    ds_red = ds_red.sel(time=epoch, method='nearest')

    # select range of epochs (ds_red does not accept "nearest")
    # epo0 = datetime.datetime.strptime(epoch, '%Y-%m-%d %H:%M:%S.%f')
    # epo1 = (epo0 + datetime.timedelta(days=366))

    if isinstance(wavelength, list):
        ds_res = ds_red.sel(  # time=slice(epo0.strftime('%Y-%m-%d'), epo1.strftime('%Y-%m-%d')),  #
            wavelength=slice(wavelength[0], wavelength[-1])).SSI.sum(dim='wavelength')
    else:
        ds_res = ds_red.sel(  # time=epo, #slice("2000-01-01", "2022-01-02"),
            wavelength=wavelength).SSI
    # plt.title("Solar Flux @1AU 115-320 nm (W/m2) ")
    # plt.show()
    # exit()
    return ds_res.time.values, ds_res.values