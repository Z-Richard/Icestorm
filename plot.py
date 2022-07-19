# Source Generated with Decompyle++
# File: plot.cpython-39.pyc (Python 3.9)

"""
Author: Richard Zhuang (hz542)
Date: June 29, 2022
"""
import os
from glob import glob

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from statistics import median

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm, ticker

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import metpy.calc as mpcalc
from metpy.units import units
import scipy.ndimage as ndimage

from multiprocessing import Pool, cpu_count
from functools import partial


# filepath = os.path.join('..', '..', '..', 'scratch', 'hz542', 'Icestorm')
# events_dir = os.path.join(filepath, 'Events_062022')
# reanalysis_dir = os.path.join(filepath, 'reanalysis')
# events_dir = 'Events_070522_eps=5_t=2.5_lon=mean-lat'
events_dir = 'Events_062022'
reanalysis_dir = 'reanalysis'

prj = ccrs.PlateCarree()
res = '110m'
lat0, lat1, lon0, lon1 = 25, 55, -110, -55

cdict = {
    -50: '#f5c4e0',
    -40: '#ad619f',
    -30: '#5c3482',
    -20: '#4e5091',
    -10: '#5192cf',
    -1: '#73dade',
    0: '#39914c',
    10: '#fafc51',
    20: '#de9c47',
    30: '#e63d27',
    40: '#990042'
}


def add_basemap(ax, land=True):
    """Add basemap to the plot."""
    border_dict = {'lw': 0.5, 'alpha': 0.5}
    fill_dict = {'lw': 0.5, 'alpha': 0.3}
    ax.add_feature(cfeature.COASTLINE.with_scale(res), **border_dict)
    ax.add_feature(cfeature.STATES.with_scale(res), **border_dict)
    if land:
        ax.add_feature(cfeature.LAND.with_scale(res), **fill_dict)
        ax.add_feature(cfeature.OCEAN.with_scale(res), **fill_dict)
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.3)


def timedelta_to_hrs(td, process_series=True):
    """Convert a timedelta object to the number of hours elapsed."""
    if process_series:
        return np.round(td.dt.components.hours + td.dt.components.minutes / 60 +
                        td.dt.components.days * 24, 2)
    return np.round(td.components.hours + td.components.minutes / 60 +
                    td.components.days * 24, 2)


def retrieve_files(time, **kwargs):
    """
    Utility function to retrieve all useful netcdf4 files at a given time. 
    """
    month = f'{time.month:02d}'
    # The variable yr is only used to index into the year folder
    yr = time.year if time.month in [11, 12] else time.year - 1
    yr_folder = os.path.join(reanalysis_dir, str(yr))

    pres_file = os.path.join(yr_folder, f'e5_{time.year}{month}_pl.nc')
    pres_temp_file = os.path.join(
        yr_folder, f'e5_{time.year}{month}_pl_temperature.nc')
    sfc_file = os.path.join(yr_folder, f'e5_{time.year}{month}_sfc.nc')

    pres_data = xr.open_dataset(pres_file, **kwargs)
    pres_temp_data = xr.open_dataset(pres_temp_file, **kwargs)
    sfc_data = xr.open_dataset(sfc_file, **kwargs)

    return pres_data, pres_temp_data, sfc_data


def find_event_median_time(event):
    """
    Find the median starting time of an event.

    Parameters
    ----------
    event : Pd.DataFrame
        The actual event. 
        Invariant: the time column of the event should already be converted
        to a datetime series and be named as 'start_time'. The event should
        also be sorted by the starting time. 

    Returns
    -------
    `datetime`
        The median starting time of an event. 
    """
    start_time = event.loc[0, 'start_time']
    time_diff = timedelta_to_hrs(event['start_time'] - start_time).tolist()
    median_time_diff = median(time_diff)
    return start_time + timedelta(hours=median_time_diff)


def find_event_bound_and_time(event, extent=[10, 10, 20, 10]):
    """
    Find the [south, north, west, east] bound of an ZR event based on
    the median latitude and longitude of the list of data points. 

    Parameters
    ----------
    event : Pd.DataFrame
        The actual event. 
        Invariant: the time column of the event should already be converted
        to a datetime series. 

    extent : list
        The extent of our reanalysis field relative to the median
        latitude and longitude of the event, in the order of
        [south, north, west, east]. 

    Returns
    -------
    `list`
        The [west, east, south, north] bound, inclusive. Should
        round up to 0.25. 

    `datetime`
        The median starting time of the event
    """
    latitude = event['lat'].tolist()
    longitude = event['lon'].tolist()

    south, north, west, east = extent[0], extent[1], extent[2], extent[3]

    med_lat = median(latitude)
    med_lon = median(longitude)

    west_bound = (med_lon // 0.25) * 0.25 - west
    east_bound = (med_lon // 0.25) * 0.25 + east
    south_bound = (med_lat // 0.25) * 0.25 - south
    north_bound = (med_lat // 0.25) * 0.25 + north

    return [west_bound, east_bound, south_bound, north_bound], find_event_median_time(event)


def plot_event(file):
    """
    Plot a single long-duration ZR event. 
    """
    df = pd.read_csv(file)
    df['start_time'] = pd.to_datetime(df['start_time'])

    start_time = df.loc[0, 'start_time']
    time_diff = df['start_time'] - start_time
    df['time_diff'] = timedelta_to_hrs(time_diff)

    fig = plt.figure(figsize=(8, 8), dpi=120)
    fig.clf()
    ax = fig.add_subplot(111, projection=prj)
    ax.set_extent((lon0, lon1, lat0, lat1), crs=prj)

    im = ax.scatter(df.loc[:, 'lon'], df.loc[:, 'lat'],
                    s=20, c=df.loc[:, 'time_diff'], transform=prj, cmap='copper')
    cax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.03,
        ax.get_position().height
    ])
    plt.colorbar(im, cax=cax)
    cax.set_ylabel('Hours after Onset', fontsize=8)
    add_basemap(ax)

    fmt = 'Start Time: %Y-%m-%d %H:%M UTC'
    title = start_time.strftime(fmt)
    ax.set_title(title, fontsize=10)

    plt.show()


def plot_events_by_year(year):
    """
    Plot a series of scatterplot for long-duration ZR events
    each year.  
    """
    year_dir = os.path.join(events_dir, str(year))
    files = os.listdir(year_dir)
    files.sort()
    for file in files:
        fp = os.path.join(year_dir, file)
        plot_event(fp)


def find_all_events_by_year(events_dir, year):
    """
    Returns all events in a year. 
    """
    year_dir = os.path.join(events_dir, str(year))
    files = os.listdir(year_dir)
    return [os.path.join(year_dir, file) for file in files]


def cdict_to_cmap(cdict):
    """
    Convert a pre-defined dictionary of color to a colormap.
    """
    keys, values = cdict.keys(), cdict.values()
    vmin, vmax = min(keys), max(keys)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norm_keys = [norm(key) for key in keys]
    return mcolors.LinearSegmentedColormap.from_list('generated cmap', list(zip(norm_keys, values)))


def create_mask_for_event(event_df, da, size):
    """
    Create a mask for the reanalysis data based on the data points.
    
    Note: This function is written in a recursive way. 
    
    Parameter
    ---------
    event : pd.DataFrame
        stores all the (latitude, longitude) information of the stations
        
    da : xr.DataArray
        A DataArray that stores all the data
        
    size : int
        1/2 of the size of a grid box
        
    Returns
    -------
    The boolean mask and the masked array. 
    """
    mask = da.coords['latitude'] >= 1e10
    for _, row in event_df.iterrows():
        lat, lon = row.loc['lat'], row.loc['lon']
        west, east = (lon // 0.25) * 0.25 - size, (lon // 0.25) * 0.25 + size
        south, north = (lat // 0.25) * 0.25 - size, (lat // 0.25) * 0.25 + size
        mask = mask | (
            (da.coords['latitude'] >= south) & 
            (da.coords['latitude'] <= north) & 
            (da.coords['longitude'] >= west) & 
            (da.coords['longitude'] <= east)
        )
    return mask
#     lat, lon = event_df.iloc[0].loc['lat'], event_df.iloc[0].loc['lon']
#     west, east = (lon // 0.25) * 0.25 - size, (lon // 0.25) * 0.25 + size
#     south, north = (lat // 0.25) * 0.25 - size, (lat // 0.25) * 0.25 + size
#     mask = (
#         (da.coords['latitude'] >= south) & 
#         (da.coords['latitude'] <= north) & 
#         (da.coords['longitude'] >= west) & 
#         (da.coords['longitude'] <= east)
#     )
#     if len(event_df) == 1:
#         return mask
#     return mask | create_mask_for_event(event_df.iloc[1:, :], da, size)


def retrieve_merge_files(start_time, end_time, bound):
    """
    Retrieve and merge files based on the start time, end time, 
    and bound.
    """
    pres_data, pres_temp_data, sfc_data = retrieve_files(start_time, cache=False)
    eend_time = end_time + timedelta(hours=6)
    sstart_time = start_time - timedelta(hours=6)
    if eend_time.month != start_time.month:
        p, t, s = retrieve_files(eend_time, cache=False)
        pres_data = xr.concat([pres_data, p], dim='time')
        pres_temp_data = xr.concat([pres_temp_data, t], dim='time')
        sfc_data = xr.concat([sfc_data, s], dim='time')
    
    if sstart_time.month != start_time.month:
        p, t, s = retrieve_files(sstart_time, cache=False)
        pres_data = xr.concat([p, pres_data], dim='time')
        pres_temp_data = xr.concat([t, pres_temp_data], dim='time')
        sfc_data = xr.concat([s, sfc_data], dim='time')
                
    pres_data = pres_data.reindex(latitude=pres_data.latitude[::-1])
    pres_temp_data = pres_temp_data.reindex(latitude=pres_temp_data.latitude[::-1])
    sfc_data = sfc_data.reindex(latitude=sfc_data.latitude[::-1])
    
    west, east, south, north = bound[0], bound[1], bound[2], bound[3]
    
    pres_data = (pres_data.sel(time=slice(sstart_time, eend_time))
                          .sel(latitude=slice(south, north))
                          .sel(longitude=slice(west, east)))
    pres_temp_data = (pres_temp_data.sel(time=slice(sstart_time, eend_time))
                                    .sel(latitude=slice(south, north))
                                    .sel(longitude=slice(west, east)))
    sfc_data = (sfc_data.sel(time=slice(sstart_time, eend_time))
                        .sel(latitude=slice(south, north))
                        .sel(longitude=slice(west, east)))
    return pres_data, pres_temp_data, sfc_data


def process_events_by_year(event_dir, year):
    """
    A function that processes all necessary events by year and output
    them to a csv file. 
    
    Parameter
    ---------
        year : int
        
        event_dir : str
            the event directory that stores all the event
    
    Returns
    -------
        pd.DataFrame
    """
    print(f'Processing: {year}')
    year_dir = os.path.join('reanalysis', str(year))
    
    events = find_all_events_by_year(event_dir, year)
    events.sort()
    
    result = []
    
    for event in events:
        df = pd.read_csv(event)
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        start_time = df.iloc[0].loc['start_time']
        end_time = df.iloc[-1].loc['start_time']
        
        bound, time = find_event_bound_and_time(df)
        west, east, south, north = bound[0], bound[1], bound[2], bound[3]
        median_lat, median_lon = south + 10, west + 20
        
        pres_data, pres_temp_data, sfc_data = retrieve_merge_files(start_time, end_time, bound)
        
        u_sfc, v_sfc, t_sfc = sfc_data.u10, sfc_data.v10, sfc_data.t2m
        u_850, v_850 = pres_data.u.sel(level=850), pres_data.v.sel(level=850)
        t_850 = pres_temp_data.t.sel(level=850)

        tadv_850 = mpcalc.advection(t_850, u_850, v_850)
        tadv_sfc = mpcalc.advection(t_sfc, u_sfc, v_sfc)
        
        tadv_850_tendency = mpcalc.first_derivative(tadv_850, axis='time')
        tadv_sfc_tendency = mpcalc.first_derivative(tadv_sfc, axis='time')

        index = ((tadv_850 - tadv_sfc) * np.exp(-(t_sfc - 273.15)) / 
                (abs(tadv_850_tendency) + abs(tadv_sfc_tendency)))
        index = xr.where(index < 0, 0, index)        

        sub_index = index.interp(time=time)        
                
        index_mask = create_mask_for_event(df, sub_index, 1.25)
        masked_index = xr.where(index_mask, sub_index, np.nan)

        index_mean = masked_index.mean().values[()]
        
        result.append([time, event, round(df['zr_hours'].mean(), 3), 
                       round(index_mean, 3), median_lat, median_lon])
        
    return pd.DataFrame(np.array(result), columns=['median_time', 'filename', 
                                                   'zr_hours', 'index', 'med_lat', 'med_lon'])


temp_cmap = cdict_to_cmap(cdict)


def acc_to_ds(dims, coords, attrs, *args):
    """
    Parameters
    ----------
    dims : array_like of str
        The dimension names for the DataArray
        
    coords : np.ndarray, or array_like
        The actual coordinate information
        
    attrs : dict
        Description for the entire dataset.
        
    arg(s) : tuple of (array_like, dict, str)
        The dict should contain the attribute information for each DataArray
    """
    updated_args = [
        (
            arg[0], 
            xr.DataArray(
                data=np.array(arg[1]),
                dims=dims,
                coords=coords,
                attrs=arg[2]
            )
        )
        for arg in args
    ]
    
    ds = xr.Dataset(data_vars=dict(updated_args), attrs=attrs)
    
    return ds


class DataModel:
    """
    
    """
    @staticmethod
    def set_time(time, *args):
        """
        Set the time for *args. 
        
        Parameters
        ----------
        time : datetime.datetime
        *args : xr.DataSet or xr.DataArray
        
        Returns
        -------
        xr.DataSet or xr.DataArray
        """
        return [arg.interp(time=time) for arg in args]
    
    @staticmethod
    def reverse_latitude(*args):
        """Reverse the latitude coordinates for *args."""
        return [arg.reindex(latitude=arg.latitude[::-1]) for arg in args]
    
    @staticmethod
    def set_extent(extent, *args):
        """
        Set the extent for *args.
        
        Parameters
        ----------
        extent : array_like (length 4)
            In the order of (west, east, south, north)
        """
        west, east, south, north = extent[0], extent[1], extent[2], extent[3]
        return [arg.sel(latitude=slice(south, north))
                   .sel(longitude=slice(west, east)) for arg in args]
    
    def __init__(self, time, extent=None, **kwargs):
        """
        Parameters
        ----------
        time : datetime.datetime
        
        extent : array_like (length 4)
            In the order of (west, east, south, north)
        
        **kwargs : arguments that will be directly passed onto self.retrieve_files
        """
        # Sets the time attribute for the data model
        self.time = time
        
        p, t, s = retrieve_files(time, **kwargs)
        p, t, s = self.reverse_latitude(p, t, s)
        p, t, s = self.set_time(time, p, t, s)
        
        if extent is not None:
            p, t, s = self.set_extent(extent, p, t, s)
            
        self.pres_data = p
        self.pres_temp_data = t
        self.sfc_data = s

        
class BasePlot:
    # Some class attributes useful for plotting
    clabeldict_ = {
        'fontsize': 6, 
        'inline': 1,
        'inline_spacing': 10,
        'fmt': '%i',
        'rightside_up': True,
        'use_clabeltext': True
    }
    
    msl_kwargs_ = dict(
        levels=np.arange(950, 1034, 4),
        colors='black',
        linewidths=1.,
        linestyles='solid',
        transform=prj
    )
    
    gph_kwargs_ = dict(
        levels=np.arange(4200, 6000, 60),
        colors='red',
        linewidths=1.,
        linestyles='dashed', 
        transform=prj, 
        alpha=0.7
    )
    
    cbar_kwargs_ = dict(
        pad=0.02, 
        shrink=0.8, 
        aspect=24, 
        labelsize=8, 
        fontsize=7
    )
    
    def __init__(self, extent, figsize=(8, 8), dpi=120, projection=prj):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection=projection)
        
        west, east, south, north = extent[0], extent[1], extent[2], extent[3]
        self.ax.set_extent((west, east, south, north), crs=prj)
        
        self.add_basemap(land=True)
        
    def add_basemap(self, land=True):
        """
        Add basemap to the plot. 
        """
        border_dict = {'lw': 0.5, 'alpha': 0.5}
        fill_dict = {'lw': 0.5, 'alpha': 0.3}
        self.ax.add_feature(cfeature.COASTLINE.with_scale(res), **border_dict)
        self.ax.add_feature(cfeature.STATES.with_scale(res), **border_dict)
        if land:
            self.ax.add_feature(cfeature.LAND.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.OCEAN.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.3)
        return self.ax
    
    def contour(self, lons, lats, data, clabel=True, clabeldict=None, **kwargs):
        """
        data - should be an `xr.DataArray` with only two dimensions `lat` and `lon`. 
        """
        msl = kwargs.pop('msl', False)
        gph = kwargs.pop('gph', False)
        
        if msl:
            kwargs.update(BasePlot.msl_kwargs_)
        elif gph:
            kwargs.update(BasePlot.gph_kwargs_)
            
        cs = self.ax.contour(lons, lats, data, **kwargs)
            
        if clabel:
            if clabeldict is not None:
                self.ax.clabel(cs, **clabeldict)
            else:
                self.ax.clabel(cs, **BasePlot.clabeldict_)
    
    @staticmethod
    def set_cmap(cdict, interval):
        import matplotlib.colors as mcolors
        import math

        lastk = -math.inf
        mpl_cdict = dict(red=[], green=[], blue=[], alpha=[])
        keys = list(cdict.keys())
        norm = mcolors.Normalize(vmin=keys[0], vmax=keys[-1])
        for k, v in cdict.items():
            if k < lastk:
                raise ValueError('Incorrect color map definition. The keys for colormaps'
                                 'should increase monotonously.')
            if isinstance(v, str):
                (r1, g1, b1, a1) = (r2, g2, b2, a2) = mcolors.to_rgba(v)
            elif isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError('If the value for a colormap is a tuple,'
                                     f'it should have a length of 2, while the current length is {len(v)}.')
                r1, g1, b1, a1 = mcolors.to_rgba(v[0])
                r2, g2, b2, a2 = mcolors.to_rgba(v[1])

            mpl_cdict['red'].append([norm(k), r1, r2])
            mpl_cdict['green'].append([norm(k), g1, g2])
            mpl_cdict['blue'].append([norm(k), b1, b2])
            mpl_cdict['alpha'].append([norm(k), a1, a2])

            lastk = k

        span = keys[-1] - keys[0]
        cmap = mcolors.LinearSegmentedColormap('cmap', mpl_cdict, N=span // interval)
        clevs = np.arange(keys[0], keys[-1] + interval, interval)
        return clevs, norm, cmap
    
    def set_cbar(self, norm, cmap, **kwargs):
        """
        Set the colorbar of the plot. 
        """
        labelsize = kwargs.pop('labelsize', 8)
        fontsize = kwargs.pop('fontsize', 7)
        orientation = kwargs.pop('orientation', 'horizontal')
        annotation = kwargs.pop('annotation', '')

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation=orientation,
                                 **kwargs)
        cbar.ax.tick_params(labelsize=labelsize)

        if orientation == 'horizontal':
            cbar.ax.set_xlabel(annotation, fontsize=fontsize)

        if orientation == 'vertical':
            cbar.ax.set_ylabel(annotation, fontsize=fontsize)
    
    def contourf(self, lons, lats, data, cdict, interval, cbar=True, cbardict=None, **kwargs):
        clevs, norm, cmap = self.set_cmap(cdict, interval)
        _ = self.ax.contourf(lons, lats, data, clevs, norm=norm, cmap=cmap, **kwargs)
        
        if cbar:
            if cbardict is not None:
                cbardict.update(BasePlot.cbar_kwargs_)
            
            self.set_cbar(norm, cmap, **cbardict)
                      

class Plot:
    """
    A plotting module that aims to make plotting the synoptic features
    of long-duration ZR events easier. 
    
    Invariants
    ----------
    1. Extent of the plot is important. 
    """
    @classmethod
    def from_csv(cls, event_csv):
        """
        Create a Plot object by reading in a csv file. 
        """
        df = pd.read_csv(event_csv)
        df['start_time'] = pd.to_datetime(df['start_time'])
        extent, time = find_event_bound_and_time(df)
        return cls(time=time, extent=extent, df=df)

    def __init__(self, time=None, figsize=(8, 8), dpi=120, extent=(-140, -55, 20, 65),
                 grid=0.25, df=None):
        """
        Initialize some fields here. 

        Arguments:
            time - `datetime.datetime` the time of the plot
            figsize - `figsize` of the `plt.figure()` method
            dpi - `dpi` of the `plt.figure()` method
            extent - (lon0, lon1, lat0, lat1) - specify the extent of the map
        """
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection=prj)

        self.grid = grid

        if time is not None:
            self.set_time(time)
            
        self.set_extent(extent)

        # Overlay the stations on top of the chart by setting zorder to a large number
        if df is not None:
            _ = self.ax.scatter(df.loc[:, 'lon'], df.loc[:, 'lat'], s=20,
                            transform=prj, c='k', zorder=10, alpha=0.5)
            
    def cla(self):
        """A wrapper around the plt.cla()."""
        self.ax.cla()

    def set_time(self, time):
        """
        Set the time for the plot object and retrieve the reanalysis data for plotting. 
        """
        self.time = time
        self._set_reanalysis(time)
        
    def set_extent(self, extent):
        """
        Set the extent for the plot object. 
        """
        self.extent = extent
        self.ax.set_extent(extent, prj)
            
        self.lons = np.arange(extent[0], extent[1] + self.grid, self.grid)
        self.lats = np.arange(extent[2], extent[3] + self.grid, self.grid)
            
        self.pres_data = self.pres_data.reindex(latitude=self.pres_data.latitude[::-1])
        self.pres_temp_data = self.pres_temp_data.reindex(latitude=self.pres_temp_data.latitude[::-1])
        self.sfc_data = self.sfc_data.reindex(latitude=self.sfc_data.latitude[::-1])
        
        self.pres_data = self.pres_data.sel(latitude=slice(extent[2], extent[3]), 
                                            longitude=slice(extent[0], extent[1]))
        self.pres_temp_data = self.pres_temp_data.sel(latitude=slice(extent[2], extent[3]), 
                                                      longitude=slice(extent[0], extent[1]))
        self.sfc_data = self.sfc_data.sel(latitude=slice(extent[2], extent[3]), 
                                          longitude=slice(extent[0], extent[1]))
        
    def _set_reanalysis(self, time):
        """
        Retrieve the reanalysis files based `time`, and then interpolate all of 
        the three files according to `time`. 
        """
        self.pres_data, self.pres_temp_data, self.sfc_data = Plot.retrieve_files(time)
        self.pres_data = self.pres_data.interp(time=time)
        self.pres_temp_data = self.pres_temp_data.interp(time=time)
        self.sfc_data = self.sfc_data.interp(time=time)

    def get_pres_data(self):
        """Retrieves pres_data."""
        return self.pres_data

    def get_pres_temp_data(self):
        """Retrieves pres_temp_data."""
        return self.pres_temp_data

    def get_sfc_data(self):
        """Retrieves sfc_data."""
        return self.sfc_data

    def add_basemap(self, land=True):
        """
        Add basemap to the plot. 
        """
        border_dict = {'lw': 0.5, 'alpha': 0.5}
        fill_dict = {'lw': 0.5, 'alpha': 0.3}
        self.ax.add_feature(cfeature.COASTLINE.with_scale(res), **border_dict)
        self.ax.add_feature(cfeature.STATES.with_scale(res), **border_dict)
        if land:
            self.ax.add_feature(cfeature.LAND.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.OCEAN.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.3)
        return self.ax

    def t_advection(self, level=850):
        """
        Calculates the temperature advection from data. 
        """
        temperature = self.pres_temp_data.sel(level=level).t
        u = self.pres_data.sel(level=level).u
        v = self.pres_data.sel(level=level).v
        return mpcalc.advection(temperature, u, v).metpy.unit_array.to(units('delta_degC/hour'))

    def t_gradient(self, level='2m'):
        """
        Calculates the temperature gradient from data.
        """
        if level == '2m':
            temperature = self.sfc_data.t2m
        else:
            temperature = self.pres_temp_data.t.sel(level=level)

        return mpcalc.gradient(temperature)

    def contour(self, data, clabel=True, clabeldict={}, **kwargs):
        """
        data - should be an `xr.DataArray` with only two dimensions `lat` and `lon`. 
        """
        cs = self.ax.contour(self.lons, self.lats, data, **kwargs)
        if clabel:
            self.ax.clabel(cs, **clabeldict)

    @staticmethod
    def set_cmap(colormap, interval, maximum, minimum=None, category='diverging', 
                 set_over=True, set_under=True):
        """
        Set the colormap for contourf. 
        """
        over = colormap.get_over()
        under = colormap.get_under()
        
        if minimum is None:
            minimum = -maximum
        
        bounds = np.arange(minimum, maximum + interval, interval)
            
        num_of_colors = len(bounds) + int(set_over) + int(set_under) - 1
        
        if category == 'diverging':
            num_of_colors -= 2
            colors = cm.get_cmap(colormap)(np.linspace(0, 1, num_of_colors))
            colors = np.concatenate([
                colors[0:len(colors) // 2],
                np.array([
                    [0, 0, 0, 0.02],
                    [0, 0, 0, 0.02]
                ]),
                colors[len(colors) // 2:]])
        elif category == 'sequential':
            colors = cm.get_cmap(colormap)(np.linspace(0, 1, num_of_colors))
            
        if set_over and list(over) == list(colors[-1]):
            over = colors[-1]
            colors = colors[:-1]
        if set_under and list(under) == list(colors[0]):
            under = colors[0]
            colors = colors[1:]
            
        cmap = mcolors.ListedColormap(colors)
        cmap.set_over(over)
        cmap.set_under(under)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
        
        return bounds, norm, cmap

    def contourf(self, data, colormap, interval=0.5, minimum=None, maximum=None,
                 category='diverging', **kwargs):
        """
        Arguments
        ---------
        data 
            should be an `xr.DataArray` with only two dimensions `lat` and `lon`. 
        colormap 
            a LinearSegmentedColormap, can optionally have over and under defined
        interval 
            the interval at which the contours will be plotted
        maximum 
            the maximum (and minimum) value of the contours
        category 
            can be set to either 'discrete', 'sequential', or 'diverging'
        """
        set_under = kwargs.pop('set_under', True)
        set_over = kwargs.pop('set_over', True)
        
        clevs, norm, cmap = Plot.set_cmap(colormap, interval, maximum, minimum=minimum,
                                          category=category, set_over=set_over, set_under=set_under)

        self.ax.contourf(self.lons, self.lats, data, clevs, cmap=cmap, norm=norm, **kwargs)
        return norm, cmap

    def imshow(self, data, **kwargs):
        """
        A wrapper around ax.imshow(). 
        """
        self.ax.imshow(data, extent=self.extent, **kwargs)

    def barbs(self, level=850, **kwargs):
        """
        Plot wind barbs at `level`. 
        """
        u = self.pres_data.sel(level=level).u
        v = self.pres_data.sel(level=level).v
        lons, lats, u = Plot.downscale(u, 4)
        _, _, v = Plot.downscale(v, 4)

        self.ax.barbs(lons, lats, u, v, **kwargs)

    def set_cbar(self, norm, cmap, orientation, annotation, **kwargs):
        """
        Set the colorbar of the plot. 
        """
        labelsize = kwargs.pop('labelsize', 8)
        fontsize = kwargs.pop('fontsize', 7)
        ticks = kwargs.pop('ticks', None)
        ticklabels = kwargs.pop('ticklabels', None)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation=orientation,
                                 **kwargs)
        cbar.ax.tick_params(labelsize=labelsize)

        if orientation == 'horizontal':
            cbar.ax.set_xlabel(annotation, fontsize=fontsize)

        if orientation == 'vertical':
            cbar.ax.set_ylabel(annotation, fontsize=fontsize)

        if ticks is not None:
            cbar.set_ticks(ticks)

        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)

    def set_title(self, plot_title):
        """
        Set the title of the plot. 
        """
        fmt = 'Valid: %Y-%m-%d %H:%M'
        time_title = self.time.strftime(fmt)
        self.ax.set_title(plot_title, loc='left', fontsize=8)
        self.ax.set_title(time_title, loc='right', fontsize=8)

    @staticmethod
    def retrieve_files(time):
        """
        Utility function to retrieve all useful netcdf4 files at a given time. 
        """
        month = f'{time.month:02d}'
        # The variable yr is only used to index into the year folder
        yr = time.year if time.month in [11, 12] else time.year - 1
        yr_folder = os.path.join(reanalysis_dir, str(yr))

        pres_file = os.path.join(yr_folder, f'e5_{time.year}{month}_pl.nc')
        pres_temp_file = os.path.join(
            yr_folder, f'e5_{time.year}{month}_pl_temperature.nc')
        sfc_file = os.path.join(yr_folder, f'e5_{time.year}{month}_sfc.nc')

        pres_data = xr.open_dataset(pres_file)
        pres_temp_data = xr.open_dataset(pres_temp_file)
        sfc_data = xr.open_dataset(sfc_file)

        return pres_data, pres_temp_data, sfc_data

    @staticmethod
    def downscale(data, scale, boundary='trim'):
        """
        data - any `xr.DataArray` to be plotted, with variables of two dimensions

        Example:
        If magnitude = 2 and grid = 0.25, regrid to 0.5 * 0.5. 

        Returns: 
        `lon` - the 1-D longitude grid after downscaling
        `lat` - the 1-D latitude grid after downscaling
        `res` - the `xr.DataArray` after downscaling 
        """
        res = (data.coarsen(latitude=scale, boundary=boundary)
               .mean()
               .coarsen(longitude=scale, boundary=boundary)
               .mean())
        lons = res.coords['longitude'].values
        lats = res.coords['latitude'].values
        return lons, lats, res
    
    
if __name__ == '__main__':
    from sklearn.cluster import KMeans
    
    metadata_dir = 'metadata'
    metadata = pd.read_csv(os.path.join(metadata_dir, 'all_event_metadata.csv'))
    metadata['median_time'] = pd.to_datetime(metadata['median_time'])
        
    X = np.array([metadata.loc[:, 'med_lat'], metadata.loc[:, 'med_lon']]).transpose()
    kmeans = KMeans(n_clusters=4).fit(X)
    
#     ld_metadata = pd.read_csv(os.path.join(metadata_dir, 'ld_event_metadata.csv'))
#     ld_metadata['median_time'] = pd.to_datetime(ld_metadata['median_time'])
#     X_ld = np.array([ld_metadata['med_lat'], ld_metadata['med_lon']]).transpose()
#     ld_metadata_label = kmeans.predict(X_ld)
    
    aggregated_dir = 'aggregated_output'
#     reanalysis_ld_dir = os.path.join(aggregated_dir, 'reanalysis_ld_n=4')
    reanalysis_all_dir = os.path.join(aggregated_dir, 'reanalysis_all_n=4')
    if not os.path.isdir(reanalysis_all_dir):
        os.makedirs(reanalysis_all_dir)

    def aggregate_metadata(label):
#         mask = ld_metadata_label == label
#         sub_metadata = ld_metadata.loc[mask, :].copy()
        mask = kmeans.labels_ == label
        sub_metadata = metadata.loc[mask, :].copy()

        tadv_850_acc, tadv_sfc_acc, msl_acc, gph_acc, tsfc_acc, tmax_acc = [], [], [], [], [], []

        count = 0
        for index, row in sub_metadata.iterrows():
            west, east = row['med_lon'] - 20, row['med_lon'] + 10
            south, north = row['med_lat'] - 10, row['med_lat'] + 10

            time = row['median_time']

            dm = DataModel(time, extent=(west, east, south, north), cache=False)

            pres_data, pres_temp_data, sfc_data = dm.pres_data, dm.pres_temp_data, dm.sfc_data

            tadv_850 = mpcalc.advection(pres_temp_data.sel(level=850).t, 
                                        pres_data.sel(level=850).u,
                                        pres_data.sel(level=850).v).metpy.unit_array.to(units('delta_degC/hour'))

            tadv_sfc = mpcalc.advection(sfc_data.t2m, 
                                        sfc_data.u10, sfc_data.v10).metpy.unit_array.to(units('delta_degC/hour'))
            tmax = pres_temp_data.t.max(dim='level')

            tadv_850_acc.append(tadv_850)
            tadv_sfc_acc.append(tadv_sfc)
            msl_acc.append(sfc_data.msl.values / 100.)
            gph_acc.append(pres_data.sel(level=500).z.values / 9.81)
            tsfc_acc.append(sfc_data.t2m.values - 273.15)
            tmax_acc.append(tmax.values - 273.15)

            count += 1
            print(f'Having processed {count}/{len(sub_metadata)} rows for label {label}...')

        mean_lat = (sub_metadata['med_lat'].mean() // 0.25) * 0.25
        mean_lon = (sub_metadata['med_lon'].mean() // 0.25) * 0.25

        latitude = np.arange(mean_lat - 10, mean_lat + 10.25, 0.25)
        longitude = np.arange(mean_lon - 20, mean_lon + 10.25, 0.25)
        time = sub_metadata['median_time']

        ds = acc_to_ds(
            ['time', 'latitude', 'longitude'],
            [time, latitude, longitude],
            dict(description=f'Aggregated data for {mean_lat}, {mean_lon}.'),
            ('tadv_850', tadv_850_acc, dict(
                description="850 hPa Temperature Advection",
                units="delta_degC/hour",
            )),
            ('tadv_sfc', tadv_sfc_acc, dict(
                description="Surface Temperature Advection",
                units="delta_degC/hour",
            )),
            ('msl', msl_acc, dict(
                description="Mean Sea-level Pressure",
                units="millibar",
            )),
            ('gph', gph_acc, dict(
                description="500 hPa Geopotential Height",
                units="meter",
            )),
            ('tsfc', tsfc_acc, dict(
                description="Surface Temperature",
                units="celsius",
            )),
            ('tmax', tmax_acc, dict(
                description="Maximum Temperature from 1000 - 700 hPa",
                units="celsius",
            )),
        )

        fp = os.path.join(reanalysis_all_dir, f'agg_{round(mean_lat)}_{round(mean_lon)}.nc')

        print(f'Having finished converting to DataSet for label {label}...')
        ds.to_netcdf(fp)

    with Pool(5) as p:
        p.map(aggregate_metadata, range(kmeans.labels_.max() + 1))
    
    
