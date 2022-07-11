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


temp_cmap = cdict_to_cmap(cdict)


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
        use_log = kwargs.pop('use_log', False)
        
        if not use_log:
            clevs, norm, cmap = Plot.set_cmap(colormap, interval, maximum, minimum=minimum,
                                              category=category, set_over=set_over, set_under=set_under)

            self.ax.contourf(self.lons, self.lats, data, clevs, cmap=cmap, norm=norm, **kwargs)
            return norm, cmap
        
        cs = self.ax.contourf(self.lons, self.lats, data, locator=ticker.LogLocator(), cmap=colormap)
        cbar = self.fig.colorbar(cs)

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
