"""
Author: Richard Zhuang (hz542)
Date: May 26, 2022
"""

import cdsapi
import os

c = cdsapi.Client()


PRES_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700]
US_SUBSET = [65, -145, 10, -40]
DAYS = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
    '13', '14', '15',
    '16', '17', '18',
    '19', '20', '21',
    '22', '23', '24',
    '25', '26', '27',
    '28', '29', '30',
    '31',
]
THREE_HOUR_INTERVAL = [
    '00:00', '03:00', '06:00',
    '09:00', '12:00', '15:00',
    '18:00', '21:00'
]
HOUR_INTERVAL = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]
NATIVE_GRID = [0.25, 0.25]
ONE_DEG_GRID = [1.0, 1.0]
SURFACE_VARIABLES = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
    '2m_temperature', 'mean_sea_level_pressure', 'precipitation_type',
    'total_column_water', 'total_column_water_vapour', 'total_precipitation',
    'vertical_integral_of_eastward_water_vapour_flux', 'vertical_integral_of_northward_water_vapour_flux',
]
VARIABLES = [
    'divergence', 'geopotential', 'potential_vorticity',
    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
    'vorticity',
]


def surface_data(folder, variables=SURFACE_VARIABLES, days=DAYS, time=THREE_HOUR_INTERVAL,
                 area=US_SUBSET, grid=NATIVE_GRID, years=[1979, 2022]):
    """
    Download ERA-5 Surface level data. 
    """
    if len(years) != 2:
        assert "The parameters years have to be a list of size 2."
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for year in range(*years):
        yr_folder = os.path.join(folder, str(year))
        if not os.path.isdir(yr_folder):
            os.makedirs(yr_folder)
        for month in [10, 11, 12, 1, 2, 3, 4]:
            yr = year if month in [10, 11, 12] else year + 1
            fn = f'e5_{yr}{month:02d}_sfc.nc'
            fp = os.path.join(yr_folder, fn)
            print('=========================================================')
            print(f'Downloading {yr}-{month:02d}')
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variables,
                    'year': f'{yr}',
                    'month': f'{month:02d}',
                    'day': days,
                    'time': time,
                    'area': area,
                    'grid': grid
                },
                fp)


def pressure_data(folder, variables, days=DAYS, time=THREE_HOUR_INTERVAL,
                  area=US_SUBSET, pressure_levels=PRES_LEVELS,
                  grid=NATIVE_GRID, years=[1979, 2022]):
    """
    Download ERA-5 Pressure level data. 
    """
    if len(years) != 2:
        assert "The parameters years have to be a list of size 2."
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for year in range(*years):
        yr_folder = os.path.join(folder, str(year))
        if not os.path.isdir(yr_folder):
            os.makedirs(yr_folder)
        for month in [10, 11, 12, 1, 2, 3, 4]:
            yr = year if month in [10, 11, 12] else year + 1
            if type(variables) == str:
                fn = f'e5_{yr}{month:02d}_pl_{variables}.nc'
            else:
                fn = f'e5_{yr}{month:02d}_pl.nc'
            fp = os.path.join(yr_folder, fn)
            print('=========================================================')
            print(f'Downloading {yr}-{month:02d}')
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variables,
                    'pressure_level': pressure_levels,
                    'year': f'{yr}',
                    'month': f'{month:02d}',
                    'day': days,
                    'time': time,
                    'area': area,
                    'grid': grid
                },
                fp)


if __name__ == '__main__':
    surface_data('reanalysis', years=[1979, 2022])
    pressure_data('reanalysis', variables=VARIABLES,
                  pressure_levels=[925, 850, 700, 500])
    pressure_data('reanalysis', 'temperature', years=[1979, 2022])
