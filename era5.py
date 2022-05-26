import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'temperature',
        'pressure_level': '850',
        'year': '2020',
        'month': '01',
        'day': [
            '01'
        ],
        'time': [
            '00:00'
        ],
        'area': [
            65, -105, 20,
            -55,
        ],
    },
    'download.nc')
