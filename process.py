"""
Author: Richard Zhuang (hz542)
Time: May 25, 2022
"""

import numpy as np
import pandas as pd
import datetime
from glob import glob
from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist
import os
import matplotlib.pyplot as plt
from scrape import get_stations_from_networks

# An incomplete list of METAR code
# Source: https://www.aviationweather.gov/metar/symbol
METAR_CODE = ['FU', 'VA', 'HZ', 'DU', 'SA', 'BLDU', 'BLSA', 'PO',
              'VCSS', 'BR', 'MIFG', 'VCTS', 'VIRGA', 'VCSH', 'TS', 'SQ', 'FC',
              'SS', '+SS', 'BLSN', 'DRSN', 'VCFG', 'BCFG', 'PRFG', 'FG', 'FZFG',
              '-DZ', 'DZ', '+DZ', '-FZDZ', 'FZDZ', '+FZDZ', '-DZRA', 'DZRA',
              '-RA', 'RA', '+RA', '-FZRA', 'FZRA', '+FZRA', '-RASN', 'RASN',
              '+RASN', '-SN', 'SN', '+SN', 'SG', 'IC', 'PE', 'PL', '-SHRA',
              'SHRA', '+SHRA', '-SHRASN', 'SHRASN', '+SHRASN', '-SHSN',
              'SHSN', '+SHSN', '-GR', 'GR', 'TSRA', 'TSGR', '+TSRA', 'MIBCFG']

# FREEZE_PRECIP includes both freezing drizzle and freezing rain
FREEZE_PRECIP = ['-FZDZ', 'FZDZ', '+FZDZ', '-FZRA', 'FZRA', '+FZRA']

# FREEZE_RAIN includes ONLY freezing rain
FREEZE_RAIN = ['-FZRA', 'FZRA', '+FZRA']

# PRECIP includes both drizzle and rain/snow but excludes blowing snow
PRECIP = ['-DZ', 'DZ', '+DZ', '-FZDZ', 'FZDZ', '+FZDZ', '-DZRA', 'DZRA',
          '-RA', 'RA', '+RA', '-FZRA', 'FZRA', '+FZRA', '-RASN', 'RASN',
          '+RASN', '-SN', 'SN', '+SN', 'SG', 'IC', 'PE', 'PL', '-SHRA',
          'SHRA', '+SHRA', '-SHRASN', 'SHRASN', '+SHRASN', '-SHSN',
          'SHSN', '+SHSN', '-GR', 'GR', 'TSRA', 'TSGR', '+TSRA']

EARTH_RADIUS = 6371
MIN_LD_STATIONS = 6


def timedelta_to_hrs(td: pd.Timedelta, process_series=True):
    """
    """
    if process_series:
        return np.round(td.dt.components.hours +
                        td.dt.components.minutes / 60 + td.dt.components.days * 24, 2)
    else:
        return np.round(td.components.hours +
                        td.components.minutes / 60 + td.components.days * 24, 2)


def is_ldzr(filename):
    """
    Argument(s): filename - the filename from which the METAR code
    would be read.

    Notice that the file should be in .csv format, with five lines of
    code debugging header at the top of the file. There should also be strictly
    three columns: station (station name), valid (time), and wxcodes.

    Returns:
    The corresponding starting and ending time of a freezing rain event
    in a `pd.DataFrame`. There will also be a key `label` indicating whether
    the event is a long-duration event (1) or a short-duration event (0).
    """
    # Read data files, convert time to python datetime object
    df = pd.read_csv(filename, header=5)
    # station = filename.split('_')[0]
    df.rename(columns={'valid': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna(subset=['wxcodes'])

    # If the METAR code CONTAINS `FZRA`,
    # then put them in the freezing rain df_zr
    df_zr = df[df['wxcodes'].str.contains('FZRA')].copy()
    # if the there is no freezing rain records
    if not len(df_zr):
        return None
    df_zr['diff'] = pd.to_timedelta(df_zr['time'].diff())
    df_zr.reset_index(inplace=True)
    df_zr.columns = ['seq', 'station', 'time', 'wxcodes', 'diff']

    # start_index, end_index of freezing rain that we use
    # to find the starting time and ending time of each event
    # We consider two events to be separate if time.diff > 24 hrs
    start_index = df_zr[(df_zr['diff'] > datetime.timedelta(days=1))
                        | (pd.isnull(df_zr['diff']))].index.tolist()

    # Invariant: len(start_index) > 0, cuz len(df_zr) > 0
    end_index = start_index[1:] + [len(df_zr)]
    start_time = df_zr.loc[start_index, 'time'].tolist()
    end_time = df_zr.loc[np.array(end_index) - 1, 'time'].tolist()
    label = []
    total_zrs = []

    for si, ei in zip(start_index, end_index):
        sseq, eseq = df_zr.loc[si, 'seq'], df_zr.loc[ei-1, 'seq']
        all_record = df.loc[sseq:eseq, :].copy()
        all_record['duration'] = pd.to_timedelta(all_record['time'].diff())
        all_record['duration'] = all_record['duration'].shift(-1)
        total_zr = all_record.loc[all_record['wxcodes'].str.contains(
            'FZRA'), 'duration'].sum()
        total_zrs.append(timedelta_to_hrs(total_zr, False))
        if total_zr >= datetime.timedelta(hours=6):
            label.append(1)
        else:
            label.append(0)

    output = pd.DataFrame.from_dict(
        {
            'start_time': start_time,
            'end_time': end_time,
            'zr_hours': total_zrs,
            'label': label
        }
    )

    return output


def make_yr_dir(folder: str, start_yr: int, end_yr: int):
    """
    Make a list of year folders from `start_yr` to `end_yr` in the directory
    `folder`. Note that this will also bring the system directory to
    `folder` -- in other words, after running this function, we are one
    step forward into the directory.

    Example:
    folder = 'LD_SD', start_yr = 1995, end_yr = 2015
    First we check if the folder exists. If it doesn't exist, create the folder.
    Then, we created the year folder `1995` to `2014` within `LD_SD`.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    for year in range(start_yr, end_yr):
        if not os.path.isdir(str(year)):
            os.makedirs(str(year))
    # os.chdir('..')


def all_ld_events(start_yr, end_yr):
    """
    This function runs through the raw data from `start_yr` to `end_yr-1`
    and then outputs the LD and SD freezing rain event to the `LD_SD/yr`
    folder.
    """
    make_yr_dir('LD_SD', start_yr, end_yr)
    for year in range(start_yr, end_yr):
        filepaths = glob(os.path.join('..', 'METAR', str(year), '*.txt'))
        for file in filepaths:
            station = file.split('\\')[-1].split('_')[0]
            print('{0} [{1}]'.format(station, year))
            output = is_ldzr(file)
            # print(output)
            if output is not None:
                op = os.path.join(str(year), '{}.csv'.format(station))
                output.to_csv(op, index=False, date_format='%Y-%m-%d %H:%M')
            else:
                continue


def stations_to_file(filename):
    """
    Write the station names and lat, lon pair obtained from
    get_stations_from_networks() to a `filename`.

    Returns: None
    """
    stations = get_stations_from_networks()
    stations.reset_index(inplace=True)
    stations = stations.rename(columns={'index': 'station'})
    stations.to_csv(filename, index=False)


def to_all_ld(folder):
    """
    Put every record into a single file sorted by year and
    starting time of the freezing rain event.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    for year in range(1979, 2022):
        print(year)
        filepaths = glob(os.path.join('..', 'LD_SD', str(year), '*.csv'))
        dfs = []
        for file in filepaths:
            df = pd.read_csv(file)
            name = file.split('\\')[-1].split('.')[0]
            df.loc[:, 'station'] = name
            df = df[['station', 'start_time', 'end_time', 'zr_hours', 'label']]
            dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        dfs['start_time'] = pd.to_datetime(dfs['start_time'])
        dfs['end_time'] = pd.to_datetime(dfs['end_time'])
        duration = dfs['end_time'] - dfs['start_time']
        dfs['duration'] = timedelta_to_hrs(duration)
        dfs = dfs.sort_values(by='start_time')
        dfs.to_csv(str(year) + '.csv', index=False,
                   date_format='%Y-%m-%d %H:%M')


def define_ld_events(folder):
    """
    Define the long-duration ZR events from 1979 to 2022.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)
    filepaths = glob(os.path.join('LD_SD_all', '*.csv'))
    stations = pd.read_csv('stations.csv').set_index('station')
    all_min_zr = min_zr_stations(1979, 2022)
    for file in filepaths:
        df = pd.read_csv(file)
        year = file.split('.')[0].split('\\')[-1]
        print(f'Processing year: {year}')
        yr_folder = os.path.join(folder, year)
        if not os.path.isdir(yr_folder):
            os.makedirs(yr_folder)
        min_zr = all_min_zr[int(year)]
        df = df.drop(columns=['end_time'])
        df['start_time'] = pd.to_datetime(df['start_time'])
        df = df.merge(stations, how='left',
                      left_on='station', right_index=True)
        df.reset_index(drop=True, inplace=True)
        df_ld = df[df['label'] == 1].copy()
        df_sd = df[df['label'] == 0].copy()
        si, ei = [], []
        indices = df_ld.index.tolist()
        for i, ind in enumerate(indices[:-1]):
            end = df_ld.loc[ind, 'start_time']
            start = df_ld.loc[indices[i+1], 'start_time']
            if start - end >= datetime.timedelta(hours=24):
                si.append(indices[i+1])
                ei.append(ind)

        si = [indices[0]] + si
        ei = ei + [indices[-1]]
        for s, e in list(zip(si, ei)):
            event = df_ld.loc[s:e, :].copy()
            if len(event) < min_zr:
                continue
            # df_event = df_sd.loc[s:e, :].copy()
            # df_75 = df_event[df_event['zr_hours'] >= 4.]
            # df_event = pd.concat([df_75, event])
            # df_event = df_event.sort_values(by='start_time')
            # df_event.reset_index(drop=True, inplace=True)
            fmt = '%Y%m%d%H%M'
            fn = df_ld.loc[s, 'start_time'].strftime(
                fmt) + '_' + df_ld.loc[e, 'start_time'].strftime(fmt) + '.csv'
            fp = os.path.join(yr_folder, fn)
            event.to_csv(fp, index=False, date_format='%Y-%m-%d %H:%M')


def plot_stations_by_year(start_yr, end_yr):
    """
    The number of stations that have at least 4 records per year.
    The result would be plotted using matplotlib.
    """
    yrs = list(range(start_yr, end_yr))
    lengths = [len(glob(os.path.join('LD_SD', str(year), '*.csv')))
               for year in yrs]
    plt.plot(yrs, lengths)
    plt.xlabel('year')
    plt.ylabel('Number of stations')
    plt.show()


def min_zr_stations(start_yr, end_yr):
    """
    Returns a dictionary that indicates the minimum number of stations that
    characterize a long-duration, impactful freezing rain event
    from `start_yr` to `end_yr`.
    """
    yrs = list(range(start_yr, end_yr))
    lengths = np.array([len(glob(os.path.join('LD_SD', str(year), '*.csv')))
                        for year in yrs])
    maxlength = np.max(lengths)
    num_stations = np.round(lengths * MIN_LD_STATIONS / maxlength)
    return dict(zip(yrs, num_stations))


def num_of_events(folder):
    """
    Returns the total number of events in a given folder. 
    """
    return sum(len(files) for _, _, files in os.walk(folder))


if __name__ == '__main__':
    # define_ld_events('Events')
    print(num_of_events('Events'))

    # for s, e in list(zip(si, ei))[:5]:
    #     length = e - s + 1
    #     event = dfs_merge.loc[s:e, :].copy()
    #     latitude = event['lat'].tolist()
    #     longitude = event['lon'].tolist()
    #     coords = list(zip(latitude, longitude))
    #     dist_mat = cdist(
    #         coords, coords, lambda lat, lon: geodist(lat, lon).kilometers)
    #     print(dist_mat)

    # print(len(si))
    # print(tuple(zip(si, ei)))
