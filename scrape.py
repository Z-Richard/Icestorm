"""
This script downloads the freezing rain data from the ASOS database.

Adapted from
https://github.com/akrherz/iem/blob/main/scripts/asos/iem_scraper_example.py.

Author: Haoyu Zhuang (hz542)
Time: Feb 28, 2022
"""

from __future__ import print_function
import json
import time
import datetime
import pandas as pd

from urllib.request import urlopen

import os

# Number of attempts to download data
MAX_ATTEMPTS = 6
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"

METAR_DIR = os.path.join('METAR')


def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check. This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def get_stations_from_networks():
    """Build a station list by using a bunch of IEM networks.

    The states are all to the east of the Rockies, where freezing rain is
    more likely to occur.

    Returns: the station list
    """
    stations = {}
    states = """AL AR CT DE FL GA IA IL IN KS KY LA MA MD ME
     MI MN MO MS NC ND NE NH NJ NY OH OK PA RI SC SD TN TX VA VT
     WI WV"""

    CA_states = """QC ON NB NF NS PE MB"""
    # IEM quirk to have Iowa AWOS sites in its own labeled network
    networks = []
    for state in states.split():
        networks.append("%s_ASOS" % (state,))

    for province in CA_states.split():
        networks.append(f'CA_{province}_ASOS')

    for network in networks:
        # Get metadata
        uri = (
            "https://mesonet.agron.iastate.edu/geojson/network/%s.geojson"
        ) % (network,)
        print(uri)
        data = urlopen(uri)
        jdict = json.load(data)
        for site in jdict["features"]:
            k, v = site["properties"]["sid"], site["geometry"]["coordinates"]
            stations[k] = v
    return pd.DataFrame.from_dict(stations, orient='index',
                                  columns=['lon', 'lat'])


def get_stations_for_one_year(stations, yr):
    """
    Arguments:
        - stations: a stations list
        - yr: we will retrieve all METAR from 11/01/yr to 03/31/yr+1

    Returns:
        - None
    """
    # timestamps in UTC to request data for
    yr_dir = os.path.join(METAR_DIR, str(yr))
    if not os.path.isdir(yr_dir):
        os.makedirs(yr_dir)

    startts = datetime.datetime(yr, 10, 1)
    endts = datetime.datetime(yr + 1, 4, 30)

    service = SERVICE + "data=wxcodes&tz=Etc/UTC&format=comma&latlon=no&"

    service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
    service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

    for station in stations:
        uri = "%s&station=%s" % (service, station)
        print("Downloading: %s (year %s)" % (station, yr))
        data = download_data(uri)
        if len(data.split('\n')) <= 100:
            continue
        outfn = "%s_%s_%s.txt" % (
            station,
            startts.strftime("%Y%m%d%H%M"),
            endts.strftime("%Y%m%d%H%M"),
        )
        fp = os.path.join(yr_dir, outfn)
        out = open(fp, "w")
        out.write(data)
        out.close()


if __name__ == '__main__':
    stations = get_stations_from_networks().index.tolist()
    for year in range(1979, 2022):
        get_stations_for_one_year(stations, year)
