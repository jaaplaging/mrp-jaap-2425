import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import copy
import time

from helper import twilight_rise_set, create_observer, index_last, opt_time, airmass, rise_set
from param_config import Configuration
from astroplan import FixedTarget
from astropy.coordinates import SkyCoord
from time import perf_counter
from astropy.time import Time
import pickle

OBSERVER = create_observer()
TIME = Time.now()
DAWN, DUSK = twilight_rise_set(observer=OBSERVER, time=TIME)


with open('mrp-jaap-2425/on_the_fly/top_eph.pkl', 'rb') as f:
    top_eph = pickle.load(f)

def get_airmass_peak(object_ephemerides):
    ra_mid = object_ephemerides['RA'][12]
    ra_mid = str(ra_mid) + 'h'
    dec_mid = object_ephemerides['dec'][12]
    dec_mid = str(dec_mid) + 'd'
    coords_mid = SkyCoord(ra_mid, dec_mid, frame='gcrs')
    target = FixedTarget(name='target', coord=coords_mid)
    rise, set = rise_set(OBSERVER, target, TIME)
    rise_ind, set_ind = time_to_ind(rise), time_to_ind(set)
    if set < DUSK > DAWN:
        return(-1, -1, -1, -1)
    else:
        time_init = perf_counter()
        peak_time = opt_time(OBSERVER, target, TIME)
        peak_ind = int((peak_time - DUSK).value*24*60)
        peak_airmass = airmass(OBSERVER, target, peak_time)
        return(peak_ind, peak_airmass, rise_ind, set_ind)
    
def time_to_ind(given_time):
    ind = int((given_time - DUSK).value*24*60)
    return(ind)

for ind_day, day in enumerate(top_eph):
    print(ind_day)
    for neo in day.keys():
        peak_ind, peak_airmass, rise_ind, set_ind = get_airmass_peak(day[neo])
        top_eph[ind_day][neo]['peak_ind'] = peak_ind
        top_eph[ind_day][neo]['peak_airmass'] = peak_airmass
        top_eph[ind_day][neo]['rise_ind'] = rise_ind
        top_eph[ind_day][neo]['set_ind'] = set_ind

with open('mrp-jaap-2425/on_the_fly/top_eph_processed.pkl', 'wb') as f:
    pickle.dump(top_eph, f)

