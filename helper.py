# file containing some helpful functions that can, e.g., calculate airmass, determine observability, etc.

import numpy as np
from astropy.coordinates import SkyCoord 
from astroplan import FixedTarget 
from astroplan import Observer 
import astropy.units as u
from astropy.coordinates import EarthLocation, get_body
from pytz import timezone
from astropy.time import Time
from param_config import Configuration

config = Configuration()


def create_target(ra, dec, frame='gcrs'):
    ''' Creates an astroplan target object for the target NEO '''
    coordinates = SkyCoord(ra, dec, frame=frame)
    return FixedTarget(name='target', coord=coordinates)

def create_observer(lat='+00d00m00.000s', long='+00d00m00.000s', elevation=1000., name='Placeholder', pressure=1., relhum=0.82, temperature=15, tz=timezone('Europe/London'), description='Placeholder'):
    ''' 
    Creates an astroplan observer object for the observatory
    '''
    location = EarthLocation.from_geodetic(long, lat, elevation*u.m)

    observer = Observer(name=name,
                        location=location,
                        pressure=pressure * u.bar,
                        relative_humidity=relhum,
                        temperature=temperature * u.deg_C,
                        timezone=tz,
                        description=description)

    return(observer)

def observable(observer, target, time):
    ''' Checks if target is observable for observer at time '''
    return(observer.target_is_up(time, target))

def night(observer, time):
    ''' Checks if it is night at time at observer location '''
    return(observer.is_night(time))

def rise_set(observer, target, time):
    ''' Returns the rise and set time of target for observer '''
    return(observer.target_rise_time(time, target, which='nearest'), observer.target_set_time(time, target, which='next'))

def twilight_rise_set(observer, time):
    ''' Returns the astronomical twilight in morning and evening at observatory '''
    return(observer.twilight_morning_astronomical(time, which='next'), observer.twilight_evening_astronomical(time, which='nearest'))

def airmass(observer, target, time):
    ''' Returns the airmass of target at time at observer'''
    return(observer.altaz(time, target).secz)

def opt_time(observer, target, time):
    ''' calculates the time when target is closest to zenith '''
    set, rise = rise_set(observer, target, time)
    return(rise+(set-rise)/2)

def near_moon(target, time):
    ''' Determines if target is too close to the moon '''
    #TODO make it so that this is part of the weights instead
    moon = get_body('moon', time)
    sep = moon.separation(target.coord)
    if sep < config.moon_dist: 
        return True
    else:
        return False

def index_last(li, item):
    for i in range(len(li)-1, -1, -1):
        if li[i] == item:
            return(i)

if __name__ == "__main__":
    time = Time(['2024-10-06 13:47:00'], scale='utc')
    target = create_target('20h41m25.9s', '+45d16m49.3s')
    observer = create_observer()



