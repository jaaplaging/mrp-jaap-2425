import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord 
from astropy.coordinates import EarthLocation, get_body
import astropy.units as u
from astroplan import FixedTarget 
from astroplan import Observer 
from pytz import timezone

from param_config import Configuration

config = Configuration()

def create_target(ra, dec, frame='gcrs'):
    ''' Creates an astroplan target object for the target NEO 
    
    Args:
        ra (str): right ascension of the target to be created, in format supported by astropy
        dec (str): declination of target to be created, in format supported by astropy
        frame (str, optional): coordinate frame, defaults to gcrs

    Returns:
        target (astroplan.FixedTarget): the created FixedTarget object 
    '''
    coordinates = SkyCoord(ra, dec, frame=frame)
    return FixedTarget(name='target', coord=coordinates)

def create_observer(lat='+00d00m00.000s', long='+00d00m00.000s', elevation=1000., name='Placeholder', pressure=1., relhum=0.82, temperature=15, tz=timezone('Europe/London'), description='Placeholder'):
    """ Creates an astroplan observer object for the observatory

    Args:
        lat (str, optional): latitude of the observer, in format supported by astropy, defaults to '+00d00m00.000s'
        long (str, optional): longitude of the observer, in format supported by astropy, defaults to '+00d00m00.000s'
        elevattion (float, optional): elevation of the observer in meters, defaults to 1000.
        name (str, optional): name of the observer, defaults to 'Placeholder'
        pressure (float, optional): air pressure at observer, in bar. Defaults to 1.
        relhum (float, optional): relative humidity at observer, defaults to 0.82
        temperature (float, optional): temperature at observer in Celsius, defaults to 15
        tz (timezone, optional): timezone at observer, defaults to timezone('Europe/London')
        description (str, optional): description of the observer, defaults to 'Placeholder'

    Returns:
        observer (astroplan.Observer): Observer object created from input
    """
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
    ''' Checks if target is observable for observer at given time 
    
    Args:
        observer (astroplan.Observer): observer for which to check if target is visible
        target (astroplan.FixedTarget): target to check if visible
        time (astropy.time.Time): time at which to check if target is visible to object

    Returns:
        is_up (bool): True if target is visible for observer at given time
    '''
    return(observer.target_is_up(time, target))

def night(observer, time):
    ''' Checks if it is night at time at observer location 
    
    Args:
        observer (astroplan.Observer): observer for which to check if it is night at given time
        time (astropy.time.Time): time at which to check if it is night
    
    Returns:
        is_night (bool): True if it is night at time at observer
    '''
    return(observer.is_night(time))

def rise_set(observer, target, time):
    ''' Returns the rise and set time of target for observer 
    
    Args:
        observer (astroplan.Observer): observer for which to get the rise and set time of target
        target (astroplan.FixedTarget): target for which to get the rise and set time
        time (astropy.time.Time): date at which to get the rise and set time

    Returns:
        rise (astropy.time.Time): rise time of target
        set (astropy.time.Time): set time of target
    '''
    return(observer.target_rise_time(time, target, which='nearest'), observer.target_set_time(time, target, which='next'))

def twilight_rise_set(observer, time):
    ''' Returns the astronomical twilight in morning and evening at observatory 
    
    Args:
        observer (astroplan.Observer): observer at which to get the astronomical twilight times
        time (astropy.time.Time): date at which to get the times

    Returns:
        twilight_morning_astronomical (astropy.time.Time): time of astronomical twilight in the morning
        twilight_evening_astronomical (astropy.time.Time): time of astronomical twilight in the evnening
    '''
    return(observer.twilight_morning_astronomical(time, which='next'), observer.twilight_evening_astronomical(time, which='nearest'))

def airmass(observer, target, time):
    ''' Returns the airmass of target at time at observer
    
    Args:
        observer (astroplan.Observer): observer at which to calculate the airmass
        target (astroplan.FixedTarget): target for which to calculate the airmass
        time (astropy.time.Time): time at which to calculate the airmass
        
    Returns:
        airmass (float): airmass of target at given time
    '''
    return(observer.altaz(time, target).secz)

def opt_time(observer, target, time):
    ''' calculates the time when target is closest to zenith 
    
    Args:
        observer (astroplan.Observer): observer for which to calculate the time when target is closest to zenith
        target (astroplan.FixedTarget): target for which to calculate the optimal time
        time (astropy.time.Time): date for which to calculate the optimal observation time

    Returns:
        peak (astropy.time.Time): time of lowest airmass 
    '''
    set, rise = rise_set(observer, target, time)
    return(rise+(set-rise)/2)

def near_moon(target, time):
    ''' Determines if target is too close to the moon 
    
    Args:
        target (astroplan.FixedTarget): target in consideration
        time (astropy.time.Time): time at which to determine closeness to moon

    Returns:
        near_moon (bool): True if object is too close to the moon
    '''
    #TODO make it so that this is part of the weights instead?
    moon = get_body('moon', time)
    sep = moon.separation(target.coord)
    if sep < config.moon_dist: 
        return True
    else:
        return False

def index_last(li, item):
    """ Finds the index of the last instance of an item in a list

    Args:
        li (list): list to search
        item (any type): item to search for

    Returns:
        index (int): index of last instance of item in list
    """
    for i in range(len(li)-1, -1, -1):
        if li[i] == item:
            return(i)



