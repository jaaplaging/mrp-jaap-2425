import numpy as np
import requests
from bs4 import BeautifulSoup
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astroplan import FixedTarget

from helper import create_observer, twilight_rise_set, rise_set, near_moon, observable
from param_config import Configuration

config = Configuration()

URL_OBJ = "https://minorplanetcenter.net/iau/NEO/neocp.txt"
URL_EPH = "https://cgi.minorplanetcenter.net/cgi-bin/confirmeph2.cgi"


def scraper(observer, time):
    """ main wrapper function for the web scraping

    Args:
        observer (astroplan.Observer): observer for which to get ephemerides
        time (astropy.time.Time): date at which to get ephemerides

    Returns:
        obj_dict (dict): dictionary containing object data
        eph_dict (dict): dictionary containing ephemerides data for each object
    """
    obj_dict = get_objects()
    obj_dict = select_objects(obj_dict, observer, time)
    eph_dict = get_ephemerides(obj_dict, observer, time)
    return(obj_dict, eph_dict)

def get_objects():
    """ Reads table from MPC website to create a dictionary of objects

    Returns:
        obj_dict (dict): dictionary of objects obtained from the website
    """
    obj_dict = {}
    objects = requests.get(URL_OBJ)
    lines = objects.text.split('\n')
    for line in lines[:-1]:
        parts = line.split(' ')
        parts = [part for part in parts if part != '']
        coordinates = SkyCoord(parts[5]+'h', parts[6]+'d', frame='gcrs')
        obj_dict[parts[0]] = {'obj': FixedTarget(name=parts[0],
                                        coord=coordinates),
                            'mag_V': parts[7]}
    return(obj_dict)


def select_objects(obj_dict, observer, time):
    ''' Function that determines which objects to get ephemerides for, based on observability
    
    Args:
        obj_dict (dict): dictionary containing objects
        observer (astroplan.Observer): observer object
        time (astropy.time.Time): date for which to select objects

    Returns:
        obj_dict (dict): filtered dictionary of objects observable
    '''
    #TODO turn into 1 for loop instead of 3
    #TODO consider making the magnitude check a 'soft check' or upping the limit
    #TODO moon check: maybe make more restrictive and check at other times during the window
    
    twilight_breaking_dawn_part_II, twilight_evening = twilight_rise_set(observer, time)
    for object in list(obj_dict.keys()):
        # magnitude check
        if float(obj_dict[object]['mag_V']) > config.mag_lim:
            del obj_dict[object]
        elif near_moon(obj_dict[object]['obj'], time):
            del obj_dict[object]
        else: 
            # visibility check
            obj_rise, obj_set = rise_set(observer, obj_dict[object]['obj'], time)
            start = np.max([obj_rise, twilight_evening])
            end = np.min([obj_set, twilight_breaking_dawn_part_II])
            if obj_rise.to_string() == '[     ———]':
                if not(observer.target_is_up(time, obj_dict[object]['obj'])):
                    del(obj_dict[object])
            if not (end-start >= config.min_visibility):
                del obj_dict[object]
    return(obj_dict)
    
def get_ephemerides(obj_dict, observer, time):
    """ Obtains the ephemerides of the objects from the MPC website

    Args:
        obj_dict (dict): dictionary containing the objects to get ephemerides for
        observer (astroplan.Observer): observer for which to get ephemerides
        time (astropy.time.Time): date for which to get ephemerides
    
    Returns:
        ephemerides (dict): dictionary containing ephemerides for each of the objects
    """  
    soup = form_post(obj_dict, observer)
    t_breaking_dawn_part_II, t_evening = twilight_rise_set(observer, time)
    ephemerides = {}

    def add_descendant(descendant, ephemerides):
        ''' Adds empty entry for the descendant 
        
        Args:
            descendant (str): name of the object
            ephemerides (dict): dictionary containing ephemerides
            
        Returns:
            ephemerides (dict): updated ephemerides dictionary
        '''
        ephemerides[descendant] = {'time': [],
                                    'coord': [],
                                    'mag_V': [],
                                    'motion': []}
        return(ephemerides)

    def parse_child_text(child, observer, t_breaking_dawn_part_II, t_evening, ephemerides):
        ''' Parses the text of the ephemeris 
        
        Args:
            child (str): text containing the ephemerides information
            observer (astroplan.Observer): observer for which to get the ephemerides
            t_breaking_dawn_part_II (astropy.time.Time): end of astronomical night
            t_evening (astropy.time.Time): start of astronomical night
            ephemerides (dict): dictionary of ephemerides to update

        Returns:
            ephemerides (dict): updated dictionary of ephemerides
            target (astroplan.FixedTarget): created target from text
        '''
        text = child.lstrip('\n')
        text = text.split(' ')
        text = [text[i] for i in range(len(text)) if text[i] != '']       
        t_eph = Time([f'{text[0][-4:]}-{text[1]}-{text[2]} {text[3]}:00:00'], scale='utc')
        coords = SkyCoord(text[4]+'h', text[5]+'d', frame='gcrs')
        target = FixedTarget(coord=coords)
        
        if observable(observer, target, t_eph):
            if t_eph <= t_breaking_dawn_part_II and t_eph >= t_evening:
                ephemerides[current]['time'].append(t_eph)
                ephemerides[current]['coord'].append(coords)
                ephemerides[current]['mag_V'].append(float(text[7]))
                ephemerides[current]['motion'].append(np.sqrt((float(text[8]))**2 + (float(text[9]))**2))   
        return(ephemerides, target)

    for descendant in soup.descendants:
        if descendant.name == 'b':
            ephemerides = add_descendant(descendant.text, ephemerides)
            current = descendant.text
        if descendant.name == 'pre':
            for child in descendant.descendants: 
                if child.name is None and 'Map' not in child.text and 'Offsets' not in child.text and '20' in child.text and 'Date' not in child.text:
                    ephemerides, target = parse_child_text(child.text, observer, t_breaking_dawn_part_II, t_evening, ephemerides)

            rise, set = rise_set(observer, target, time) 
            ephemerides[current]['start_time'] = np.max([rise, t_evening])
            ephemerides[current]['end_time'] = np.min([set, t_breaking_dawn_part_II])
    for object in list(ephemerides.keys()):
        if len(ephemerides[object]['time']) < 1:
            del ephemerides[object]

    return(ephemerides)


def form_post(obj_dict, observer):
    """ Creates a form post for the ephemerides form on the MPC site and obtains the resulting website information

    Args:
        obj_dict (dict): dictionary containing the objects in consideration
        observer (astroplan.Observer): observer for which to get ephemerides

    Returns:
        soup (BeautifulSoup): obtained website information
    """
    obj_list = list(obj_dict.keys())

    #TODO look at 'int' (intervals), 'start' (starting time), geocentric?
    form_data = {'W':'j',
             'mb': '-30', 'mf': '30',
             'dl': '-90', 'du': '+90',
             'nl': '0', 'nu': '100',
             'Parallax': '0', 'obscode': '500',
             'int': '0', 'start': '0',
             'raty': 'd', 'mot': 'm',
             'dmot': 'r', 'out': 'f',
             'sun': 'x', 'oalt': '20',
             'long': f'{observer.longitude.deg}',
             'lat': f'{observer.latitude.deg}',
             'alt': f'{observer.elevation}',
             'obj': obj_list,
             'sort': 'd'}

    response = requests.post(URL_EPH, data=form_data)
    soup = BeautifulSoup(response.content, 'html.parser')

    return(soup)



