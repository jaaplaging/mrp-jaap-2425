# file containing functions that scrape the contents of the MPC site 
import requests
from bs4 import BeautifulSoup
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroplan import FixedTarget
from helper import create_observer, twilight_rise_set, rise_set, near_moon, observable
from astropy.time import Time
import numpy as np
import warnings
from param_config import Configuration

config = Configuration()

URL_OBJ = "https://minorplanetcenter.net/iau/NEO/neocp.txt"
URL_EPH = "https://cgi.minorplanetcenter.net/cgi-bin/confirmeph2.cgi"


def scraper(observer, time):
    # function that calls other functions
    obj_dict = get_objects()
    obj_dict = select_objects(obj_dict, observer, time)
    eph_dict = get_ephemerides(obj_dict, observer, time)
    return(obj_dict, eph_dict)

def get_objects():
    # function that reads the main site and returns them as a dict to scraper
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
    ''' function that determines which objects to get ephemerides for '''
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
    # function that returns a clean dict with ephemerides to scaper
    soup = form_post(obj_dict, observer)
    t_breaking_dawn_part_II, t_evening = twilight_rise_set(observer, time)
    ephemerides = {}

    def add_descendant(descendant):
        ''' Adds empty entry for the descendant '''
        ephemerides[descendant] = {'time': [],
                                    'coord': [],
                                    'mag_V': [],
                                    'motion': []}
        return(ephemerides)

    def parse_child_text(child, observer, t_breaking_dawn_part_II, t_evening, ephemerides):
        ''' Parses the text of the ephemeris '''
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
            ephemerides = add_descendant(descendant.text)
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
    # function that performs a request.post to get the ephemerides
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



if __name__ == '__main__':
    observer = create_observer()
    time = Time(['2024-10-11 18:00:00'], scale='utc')
    print(scraper(observer, time))