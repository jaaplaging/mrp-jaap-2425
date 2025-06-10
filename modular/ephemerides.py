# Module containing the different ephemerides classes that can be used with all other classes in this directory

import pickle
import numpy as np
import sys
from mpc_scraper import scraper
from helper import create_observer
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget
from helper import rise_set, twilight_rise_set, opt_time, airmass


FILE_EPH_DICT = 'mrp-jaap-2425/modular/top_eph_processed.pkl'

class EphemeridesDummy():
    """Epherides class using dummy values for the NEO variables
    """
    def __init__(self, observer, time, config, random=False):
        """Initializes the ephemerides object

        Args:
            observer (astroplan.Observer): Observer used to determine visibility of NEOs
            time (astropy.time.Time): Time used to determine visibility of NEOs
            config (param_config.Configuration): Object containing all parameters used
            random (bool, optional): Determines whether the ephemerides returned by get_daily_ephemerides 
            are random every time. Defaults to False.
        """
        self.random = random
        self.config = config
        if not random:
            self.eph_dict = self.generate_constant_ephemerides()

        np.random.seed(self.config.seed)

    def generate_constant_ephemerides(self):
        """Creates a dictionary of ephemerides for n_objects objects

        Returns:
            eph_dict (dict): Dictionary with ephemerides per object
        """
        eph_dict = {}
        # Values determined based on actual values from the MPC object table
        for i in range(self.config.n_objects):
            eph_dict[str(i)] = {}
            eph_dict[str(i)]['RA'] = np.linspace(i, i+1, 24)
            eph_dict[str(i)]['dec'] = np.linspace(i, i+5, 24)
            eph_dict[str(i)]['total motion'] = np.linspace(10*i, 10*i, 24)
            eph_dict[str(i)]['RA motion'] = np.linspace(7.07*i, 7.07*i, 24)
            eph_dict[str(i)]['dec motion'] = np.linspace(7.07*i, 7.07*i, 24)
            eph_dict[str(i)]['apparent magnitude'] = np.linspace(15 + i, 15 + i, 24)
            eph_dict[str(i)]['peak_ind'] = int(i*self.config.state_length/10 + self.config.state_length/10)
            eph_dict[str(i)]['peak_airmass'] = 1 + i
            eph_dict[str(i)]['rise_ind'] = 0
            eph_dict[str(i)]['set_ind'] = self.config.state_length
        return(eph_dict)

    def get_daily_ephemerides(self):
        """ Either returns the ephemerides dict if not self.random or randomly generates a new one if self.random

        Returns:
            eph_dict (dict): Dictionary of ephemerides per NEO
        """
        eph_dict = {}
        if not self.random:
            return(self.eph_dict)
        else:
            for i in range(self.config.n_objects):
                eph_dict[str(i)] = {}
                eph_dict[str(i)]['RA'] = np.random.uniform(i, i+1, 24)
                eph_dict[str(i)]['dec'] = np.random.uniform(i, i+5, 24)
                eph_dict[str(i)]['total motion'] = np.random.uniform(0, 100, 24)
                eph_dict[str(i)]['RA motion'] = np.random.uniform(0, 100, 24)
                eph_dict[str(i)]['dec motion'] = np.random.uniform(0, 100, 24)
                eph_dict[str(i)]['apparent magnitude'] = np.random.uniform(15, 21.5, 24)
                eph_dict[str(i)]['peak_ind'] = np.random.uniform(-100, self.config.state_length+100)
                eph_dict[str(i)]['peak_airmass'] = np.random.uniform(1, 10)
                eph_dict[str(i)]['rise_ind'] = np.random.uniform(-200, eph_dict[str(i)]['peak_ind']-10)
                eph_dict[str(i)]['set_ind'] = np.random.uniform(eph_dict[str(i)]['peak_ind']+10, self.config.state_length+200)
        return(eph_dict)


class EphemeridesSimulated():
    """Ephemerides class using simulated NEOs"""
    def __init__(self, observer, time, config, random=False, file_eph = FILE_EPH_DICT, eph_ind=0):
        """Initializes the ephemerides object

        Args:
            observer (astroplan.observer): Observer object to determine visibility
            time (astropy.time.Time): Time at which to determine object visibility
            config (param_config.Configuration): Configuration object containing parameter values
            random (bool, optional): Determines if get_daily_ephemerides should return random ephemerides every time. Defaults to False.
            file_eph (string, optional): Path to the simulated ephemerides file. Defaults to FILE_EPH_DICT.
            eph_ind (int, optional): Determines the index of ephemerides to return if random=False. Defaults to 0.
        """
        self.random = random
        self.config = config
        self.file_eph = file_eph
        self.eph_ind = eph_ind
        with open(file_eph, 'rb') as f:
            self.eph_dict = pickle.load(f)

        np.random.seed(self.config.seed)

    def get_daily_ephemerides(self):
        """Either returns the ephemerides from the list at self.eph_ind of the list or randomly selects them
        from the list.

        Returns:
            dict: Dictionary containing the ephemerides selected.
        """ 
        if self.random:
            available_objects = False
            while not available_objects:
                eph_dict_try = np.random.choice(self.eph_dict)
                eph_dict_copy = {}
                for key in eph_dict_try.keys():
                    if eph_dict_try[key]['rise_ind'] < self.config.state_length and eph_dict_try[key]['set_ind'] > 0:
                        eph_dict_copy[key] = eph_dict_try[key]
                        available_objects = True
            return(eph_dict_copy)
        else:
            return(self.eph_dict[self.eph_ind])


class EphemeridesReal():
    """Ephemerides class for creating ephemerides from the actual MPC NEO table
    """
    def __init__(self, observer, time, config, random = False):
        """Initializes the ephemerides object
        
        Args:
            observer (astroplan.Observer): Observer object to determine the visibility of the NEOs
            time (astropy.time.Time): Time to determine the visibility of the NEOs
            config (param_config.Configuration): Configuration class holding the parameter values
            random (bool, optional): Not used in this class. Defaults to False.
        """
        self.config = config
        self.random = random
        self.observer = observer
        self.time = time
        self.dawn, self.dusk = twilight_rise_set(self.observer, self.time)
        self.scrape_ephemerides()

    def time_to_ind(self, given_time):
        """Determines the index in the schedule of a given time.

        Args:
            given_time (astropy.time.Time): Time to be converted to an index.

        Returns:
            ind (int): Index corresponding to the given time
        """
        ind = int((given_time - self.dusk).value*24*60)
        return(ind)

    def get_airmass_peak(self, object_ephemerides):
        """Determines the index of the best airmass, the value of the best airmass, the index of the 
        rising of the object and the index of the setting of the object.

        Args:
            object_ephemerides (dict): Dictionary of the object
        
        Returns:
            peak_ind (int): Index of the best airmass of the object
            peak_airmass (float): Best airmass of the object
            rise_ind (int): Index of the rise of the object
            set_ind (int): Index of the set of the object
        """
        ra_mid = object_ephemerides['coord'][len(object_ephemerides['coord'])//2].ra.value
        ra_mid = str(ra_mid) + 'h'
        dec_mid = object_ephemerides['coord'][len(object_ephemerides['coord'])//2].dec.value
        dec_mid = str(dec_mid) + 'd'
        coords_mid = SkyCoord(ra_mid, dec_mid, frame='gcrs')
        target = FixedTarget(name='target', coord=coords_mid)
        rise, set = rise_set(self.observer, target, self.time)
        rise_ind, set_ind = self.time_to_ind(rise), self.time_to_ind(set)
        if set < self.dusk or rise > self.dawn or set_ind < 0 or rise_ind > self.config.state_length:  # If the object is never visible
            return(-1, -1, -1, -1)
        else:
            peak_time = opt_time(self.observer, target, self.time)
            peak_ind = int((peak_time - self.dusk).value*24*60)
            peak_airmass = airmass(self.observer, target, peak_time)
            if peak_airmass <= 0:
                return(-1, -1, -1, -1)
            return(peak_ind, peak_airmass, rise_ind, set_ind)

    def scrape_ephemerides(self):
        """Gets the ephemerides from the MPC table and adds some other variables for each of the NEOs
        """
        _, self.eph_dict = scraper(observer=self.observer, time=self.time)
        for key in self.eph_dict.keys():
            peak_ind, peak_airmass, rise_ind, set_ind = self.get_airmass_peak(self.eph_dict[key])
            self.eph_dict[key]['peak_ind'] = peak_ind
            self.eph_dict[key]['peak_airmass'] = peak_airmass
            self.eph_dict[key]['rise_ind'] = rise_ind
            self.eph_dict[key]['set_ind'] = set_ind

    def get_daily_ephemerides(self):
        """Returns the ephemerides.

        Returns:
            self.eph_dict (dict): Dictionary of real ephemerides.
        """ 
        return(self.eph_dict)
        
        
    
