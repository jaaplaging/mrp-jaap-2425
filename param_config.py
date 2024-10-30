import astropy.units as u

class Configuration:
    def __init__(self):
        """ Initializes the configuration parameters """
        self.moon_dist = 2.5*u.deg 
        self.mag_lim = 20.5
        self.min_visibility = 1/24*u.day 
        self.t_setup = 2  # minutes
        self.t_obs = 10  # minutes
        self.t_int = 45  # minutes
        self.t_int_min = 30  # minutes
        self.t_replace = 15  # minutes
        self.airmass_window = 6/24  # days
        self.init_attempts = 90
        self.w_add_object = 40
        self.w_remove_object = 2
        self.w_add_obs = 25
        self.w_remove_obs = 5
        self.w_replace = 20
        self.max_iter = 500
        self.init_fill = 0.5
        self.n_sub_iter = 45
        self.add_attempts = 10