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
        self.init_attempts = 130
        self.w_add_object = 35
        self.w_remove_object = 2.5
        self.w_add_obs = 25
        self.w_remove_obs = 5
        self.w_replace = 32.5
        self.max_iter = 100
        self.init_fill = 0.4
        self.n_sub_iter = 35
        self.add_attempts = 13

        self.learning_rate = 1e-4
        self.discount_factor = 0.75
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99
        self.memory_size = 10000
        self.batch_size = 64
        self.target_update_freq = 10
        self.episodes = 100