# Module that contains the configuration class, which contains parameters used by the algorithm.

import astropy.units as u
import numpy as np

class Configuration:
    """Class that holds all parameters to be used by the code
    """
    def __init__(self):
        """ Initializes the configuration parameters """
        
        self.seed = np.random.randint(0,17000)
        
        self.moon_dist = 2.5*u.deg 
        self.mag_lim = 21.5
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
        self.w_empty_add = 1e-5
        self.max_iter = 100
        self.init_fill = 0.4
        self.n_sub_iter = 35
        self.n_sub_iter_otf = 3
        self.add_attempts = 13

        # RL general
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        self.memory_size = 10000
        self.batch_size = 32
        self.target_update_freq = 10  # episodes
        self.episodes = 1000  # episodes per run
        self.exploration_rate = 1.0
        self.exploration_decay = 0.1 ** (1/self.episodes)  # so that at final timestep, exploration rate equals 0.1
        self.steps = 200  # steps per episode
        self.masking_duration = 2  # steps

        self.layer_size = 256
        self.layer_size_critic = 32

        self.state_length = 360  # minutes
        self.n_objects = 5  # objects

        # PPO 
        self.clip_ratio = 0.2
        self.n_epochs = 10  # epochs

        # on the fly reward weights
        self.w_time_from_peak = 0.5
        self.w_airmass = 0.5
        self.w_magnitude = 0.5
        self.w_motion = 0.5
        self.w_n_observations = 2
        self.w_time_gap = 1
        self.w_all_viewed = 1

        self.w_total = self.w_time_from_peak \
                     + self.w_airmass \
                     + self.w_magnitude \
                     + self.w_motion \
                     + self.w_n_observations \
                     + self.w_time_gap \
                     + self.w_all_viewed

        # used for scale of determining magnitude scale
        self.mag_min = 15
        self.motion_scale = 100  # arcsec/min

        # for evaluation runs
        self.evaluation_interval = 50  # episodes