# this will contain the gradient descent agent, following Kutay's method

from helper import airmass, rise_set, twilight_rise_set
from astroplan import FixedTarget 
from astropy.coordinates import SkyCoord
import numpy as np
from param_config import Configuration
from astropy.time import Time
import copy
import time

class GDAgent():

    def __init__(self, observer, eph_dict, time, env, config = Configuration()):
        self.observer = observer
        self.eph_dict = eph_dict
        self.time = time
        self.env = env
        self.object_weights = {}
        self.__create_init_weights()
        self.action_weights = [config.w_add_object, config.w_remove_object, config.w_add_obs, config.w_remove_obs, config.w_replace]
        self.config = config

    def create_init_state(self):
        ''' Tries to insert a certain number of observations for the initial state '''
        init_weights = self.object_weights.copy()

        def select_object(init_weights):
            ''' Selects an object based on a weighted random choice '''
            obj = np.random.choice(list(init_weights.keys()), p=np.array(list(init_weights.values()))/np.sum(list(init_weights.values())))
            return(obj)
        
        def observation_window(object):
            ''' Finds the first and last time at which object can be observed '''
            rise, set = rise_set(self.observer, FixedTarget(coord=SkyCoord(self.eph_dict[object]['coord'][0])), self.time)
            twilight_morning, twilight_evening = twilight_rise_set(self.observer, self.time)
            start = np.max([rise, twilight_evening])
            end = np.min([set, twilight_morning])
            if end-start > self.config.airmass_window:
                peak = rise + (set-rise)/2
                if peak - start < self.config.airmass_window/2:
                    end = start + self.config.airmass_window
                elif end - peak < self.config.airmass_window/2:
                    start = end - self.config.airmass_window
                else:
                    start = peak - self.config.airmass_window/2
                    end = peak + self.config.airmass_window/2
            return(start, end)
    
        def add_attempt_loop(object, start, end):
            ''' Loops until object is added '''
            success = False
            attempt = 0
            while not success and attempt < self.config.init_attempts:
                add_time = Time(str(np.random.uniform(start.value, end.value)), format='jd')
                success = self.env.add_object(object, add_time)
                attempt += 1
            return(success)

        while np.sum(self.env.rewards)/len(self.env.rewards) < self.config.init_fill and len(list(init_weights.keys())) > 0:
            object = select_object(init_weights)
            start, end = observation_window(object)
            success = add_attempt_loop(object, start, end)
            if success:
                del init_weights[object]


    def __create_init_weights(self):
        ''' Creates the initial weights for the objects '''
        total_weights = 0

        def calculate_averages(obj):
            ''' Calculates average airmass, motion and magnitudes for calculating weights '''
            eph = self.eph_dict[obj]
            t_eph = self.eph_dict[obj]['time']
            avg_airmass = np.mean([airmass(self.observer, FixedTarget(coord=eph['coord'][i]), t_eph[i]) for i in range(len(t_eph))])
            avg_motion = np.mean(eph['motion'])
            avg_magnitude = np.mean(eph['mag_V'])
            return(avg_airmass, avg_motion, avg_magnitude)


        for obj in self.eph_dict.keys():
            avg_airmass, avg_motion, avg_magnitude = calculate_averages(obj)

            weight = int(10 * (1/avg_airmass) * (1+np.log10(avg_motion)) * avg_magnitude)
            self.object_weights[obj] = weight
            total_weights += weight

        for obj in self.eph_dict.keys():
            self.object_weights[obj] /= total_weights
    
    def gradient_descent(self):
        ''' Performs the gradient descent '''
        iteration = 0

        def sample_action(env):
            ''' Calculates weights for actions and picks one of them '''
            w_action = copy.deepcopy(self.action_weights)

            if all([env.obs_objects[object] for object in self.eph_dict.keys()]):
                w_action[0] = 0
            if not any([env.obs_objects[object] for object in self.eph_dict.keys()]):
                w_action[1:5] = [0,0,0,0]
            if all([env.obs_count[object] <= 2 for object in self.eph_dict.keys()]):
                w_action[3] = 0
            return(np.random.choice([1,2,3,4,5], p=np.array(w_action)/np.sum(w_action)))
        
        def sample_object(env, invert=False, add_object=False, remove_obs=False):
            ''' Calculates weights for objects and picks one of them '''
            w_object = self.object_weights.copy()
            for obj in w_object.keys():
                if invert:
                    w_object[obj] = 1/w_object[obj]
                if not env.obs_objects[obj] and not add_object:
                    w_object[obj] = 0
                if env.obs_objects[obj] and add_object:
                    w_object[obj] = 0
                if env.obs_count[obj] <= 2 and remove_obs:
                    w_object[obj] = 0
            return(np.random.choice(list(self.eph_dict.keys()), p=np.array(list(w_object.values())) / np.sum(list(w_object.values()))))
        
        def perform_action(action, next_env):
            ''' Handles the chosen action '''
            if action == 1:  # add object  
                object = sample_object(next_env, add_object=True)
                success = False
                attempt = 0
                while not success and attempt < self.config.add_attempts:
                    attempt += 1
                    add_time = Time(str(np.random.uniform(
                        self.eph_dict[object]['start_time'].value, 
                        self.eph_dict[object]['end_time'].value)), format='jd')
                    success = next_env.add_object(object, add_time)

            if action == 2:  # remove object  
                object = sample_object(next_env, invert=True)
                success = next_env.remove_object(object)

            if action == 3:  # add observation   
                object = sample_object(next_env)
                success = next_env.add_observation(object)


            if action == 4:  # remove observation 
                object = sample_object(next_env, invert=True, remove_obs=True)
                observations_current = []
                for i in range(len(next_env.obs_starts)):
                    if next_env.obs_starts[i] == object:
                        observations_current.append(i)
                observation = np.random.choice(observations_current)
                success = next_env.remove_observation(object, observation)


            if action == 5:  # replace observation   
                object = sample_object(next_env, invert=True)
                observations_current = []
                for i in range(len(next_env.obs_starts)):
                    if next_env.obs_starts[i] == object:  
                        observations_current.append(i)
                observation = np.random.choice(observations_current)
                success = next_env.replace_observation(object, observation)

        while iteration < self.config.max_iter and self.env.calculate_reward() < 0.9:
            next_env = copy.deepcopy(self.env)
            for sub_iter in range(self.config.n_sub_iter):
                action = sample_action(next_env)            
                perform_action(action, next_env)


            if next_env.calculate_reward() > self.env.calculate_reward():
                self.env = copy.deepcopy(next_env)
            iteration += 1
