import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import matplotlib.pyplot as plt
import copy
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroplan import FixedTarget 

from helper import airmass, rise_set, twilight_rise_set
from param_config import Configuration

class GDAgent():

    def __init__(self, observer, eph_dict, time, env, config = Configuration()):
        """Initializes the gradient descent 'agent'

        Args:
            observer (astroplan.Observer): Observer representing the observatory
            eph_dict (dict): dictionary containing the ephemerides for the NEO's
            time (astropy.time.Time): date at which the observations take place
            env (env_table.ObservationScheduleEnv): schedule to be filled with observations
            config (param_config.Configuration, optional): hyperparameters to use. Defaults to Configuration().
        """        
        self.observer = observer
        self.eph_dict = eph_dict
        self.time = time
        self.env = env
        self.object_weights = {}
        self.__create_init_weights()
        self.action_weights = [config.w_add_object, config.w_remove_object, config.w_add_obs, config.w_remove_obs, config.w_replace]
        self.config = config
        self.history = []

    def create_init_state(self):
        ''' Tries to insert a certain number of observations for the initial state '''
        init_weights = self.object_weights.copy()

        def select_object(init_weights):
            ''' Selects an object based on a weighted random choice 
            
            Args: 
                init_weights (dict): dictionary containing weights of the objects to be sampled

            Returns:
                obj (str): key of the object that was sampled weighted randomly
            '''
            obj = np.random.choice(list(init_weights.keys()), p=np.array(list(init_weights.values()))/np.sum(list(init_weights.values())))
            return(obj)
        
        def observation_window(object):
            ''' Finds the first and last time at which object can be observed 
            
            Args:
                object (str): key of object in consideration

            Returns:
                start (astropy.time.Time): first possible time to observe the object
                end (astropy.time.TIme): last possible time to observe the object
            '''
            rise, set = rise_set(self.observer, FixedTarget(coord=SkyCoord(self.eph_dict[object]['coord'][0])), self.time)
            twilight_morning, twilight_evening = twilight_rise_set(self.observer, self.time)
            start = np.max([rise, twilight_evening])
            end = np.min([set, twilight_morning])

            # find the best hours of observation time depending on the object's airmass
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
            ''' Loops until object is added or a maximum number of iterations is reached
            
            Args:
                object (str): key of object to be added
                start (astropy.time.Time): start of best airmass window
                end (astropy.time.Time): end of best airmass window

            Returns:
                success (bool): True if the object was successfully added
            '''
            success = False
            attempt = 0
            while not success and attempt < self.config.init_attempts:
                add_time = Time(str(np.random.uniform(start.value, end.value)), format='jd')
                success = self.env.add_object(object, add_time)
                attempt += 1
            return(success)

        # attempt to add new objects until a certain threshold is reached or we run out of objects
        #TODO make it so that this doesn't become an infinite loop in some circumstances
        while np.sum(self.env.rewards)/len(self.env.rewards) < self.config.init_fill and len(list(init_weights.keys())) > 0:
            object = select_object(init_weights)
            start, end = observation_window(object)
            success = add_attempt_loop(object, start, end)
            if success:
                del init_weights[object]
        self.history.append(self.env.calculate_reward())


    def __create_init_weights(self):
        ''' Creates the initial weights for the objects '''
        total_weights = 0

        def calculate_averages(obj):
            ''' Calculates average airmass, motion and magnitudes for calculating weights 
            
            Args:
                obj (str): key of object to calculate averages for

            Returns:
                avg_airmass (float): the average airmass during the observation night
                avg_motion (float): the average motion of the object during the observation night
                avg_magnitude (float): the average magnitude of the object during the observation night
            '''
            eph = self.eph_dict[obj]
            t_eph = self.eph_dict[obj]['time']
            avg_airmass = np.mean([airmass(self.observer, FixedTarget(coord=eph['coord'][i]), t_eph[i]) for i in range(len(t_eph))])
            avg_motion = np.mean(eph['motion'])
            avg_magnitude = np.mean(eph['mag_V'])
            return(avg_airmass, avg_motion, avg_magnitude)

        for obj in self.eph_dict.keys():
            avg_airmass, avg_motion, avg_magnitude = calculate_averages(obj)

            weight = (10 * (1/avg_airmass) * (1+np.log10(avg_motion)) * avg_magnitude)
            self.object_weights[obj] = weight
            total_weights += weight

        for obj in self.eph_dict.keys():
            self.object_weights[obj] /= total_weights
    
    def gradient_descent(self):
        ''' Fills the schedule as much as possible using gradient descent '''
        iteration = 0

        def sample_action(env):
            ''' Calculates weights for actions and picks one of them 
            
            Args:
                env (env_table.ObservationScheduleEnv): schedule for the observation nights

            Returns:
                action (int): the chosen action based on weighted random choice
            '''
            w_action = copy.deepcopy(self.action_weights)

            # disable certain actions based on current state of the schedule
            if all([env.obs_objects[object] for object in self.eph_dict.keys()]):
                w_action[0] = 0
            if not any([env.obs_objects[object] for object in self.eph_dict.keys()]):
                w_action[1:5] = [0,0,0,0]
            if all([env.obs_count[object] <= 2 for object in self.eph_dict.keys()]):
                w_action[3] = 0
            return(np.random.choice([1,2,3,4,5], p=np.array(w_action)/np.sum(w_action)))
        
        def sample_object(env, invert=False, add_object=False, remove_obs=False):
            ''' Calculates weights for objects and picks one of them 
            
            Args:
                env (env_table.ObservationScheduleEnv): schedule of the observation night
                invert (bool, optional): inverts the weights of the objects. Defaults to False.
                add_object (bool, optional): determines which objects can be added depending if they're already in schedule. Defaults to False.
                remove_obs (bool, optional): determines which objects can have observations removed. Defaults to False.

            Returns:
                object (str): key of the object chosen weighted randomly.
            '''
            w_object = self.object_weights.copy()
            for obj in w_object.keys():
                if invert:
                    if w_object[obj] > 0:
                        w_object[obj] = 1/w_object[obj]
                if not env.obs_objects[obj] and not add_object:
                    w_object[obj] = 0
                if env.obs_objects[obj] and add_object:
                    w_object[obj] = 0
                if env.obs_count[obj] <= 2 and remove_obs:
                    w_object[obj] = 0
            return(np.random.choice(list(self.eph_dict.keys()), p=np.array(list(w_object.values())) / np.sum(list(w_object.values()))))
        
        def perform_action(action, next_env):
            ''' Handles the chosen action 
            
            Args:
                action (int): indicates which action to perform
                next_env (env_table.ObservationScheduleEnv): schedule of the observation night
            '''
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

        # keep performing actions until a certain number of iterations is reached or a certain fill factor is reached
        while iteration < self.config.max_iter: # and self.env.calculate_reward() < 0.95:
            next_env = copy.deepcopy(self.env)
            for sub_iter in range(self.config.n_sub_iter):
                action = sample_action(next_env)            
                perform_action(action, next_env)

            if next_env.calculate_reward() > self.env.calculate_reward():
                self.env = copy.deepcopy(next_env)
            self.history.append(self.env.calculate_reward())
            iteration += 1

    def plot_history(self):
        """ Generates a plot of the fill factor through the gradient descent """
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Fill factor')
        plt.grid()
        plt.savefig('convergence.png')
        plt.show()
