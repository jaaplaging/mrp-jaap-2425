# Module containing the environments that can be used together will all other classes in this directory.

import sys
import numpy as np
import copy

from helper import twilight_rise_set, create_observer, index_last, opt_time, airmass, rise_set
from astroplan import FixedTarget
from time import perf_counter


class ScheduleEnv():
    """Class containing the schedule environment
    """
    def __init__(self, observer, time, eph_class, reward_class, config):
        """ Initializes the observation schedule environment

        Args:
            observer (astroplan.Observer): observatory for which to schedule observations
            time (astropy.time.Time): date for which to create the schedule
            eph_class (ephemerides object): Any of the classes from the ephemerides.py module (initiated)
            reward_class (reward object): An initiated Reward object from the rewards.py module
            config (param_config.Configuration): Initiated Configuration object
        """  
        self.config = config      
        self.observer = observer
        self.time = time
        start_time, minutes = self.__calculate_start_length()
        self.length = minutes
        self.start_time = start_time

        self.state_space_a = (self.config.n_objects * 9)
        self.state_size_a = self.config.n_objects * 9
        self.state_type_a = 'float32'

        self.state_space_b = (self.config.state_length)
        self.state_size_b = self.config.state_length
        self.state_type_b = 'int32'

        self.action_space = (self.config.n_objects * self.config.state_length * 5)
        self.action_size = self.config.n_objects * self.config.state_length * 5

        self.reward = reward_class
        self.empty_flag = 0

        self.eph_names = []

        self.reset(eph_class)

        np.random.seed(self.config.seed)

        #self.twilight_evening_ind, self.twilight_morning_ind = self.__calculate_twilight()
        #self.state[:self.twilight_evening_ind] = 0
        #self.state[self.twilight_morning_ind:] = 0


    def reset(self, eph_class):
        """ Resets the environment schedule to empty 
        
        Args:
            eph_class (ephemerides class): ephemerides object used to generate the object state

        Return: 
            list: the state representation, a list of the objects and the schedule
        """        
        self.schedule = np.full((self.length), 0, dtype='int32')
        self.obs_starts = [None]*self.length
        self.observations = [[] for i in range(self.config.n_objects)]
        self.obs_objects = {}
        self.obs_count = {}
        self.steps_taken = 0
        self.daily_ephemerides = eph_class.get_daily_ephemerides()
        self.object_to_ind = {}
        self.object_state, self.object_state_norm = self.__create_object_state()
        for object in range(len(self.object_state)):
            self.obs_objects[object] = False  # keeps track which objects are being observed
            self.obs_count[object] = 0  # keeps track how often objects are observed
        self.__create_action_masks()
        return([self.object_state_norm, self.schedule])

    def __create_action_masks(self):
        """Creates masks for action possibilities to be applied to the output of the neural network
        """ 
        # Mask that is True when night and False when day
        self.mask_day = np.full((self.config.n_objects, self.config.state_length, 5), True)

        # Mask that keeps track of the visibility of objects based on when they rise and set
        self.mask_object_visibility = np.full((self.config.n_objects,self.config.state_length,5), False)
        for ind, obj in enumerate(self.object_state):
            self.mask_object_visibility[ind, np.max([0,int(obj[1])]):np.min([self.config.state_length,int(obj[2])]), :] = True

        self.base_mask = self.mask_day & self.mask_object_visibility

        # Mask that keeps track of the availability of the different possible actions (add, remove, replace, etc.)
        self.action_availability_mask = np.full((self.config.state_length, 5), True)
        self.action_availability_mask[-self.config.t_obs*2-self.config.t_setup*2-self.config.t_int:,0] = False

        self.action_availability_mask[self.mask_day.shape[0]:,1] = False

        self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False
        self.action_availability_mask[:,3] = False
        self.action_availability_mask[:,4] = False

        # Mask that keeps track of which types of actions are possible for which objects
        self.object_unavailability_mask = np.full((self.config.n_objects, 5), False)
        self.object_unavailability_mask[:,0] = True

        # Mask that keeps track of the starting indices of all observations
        self.observation_mask = np.full((self.config.n_objects,self.config.state_length), False)

        # Mask that prevents the agents from repeating the same actions constantly
        self.taken_actions_mask = np.full((self.config.n_objects,self.config.state_length,5), True)
        self.taken_actions_countdown = np.full((self.config.n_objects,self.config.state_length,5), 0)

        # Combine to create the full mask to be applied to the output of the agent networks
        self.total_mask = self.base_mask & np.einsum('ij,kj -> kij', self.action_availability_mask, self.object_unavailability_mask) & self.taken_actions_mask

    def __update_total_mask(self):
        """Updates the total mask after an action taken
        """
        self.total_mask = self.base_mask & np.einsum('ij,kj -> kij', self.action_availability_mask, self.object_unavailability_mask)

        # Rebuild the availability of the remove observation and the replace observation action
        for object in range(self.total_mask.shape[0]):
            if self.object_unavailability_mask[object,3]:
                self.total_mask[object,:,3] = self.observation_mask[object,:]
            if self.object_unavailability_mask[object,4]:
                replace_available = np.full((self.config.state_length), False)
                for i in range(len(self.schedule)-self.config.t_obs-self.config.t_setup):
                    if not (self.schedule[i] == object + 1 and self.schedule[i+self.config.t_obs+self.config.t_setup-1] == object + 1):
                        if all(self.schedule[j] in [0, object+1] for j in range(i,i+self.config.t_obs+self.config.t_setup)):
                            if not all(self.schedule[j] == object+1 for j in range(i,i+self.config.t_obs+self.config.t_setup)):
                                replace_available[i] = True
                self.total_mask[object,:,4] = replace_available & self.base_mask[object,:,4]
        
        # self.taken_actions_countdown = np.where(self.taken_actions_countdown >= 1, self.taken_actions_countdown - 1, self.taken_actions_countdown)
        # self.taken_actions_mask = np.where(self.taken_actions_countdown >= 1, False, True)
        # self.total_mask = self.total_mask & self.taken_actions_mask

    def create_mask(self):
        """Returns that total mask to be applied to the logits of the agent networks

        Returns:
            np.ndarray: the flattened total mask
        """
        return(self.total_mask.flatten())

    def __create_object_state(self):
        """Creates an array containing the features of the NEOs being considered

        Returns: 
            object_state (np.ndarray): Array containing the features of the NEOs being considered
            object_state_norm (np.ndarray): object_state normalized
        """
        object_state = np.empty((self.config.n_objects, 9))
    
        def normalize(object_state):
            """Normalizes the object state

            Args:
                object_state (np.ndarray): Array containing the features of the NEOs being considered

            Returns:
                object_state (np.ndarray): Array containing the features of the NEOs being considered
                object_state_norm (np.ndarray): object_state normalized
            """
            object_state_norm = object_state.copy()
            object_state_norm[:,:3] = object_state_norm[:,:3] / (0.5 * self.config.state_length) - 1 
            object_state_norm[:,3] = (object_state_norm[:,3] - 5) / 5
            object_state_norm[:,4] = (object_state_norm[:,4] - 17.5) / 5
            object_state_norm[:,5] = (object_state_norm[:,5] - 50) / 25
            return(object_state, object_state_norm)

        state_ind = 0

        # Extract the NEO features from the ephemerides object
        for ind, object in enumerate(self.daily_ephemerides.keys()):
            peak_ind = self.daily_ephemerides[object]['peak_ind']
            peak_airmass = self.daily_ephemerides[object]['peak_airmass']
            rise_ind = self.daily_ephemerides[object]['rise_ind']
            set_ind = self.daily_ephemerides[object]['set_ind']
            if peak_airmass == -1:
                continue
            else:
                mag = np.mean(self.daily_ephemerides[object]['apparent magnitude'])
                motion = np.mean(self.daily_ephemerides[object]['total motion'])
                observations = 0  # Keeps track of number of observations per object
                previous_observation = -2  # Keeps track of the previous observation
                flag = 1  # Indicates the object is an NEO and not empty
                object_state[state_ind,:] = [peak_ind, rise_ind, set_ind, peak_airmass, mag, motion, observations, previous_observation, flag]
                self.object_to_ind[object] = ind
                state_ind += 1
                self.eph_names.append(object)
            if state_ind >= self.config.n_objects:
                break
        
        # Fill the remaining rows with empty NEOs
        for i in range(state_ind, self.config.n_objects):
            object_state[i, :] = [0,0,0,0,0,0,0,0,0]
        
        object_state, object_state_norm = normalize(object_state)

        return(object_state, object_state_norm)


    def __calculate_start_length(self):
        """ Calculates the time of noon and how long the night takes in minutes

        Returns:
            noon (astropy.time.Time): time of astronomical noon
            minutes (int): length of a the schedule in minutes
        """        
        noon = self.observer.noon(self.time, which='nearest')
        minutes = self.config.state_length
        return(noon, minutes)

    # def __calculate_twilight(self):
    #     """Gives the indices in the state for when the astronomical night starts and ends

    #     Returns:
    #         ind_set (int): index of astronomical twilight in the evening
    #         ind_rise (int): index of astrononmical twilight in the morning
    #     """
    #     twilight_rise, twilight_set = twilight_rise_set(self.observer, self.time)
    #     ind_set = int((twilight_set - self.start_time).value*24*60)
    #     ind_rise = int((twilight_rise - self.start_time).value*24*60)
    #     return(ind_set, ind_rise)

    # def __time_to_index(self, given_time):
    #     ''' Converts inserted time to index in schedule 
        
    #     Args:
    #         given_time (astropy.time.Time): time to convert to an index

    #     Returns:
    #         index (int): index in the schedule corresponding to the given time
    #     '''
    #     index = int((given_time-self.start_time).value*2*60)
    #     return(index)


    def step(self, action):
        """ Performs a given action within the schedule

        Args:
            action (int): action index (to be unraveled) of the action taken

        Returns:
            list: Object state and schedule
            reward (float): reward returned by the reward object
            done (int): 1 if a certain number of steps have been taken, 0 otherwise
        """        
        if action < 0 or action >= self.action_size:
            return([self.object_state_norm, self.schedule], -2, 0)
        
        self.steps_taken += 1
        # Obtain the object, index and action type taken from the index
        obj, ind, action = np.unravel_index(action, (self.config.n_objects, self.config.state_length, 5))    

        # add object
        if action == 0:
            self.add_object(obj, ind)
        
        # remove object
        elif action == 1:
            self.remove_object(obj)

        # add observation
        elif action == 2:
            self.add_observation(obj, ind)
        
        # remove observation
        elif action == 3:
            self.remove_observation(obj, ind)
        
        # replace observation
        elif action == 4:
            self.replace_observation(obj, ind)
            
        for ind, _ in enumerate(self.observations):
            self.observations[ind].sort()
        reward = self.reward.get_reward(self, 0)

        if self.steps_taken >= self.config.steps-1:
            done = 1
        else:
            done = 0

        return([self.object_state_norm, self.schedule], reward, done)


    # def __index_to_object(self, index):
    #     ''' Translates input object index of object to actual object key 
        
    #     Args:
    #         index (int): index of the object

    #     Returns:
    #         object (str): name of the object
    #     '''
    #     return(list(self.daily_ephemerides.keys())[index])


    # def __is_interval_free(self, start, end):
    #     """ Checks whether a certain time period in the schedule is empty

    #     Args:
    #         start (int): index of start of period
    #         end (int): index of end of period

    #     Returns:
    #         free (bool): True if time interval is empty in schedule
    #     """        
    #     return(all([self.schedule[i] == 0 for i in range(start,end)]))


    def add_object(self, object, ind_start):
        ''' Insert object in schedule at the given index and adds a second observation of the object after 
        a certain interval given in the config 
        
        Args:
            object (int): index of object to add
            ind_start (int): index at which to add the object in the schedule
        '''
        # Define some indices
        ind_end_obs_1 = ind_start+self.config.t_setup+self.config.t_obs
        ind_start_obs_2 = ind_start+self.config.t_obs+self.config.t_int
        ind_end = ind_start + self.config.t_setup + self.config.t_obs *2 + self.config.t_int
        #TODO make the 45 minute interval variable by +/- 5 minutes
        #TODO reconsider rewards (make weights a factor as well)
        
        # Add the object in the schedule and change other lists
        self.schedule[ind_start:ind_end_obs_1] = object+1
        self.schedule[ind_start_obs_2:ind_end] = object+1
        self.obs_starts[ind_start] = object+1
        self.observations[object].append(ind_start)
        self.obs_starts[ind_start_obs_2] = object+1
        self.observations[object].append(ind_start_obs_2)
        self.obs_objects[object] = True
        self.obs_count[object] = 2
        self.object_state[object, 6] = 2
        self.object_state_norm[object, 6] = 0.2

        # Update the masks
        if ind_start-self.config.t_obs*2-self.config.t_setup-self.config.t_int >= 0 and ind_start-self.config.t_int+self.config.t_setup >= 0:
            self.action_availability_mask[ind_start-self.config.t_obs*2-self.config.t_setup-self.config.t_int:ind_start-self.config.t_int+self.config.t_setup,0] = False
        elif ind_start-self.config.t_int+self.config.t_setup >= 0:
            self.action_availability_mask[:ind_start-self.config.t_int+self.config.t_setup,0] = False
        if ind_start-self.config.t_obs-self.config.t_setup >= 0:
            self.action_availability_mask[ind_start-self.config.t_obs-self.config.t_setup:ind_end_obs_1,0] = False
            self.action_availability_mask[ind_start-self.config.t_obs-self.config.t_setup:ind_end_obs_1,2] = False
        else:
            self.action_availability_mask[:ind_end_obs_1,0] = False
            self.action_availability_mask[:ind_end_obs_1,2] = False
        self.action_availability_mask[ind_start_obs_2-self.config.t_obs-self.config.t_setup:ind_end,0] = False
        #self.action_availability_mask[:,1] = True
        self.action_availability_mask[ind_start_obs_2-self.config.t_obs-self.config.t_setup:ind_end,2] = False
        self.action_availability_mask[ind_start,3] = True
        self.action_availability_mask[ind_start_obs_2,3] = True
        #self.action_availability_mask[ind_start,4] = True
        #self.action_availability_mask[ind_start_obs_2,4] = True

        self.object_unavailability_mask[object,0] = False
        self.object_unavailability_mask[object,1] = True
        self.object_unavailability_mask[object,2] = True
        self.object_unavailability_mask[object,4] = True

        self.observation_mask[object,ind_start] = True
        self.observation_mask[object,ind_start_obs_2] = True

        self.__update_total_mask()

    def remove_object(self, object):
        ''' Remove object from the schedule 
        
        Args: 
            object (int): object to remove completely from the schedule
        '''
        # Remove the object from the schedule and update other states
        self.schedule = np.where(self.schedule == object+1, 0, self.schedule)
        self.obs_starts = [None if self.obs_starts[i] == object+1 else self.obs_starts[i] for i in range(len(self.obs_starts))]
        self.observations[object] = []
        self.obs_objects[object] = False
        self.obs_count[object] = 0
        self.object_state[object, 6] = 0
        self.object_state_norm[object, 6] = 0

        # Update masks
        for ind in range(len(self.schedule)):
            if np.all(self.schedule[ind:ind+self.config.t_obs+self.config.t_setup] == 0):
                self.action_availability_mask[ind,2] = True
                if np.all(self.schedule[ind+self.config.t_int+self.config.t_obs:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup] == 0):
                    self.action_availability_mask[ind,0] = True
            if self.action_availability_mask[ind,3] and self.schedule[ind] == 0:
                self.action_availability_mask[ind,3] = False
            #if self.action_availability_mask[ind,4] and self.state[ind] == 1:
            #    self.action_availability_mask[ind,4] = False
        self.action_availability_mask[-self.config.t_int-self.config.t_obs*2-self.config.t_setup:,0] = False
        self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False

        self.object_unavailability_mask[object,0] = True
        self.object_unavailability_mask[object,1] = False
        self.object_unavailability_mask[object,2] = False
        self.object_unavailability_mask[object,3] = False
        self.object_unavailability_mask[object,4] = False

        self.observation_mask[object,:] = False

        self.__update_total_mask()


    def add_observation(self, object, ind):
        ''' Adds an observation in the observation schedule if the object already exists in the schedule 
        
        Args:
            object (int): index of object to add an observation for
            ind (int): index of where to put the observation in the schedule
        '''
        # Define index of the end of the observation
        ind_end = ind+self.config.t_obs+self.config.t_setup

        # Add the observation to the schedule and update other states
        self.schedule[ind:ind_end] = object+1
        self.obs_starts[ind] = object+1
        self.observations[object].append(ind)
        self.obs_count[object] += 1
        self.object_state[object, 6] += 1
        self.object_state_norm[object, 6] += 0.1

        # Update the masks
        if ind-self.config.t_int-self.config.t_obs*2-self.config.t_setup >= 0:
            self.action_availability_mask[ind-self.config.t_int-self.config.t_obs*2-self.config.t_setup:ind-self.config.t_int+self.config.t_setup,0] = False
        elif ind-self.config.t_int+self.config.t_setup >= 0:
            self.action_availability_mask[:ind-self.config.t_int+self.config.t_setup,0] = False
        if ind-self.config.t_obs-self.config.t_setup >= 0:
            self.action_availability_mask[ind-self.config.t_obs-self.config.t_setup:ind_end,0] = False
            self.action_availability_mask[ind-self.config.t_obs-self.config.t_setup:ind_end,2] = False
        else:
            self.action_availability_mask[:ind_end,0] = False
            self.action_availability_mask[:ind_end,2] = False
        self.action_availability_mask[ind,3] = True
        #self.action_availability_mask[ind,4] = True

        if self.obs_count[object] >= 3:
            self.object_unavailability_mask[object,1] = False
            self.object_unavailability_mask[object,3] = True

        self.observation_mask[object,ind] = True

        self.__update_total_mask()
        

    def remove_observation(self, object, ind_start):
        ''' Removes an observation from the schedule 
        
        Args:
            object (int): index of the object to remove an observation for
            ind_start (int): index at which the to be removed observation starts
        '''
        # Define the index of the end of the to be removed observation
        ind_end = ind_start+self.config.t_obs+self.config.t_setup

        # Remove the observation from the schedule and update the other states
        self.schedule[ind_start:ind_end] = 0
        self.obs_starts[ind_start] = None
        for ind, obs in enumerate(self.observations[object]):
            if obs == object:
                self.observations[object].pop(ind)
        self.obs_count[object] -= 1
        self.object_state[object, 6] -= 1
        self.object_state_norm[object, 6] -= 0.1

        # Update the masks
        for ind in range(ind_start-self.config.t_int-self.config.t_obs-self.config.t_setup+1,ind_end):
            if ind < len(self.schedule) and ind >= 0:
                if np.all(self.schedule[ind:ind+self.config.t_obs+self.config.t_setup] == 0):
                    self.action_availability_mask[ind,2] = True
                    if np.all(self.schedule[ind+self.config.t_int+self.config.t_obs:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup] == 0):
                        self.action_availability_mask[ind,0] = True
                if self.action_availability_mask[ind,3] and self.schedule[ind] == 0:
                    self.action_availability_mask[ind,3] = False
                # if self.action_availability_mask[ind,4] and self.state[ind] == 1:
                #     self.action_availability_mask[ind,4] = False
        self.action_availability_mask[-self.config.t_int-self.config.t_obs*2-self.config.t_setup:,0] = False
        self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False

        if self.obs_count[object] < 3:
            self.object_unavailability_mask[object,3] = False
            self.object_unavailability_mask[object,1] = True

        self.observation_mask[object, ind_start] = False

        self.__update_total_mask()

    def replace_observation(self, object, ind_target):
        ''' Moves an observation to a different index in the schedule 
        
        Args:
            object (int): index of the object to have an observation replaced
            ind_target (int): index at which an observation should be moved to
        '''
        # Choose the observation to move to the new index
        chosen = False
        if self.schedule[ind_target] == object+1:
            for i in range(ind_target+self.config.t_obs+self.config.t_setup-1,ind_target-1,-1):
                if self.schedule[i] == object+1:
                    chosen = True
                    ind_source = i-self.config.t_obs-self.config.t_setup+1
                    break
        if not chosen:
            for i in range(ind_target,ind_target+self.config.t_obs+self.config.t_setup):
                if self.schedule[i] == object+1:
                    chosen=True
                    ind_source = i
                    break
        i = 0
        n = 1000
        while not chosen:
            try:
                if ind_target - 1 - i >= 0:
                    if self.schedule[ind_target - i - 1] == object+1:
                        chosen = True
                        ind_source = ind_target - i - self.config.t_obs - self.config.t_setup
            except IndexError:
                pass
            try:
                if self.schedule[ind_target + self.config.t_obs + self.config.t_setup + i] == object+1 and not chosen:
                    chosen = True
                    ind_source = ind_target + self.config.t_obs + self.config.t_setup + i
            except IndexError:
                pass
            i += 1
            n -= 1

        self.remove_observation(object, ind_source)
        self.add_observation(object, ind_target)




class OnTheFlyEnv():
    """Class for the On The Fly version of the environment
    """
    def __init__(self, observer, time, eph_class, reward_class, config):
        """Initializes the class

        Args:
            observer (astroplan.Observer): observatory for which to plan observations
            time (asropy.time.Time): Time for which to create the schedule
            eph_class (ephemerides class): Any initiated object from the ephemerides.py module
            reward_class (rewards class): Initiated Reward object from the rewards.py module
            config (param_config.Configuration): Object containing parameter values
        """
        self.config = config
        self.observer = observer
        self.time = time
        start_time, minutes = self.__calculate_start_length()
        self.length = minutes
        self.start_time = start_time

        self.state_space_a = (self.config.n_objects * 9)
        self.state_size_a = self.config.n_objects * 9
        self.state_type_a = 'float32'

        self.state_space_b = (2)
        self.state_size_b = 2
        self.state_type_b = 'float'

        self.reward = reward_class

        self.action_space = (self.config.n_objects + 1)
        self.action_size = self.config.n_objects + 1
    
        self.dawn, self.dusk = twilight_rise_set(self.observer, self.time)
        self.empty_flag = -1

        self.eph_names = []

        self.reset(eph_class)

        np.random.seed(self.config.seed)
        

    def reset(self, eph_class):
        """Resets the environment to empty

        Args:
            eph_class (ephemerides object): Initiated ephemerides object from the ephemerides.py module
        
        Returns:
            list: List containing the normalized object state and an array containing the current progression of filling the schedule
            and the total length of the schedule, normalized to go from -1 to 1
        """
        self.schedule = []
        self.object_to_ind = {}
        self.daily_ephemerides = eph_class.get_daily_ephemerides()
        self.object_state, self.object_state_norm = self.__create_object_state()
        self.insert_time = 0
        self.observations = [[] for i in range(self.config.n_objects)]
        return([self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])])

    def __calculate_start_length(self):
        """Determines the time of noon and the length of the schedule in minutes
        
        Returns:
            noon (astropy.time.Time): Time of noon
            minutes (int): length of the schedule
        """
        noon = self.observer.noon(self.time, which='nearest')
        minutes = self.config.state_length
        return(noon, minutes)

    def __create_object_state(self):
        """Creates an array containing different variables for each of the NEOs

        Returns:
            object_state (np.ndarray): Array containing certain variables for each of the NEOs
            object_state_norm (np.ndarray): Normalized object state
        """
        object_state = np.empty((self.config.n_objects, 9))

        def normalize(object_state):
            """Normalizes the object state

            Args:
                object_state (np.ndarray): Array containing certain variables for each of the NEOs

            Returns:
                object_state (np.ndarray): Array containing certain variables for each of the NEOs
                object_state_norm (np.ndarray): Normalized object state
            """
            object_state_norm = object_state.copy()
            object_state_norm[:,:3] = object_state_norm[:,:3] / (0.5 * self.config.state_length) - 1 
            object_state_norm[:,3] = (object_state_norm[:,3] - 5) / 5
            object_state_norm[:,4] = (object_state_norm[:,4] - 17.5) / 5
            object_state_norm[:,5] = (object_state_norm[:,5] - 50) / 25
            return(object_state, object_state_norm)

        state_ind = 0
        # Extract the features for the NEOs from the ephemerides object
        for ind, object in enumerate(self.daily_ephemerides.keys()):
            #peak_ind, peak_airmass, rise_ind, set_ind = get_airmass_peak(self.daily_ephemerides[object])
            peak_ind = self.daily_ephemerides[object]['peak_ind']
            peak_airmass = self.daily_ephemerides[object]['peak_airmass']
            rise_ind = self.daily_ephemerides[object]['rise_ind']
            set_ind = self.daily_ephemerides[object]['set_ind']
            if peak_airmass == -1:
                continue
            else:
                mag = np.mean(self.daily_ephemerides[object]['apparent magnitude'])
                motion = np.mean(self.daily_ephemerides[object]['total motion'])
                observations = 0
                previous_observation = -2
                flag = 1
                object_state[state_ind,:] = [peak_ind, rise_ind, set_ind, peak_airmass, mag, motion, observations, previous_observation, flag]
                self.object_to_ind[object] = ind
                state_ind += 1
                self.eph_names.append(object)
            if state_ind >= self.config.n_objects:
                break
        
        # Fill rest of the object state with empty objects
        for i in range(state_ind, self.config.n_objects):
            object_state[i, :] = [0,0,0,0,0,0,0,0,0]
        
        object_state, object_state_norm = normalize(object_state)

        return(object_state, object_state_norm)

    # def __time_to_ind(self, given_time):
    #     ind = int((given_time - self.dusk).value*24*60)
    #     return(ind)

    def create_mask(self):
        """Creates a mask to be applied to the output of the agent networks based on which objects
        are visibile.

        Returns:
            total_mask (np.ndarray): Mask to be used by the agent to determine validity of actions.
        """
        no_object_mask = self.object_state[:,8] == 1
        time_mask = (self.insert_time >= self.object_state[:,1]) & (self.insert_time+12 <= self.object_state[:,2])
        total_mask = no_object_mask & time_mask
        total_mask = np.concat([np.array([True]), total_mask]).reshape(1, self.config.n_objects + 1)
        return(total_mask)

    def step(self, obj):
        """Performs a given action on the environment and returns the state, reward and done values

        Args:
            obj (int): The object for which to add an observation to the end of the schedule

        Returns:
            state (list): list of the normalized object state and an array containing the current progression
            of the schedule and the total length of the schedule (normalized)
            reward (float): Reward returned by the reward object if done, -1 otherwise
            done (int): 1 if the end of the schedule has been reached, 0 otherwise
        """
        if obj < 0 or obj >= self.action_size:
            return([self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])], -2, 1)

        obj -= 1
        if obj == -1:
            self.schedule.extend([-1])
            self.insert_time += 1
        else:
            self.schedule.extend([int(obj) for _ in range(self.config.t_obs+self.config.t_setup)])
            self.object_state[obj, 6] += 1
            self.object_state_norm[obj, 6] += 0.1
            self.object_state[obj, 7] = self.insert_time
            self.object_state_norm[obj, 7] = self.insert_time / (0.5 * self.config.state_length) - 1
            self.observations[obj].append(self.insert_time)
            self.insert_time += self.config.t_obs + self.config.t_setup

        if self.insert_time + self.config.t_setup + self.config.t_obs > self.length:
            done = 1
            self.schedule = self.schedule[:self.length]
            reward = self.reward.get_reward(self, -1)
        else:
            done = 0
            reward = -1
        state = [self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])]

        return(state, reward, done)
   