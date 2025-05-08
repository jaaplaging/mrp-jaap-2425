import sys
import numpy as np
import copy

from helper import twilight_rise_set, create_observer, index_last, opt_time, airmass, rise_set
from astroplan import FixedTarget
from time import perf_counter


class ScheduleEnv():
    
    def __init__(self, observer, time, eph_class, reward_class, config):
        """ Initializes the observation schedule environment

        Args:
            observer (astroplan.Observer): observatory for which to schedule observations
            time (astropy.time.Time): date at which to create the schedule
            obj_dict (dict): dictionary with targets
            eph_dict (dict): dictionary containing ephemerides for each of the targets for the night
            config (Configuration, optional): configuration class, see param_config.py. Defaults to param_config.Configuration().
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

        self.times = [0 for i in range(20)]
        self.reset(eph_class)

        np.random.seed(self.config.seed)

        #self.twilight_evening_ind, self.twilight_morning_ind = self.__calculate_twilight()
        #self.state[:self.twilight_evening_ind] = 0
        #self.state[self.twilight_morning_ind:] = 0


    def reset(self, eph_class):
        """ Resets the environment schedule to empty """        
        self.schedule = np.full((self.length), 0, dtype='int32')
        self.obs_starts = [None]*self.length
        self.observations = [[] for i in range(self.config.n_objects)]
        self.obs_objects = {}
        self.obs_count = {}
        self.steps_taken = 0
        self.daily_ephemerides = eph_class.get_daily_ephemerides()
        self.object_to_ind = {}
        self.object_state, self.object_state_norm = self.__create_object_state()
        for object in self.daily_ephemerides.keys():
            self.obs_objects[object] = False  # keeps track which objects are being observed
            self.obs_count[object] = 0  # keeps track how often objects are observed
        self.__create_action_masks()
        return([self.object_state_norm, self.schedule])

    def __create_action_masks(self):
        """Creates masks for action possibilities to be applied to the output of the neural network
        """ 

        self.mask_day = np.full((self.config.n_objects, self.config.state_length, 5), True)

        self.mask_object_visibility = np.full((self.config.n_objects,self.config.state_length,5), False)
        for ind, obj in enumerate(self.object_state):
            self.mask_object_visibility[ind, np.max([0,int(obj[1])]):np.min([self.config.state_length,int(obj[2])]), :] = True

        self.base_mask = self.mask_day & self.mask_object_visibility

        self.action_availability_mask = np.full((self.config.state_length, 5), True)
        self.action_availability_mask[-self.config.t_obs*2-self.config.t_setup*2-self.config.t_int:,0] = False

        #test:
        self.action_availability_mask[self.mask_day.shape[0]:,1] = False

        self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False
        self.action_availability_mask[:,3] = False
        self.action_availability_mask[:,4] = False
        #self.action_availability_mask[self.twilight_morning_ind-self.config.t_obs*2-self.config.t_setup*2-self.config.t_int:,0] = False

        self.object_unavailability_mask = np.full((self.config.n_objects, 5), False)
        self.object_unavailability_mask[:,0] = True

        self.observation_mask = np.full((self.config.n_objects,self.config.state_length), False)

        #test to try to fix repetition of actions
        self.taken_actions_mask = np.full((self.config.n_objects,self.config.state_length,5), True)
        self.taken_actions_countdown = np.full((self.config.n_objects,self.config.state_length,5), 0)

        self.total_mask = self.base_mask & np.einsum('ij,kj -> kij', self.action_availability_mask, self.object_unavailability_mask) & self.taken_actions_mask

    def __update_total_mask(self):
        """Updates the total mask after an action taken
        """
        self.total_mask = self.base_mask & np.einsum('ij,kj -> kij', self.action_availability_mask, self.object_unavailability_mask)

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
                self.total_mask[object,:,4] = replace_available
        
        self.taken_actions_countdown = np.where(self.taken_actions_countdown >= 1, self.taken_actions_countdown - 1, self.taken_actions_countdown)
        self.taken_actions_mask = np.where(self.taken_actions_countdown >= 1, False, True)
        self.total_mask = self.total_mask & self.taken_actions_mask

    def create_mask(self):
        return(self.total_mask.flatten())

    def __create_object_state(self):
        object_state = np.empty((self.config.n_objects, 9))
    
        def normalize(object_state):
            object_state_norm = object_state.copy()
            object_state_norm[:,:3] = object_state_norm[:,:3] / (0.5 * self.config.state_length) - 1 
            object_state_norm[:,3] = (object_state_norm[:,4] - 5) / 5
            object_state_norm[:,4] = (object_state_norm[:,5] - 17.5) / 5
            object_state_norm[:,5] = (object_state_norm[:,6] - 50) / 25
            return(object_state, object_state_norm)

        state_ind = 0
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
                observations = 0
                previous_observation = -2
                flag = 1
                object_state[state_ind,:] = [peak_ind, rise_ind, set_ind, peak_airmass, mag, motion, observations, previous_observation, flag]
                self.object_to_ind[object] = ind
                state_ind += 1
            if state_ind >= self.config.n_objects:
                break
        
        for i in range(state_ind, self.config.n_objects):
            object_state[i, :] = [0,0,0,0,0,0,0,0,0]
        
        object_state, object_state_norm = normalize(object_state)

        return(object_state, object_state_norm)


    def __calculate_start_length(self):
        """ Calculates the time of noon and how long a day lasts in minutes

        Returns:
            noon (astropy.time.Time): time of astronomical noon
            minutes (int): length of a day in minutes
        """        
        noon = self.observer.noon(self.time, which='nearest')
        minutes = self.config.state_length
        return(noon, minutes)

    def __calculate_twilight(self):
        """Gives the indices in the state for when the astronomical night starts and ends

        Returns:
            ind_set (int): index of astronomical twilight in the evening
            ind_rise (int): index of astrononmical twilight in the morning
        """
        twilight_rise, twilight_set = twilight_rise_set(self.observer, self.time)
        ind_set = int((twilight_set - self.start_time).value*24*60)
        ind_rise = int((twilight_rise - self.start_time).value*24*60)
        return(ind_set, ind_rise)

    def __time_to_index(self, given_time):
        ''' Converts inserted time to index in schedule 
        
        Args:
            given_time (astropy.time.Time): time to convert to an index

        Returns:
            index (int): index in the schedule corresponding to the given time
        '''
        index = int((given_time-self.start_time).value*2*60)
        return(index)


    def step(self, action):
        """ Performs a given action within the schedule

        Args:
            obj (int): object on which the action is performed
            ind (int): time at which to take the action
            action (int): action to take
            step (int): RL step

        Returns:
            list: schedule and object states
            self.calculate_reward() (float): fill factor of the schedule after taking action
            done (bool): True if some condition is met
        """        
        if action < 0 or action >= self.action_size:
            return([self.object_state_norm, self.schedule], -2, 0)
        
        self.steps_taken += 1
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


    def __index_to_object(self, index):
        ''' Translates input object index of object to actual object key 
        
        Args:
            index (int): index of the object

        Returns:
            object (str): name of the object
        '''
        return(list(self.daily_ephemerides.keys())[index])


    def __is_interval_free(self, start, end):
        """ Checks whether a certain time period in the schedule is empty

        Args:
            start (int): index of start of period
            end (int): index of end of period

        Returns:
            free (bool): True if time interval is empty in schedule
        """        
        return(all([self.schedule[i] == 0 for i in range(start,end)]))


    def add_object(self, object, ind_start):
        ''' Insert object in schedule at time 
        
        Args:
            object (int): index of object to add
            ind_start (int): time at which to add the object in the schedule
        '''
        ind_end_obs_1 = ind_start+self.config.t_setup+self.config.t_obs
        ind_start_obs_2 = ind_start+self.config.t_obs+self.config.t_int
        ind_end = ind_start + self.config.t_setup + self.config.t_obs *2 + self.config.t_int
        #TODO make the 45 minute interval variable by +/- 5 minutes
        #TODO reconsider rewards (make weights a factor as well)
        
        self.schedule[ind_start:ind_end_obs_1] = object+1
        self.schedule[ind_start_obs_2:ind_end] = object+1
        self.obs_starts[ind_start] = self.__index_to_object(object)
        self.observations[object].append(ind_start)
        self.obs_starts[ind_start_obs_2] = self.__index_to_object(object)
        self.observations[object].append(ind_start_obs_2)
        self.obs_objects[self.__index_to_object(object)] = True
        self.obs_count[self.__index_to_object(object)] = 2
        self.object_state[object, 4] = 2
        self.object_state_norm[object, 4] = 0.2

        if ind_start-self.config.t_obs-self.config.t_setup-self.config.t_int+1 >= 0 and ind_start-self.config.t_int+self.config.t_setup+self.config.t_obs >= 0:
            self.action_availability_mask[ind_start-self.config.t_obs-self.config.t_setup-self.config.t_int+1:ind_start-self.config.t_int+self.config.t_setup+self.config.t_obs,0] = False
        elif ind_start-self.config.t_int+self.config.t_setup+self.config.t_obs >= 0:
            self.action_availability_mask[:ind_start-self.config.t_int+self.config.t_setup+self.config.t_obs,0] = False
        if ind_start-self.config.t_obs-self.config.t_setup+1 >= 0:
            self.action_availability_mask[ind_start-self.config.t_obs-self.config.t_setup+1:ind_end_obs_1,0] = False
            self.action_availability_mask[ind_start-self.config.t_obs-self.config.t_setup+1:ind_end_obs_1,2] = False
        else:
            self.action_availability_mask[:ind_end_obs_1,0] = False
            self.action_availability_mask[:ind_end_obs_1,2] = False
        self.action_availability_mask[ind_start_obs_2-self.config.t_obs-self.config.t_setup+1:ind_end,0] = False
        #self.action_availability_mask[:,1] = True
        self.action_availability_mask[ind_start_obs_2-self.config.t_obs-self.config.t_setup+1:ind_end,2] = False
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
        ''' Remove object in schedule 
        
        Args: 
            object (int): object to remove completely from the schedule
        '''
        obj_str = self.__index_to_object(object)
        self.schedule = np.where(self.schedule == object+1, 0, self.schedule)
        self.obs_starts = [None if self.obs_starts[i] == obj_str else self.obs_starts[i] for i in range(len(self.obs_starts))]
        self.observations[object] = []
        self.obs_objects[obj_str] = False
        self.obs_count[obj_str] = 0
        self.object_state[object, 4] = 0
        self.object_state_norm[object, 4] = 0

        for ind in range(len(self.schedule)):
            if np.all(self.schedule[ind:ind+self.config.t_obs*2+self.config.t_setup*2] == 0):
                self.action_availability_mask[ind,2] = True
                if np.all(self.schedule[ind+self.config.t_int:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup*2] == 0):
                    self.action_availability_mask[ind,0] = True
            if self.action_availability_mask[ind,3] and self.schedule[ind] == 0:
                self.action_availability_mask[ind,3] = False
            #if self.action_availability_mask[ind,4] and self.state[ind] == 1:
            #    self.action_availability_mask[ind,4] = False
        self.action_availability_mask[-self.config.t_int-self.config.t_obs*2-self.config.t_setup*2:,0] = False
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
            object (int): key of object to add an observation for
            ind (int): index of where to put the observation in the schedule
        '''

        ind_end = ind+self.config.t_obs+self.config.t_setup
        obj_str = self.__index_to_object(object)

        self.schedule[ind:ind_end] = object+1
        self.obs_starts[ind] = obj_str
        self.observations[object].append(ind)
        self.obs_count[obj_str] += 1
        self.object_state[object, 4] += 1
        self.object_state_norm[object, 4] += 0.1

        if ind-self.config.t_int-self.config.t_obs-self.config.t_setup+1 >= 0:
            self.action_availability_mask[ind-self.config.t_int-self.config.t_obs-self.config.t_setup+1:ind-self.config.t_int+self.config.t_obs+self.config.t_setup,0] = False
        elif ind-self.config.t_int+self.config.t_obs+self.config.t_setup >= 0:
            self.action_availability_mask[:ind-self.config.t_int+self.config.t_obs+self.config.t_setup,0] = False
        if ind-self.config.t_obs-self.config.t_setup+1 >= 0:
            self.action_availability_mask[ind-self.config.t_obs-self.config.t_setup+1:ind_end,0] = False
            self.action_availability_mask[ind-self.config.t_obs-self.config.t_setup+1:ind_end,2] = False
        else:
            self.action_availability_mask[:ind_end,0] = False
            self.action_availability_mask[:ind_end,2] = False
        self.action_availability_mask[ind,3] = True
        #self.action_availability_mask[ind,4] = True

        if self.obs_count[obj_str] >= 3:
            self.object_unavailability_mask[object,1] = False
            self.object_unavailability_mask[object,3] = True

        self.observation_mask[object,ind] = True

        self.__update_total_mask()
        

    def remove_observation(self, object, ind_start):
        ''' Removes an observation from the schedule 
        
        Args:
            object (str): key of the object to remove an observation for
            ind_start (int): index at which the to be removed observation starts
        '''
        ind_end = ind_start+self.config.t_obs+self.config.t_setup
        obj_str = self.__index_to_object(object)

        self.schedule[ind_start:ind_end] = 0
        self.obs_starts[ind_start] = None
        for ind, obs in enumerate(self.observations[object]):
            if obs == object:
                self.observations[object].pop(ind)
        self.obs_count[self.__index_to_object(object)] -= 1
        self.object_state[object, 4] -= 1
        self.object_state_norm[object, 4] -= 0.1

        for ind in range(ind_start-self.config.t_int-self.config.t_obs-self.config.t_setup+1,ind_end):
            if ind < len(self.schedule) and ind >= 0:
                time_init = perf_counter()
                if np.all(self.schedule[ind:ind+self.config.t_obs*2+self.config.t_setup*2] == 0):
                    self.action_availability_mask[ind,2] = True
                    if np.all(self.schedule[ind+self.config.t_int:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup*2] == 0):
                        self.action_availability_mask[ind,0] = True
                self.times[0] += perf_counter() - time_init
                if self.action_availability_mask[ind,3] and self.schedule[ind] == 0:
                    self.action_availability_mask[ind,3] = False
                # if self.action_availability_mask[ind,4] and self.state[ind] == 1:
                #     self.action_availability_mask[ind,4] = False
        self.action_availability_mask[-self.config.t_int-self.config.t_obs*2-self.config.t_setup*2:,0] = False
        self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False

        if self.obs_count[obj_str] < 3:
            self.object_unavailability_mask[object,3] = False
            self.object_unavailability_mask[object,1] = True

        self.observation_mask[object, ind_start] = False

        self.__update_total_mask()

    def replace_observation(self, object, ind_target):
        ''' Tries to move an observation to a different place in the schedule 
        
        Args:
            object (str): key of the object to have an observation replaced
            ind_target (int): index at which an observation should be moved to
        '''
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
            if n == 0:
                print(self.schedule)
                print(object)
                print(ind_target)
                print(self.schedule)
                print(self.obs_count)
                print(self.total_mask[object,:,4])
                raise('here you go.')

        self.remove_observation(object, ind_source)
        self.add_observation(object, ind_target)




class OnTheFlyEnv():

    def __init__(self, observer, time, eph_class, reward_class, config):
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

        #self.times = [[] for _ in range(10)]
        self.reset(eph_class)

        np.random.seed(self.config.seed)
        

    def reset(self, eph_class):
        self.schedule = []
        self.object_to_ind = {}
        self.daily_ephemerides = eph_class.get_daily_ephemerides()
        self.object_state, self.object_state_norm = self.__create_object_state()
        self.insert_time = 0
        self.observations = [[] for i in range(self.config.n_objects)]
        return([self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])])

    def __calculate_start_length(self):
        noon = self.observer.noon(self.time, which='nearest')
        minutes = self.config.state_length
        return(noon, minutes)

    def __create_object_state(self):
        object_state = np.empty((self.config.n_objects, 9))

        def normalize(object_state):
            object_state_norm = object_state.copy()
            object_state_norm[:,:3] = object_state_norm[:,:3] / (0.5 * self.config.state_length) - 1 
            object_state_norm[:,3] = (object_state_norm[:,4] - 5) / 5
            object_state_norm[:,4] = (object_state_norm[:,5] - 17.5) / 5
            object_state_norm[:,5] = (object_state_norm[:,6] - 50) / 25
            return(object_state, object_state_norm)

        state_ind = 0
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
            if state_ind >= self.config.n_objects:
                break
        
        for i in range(state_ind, self.config.n_objects):
            object_state[i, :] = [0,0,0,0,0,0,0,0,0]
        
        object_state, object_state_norm = normalize(object_state)

        return(object_state, object_state_norm)

    def __time_to_ind(self, given_time):
        ind = int((given_time - self.dusk).value*24*60)
        return(ind)

    def create_mask(self):
        no_object_mask = self.object_state[:,8] == 1
        time_mask = (self.insert_time >= self.object_state[:,1]) & (self.insert_time+12 <= self.object_state[:,2])
        total_mask = no_object_mask & time_mask
        total_mask = np.concat([np.array([True]), total_mask]).reshape(1, self.config.n_objects + 1)
        return(total_mask)

    def step(self, obj):
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
            reward = self.reward.get_reward(self, -1)
        else:
            done = 0
            reward = -1
        state = [self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])]

        return(state, reward, done)
   