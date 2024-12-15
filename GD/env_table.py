import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import copy

from helper import twilight_rise_set, create_observer, index_last
from param_config import Configuration

class ObservationScheduleEnv():
    
    def __init__(self, observer, time, obj_dict, eph_dict, config = Configuration()):
        """ Initializes the observation schedule environment

        Args:
            observer (astroplan.Observer): observatory for which to schedule observations
            time (astropy.time.Time): date at which to create the schedule
            obj_dict (dict): dictionary with targets
            eph_dict (dict): dictionary containing ephemerides for each of the targets for the night
            config (Configuration, optional): configuration class, see param_config.py. Defaults to param_config.Configuration().
        """        
        self.observer = observer
        self.time = time
        start_time, minutes = self.__calculate_start_length()
        self.length = int(minutes.value)
        self.start_time = start_time
        self.objects = obj_dict
        self.ephemerides = eph_dict
        self.reset()
        self.config = config


    def reset(self):
        """ Resets the environment schedule to empty """        
        self.state = [None]*self.length
        self.rewards = [0]*self.length
        self.obs_starts = [None]*self.length
        self.obs_objects = {}
        self.obs_count = {}
        for object in self.ephemerides.keys():
            self.obs_objects[object] = False  # keeps track which objects are being observed
            self.obs_count[object] = 0  # keeps track how often objects are observed


    def __calculate_start_length(self):
        """ Calculates the time at which the astronomical night starts and how long it lasts

        Returns:
            twilight_evening (astropy.time.Time): time at which astronomical night begins
            minutes (int): length of the astronomical night in minutes
        """        
        twilight_morning, twilight_evening = twilight_rise_set(self.observer, self.time)
        minutes = (twilight_morning-twilight_evening)*24*60
        return(twilight_evening, minutes)


    def __time_to_index(self, given_time):
        ''' Converts inserted time to index in schedule 
        
        Args:
            given_time (astropy.time.Time): time to convert to an index

        Returns:
            index (int): index in the schedule corresponding to the given time
        '''
        index = int((given_time-self.start_time).value*24*60)
        return(index)


    def step(self, action):
        """ Performs a given action within the schedule

        Args:
            action (int): Indicates which action to take

        Returns:
            self.state (list): state of the schedule after taking the action
            self.calculate_reward() (float): fill factor of the schedule after taking action
            done (bool): True if some condition is met
            success (bool): True if the intended action was successful
        """        
        act, ind, t = action
        obj = self.__index_to_object(ind)
        t_ind = self.__time_to_index(t)

        # add object
        if act == 0:
            success = self.add_object(obj, t_ind)
        
        # remove object
        elif act == 1:
            success = self.remove_object(obj)

        # add observation
        elif act == 2:
            success = self.add_observation(obj, t_ind)
        
        # remove observation
        elif act == 3:
            success = self.remove_observation(obj, t_ind)
        
        # replace observation
        elif act == 4:
            success = self.replace_observation(obj, t_ind)
            
        done = False  # TODO when are we done?
        return(self.state, self.calculate_reward(), done, success)


    def calculate_reward(self):
        ''' Calculates reward of current state 
        
        Returns:
            reward (float): fill factor of the schedule
        '''
        #TODO maybe change if necessary
        return(np.sum(self.rewards)/len(self.rewards))


    def __index_to_object(self, index):
        ''' Translates input object index of object to actual object key 
        
        Args:
            index (int): index of the object

        Returns:
            object (str): name of the object
        '''
        return(list(self.obj_dict.keys())[index])


    def __is_interval_free(self, start, end):
        """ Checks whether a certain time period in the schedule is empty

        Args:
            start (int): index of start of period
            end (int): index of end of period

        Returns:
            free (bool): True if time interval is empty in schedule
        """        
        return(all([self.state[i] == None for i in range(start,end)]))


    def add_object(self, object, time):
        ''' Insert object in schedule at time 
        
        Args:
            object (str): key of object to add
            time (astropy.time.Time): time at which to add the object in the schedule

        Returns:
            success (bool): True if add attempt was successful
        '''
        ind_start = self.__time_to_index(time)
        ind_end_obs_1 = ind_start+self.config.t_setup+self.config.t_obs
        ind_start_obs_2 = ind_start+self.config.t_obs+self.config.t_int
        ind_end = ind_start + self.config.t_setup + self.config.t_obs *2 + self.config.t_int
        obs_length = self.config.t_setup + self.config.t_obs
        #TODO make the 45 minute interval variable by +/- 5 minutes
        #TODO reconsider rewards (make weights a factor as well)
        
        # make sure that object not already in scheudle, time period within night and if interval is empty
        if not self.obs_objects[object]:
            if ind_end < len(self.state) and ind_start >= 0:
                if self.__is_interval_free(ind_start, ind_end_obs_1):
                    if self.__is_interval_free(ind_start_obs_2, ind_end):
                        self.state[ind_start:ind_end_obs_1] = [object]*obs_length
                        self.state[ind_start_obs_2:ind_end] = [object]*obs_length
                        self.rewards[ind_start:ind_end_obs_1] = [1]*obs_length
                        self.rewards[ind_start_obs_2:ind_end] = [1]*obs_length
                        self.obs_starts[ind_start] = object
                        self.obs_starts[ind_start_obs_2] = object
                        self.obs_objects[object] = True
                        self.obs_count[object] = 2
                        return(True)
        return(False)


    def remove_object(self, object):
        ''' Remove object in schedule 
        
        Args: 
            object (str): object to remove completely from the schedule
        
        Returns:
            success (bool): True if object was removed
        '''
        self.rewards = [0 if self.state[i] == object else self.rewards[i] for i in range(len(self.state))]
        self.state = [None if self.state[i] == object else self.state[i] for i in range(len(self.state))]
        self.obs_starts = [None if self.obs_starts[i] == object else self.obs_starts[i] for i in range(len(self.obs_starts))]
        self.obs_objects[object] = False
        self.obs_count[object] = 0
        return(True)


    def add_observation(self, object):
        ''' Adds an observation in the observation schedule if the object already exists in the schedule 
        
        Args:
            object (str): key of object to add an observation for

        Returns:
            success (bool): True if observation was added
        '''
        ind_first = self.state.index(object)
        ind_last = index_last(self.state, object)
        ind_before_start, ind_before_end = ind_first-self.config.t_int_min*2-self.config.t_obs-self.config.t_setup, ind_first-self.config.t_int_min-self.config.t_obs-self.config.t_setup
        ind_after_start, ind_after_end = ind_last+self.config.t_int_min, ind_last+self.config.t_int_min*2
        obs_length = self.config.t_setup+self.config.t_obs

        #  check if object already being observed
        if self.obs_objects[object]:
            start_ind, end_ind = self.__time_to_index(self.ephemerides[object]['start_time']), self.__time_to_index(self.ephemerides[object]['end_time'])
            new_times_before = np.arange(ind_before_start, ind_before_end)
            new_times_after = np.arange(ind_after_start, ind_after_end)
            new_times = np.concatenate((new_times_before, new_times_after))
            np.random.shuffle(new_times[(new_times > start_ind) & (new_times < end_ind)])

            #  randomly attempt to add new observations before and after the already present observations
            for time_new in new_times:
                time_new_end = time_new+self.config.t_obs+self.config.t_setup
                if time_new >= 0 and time_new_end < len(self.state):
                    if all([self.state[i] == None for i in range(time_new, time_new_end)]):
                        self.state[time_new:time_new_end] = [object] * obs_length
                        self.rewards[time_new:time_new_end] = [1] * obs_length
                        self.obs_starts[time_new] = object
                        self.obs_count[object] += 1
                        return(True)
        return(False)       
       

    def remove_observation(self, object, ind_start):
        ''' Removes an observation from the schedule 
        
        Args:
            object (str): key of the object to remove an observation for
            ind_start (int): index at which the to be removed observation starts

        Returns:
            success (bool): True if removal was successful
        '''
        ind_end = ind_start+self.config.t_obs+self.config.t_setup
        obs_length = self.config.t_obs+self.config.t_setup

        # check if object is present at start index and being observed more than twice
        if self.state[ind_start] == object:
            if self.obs_count[object] > 2:
                backup_schedule = copy.deepcopy(self.state)
                backup_rewards = copy.deepcopy(self.rewards)
                backup_obs_starts = copy.deepcopy(self.obs_starts)
                self.state[ind_start:ind_end] = [None]*obs_length
                self.rewards[ind_start:ind_end] = [0]*obs_length
                self.obs_starts[ind_start] = None
                self.obs_count[object] -= 1
                if index_last(self.state, object) - self.state.index(object) > self.config.t_setup * 2 + self.config.t_obs * 2 + self.config.t_int + 10:
                    self.state = backup_schedule
                    self.rewards = backup_rewards
                    self.obs_starts = backup_obs_starts
                    self.obs_count[object] += 1
                else:
                    return(True)
        return(False)
                    

    def replace_observation(self, object, ind_init):
        ''' Tries to move an observation to a different place in the schedule 
        
        Args:
            object (str): key of the object to have an observation replaced
            ind_init (int): index at which the observation initially takes place

        Returns:
            success (bool): True if replacement was successful
        '''
        ind_init_end = ind_init + self.config.t_obs + self.config.t_setup
        obs_length = self.config.t_setup+self.config.t_obs

        # check if the observation takes place at initial index
        if self.obs_starts[ind_init] == object:
            init_state = copy.deepcopy(self.state)
            init_rewards = copy.deepcopy(self.rewards)
            init_obs_starts = copy.deepcopy(self.obs_starts)
            replaced = False
            self.state[ind_init:ind_init_end] = [None] * obs_length
            self.rewards[ind_init:ind_init_end] = [0] * obs_length
            self.obs_starts[ind_init] = None

            start_ind, end_ind = self.__time_to_index(self.ephemerides[object]['start_time']), self.__time_to_index(self.ephemerides[object]['end_time'])
            new_times = np.arange(ind_init-self.config.t_replace, ind_init+self.config.t_replace+1)
            np.random.shuffle(new_times[(new_times > start_ind) & (new_times < end_ind)])

            # randomly attempt to replace the observation to a new time
            for time_new in new_times:
                time_new_end = time_new+self.config.t_obs+self.config.t_setup
                if time_new >= 0 and time_new_end < len(self.state):
                    if all([self.state[i] == None for i in range(time_new, time_new_end)]):
                        self.state[time_new:time_new_end] = [object] * obs_length
                        self.rewards[time_new:time_new_end] = [1] * obs_length
                        self.obs_starts[time_new] = object
                        replaced = True
                        break

            if not replaced:
                self.state = init_state
                self.obs_starts = init_obs_starts
                self.rewards = init_rewards
            else:
                return(True)
        return(False)


    def __str__(self):
        """Generates a table of the observation schedule to print

        Returns:
            string (str): string representation of the schedule
        """
        string = 'Time   | Object   | RA        | Dec      | Mag (V) | Motion  \n--------------------------------------------------------\n'
        for ind in range(len(self.obs_starts)):
            if self.obs_starts[ind] != None:
                time = self.__ind_to_time(ind)
                obj = self.obs_starts[ind]
                eph_ind = self.__get_ephemeris(obj, time)
                ra = np.round(self.ephemerides[obj]['coord'][eph_ind].ra, 2)
                dec = np.round(self.ephemerides[obj]['coord'][eph_ind].dec, 2)
                mag = self.ephemerides[obj]['mag_V'][eph_ind]
                motion = np.round(self.ephemerides[obj]['motion'][eph_ind], 1)
                string += f'{time}  | {obj}  | {ra} | {dec} | {mag}    | {motion}  \n'
        return(string)


    def __ind_to_time(self, ind):
        """ Converts index in schedule to time string

        Args:
            ind (int): index in the schedule

        Returns:
            time (str): time of index
        """
        hours = int(np.floor(ind/60))
        minutes = ind % 60
        start_hour, start_minutes = int(self.start_time.fits[11:13]), int(self.start_time.fits[14:16])
        hours += start_hour
        minutes += start_minutes
        if minutes >= 60:
            minutes -= 60
            hours += 1
        if hours >= 24:
            hours -= 24
        hours = str(hours)
        minutes = str(minutes)
        if len(hours) == 1:
            hours = '0' + hours
        if len(minutes) == 1:
            minutes = '0' + minutes
        time = f'{hours}:{minutes}'
        return(time)
    

    def __get_ephemeris(self, obj, time):
        """ Returns the index of the ephemeris of an object given a time

        Args:
            obj (str): key of object
            time (str): string representation of time

        Returns:
            ephemeris_ind (int): index of ephemeris of object corresponding to given time
        """
        for ephemeris_ind in range(len(self.ephemerides[obj]['time'])):
            if int(time[:2]) == int(self.ephemerides[obj]['time'][ephemeris_ind][0].fits[11:13]) or 1 + int(time[:2]) == int(self.ephemerides[obj]['time'][ephemeris_ind][0].fits[11:13]):
                return(ephemeris_ind)



    


