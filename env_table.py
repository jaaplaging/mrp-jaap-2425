# Here we will have the environment

from helper import twilight_rise_set, create_observer, index_last
from astropy.time import Time
import astropy.units as u
import numpy as np
from param_config import Configuration
import copy


class ObservationScheduleEnv():
    
    def __init__(self, observer, time, obj_dict, eph_dict, config = Configuration()):
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
        self.state = [None]*self.length
        self.rewards = [0]*self.length
        self.obs_starts = [None]*self.length
        self.obs_objects = {}
        self.obs_count = {}
        for object in self.ephemerides.keys():
            self.obs_objects[object] = False
            self.obs_count[object] = 0

    def __calculate_start_length(self):
        twilight_morning, twilight_evening = twilight_rise_set(self.observer, self.time)
        minutes = (twilight_morning-twilight_evening)*24*60
        return(twilight_evening, minutes)

    def __time_to_index(self, given_time):
        ''' Converts inserted time to index in schedule '''
        index = int((given_time-self.start_time).value*24*60)
        return(index)

    def step(self, action):
        ''' Performs action and calculates reward '''
        # action consists of 5 options for the action, n number of objects and m number of times the action can be applied
        # action should be a tuple, list or array with len = 3, corresponding to (action, object, time)
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
        ''' Calculates reward of current state '''
        #TODO maybe change if necessary
        return(np.sum(self.rewards)/len(self.rewards))

    def __index_to_object(self, index):
        ''' Translates input object index of object to actual object key '''
        return(list(self.obj_dict.keys())[index])

    def __is_interval_free(self, start, end):
        return(all([self.state[i] == None for i in range(start,end)]))

    def add_object(self, object, time):
        ''' Insert object in schedule at time '''
        ind_start = self.__time_to_index(time)
        ind_end_obs_1 = ind_start+self.config.t_setup+self.config.t_obs
        ind_start_obs_2 = ind_start+self.config.t_obs+self.config.t_int
        ind_end = ind_start + self.config.t_setup + self.config.t_obs *2 + self.config.t_int
        obs_length = self.config.t_setup + self.config.t_obs
        #TODO make the 45 minute interval variable by +/- 5 minutes
        #TODO reconsider rewards (make weights a factor as well)
        
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
        return(True)

    def remove_object(self, object):
        ''' Remove object in schedule '''
        self.rewards = [0 if self.state[i] == object else self.rewards[i] for i in range(len(self.state))]
        self.state = [None if self.state[i] == object else self.state[i] for i in range(len(self.state))]
        self.obs_starts = [None if self.obs_starts[i] == object else self.obs_starts[i] for i in range(len(self.obs_starts))]
        self.obs_objects[object] = False
        self.obs_count[object] = 0
        return(True)


    def add_observation(self, object):
        ''' Adds an observation in the observation schedule if the object already exists in the schedule '''
        ind_first = self.state.index(object)
        ind_last = index_last(self.state, object)
        ind_before_start, ind_before_end = ind_first-self.config.t_int_min*2-self.config.t_obs-self.config.t_setup, ind_first-self.config.t_int_min-self.config.t_obs-self.config.t_setup
        ind_after_start, ind_after_end = ind_last+self.config.t_int_min, ind_last+self.config.t_int_min*2
        obs_length = self.config.t_setup+self.config.t_obs

        if self.obs_objects[object]:
            start_ind, end_ind = self.__time_to_index(self.ephemerides[object]['start_time']), self.__time_to_index(self.ephemerides[object]['end_time'])
            new_times_before = np.arange(ind_before_start, ind_before_end)
            new_times_after = np.arange(ind_after_start, ind_after_end)
            new_times = np.concatenate((new_times_before, new_times_after))
            np.random.shuffle(new_times[(new_times > start_ind) & (new_times < end_ind)])
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
        ''' Removes an observation from the schedule '''
        ind_end = ind_start+self.config.t_obs+self.config.t_setup
        obs_length = self.config.t_obs+self.config.t_setup

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
        ''' Tries to move an observation to a different place in the schedule '''
        ind_init_end = ind_init + self.config.t_obs + self.config.t_setup
        obs_length = self.config.t_setup+self.config.t_obs

        if self.state[ind_init] == object:
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
        pass


if __name__ == '__main__':
    # observer = create_observer()
    # time = Time.now()
    # env = ObservationScheduleEnv(observer, time)
    li = [None, None, None]
    li[:2] = [1]*2
    print(li)
    


