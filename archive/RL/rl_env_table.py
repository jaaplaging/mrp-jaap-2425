import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import copy

from helper import twilight_rise_set, create_observer, index_last, opt_time, airmass, rise_set
from param_config import Configuration
from astroplan import FixedTarget


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
        self.config = config      
        self.observer = observer
        self.time = time
        start_time, minutes = self.__calculate_start_length()
        self.length = minutes
        self.start_time = start_time
        self.objects = obj_dict
        self.ephemerides = eph_dict
        self.reset()
        #self.twilight_evening_ind, self.twilight_morning_ind = self.__calculate_twilight()
        #self.state[:self.twilight_evening_ind] = 0
        #self.state[self.twilight_morning_ind:] = 0


    def reset(self):
        """ Resets the environment schedule to empty """        
        self.state = np.full((self.length), 0, dtype='int32')
        self.rewards = np.zeros((self.length))
        self.obs_starts = [None]*self.length
        self.obs_objects = {}
        self.obs_count = {}
        self.object_state, self.object_state_norm = self.__create_object_state()
        self.object_to_ind = {}
        for object in self.objects.keys():
            self.obs_objects[object] = False  # keeps track which objects are being observed
            self.obs_count[object] = 0  # keeps track how often objects are observed
        self.__create_action_masks()

    def __create_action_masks(self):
        """Creates masks for action possibilities to be applied to the output of the neural network
        """  
        # TODO optional a mask for moon visibility
        
        # self.mask_day = np.full((20, 1440, 5), False)
        # self.mask_day[:, self.twilight_evening_ind:self.twilight_morning_ind-self.config.t_obs-self.config.t_setup, :] = True

        # self.mask_object_visibility = np.full((20, 1440, 5), False)
        # for ind, object in enumerate(self.ephemerides.keys()):
        #     coords_mid = self.ephemerides[object]['coord'][len(self.ephemerides[object]['coord'])//2]
        #     rise, set = rise_set(self.observer, FixedTarget(coord=coords_mid), self.time)
        #     rise_ind, set_ind = self.__time_to_index(rise), self.__time_to_index(set)
        #     self.mask_object_visibility[ind, rise_ind:set_ind, :] = True

        self.mask_day = np.full((self.config.n_objects, self.config.state_length, 5), True)

        self.mask_object_visibility = np.full((self.config.n_objects,self.config.state_length,5), True)

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
                for i in range(len(self.state)-self.config.t_obs-self.config.t_setup):
                    if not (self.state[i] == object + 1 and self.state[i+self.config.t_obs+self.config.t_setup-1] == object + 1):
                        if all(self.state[j] in [0, object+1] for j in range(i,i+self.config.t_obs+self.config.t_setup)):
                            if not all(self.state[j] == object+2 for j in range(i,i+self.config.t_obs+self.config.t_setup)):
                                replace_available[i] = True
                self.total_mask[object,:,4] = replace_available
        
        self.taken_actions_countdown = np.where(self.taken_actions_countdown >= 1, self.taken_actions_countdown - 1, self.taken_actions_countdown)
        self.taken_actions_mask = np.where(self.taken_actions_countdown >= 1, False, True)
        self.total_mask = self.total_mask & self.taken_actions_mask



    def __create_object_state(self):
        object_state = np.empty((self.config.n_objects, 6))

        def normalize(object_state):
            object_state_norm = object_state.copy()
            object_state_norm[:,0] = object_state_norm[:,0] / (0.5 * self.config.state_length) - 1 
            object_state_norm[:,1] = (object_state_norm[:,1] - 5) / 5
            object_state_norm[:,2] = (object_state_norm[:,2] - 17.5) / 5
            object_state_norm[:,3] = (object_state_norm[:,3] - 50) / 25
            return(object_state_norm)

        for i in range(self.config.n_objects):
            object_state[i] = [30, 2., 20., 15., 0, 1]
        #object_state[1] = [60, 3., 20.3, 20., 0, 1]
        #object_state[2] = [75, 1.5, 19., 5., 0, 1]
        # def get_airmass_peak(object_ephemerides):
        #     coords_mid = object_ephemerides['coord'][len(object_ephemerides['coord'])//2]
        #     peak_time = opt_time(self.observer, FixedTarget(name='target', coord=coords_mid), self.time)
        #     peak_ind = self.__time_to_index(peak_time)
        #     peak_airmass = airmass(self.observer, FixedTarget(name='target', coord=coords_mid), peak_time)
        #     return(peak_ind, peak_airmass)



        # for ind, object in enumerate(self.ephemerides.keys()):
        #     peak_ind, peak_airmass = get_airmass_peak(self.ephemerides[object])
        #     mag = np.mean(self.ephemerides[object]['mag_V'])
        #     motion = np.mean(self.ephemerides[object]['motion'])
        #     observations = 0
        #     flag = 1  # indicates that there is an object present in this row
        #     object_state[ind, :] = [peak_ind, peak_airmass, mag, motion, observations, flag]
        #     self.object_to_ind[object] = ind
        
        # for i in range(len(self.ephemerides.keys()), 20):
        #     object_state[i, :] = [0, 0, 0, 0, 0, 0]
        
        object_state_norm = normalize(object_state)

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


    def step(self, obj, ind, action, step):
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
            
        reward = self.calculate_reward()

        if step >= self.config.steps-1:
            done = 1
        else:
            done = 0
        return([self.object_state_norm, self.state], reward, done)


    def calculate_reward(self):
        ''' Calculates reward of current state 
        
        Returns:
            reward (float): fill factor of the schedule
        '''
        # Threshold #
        # if np.sum(self.rewards)/len(self.rewards) > 0.5:
        #     reward = (np.sum(self.rewards)/len(self.rewards) - 0.5) * 10
        # else:
        #     reward = -1

        #  Linear  #
        reward = np.sum(self.rewards)/len(self.rewards)*2-1
        
        #  Power  #
        #reward = (np.sum(self.rewards)/len(self.rewards)*2-1)**3
        
        return(reward)


    def __index_to_object(self, index):
        ''' Translates input object index of object to actual object key 
        
        Args:
            index (int): index of the object

        Returns:
            object (str): name of the object
        '''
        return(list(self.objects.keys())[index])


    def __is_interval_free(self, start, end):
        """ Checks whether a certain time period in the schedule is empty

        Args:
            start (int): index of start of period
            end (int): index of end of period

        Returns:
            free (bool): True if time interval is empty in schedule
        """        
        return(all([self.state[i] == 0 for i in range(start,end)]))


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
        
        self.state[ind_start:ind_end_obs_1] = object+1
        self.state[ind_start_obs_2:ind_end] = object+1
        self.rewards[ind_start:ind_end_obs_1] = 1
        self.rewards[ind_start_obs_2:ind_end] = 1
        self.obs_starts[ind_start] = self.__index_to_object(object)
        self.obs_starts[ind_start_obs_2] = self.__index_to_object(object)
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
        self.rewards = np.where(self.state == object+1, 0, self.rewards)
        self.state = np.where(self.state == object+1, 0, self.state)
        self.obs_starts = [None if self.obs_starts[i] == obj_str else self.obs_starts[i] for i in range(len(self.obs_starts))]
        self.obs_objects[obj_str] = False
        self.obs_count[obj_str] = 0
        self.object_state[object, 4] = 0
        self.object_state_norm[object, 4] = 0

        for ind in range(len(self.state)):
            if np.all(self.state[ind:ind+self.config.t_obs*2+self.config.t_setup*2] == 0):
                self.action_availability_mask[ind,2] = True
                if np.all(self.state[ind+self.config.t_int:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup*2] == 0):
                    self.action_availability_mask[ind,0] = True
            if self.action_availability_mask[ind,3] and self.state[ind] == 0:
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

        self.state[ind:ind_end] = object+1
        self.rewards[ind:ind_end] = 1
        self.obs_starts[ind] = obj_str
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

        self.state[ind_start:ind_end] = 0
        self.rewards[ind_start:ind_end] = 0
        self.obs_starts[ind_start] = None
        self.obs_count[self.__index_to_object(object)] -= 1
        self.object_state[object, 4] -= 1
        self.object_state_norm[object, 4] -= 0.1

        for ind in range(ind_start-self.config.t_int-self.config.t_obs-self.config.t_setup+1,ind_end):
            if ind < len(self.state) and ind >= 0:
                if np.all(self.state[ind:ind+self.config.t_obs*2+self.config.t_setup*2] == 0):
                    self.action_availability_mask[ind,2] = True
                    if np.all(self.state[ind+self.config.t_int:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup*2] == 0):
                        self.action_availability_mask[ind,0] = True
                if self.action_availability_mask[ind,3] and self.state[ind] == 0:
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
        if self.state[ind_target] == object+1:
            for i in range(ind_target+self.config.t_obs+self.config.t_setup-1,ind_target-1,-1):
                if self.state[i] == object+1:
                    chosen = True
                    ind_source = i-self.config.t_obs-self.config.t_setup+1
                    break
        if not chosen:
            for i in range(ind_target,ind_target+self.config.t_obs+self.config.t_setup):
                if self.state[i] == object+1:
                    chosen=True
                    ind_source = i
                    break
        i = 0
        n = 1000
        while not chosen:
            try:
                if ind_target - 1 - i >= 0:
                    if self.state[ind_target - i - 1] == object+1:
                        chosen = True
                        ind_source = ind_target - i - self.config.t_obs - self.config.t_setup
            except IndexError:
                pass
            try:
                if self.state[ind_target + self.config.t_obs + self.config.t_setup + i] == object+1 and not chosen:
                    chosen = True
                    ind_source = ind_target + self.config.t_obs + self.config.t_setup + i
            except IndexError:
                pass
            i += 1
            n -= 1
            if n == 0:
                print(self.state)
                print(object)
                print(ind_target)
                print(self.object_state)
                print(self.obs_count)
                print(self.total_mask[object,:,4])
                raise('here you go.')
        
        self.remove_observation(object, ind_source)
        self.add_observation(object, ind_target)

        ####################
        # OLD METHOD BELOW #
        ####################
        # ind_init_end = ind_init + self.config.t_obs + self.config.t_setup
        # if ind_init_end >= len(self.state):
        #     ind_init_end = 119
        # obj_str = self.__index_to_object(object)

        # # check if the observation takes place at initial index
        # if self.obs_starts[ind_init] == obj_str:
        #     init_state = np.copy(self.state)
        #     init_rewards = np.copy(self.rewards)
        #     init_obs_starts = copy.deepcopy(self.obs_starts)
        #     replaced = False
        #     self.state[ind_init:ind_init_end] = 1
        #     self.rewards[ind_init:ind_init_end] = 0
        #     self.obs_starts[ind_init] = None

        #     start_ind, end_ind = 0, 120
        #     # start_ind, end_ind = self.__time_to_index(self.ephemerides[obj_str]['start_time']), self.__time_to_index(self.ephemerides[obj_str]['end_time'])
        #     new_times = np.arange(ind_init-self.config.t_replace, ind_init+self.config.t_replace+1)
        #     new_times = new_times[(new_times > start_ind) & (new_times < end_ind)]
        #     np.random.shuffle(new_times)
        #     # randomly attempt to replace the observation to a new time
        #     for time_new in new_times:
        #         time_new_end = time_new+self.config.t_obs+self.config.t_setup
        #         if time_new_end < len(self.state):
        #             if all([self.state[i] == 1 for i in range(time_new, time_new_end)]):
        #                 self.state[time_new:time_new_end] = object+2
        #                 self.rewards[time_new:time_new_end] = 1
        #                 self.obs_starts[time_new] = obj_str
        #                 replaced = True

        #                 for ind in range(ind_init-self.config.t_int-self.config.t_obs-self.config.t_setup+1,ind_init_end):
        #                     if np.all(self.state[ind:ind+self.config.t_obs*2+self.config.t_setup*2] == 1):
        #                         self.action_availability_mask[ind,2] = True
        #                         if np.all(self.state[ind+self.config.t_int:ind+self.config.t_int+self.config.t_obs*2+self.config.t_setup*2] == 1):
        #                             self.action_availability_mask[ind,0] = True
        #                     # if self.action_availability_mask[ind,3] and self.state[ind] == 1:
        #                     #     self.action_availability_mask[ind,3] = False
        #                     # if self.action_availability_mask[ind,4] and self.state[ind] == 1:
        #                     #     self.action_availability_mask[ind,4] = False
        #                 self.action_availability_mask[ind_init,3] = False
        #                 self.action_availability_mask[ind_init,4] = False

        #                 self.action_availability_mask[time_new-self.config.t_int-self.config.t_obs-self.config.t_setup+1:time_new-self.config.t_int+self.config.t_obs+self.config.t_setup,0] = False
        #                 self.action_availability_mask[time_new-self.config.t_obs-self.config.t_setup+1:time_new_end,0] = False
        #                 self.action_availability_mask[time_new-self.config.t_obs-self.config.t_setup+1:time_new_end,2] = False
        #                 self.action_availability_mask[time_new,3] = True
        #                 self.action_availability_mask[time_new,4] = True

        #                 self.action_availability_mask[-self.config.t_int-self.config.t_obs*2-self.config.t_setup*2:,0] = False
        #                 self.action_availability_mask[-self.config.t_obs-self.config.t_setup:,2] = False

        #                 self.observation_mask[object,ind_init] = False
        #                 self.observation_mask[object,time_new] = True

        #                 self.__update_total_mask()
        #                 break

        #     if not replaced:
        #         self.state = init_state
        #         self.obs_starts = init_obs_starts
        #         self.rewards = init_rewards


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



    


