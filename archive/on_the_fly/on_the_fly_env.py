import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import copy

from helper import twilight_rise_set, create_observer, index_last, opt_time, airmass, rise_set
from param_config import Configuration
from astroplan import FixedTarget
from astropy.coordinates import SkyCoord
from time import perf_counter

class OnTheFlyEnv():

    def __init__(self, observer, time, eph_dict, config = Configuration()):
        self.config = config
        self.observer = observer
        self.time = time
        start_time, minutes = self.__calculate_start_length()
        self.length = minutes
        self.start_time = start_time
        self.ephemerides = eph_dict
        self.dawn, self.dusk = twilight_rise_set(self.observer, self.time)

        #self.times = [[] for _ in range(10)]
        self.reset()
        

    def reset(self):
        self.schedule = []
        self.object_to_ind = {}
        self.daily_ephemerides = self.__get_daily_ephemerides()
        self.object_state, self.object_state_norm = self.__create_object_state()
        self.insert_time = 0
        self.obs_starts = [[] for i in range(10)]

    def __calculate_start_length(self):
        noon = self.observer.noon(self.time, which='nearest')
        minutes = self.config.state_length
        return(noon, minutes)

    def __get_daily_ephemerides(self):
        '''Picks a single day of ephemerides from self.ephemerides at random  '''
        return(self.ephemerides[0])
        #return(np.random.choice(self.ephemerides))

    def __create_object_state(self):
        object_state = np.empty((10, 9))
        
        # def get_airmass_peak(object_ephemerides):
        #     time_init = perf_counter()
        #     ra_mid = object_ephemerides['RA'][12]
        #     ra_mid = str(ra_mid) + 'h'
        #     dec_mid = object_ephemerides['dec'][12]
        #     dec_mid = str(dec_mid) + 'd'
        #     self.times[0].append(perf_counter() - time_init)
        #     time_init = perf_counter()
        #     coords_mid = SkyCoord(ra_mid, dec_mid, frame='gcrs')
        #     self.times[1].append(perf_counter() - time_init)
        #     time_init = perf_counter()
        #     target = FixedTarget(name='target', coord=coords_mid)
        #     self.times[2].append(perf_counter() - time_init)
        #     time_init = perf_counter()
        #     rise, set = rise_set(self.observer, target, self.time)
        #     self.times[3].append(perf_counter() - time_init)
        #     time_init = perf_counter()
        #     rise_ind, set_ind = self.__time_to_ind(rise), self.__time_to_ind(set)
        #     self.times[4].append(perf_counter() - time_init)
        #     if set < self.dusk or rise > self.dawn:
        #         return(-1, -1, -1, -1)
        #     else:
        #         time_init = perf_counter()
        #         peak_time = opt_time(self.observer, target, self.time)
        #         peak_ind = int((peak_time - self.dusk).value*24*60)
        #         peak_airmass = airmass(self.observer, target, peak_time)
        #         self.times[5].append(perf_counter() - time_init)
        #         return(peak_ind, peak_airmass, rise_ind, set_ind)

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
            if state_ind >= 10:
                break
        
        for i in range(state_ind, 10):
            object_state[i, :] = [0,0,0,0,0,0,0,0,0]
        
        object_state, object_state_norm = normalize(object_state)

        return(object_state, object_state_norm)

    def __time_to_ind(self, given_time):
        ind = int((given_time - self.dusk).value*24*60)
        return(ind)

    def create_mask(self):
        no_object_mask = self.object_state[:,8] == 1
        time_mask = (self.insert_time > self.object_state[:,1]) & (self.insert_time+12 < self.object_state[:,2])
        total_mask = no_object_mask & time_mask
        total_mask = np.concat([np.array([True]), total_mask]).reshape(1, 11)
        return(total_mask)

    def step(self, obj):
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
            self.obs_starts[obj].append(self.insert_time)
            self.insert_time += self.config.t_obs + self.config.t_setup

        if self.insert_time + self.config.t_setup + self.config.t_obs > self.length:
            done = 1
            reward = self.calculate_reward()
        else:
            done = 0
            reward = -1
        state = [self.object_state_norm, np.array([self.insert_time/(0.5*self.config.state_length)-1, self.length/(0.5*self.config.state_length)-1])]

        return(state, reward, done)
    
    def calculate_reward(self):
        # need to consider: 
        # time difference from peak, airmass, magnitude, motion, at least 2 observations, 45 minutes between observations, preferably every object viewed
        # visibility is already taken into account by masking
        
        r_time_from_peak = 0
        r_airmass = 0
        r_magnitude = 0
        r_motion = 0
        r_all_viewed = 0

        ind_eval = 0
        n_obs = 0
        while ind_eval+1 < len(self.schedule):
            obj = self.schedule[ind_eval]
            if obj == -1:
                ind_eval += 1
            else:
                r_time_from_peak += self.__get_reward_time_from_peak(ind_eval, obj)
                r_airmass += self.__get_reward_airmass(ind_eval, obj)
                r_magnitude += self.__get_reward_magnitude(ind_eval, obj)
                r_motion += self.__get_reward_motion(ind_eval, obj)

                ind_eval += self.config.t_setup + self.config.t_obs
                n_obs += 1

        if n_obs > 0:
            r_time_from_peak /= n_obs
            r_airmass /= n_obs
            r_magnitude /= n_obs
            r_motion /= n_obs

        r_n_observations = self.__get_reward_n_observations()
        r_time_gap = self.__get_reward_time_gap()
        r_all_viewed = self.__get_reward_all_viewed()

        r_total = self.config.w_time_from_peak * r_time_from_peak \
                + self.config.w_airmass * r_airmass \
                + self.config.w_magnitude * r_magnitude \
                + self.config.w_motion * r_motion \
                + self.config.w_n_observations * r_n_observations \
                + self.config.w_time_gap * r_time_gap \
                + self.config.w_all_viewed * r_all_viewed
        
        # scale total reward from -1 to 1, with lower limit
        r_total = r_total / self.config.w_total
        if r_total < 0.25: # at least this much needed
            r_total = -1
        else:
            r_total -= 0.25
            r_total *= 4
            r_total -= 1

        return(r_total)

    def __get_reward_time_from_peak(self, ind_eval, obj):
        peak_time = self.object_state[obj, 0]
        dt = np.abs(peak_time - (ind_eval + self.config.t_obs//2 + self.config.t_setup//2))
        scale = self.object_state[obj, 2] - peak_time
        dt_scaled = 1 - dt / scale
        
        if dt_scaled < 0:
            dt_scaled = 0
        
        return(dt_scaled)
    
    def __get_reward_airmass(self, ind_eval, obj):
        airmass = self.object_state[obj, 3]
        r_airmass = 1 / airmass
        return(r_airmass)
    
    def __get_reward_magnitude(self, ind_eval, obj):
        magnitude = self.object_state[obj, 4]
        if magnitude > self.config.mag_lim:
            return(0)
        else:
            dm = self.config.mag_lim - magnitude
            scale = self.config.mag_lim - self.config.mag_min
            dm_scaled = 1 - dm / scale
            
            if dm_scaled < 0:
                dm_scaled = 0
            
            return(dm_scaled)

    def __get_reward_motion(self, ind_eval, obj):
        motion = self.object_state[obj, 5]
        scale = self.config.motion_scale
        motion_scaled = np.min([motion / scale, 1.])
        return(motion_scaled)
    
    def __get_reward_n_observations(self):
        r = []
        for number in self.object_state[:,6]:
            if number >= 2:
                r_number = 0.75 + (number - 2)/20
                if r_number > 1:
                    r_number = 1
                r.append(r_number)
            elif number == 1:
                r.append(0)
        if len(r) > 0:
            return(np.mean(r))
        else:
            return(0)
            
    def __get_reward_time_gap(self):
        r = []
        for object in self.obs_starts:
            if len(object) > 1:
                for i in range(len(object)-1):
                    gap = object[i+1] - object[i]
                    if gap <= self.config.t_int_min or gap > self.config.t_int * 2:
                        r.append(0)
                    elif gap > self.config.t_int_min and gap <= self.config.t_int:
                        score = 1 - (self.config.t_int - gap) / (self.config.t_int - self.config.t_int_min)
                        r.append(score)
                    else:
                        score = 1 - (gap - self.config.t_int)  / self.config.t_int      
                        r.append(score)
        if len(r) > 0:
            return(np.mean(r))
        else:
            return(0)
            
    def __get_reward_all_viewed(self):
        r = 0
        count = 0
        for object in self.object_state:
            if object[8] == 1:
                count += 1
                if object[6] > 0:
                    r += 1
        if count == 0:
            return(0)
        else:
            return(r / count)



