import numpy as np

class Reward():
    
    def __init__(self, weight_fill_factor):
        self.w_fill_factor = weight_fill_factor
        self.fill_factor_reward = -1
        self.observations_reward = -1

        self.r_time_from_peak = -1
        self.r_airmass = -1
        self.r_magnitude = -1
        self.r_motion = -1
        self.r_all_viewed = -1
        self.r_n_observations = -1
        self.r_time_gap = -1


    def get_reward(self, env, empty_flag):
        if self.w_fill_factor > 0:
            self.update_fill_factor_reward(env, empty_flag)
        if self.w_fill_factor < 1:
            self.update_observations_reward(env)

        self.total_reward = self.w_fill_factor * self.fill_factor_reward \
                          + (1 - self.w_fill_factor) * self.observations_reward
        
        return(self.total_reward)
    
    def update_fill_factor_reward(self, env, empty_flag):
        if type(env.schedule) == type([]):
            fill = np.sum(np.where(np.array(env.schedule) > empty_flag, 1, 0))
        else:
            fill = np.sum(np.where(env.schedule > empty_flag, 1, 0))
        fill_factor = fill / env.config.state_length
        self.fill_factor_reward = fill_factor * 2 - 1

    def update_observations_reward(self, env):
        self.r_time_from_peak = 0
        self.r_airmass = 0
        self.r_magnitude = 0
        self.r_motion = 0
        self.r_all_viewed = 0

        ind_eval = 0
        n_obs = 0
        while ind_eval+1 < len(env.schedule):
            obj = env.schedule[ind_eval]
            if obj == env.empty_flag:
                ind_eval += 1
            else:
                self.r_time_from_peak += self.__get_reward_time_from_peak(env, ind_eval, obj)
                self.r_airmass += self.__get_reward_airmass(env, ind_eval, obj)
                self.r_magnitude += self.__get_reward_magnitude(env, ind_eval, obj)
                self.r_motion += self.__get_reward_motion(env, ind_eval, obj)

                ind_eval += env.config.t_setup + env.config.t_obs
                n_obs += 1

        if n_obs > 0:
            self.r_time_from_peak /= n_obs
            self.r_airmass /= n_obs
            self.r_magnitude /= n_obs
            self.r_motion /= n_obs

        self.r_n_observations = self.__get_reward_n_observations(env)
        self.r_time_gap = self.__get_reward_time_gap(env)
        self.r_all_viewed = self.__get_reward_all_viewed(env)

        self.observations_reward = env.config.w_time_from_peak * self.r_time_from_peak \
                + env.config.w_airmass * self.r_airmass \
                + env.config.w_magnitude * self.r_magnitude \
                + env.config.w_motion * self.r_motion \
                + env.config.w_n_observations * self.r_n_observations \
                + env.config.w_time_gap * self.r_time_gap \
                + env.config.w_all_viewed * self.r_all_viewed
        
        # scale total reward from -1 to 1, with lower limit
        # self.observations_reward = self.observations_reward / env.config.w_total
        # if  < 0.25: # at least this much needed
        #     r_total = -1
        # else:
        #     r_total -= 0.25
        #     r_total *= 4
        #     r_total -= 1
        self.observations_reward /= env.config.w_total
        self.observations_reward = self.observations_reward * 2 - 1

    def __get_reward_time_from_peak(self, env, ind_eval, obj):
        peak_time = env.object_state[obj-1-env.empty_flag, 0]
        dt = np.abs(peak_time - (ind_eval + env.config.t_obs//2 + env.config.t_setup//2))
        scale = env.object_state[obj-1-env.empty_flag, 2] - peak_time
        
        if scale <= 0:
            dt_scaled = 0
            return(dt_scaled)
        dt_scaled = 1 - dt / scale
        
        if dt_scaled < 0:
            dt_scaled = 0
        
        return(dt_scaled)
    
    def __get_reward_airmass(self, env, ind_eval, obj):
        airmass = env.object_state[obj-1-env.empty_flag, 3]
        r_airmass = 1 / airmass
        
        return(r_airmass)
    
    def __get_reward_magnitude(self, env, ind_eval, obj):
        magnitude = env.object_state[obj-1-env.empty_flag, 4]
        if magnitude > env.config.mag_lim:
            return(0)
        else:
            dm = env.config.mag_lim - magnitude
            scale = env.config.mag_lim - env.config.mag_min
            dm_scaled = 1 - dm / scale
            
            if dm_scaled < 0:
                dm_scaled = 0
            
            return(dm_scaled)

    def __get_reward_motion(self, env, ind_eval, obj):
        motion = env.object_state[obj-1-env.empty_flag, 5]
        scale = env.config.motion_scale
        motion_scaled = np.min([motion / scale, 1.])
        return(motion_scaled)
    
    def __get_reward_n_observations(self, env):
        r = []
        for number in env.object_state[:,6]:
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
            
    def __get_reward_time_gap(self, env):
        r = []
        for object in env.observations:
            if len(object) > 1:
                for i in range(len(object)-1):
                    gap = object[i+1] - object[i]
                    if gap <= env.config.t_int_min or gap > env.config.t_int * 2:
                        r.append(0)
                    elif gap > env.config.t_int_min and gap <= env.config.t_int:
                        score = 1 - (env.config.t_int - gap) / (env.config.t_int - env.config.t_int_min)
                        r.append(score)
                    else:
                        score = 1 - (gap - env.config.t_int)  / env.config.t_int      
                        r.append(score)
        if len(r) > 0:
            return(np.mean(r))
        else:
            return(0)
            
    def __get_reward_all_viewed(self, env):
        r = 0
        count = 0
        for object in env.object_state:
            if object[8] == 1:
                count += 1
                if object[6] > 0:
                    r += 1
        if count == 0:
            return(0)
        else:
            return(r / count)
        
