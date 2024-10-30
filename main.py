# main python file that runs the entire code

from mpc_scraper import scraper
from helper import create_observer
from astropy.time import Time
from env_table import ObservationScheduleEnv
from gd_agent import GDAgent
from astroplan import Observer, FixedTarget
from astropy.coordinates import SkyCoord
from astroplan.plots import plot_airmass
import matplotlib.pyplot as plt
import copy
from param_config import Configuration
import numpy as np
import pickle

def main(observer=create_observer(), time=Time.now()):
    obj_dict, eph_dict = scraper(observer, time)


    config = Configuration()
    init_attempts = np.array([50, 150], dtype=np.int32)
    w_add_object = np.array([25, 75])
    w_remove_object = np.array([1, 5])
    w_add_obs = np.array([12.5, 37.5])
    w_remove_obs = np.array([2.5, 12.5])
    w_replace = np.array([12.5, 37.5])
    max_iter = np.array([250, 750], dtype=np.int32)
    n_sub_iter = np.array([38, 50], dtype=np.int32)
    add_attempts = np.array([5, 15], dtype=np.int32)
    
    results = np.zeros((30, 11))

    for i in range(30):  # number of different configurations
        fill_factors = []

        config.init_attempts = np.random.randint(init_attempts[0], init_attempts[1]+1)
        config.w_add_object = np.random.uniform(w_add_object[0], w_add_object[1])
        config.w_remove_object = np.random.uniform(w_remove_object[0], w_remove_object[1])
        config.w_add_obs = np.random.uniform(w_add_obs[0], w_add_obs[1])
        config.w_remove_obs = np.random.uniform(w_remove_obs[0], w_remove_obs[1])
        config.w_replace = np.random.uniform(w_replace[0], w_replace[1])
        config.max_iter = np.random.randint(max_iter[0], max_iter[1]+1)
        config.n_sub_iter = np.random.randint(n_sub_iter[0], n_sub_iter[1]+1)
        config.add_attempts = np.random.randint(add_attempts[0], add_attempts[1]+1)

        print(f'Setting: \n \
               init attempts: {config.init_attempts}, w_add_object: {config.w_add_object}\n \
                w_remove_object: {config.w_remove_object}, w_add_obs: {config.w_add_obs}\n \
                w_remove_obs: {config.w_remove_obs}, w_replace: {config.w_replace}\n \
                max_iter: {config.max_iter}, n_sub_iter: {config.n_sub_iter}\n \
                add_attempts: {config.add_attempts}')

        for n in range(5):  # take mean for this to negate randomness
            env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
            agent = GDAgent(observer, eph_dict, time, env, config=config)
            agent.create_init_state()
            agent.gradient_descent()
            final_reward = agent.env.calculate_reward()
            print(f'Final fill factor: {final_reward}')
            fill_factors.append(final_reward)
        
        print(f'Mean fill factors: {np.mean(fill_factors)}')
        
        results[i,0] = config.init_attempts
        results[i,1] = config.w_add_object
        results[i,2] = config.w_remove_object
        results[i,3] = config.w_add_obs
        results[i,4] = config.w_remove_obs
        results[i,5] = config.w_replace
        results[i,6] = config.max_iter
        results[i,7] = config.n_sub_iter
        results[i,8] = config.add_attempts
        results[i,9] = np.mean(fill_factors)
        results[i,10] = np.std(fill_factors)
        
    with open('random_search_results_continuous_2.pkl', 'wb') as f:
        pickle.dump(results, f)

 

if __name__ == '__main__':
    main()