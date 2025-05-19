from astropy.time import Time
import numpy as np
from env_table import ObservationScheduleEnv
from gd_agent import GDAgent
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
import pickle

def main(observer=create_observer(), time=Time.now()):
    obj_dict, eph_dict = scraper(observer, time)


    config = Configuration()

    results = np.zeros((40, 10))

    for i in range(40):
        config.init_attempts = np.random.randint(50, 150)
        config.w_add_object = np.random.uniform(25, 75)
        config.w_remove_object = np.random.uniform(1,8)
        config.w_add_obs = np.random.uniform(12.5,37.5)
        config.w_remove_obs = np.random.uniform(2.5,22.5)
        config.w_replace = np.random.uniform(12.5,37.5)
        config.max_iter = np.random.randint(250,750)
        config.n_sub_iter = np.random.randint(15,50)
        config.add_attempts = np.random.randint(5, 15)

        env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
        agent = GDAgent(observer, eph_dict, time, env, config=config)
        agent.create_init_state()
        agent.gradient_descent()
        final_reward = agent.env.calculate_reward()
        print(f'Final fill factor: {final_reward}, iteration: {i}')

        results[i,:] = np.array([config.init_attempts, config.w_add_object, config.w_remove_object,
                                config.w_add_obs, config.w_remove_obs, config.w_replace,
                                config.max_iter, config.n_sub_iter, config.add_attempts, final_reward])
        
    with open('new_results_4.pkl', 'wb') as f:
        pickle.dump(results, f)




 

if __name__ == '__main__':
    main()