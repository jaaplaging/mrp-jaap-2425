import sys
sys.path.append('mrp-jaap-2425/')
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from env_table import ObservationScheduleEnv
from gd_agent import GDAgent
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
import pickle
from time import perf_counter


def main(observer=create_observer(), time=Time.now()):
    obj_dict, eph_dict = scraper(observer, time)

    config = Configuration()

    env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
    agent = GDAgent(observer, eph_dict, time, env, config=config)
    agent.create_init_state()
    agent.gradient_descent()
    final_reward = agent.env.calculate_reward()
    print(f'Final fill factor: {final_reward}')
    print(agent.env)
    agent.plot_history()

def run_multiple(observer=create_observer(), time=Time.now(), iterations=40):
    obj_dict, eph_dict = scraper(observer, time)

    config = Configuration()

    init_fill = []
    final_fill = []
    runtime = []
    for i in range(iterations):
        time_init = perf_counter()
        env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
        agent = GDAgent(observer, eph_dict, time, env, config=config)
        agent.create_init_state()
        agent.gradient_descent()
        final_reward = agent.env.calculate_reward()
        print(f'Final fill factor: {final_reward}')
        init_fill.append(agent.history[0])
        final_fill.append(agent.history[-1])
        runtime.append(perf_counter()-time_init)
    
    with open('init_fill.pkl', 'rb') as file:
        init_fill_before = pickle.load(file)

    with open('final_fill.pkl', 'rb') as file:
        final_fill_before = pickle.load(file)

    with open('runtime.pkl', 'rb') as file:
        runtime_before = pickle.load(file)

    init_fill.extend(init_fill_before)
    final_fill.extend(final_fill_before)
    runtime.extend(runtime_before)

    print(init_fill, final_fill, runtime)

    with open('init_fill.pkl', 'wb') as file:
        pickle.dump(init_fill, file)
    
    with open('final_fill.pkl', 'wb') as file:
        pickle.dump(final_fill, file)
    
    with open('runtime.pkl', 'wb') as file:
        pickle.dump(runtime, file)


def run_no_limit(observer=create_observer(), time=Time.now(), iterations=10):
    obj_dict, eph_dict = scraper(observer, time)

    config = Configuration()
    config.max_iter = 2000

    histories = []
    for i in range(iterations):
        env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
        agent = GDAgent(observer, eph_dict, time, env, config=config)
        agent.create_init_state()
        agent.gradient_descent()
        final_reward = agent.env.calculate_reward()
        print(f'Final fill factor: {final_reward}')
        histories.append(agent.history)
    
    with open('histories.pkl', 'rb') as file:
        histories_before = pickle.load(file)

    histories.extend(histories_before)

    print(len(histories))

    with open('histories.pkl', 'wb') as file:
        pickle.dump(histories, file)

    


if __name__ == '__main__':
    main()
    #run_multiple()
    #run_no_limit()