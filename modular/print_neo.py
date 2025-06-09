import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt
from keras.models import load_model

sys.path.append('mrp-jaap-2425/')

from param_config import Configuration
from ephemerides import EphemeridesReal
from environments import OnTheFlyEnv, ScheduleEnv
from agents import PPOAgent, DQNAgent, GDAgent
from rewards import Reward
from helper import create_observer
from astropy.time import Time
import tensorflow as tf

observer = create_observer()
time = Time.now()
config = Configuration()




for file_str in os.listdir('results_models/results_models/'):
    if 'critic' in file_str:
        continue

    if 'WFF0_' in file_str:
        W_FF = 0
    elif 'WFF0.5_' in file_str:
        W_FF = 0.5
    else:
        W_FF = 1
    
    ephemerides = EphemeridesReal(observer, time, config=config)
    reward = Reward(weight_fill_factor=W_FF)

    if 'sched' in file_str:
        env = ScheduleEnv(observer=observer, time=time, eph_class=ephemerides, reward_class=reward, config=config) 
    else:
        env = ScheduleEnv(observer=observer, time=time, eph_class=ephemerides, reward_class=reward, config=config)
    if 'dqn' in file_str:
        agent = DQNAgent(env=env, eph_class = ephemerides)
        agent.q_network = load_model(f'results_models/results_models/{file_str}')
        agent.q_network_func = tf.function(agent.q_network)
    elif 'ppo' in file_str:
        agent = PPOAgent(env=env, eph_class = ephemerides)
        agent.actor_network = load_model(f'results_models/results_models/{file_str}')
        agent.actor_network_func = tf.function(agent.actor_network)
        agent.critic_network = load_model(f'results_models/results_models/{file_str.replace('actor', 'critic')}')
        agent.critic_network_func = tf.function(agent.critic_network)
    else:
        continue

    # create table with ephemerides info
    print("NEO & $i_\\text{peak}$ & $i_\\text{rise}$ & $i_\\text{set}$ & $X_\\text{peak}$ & $m$ & $\mu$ \\\ \hline")
    for i in range(len(agent.env.object_state)):
        print(f"{agent.env.eph_names[i]} & {agent.env.object_state[i,0]} & {agent.env.object_state[i,1]} & {agent.env.object_state[i,2]} & {np.round(agent.env.object_state[i,3],2)} & {np.round(agent.env.object_state[i,4], 1)} & {np.round(agent.env.object_state[i,5], 1)} \\\ \hline")


    

