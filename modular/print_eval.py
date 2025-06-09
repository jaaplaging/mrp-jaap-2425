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


COLORS = ['f2d7d5',
          'd4e6f1',
          'd4efdf',
          'f4ecf7',
          'fef9e7']

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
        env = OnTheFlyEnv(observer=observer, time=time, eph_class=ephemerides, reward_class=reward, config=config)
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

    rewards, actions_taken, final_schedule = agent.evaluate()
    

    # string_schedule = ''
    # count = 0
    # previous = final_schedule[0]
    # for ind, neo in enumerate(final_schedule):
    #     neo = neo - 1 - agent.env.empty_flag
    #     current = neo
    #     count += 1
    #     if current != previous:
    #         if neo >= 0:
    #             string_schedule += ' & \cellcolor[HTML]{'+COLORS[neo]+'} \multicol{'+str(count)+'}{|c|}{'+agent.env.eph_names[neo]+'}' 
    #         else:
    #             string_schedule += ' & \multicol{'+str(count)+'}{|c|}{}' 
    #         count = 0
    #     previous = current
    # string_schedule += ' & \cellcolor[HTML]{'+COLORS[neo]+'} \multicol{'+str(count)+'}{|c|}{'+agent.env.eph_names[neo]+'}'
    # string_schedule += '\\\ \hline'
    # print(string_schedule)

    import matplotlib.patches as mpatches


    neos = sorted(set(final_schedule))
    colors = plt.cm.tab10(np.linspace(0, 1, len(neos)))

    neo_color_map = {neo: colors[i] for i, neo in enumerate(neos)}

    # Create a color list for the plot
    color_row = [neo_color_map[neo] for neo in final_schedule]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 2))

    # Display the timeline as an image (1-row matrix)
    ax.imshow([color_row], aspect='auto')

    # Create a legend
    legend_elements = [
        mpatches.Patch(color=neo_color_map[neo], label=f'Task {neo}')
        for neo in neos if neo != 0
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=len(legend_elements))

    # Set ticks and labels
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, len(final_schedule), max(1, len(final_schedule) // 10)))
    ax.set_xlabel("Minutes")

    plt.tight_layout()
    plt.show()
    

