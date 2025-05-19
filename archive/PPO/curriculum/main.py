import sys
sys.path.append('mrp-jaap-2425/')
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from ppo_env_table_curriculum import ObservationScheduleEnvCurriculum
from ppo_agent_curriculum import PPOAgentCurriculum
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
import pickle
from time import perf_counter
from keras.saving import load_model
import tensorflow as tf


def main(observer=create_observer(), time=Time.now()):
    #obj_dict, eph_dict = scraper(observer, time)


    config = Configuration()
    obj_dict = {'A': {},
                'B': {},
                'C': {}}
    eph_dict = {'A': {},
                'B': {},
                'C': {}}
    
    env = ObservationScheduleEnvCurriculum(observer, time, obj_dict, eph_dict, config = config)

    agent = PPOAgentCurriculum(env)
    f_factors_max, f_factors_mean, f_factors_final, actions_taken_total, actions_logits_mean = agent.train()

    plt.plot(f_factors_max, label='Max reward',color='red')
    plt.plot(f_factors_mean, label='Mean reward',color='blue')
    plt.plot(f_factors_final, label='Final reward', color='green')
    plt.title(f'absolute rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rewards.png')
    plt.show()

    plt.plot(actions_taken_total[0], label='Add object',color='red')
    plt.plot(actions_taken_total[1], label='Remove object',color='orange')
    plt.plot(actions_taken_total[2], label='Add observation',color='green')
    plt.plot(actions_taken_total[3], label='Remove observation',color='blue')
    plt.plot(actions_taken_total[4], label='Replace observation',color='pink')
    plt.title(f'actions taken')
    plt.xlabel('Episode')
    plt.ylabel('Number of actions')
    plt.legend(bbox_to_anchor=(1.1,1))
    plt.tight_layout()
    plt.savefig('actions.png')
    plt.show()

    plt.plot(actions_logits_mean[0], label='Add object',color='red')
    plt.plot(actions_logits_mean[1], label='Remove object',color='orange')
    plt.plot(actions_logits_mean[2], label='Add observation',color='green')
    plt.plot(actions_logits_mean[3], label='Remove observation',color='blue')
    plt.plot(actions_logits_mean[4], label='Replace observation',color='pink')
    plt.title(f'Action logits values')
    plt.xlabel('Episode')
    plt.ylabel('Mean logit values')
    plt.legend(bbox_to_anchor=(1.1,1))
    plt.tight_layout()
    plt.savefig('logits.png')
    plt.show()


def run_4(observer=create_observer(), time=Time.now(), iterations=40):
    #obj_dict, eph_dict = scraper(observer, time)

    f_factors_max_list, f_factors_mean_list, f_factors_final_list, actions_taken_total_list, actions_logits_mean_list = [],[],[],[],[]
    rewards_mean_list = []
    steps_taken_list = []
    for i in range(4):
        print(f'Run {i+1} starting...')
        config = Configuration()
        obj_dict = {'A': {}}#,
                  #  'B': {},
                  #  'C': {}}
        eph_dict = {'A': {}}#,
                  #  'B': {},
                  #  'C': {}}
        target_fill = 0.7

        env = ObservationScheduleEnvCurriculum(observer, time, obj_dict, eph_dict, target_fill, config = config)

        agent = PPOAgentCurriculum(env)

        agent.actor_network = load_model(f'mrp-jaap-2425/PPO/curriculum/models/ppo_actor_network_0_60_{i}.keras')
        agent.critic_network = load_model(f'mrp-jaap-2425/PPO/curriculum/models/ppo_critic_network_0_60_{i}.keras')
        agent.actor_network_func = tf.function(agent.actor_network, reduce_retracing=True)
        agent.critic_network_func = tf.function(agent.critic_network, reduce_retracing=True)

        rewards_mean, f_factors_mean, actions_taken_total, actions_logits_mean, steps_taken = agent.train()
        f_factors_mean_list.append(f_factors_mean)
        actions_taken_total_list.append(actions_taken_total)
        actions_logits_mean_list.append(actions_logits_mean)
        rewards_mean_list.append(rewards_mean)
        steps_taken_list.append(steps_taken)

        agent.actor_network.save(f'mrp-jaap-2425/PPO/curriculum/models/ppo_actor_network_0_70_{i}.keras')
        agent.critic_network.save(f'mrp-jaap-2425/PPO/curriculum/models/ppo_critic_network_0_70_{i}.keras')


    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(steps_taken_list[i], label='steps taken', color='blue')
    plt.savefig('rewards.png')
    plt.show()

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(actions_taken_total_list[i][0], label='Add object',color='red')
        plt.plot(actions_taken_total_list[i][1], label='Remove object',color='orange')
        plt.plot(actions_taken_total_list[i][2], label='Add observation',color='green')
        plt.plot(actions_taken_total_list[i][3], label='Remove observation',color='blue')
        plt.plot(actions_taken_total_list[i][4], label='Replace observation',color='pink')
    plt.savefig('actions.png')
    plt.show()

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(actions_logits_mean_list[i][0], label='Add object',color='red')
        plt.plot(actions_logits_mean_list[i][1], label='Remove object',color='orange')
        plt.plot(actions_logits_mean_list[i][2], label='Add observation',color='green')
        plt.plot(actions_logits_mean_list[i][3], label='Remove observation',color='blue')
        plt.plot(actions_logits_mean_list[i][4], label='Replace observation',color='pink')
    plt.savefig('logits.png')
    plt.show()


def run_9(observer=create_observer(), time=Time.now(), iterations=40):
    #obj_dict, eph_dict = scraper(observer, time)

    f_factors_max_list, f_factors_mean_list, f_factors_final_list, actions_taken_total_list, actions_logits_mean_list = [],[],[],[],[]
    rewards_mean_list = []
    steps_taken_list = []
    for i in range(9):
        print(f'Run {i+1} starting...')
        config = Configuration()
        obj_dict = {'A': {},
                    'B': {},
                    'C': {}}
        eph_dict = {'A': {},
                    'B': {},
                    'C': {}}
        target_fill = 0

        env = ObservationScheduleEnvCurriculum(observer, time, obj_dict, eph_dict, target_fill, config = config)

        agent = PPOAgentCurriculum(env)
        rewards_mean, f_factors_mean, actions_taken_total, actions_logits_mean, steps_taken = agent.train()
        f_factors_mean_list.append(f_factors_mean)
        actions_taken_total_list.append(actions_taken_total)
        actions_logits_mean_list.append(actions_logits_mean)
        rewards_mean_list.append(rewards_mean)
        steps_taken_list.append(steps_taken)

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(steps_taken_list[i], label='steps taken', color='blue')
    plt.savefig('rewards.png')
    plt.show()

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(actions_taken_total_list[i][0], label='Add object',color='red')
        plt.plot(actions_taken_total_list[i][1], label='Remove object',color='orange')
        plt.plot(actions_taken_total_list[i][2], label='Add observation',color='green')
        plt.plot(actions_taken_total_list[i][3], label='Remove observation',color='blue')
        plt.plot(actions_taken_total_list[i][4], label='Replace observation',color='pink')
    plt.savefig('actions.png')
    plt.show()

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(actions_logits_mean_list[i][0], label='Add object',color='red')
        plt.plot(actions_logits_mean_list[i][1], label='Remove object',color='orange')
        plt.plot(actions_logits_mean_list[i][2], label='Add observation',color='green')
        plt.plot(actions_logits_mean_list[i][3], label='Remove observation',color='blue')
        plt.plot(actions_logits_mean_list[i][4], label='Replace observation',color='pink')
    plt.savefig('logits.png')
    plt.show()


def run_no_limit(observer=create_observer(), time=Time.now(), iterations=10):
    obj_dict, eph_dict = scraper(observer, time)

    config = Configuration()
    config.max_iter = 2000

    histories = []
    for i in range(iterations):
        env = ObservationScheduleEnvCurriculum(observer, time, obj_dict, eph_dict, config=config)
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
    #main()
    run_4()
    #run_no_limit()