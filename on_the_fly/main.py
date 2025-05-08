import numpy as np
import astropy
import astroplan
import pickle
import sys
sys.path.append('mrp-jaap-2425/')
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from on_the_fly_env import OnTheFlyEnv
from on_the_fly_agent import OnTheFlyAgentPPO, OnTheFlyAgentDQN
from helper import create_observer
from param_config import Configuration


FILE_EPH_DICT = 'mrp-jaap-2425/on_the_fly/top_eph_processed.pkl'

def get_eph_dict():
    with open(FILE_EPH_DICT, 'rb') as f:
        eph_dict = pickle.load(f)
    return(eph_dict)

def run(eph_dict, observer=create_observer(), time=Time.now()):
    config = Configuration()
    env = OnTheFlyEnv(observer, time, eph_dict, config=config)
    agent = OnTheFlyAgentPPO(env)
    #agent = OnTheFlyAgentDQN(env)

    rewards, rewards_eval = agent.train()

    plt.plot(rewards, label='Final rewards')
    plt.title('Final rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.1,1))
    plt.tight_layout()
    plt.savefig('onthefly.png')
    plt.show()

    ep = np.arange(0, config.episodes, config.evaluation_interval)
    plt.plot(ep, rewards_eval, label='Eval fill factor', color='green')
    plt.title('Evaluation episodes results')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('ontheflyeval.png')
    plt.show()

def run_multiple(eph_dict, observer=create_observer(), time=Time.now()):
    config = Configuration()

    rewards_list = []
    rewards_eval_list = []

    for run in range(9):
        env = OnTheFlyEnv(observer, time, eph_dict, config=config)
        #agent = OnTheFlyAgentPPO(env)
        agent = OnTheFlyAgentDQN(env)

        rewards, rewards_eval = agent.train()

        rewards_list.append(rewards)
        rewards_eval_list.append(rewards_eval)

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(rewards_list[i], label='Final rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
    plt.title('Final rewards')
    plt.tight_layout()
    plt.savefig('onthefly.png')
    plt.show()

    for i in range(9):
        plt.subplot(3,3,i+1)
        ep = np.arange(0, config.episodes, config.evaluation_interval)
        plt.plot(ep, rewards_eval_list[i], label='Eval fill factor', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
    plt.title('Evaluation episodes results')
    plt.savefig('ontheflyeval.png')
    plt.show()







if __name__ == '__main__':
    eph_dict = get_eph_dict()
    run(eph_dict)
    #run_multiple(eph_dict)