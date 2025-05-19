import sys
sys.path.append('mrp-jaap-2425/')
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from rl_env_table import ObservationScheduleEnv
from rl_agent import RLAgent
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
import pickle
from time import perf_counter


def main(observer=create_observer(), time=Time.now()):
    #obj_dict, eph_dict = scraper(observer, time)

    config = Configuration()
    obj_dict = {}
    eph_dict = {}
    for i in range(config.n_objects):
        obj_dict[str(i)] = {}
        eph_dict[str(i)] = {}
    
    env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config = config)

    agent = RLAgent(env)
    f_factors_max, f_factors_mean, f_factors_final, actions_taken_total, actions_logits_mean, f_factors_eval = agent.train()

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

    ep = np.arange(0, config.episodes, config.evaluation_interval)
    plt.plot(ep, f_factors_eval, label='Eval fill factor', color='green')
    plt.title('Evaluation episodes results')
    plt.xlabel('Episode')
    plt.ylabel('Fill factor')
    plt.savefig('evaluation.png')
    plt.show()




def run_multiple(observer=create_observer(), time=Time.now(), iterations=40):
    #obj_dict, eph_dict = scraper(observer, time)

    f_factors_max_list, f_factors_mean_list, f_factors_final_list, actions_taken_total_list, actions_logits_mean_list, f_factors_eval_list = [],[],[],[],[],[]
    for i in range(9):
        print(f'Run {i+1} starting...')
        config = Configuration()
        obj_dict = {}
        eph_dict = {}
        for i in range(config.n_objects):
            obj_dict[str(i)] = {}
            eph_dict[str(i)] = {}

        env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config = config)

        agent = RLAgent(env)
        f_factors_max, f_factors_mean, f_factors_final, actions_taken_total, actions_logits_mean, f_factors_eval = agent.train()
        f_factors_max_list.append(f_factors_max)
        f_factors_mean_list.append(f_factors_mean)
        f_factors_final_list.append(f_factors_final)
        actions_taken_total_list.append(actions_taken_total)
        actions_logits_mean_list.append(actions_logits_mean)
        f_factors_eval_list.append(f_factors_eval)

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(f_factors_max_list[i], label='Max reward',color='red')
        plt.plot(f_factors_mean_list[i], label='Mean reward',color='blue')
        plt.plot(f_factors_final_list[i], label='Final reward', color='green')
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
    
    ep = np.arange(0, config.episodes, config.evaluation_interval)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(ep, f_factors_eval_list[i], label='Eval fill factor', color='green')
    plt.savefig('evaluation.png')
    plt.show()



    


if __name__ == '__main__':
    #main()
    run_multiple()
    #run_no_limit()