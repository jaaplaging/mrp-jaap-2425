import sys
import pickle
import numpy as np

sys.path.append('mrp-jaap-2425/')

from param_config import Configuration
from helper import create_observer
from astropy.time import Time
from environments import ScheduleEnv, OnTheFlyEnv
from ephemerides import EphemeridesDummy, EphemeridesReal, EphemeridesSimulated
from rewards import Reward
from agents import PPOAgent, DQNAgent, GDAgent

import optuna

def run(config, eph_class, env_class, agent_class, source, random=False):
    try:
        observer = create_observer()
        time = Time.now()

        # idea: try and except around loop so that the batch doesnt fail entirely

        ephemerides = eph_class(observer, time, config=config, random=random)
        reward = Reward(weight_fill_factor=W_FF)
        env = env_class(observer=observer, time=time, eph_class=ephemerides, reward_class=reward, config=config)
        agent = agent_class(env=env, eph_class = ephemerides)

        final_rewards_env, other_rewards = agent.train()

        results = [config, 
                final_rewards_env, 
                other_rewards]
        
        time_current = Time.now().isot[:-4]
        time_current = time_current.replace(':', '_').replace('-', '_')
        with open(f'mrp-jaap-2425/modular/results/{source}_WFF{W_FF}_{time_current}.pkl', 'wb') as f:
            pickle.dump(results, f)

        return(final_rewards_env)
    except:
        string = 'failed during run'
        with open(f'mrp-jaap-2425/modular/results/FAIL_{source}_WFF{W_FF}_{time_current}.pkl', 'wb') as f:
            pickle.dump(string, f)



def objective_dummy_sched_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesDummy, ScheduleEnv, DQNAgent, 'objective_dummy_sched_dqn')

    return(-final_rewards_env[-1])

def objective_sim_sched_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, DQNAgent, 'objective_sim_sched_dqn')

    return(-final_rewards_env[-1])

def objective_sim_rnd_sched_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, DQNAgent, 'objective_sim_rnd_sched_dqn', random=True)

    return(-final_rewards_env[-1])

def objective_dummy_otf_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesDummy, OnTheFlyEnv, DQNAgent, 'objective_dummy_otf_dqn')

    return(-final_rewards_env[-1])

def objective_sim_otf_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, DQNAgent, 'objective_sim_otf_dqn')

    return(-final_rewards_env[-1])

def objective_sim_rnd_otf_dqn(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.memory_size = trial.suggest_int('mem', 5000, 15000)
    config.batch_size = int(2 ** trial.suggest_int('batch power', 4, 6))
    config.target_update_freq = trial.suggest_int('update_freq', 5, 15)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, DQNAgent, 'objective_sim_rnd_otf_dqn', random=True)

    return(-final_rewards_env[-1])

## PPO ##

def objective_dummy_sched_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesDummy, ScheduleEnv, PPOAgent, 'objective_dummy_sched_ppo')

    return(-final_rewards_env[-1])

def objective_sim_sched_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, PPOAgent, 'objective_sim_sched_ppo')

    return(-final_rewards_env[-1])

def objective_sim_rnd_sched_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.steps = trial.suggest_int('steps', 50, 200)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, PPOAgent, 'objective_sim_rnd_sched_dqn', random=True)

    return(-final_rewards_env[-1])

def objective_dummy_otf_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesDummy, OnTheFlyEnv, PPOAgent, 'objective_dummy_otf_ppo')

    return(-final_rewards_env[-1])

def objective_sim_otf_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, PPOAgent, 'objective_sim_otf_ppo')

    return(-final_rewards_env[-1])

def objective_sim_rnd_otf_ppo(trial):
    config = Configuration()

    config.discount_factor = trial.suggest_float('gamma', 0.75, 0.9999)
    config.learning_rate = trial.suggest_float('lr', 1e-5, 3e-4)
    config.layer_size = int(2 ** trial.suggest_int('layer power', 7, 9))
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)
    config.clip_ratio = trial.suggest_float('clip', 0, 0.3)
    config.n_epochs = trial.suggest_int('epoch', 5, 15)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, PPOAgent, 'objective_sim_rnd_otf_ppo', random=True)

    return(-final_rewards_env[-1])

def objective_dummy_sched_gd(trial):
    config = Configuration()

    config.init_attempts = trial.suggest_int('init attempts', 100, 150)
    config.w_add_object = trial.suggest_float('w_add_object', 20, 50)
    config.w_remove_object = trial.suggest_float('w_remove_object', 1, 5)
    config.w_add_obs = trial.suggest_float('w_add_obs', 15, 40)
    config.w_remove_obs = trial.suggest_float('w_remove_obs', 2, 8)
    config.w_replace = trial.suggest_float('w_replace', 25, 50)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.init_fill = trial.suggest_float('init_fill', 0.2, 0.5)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesDummy, ScheduleEnv, GDAgent, 'objective_dummy_sched_gd')
    
    return(-final_rewards_env) 

def objective_sim_sched_gd(trial):
    config = Configuration()

    config.init_attempts = trial.suggest_int('init attempts', 100, 150)
    config.w_add_object = trial.suggest_float('w_add_object', 20, 50)
    config.w_remove_object = trial.suggest_float('w_remove_object', 1, 5)
    config.w_add_obs = trial.suggest_float('w_add_obs', 15, 40)
    config.w_remove_obs = trial.suggest_float('w_remove_obs', 2, 8)
    config.w_replace = trial.suggest_float('w_replace', 25, 50)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.init_fill = trial.suggest_float('init_fill', 0.2, 0.5)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, GDAgent, 'objective_sim_sched_gd')
    
    return(-final_rewards_env) 

def objective_sim_rnd_sched_gd(trial):
    config = Configuration()

    config.init_attempts = trial.suggest_int('init attempts', 100, 150)
    config.w_add_object = trial.suggest_float('w_add_object', 20, 50)
    config.w_remove_object = trial.suggest_float('w_remove_object', 1, 5)
    config.w_add_obs = trial.suggest_float('w_add_obs', 15, 40)
    config.w_remove_obs = trial.suggest_float('w_remove_obs', 2, 8)
    config.w_replace = trial.suggest_float('w_replace', 25, 50)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.init_fill = trial.suggest_float('init_fill', 0.2, 0.5)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, ScheduleEnv, GDAgent, 'objective_sim_rnd_sched_gd', random=True)
    
    return(-final_rewards_env) 

def objective_dummy_otf_gd(trial):
    config = Configuration()

    config.w_empty_add = trial.suggest_float('w_empty_add', 0, 1e-2)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesDummy, OnTheFlyEnv, GDAgent, 'objective_dummy_otf_gd')
    
    return(-final_rewards_env) 

def objective_sim_otf_gd(trial):
    config = Configuration()

    config.w_empty_add = trial.suggest_float('w_empty_add', 0, 1e-2)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, GDAgent, 'objective_sim_otf_gd')
    
    return(-final_rewards_env) 

def objective_sim_rnd_otf_gd(trial):
    config = Configuration()

    config.w_empty_add = trial.suggest_float('w_empty_add', 0, 1e-2)
    config.max_iter = trial.suggest_int('max_iter', 50, 250)
    config.n_sub_iter = trial.suggest_int('n_sub_iter', 20, 50)
    config.add_attempts = trial.suggest_int('add attempts', 5, 20)
    config.state_length = 120 * trial.suggest_int('state times', 1, 5)
    config.n_objects = trial.suggest_int('n_obj', 1, 10)

    final_rewards_env = run(config, EphemeridesSimulated, OnTheFlyEnv, GDAgent, 'objective_sim_rnd_otf_gd', random=True)

    return(-final_rewards_env) 

if __name__ == '__main__':
    eph, env, agent = sys.argv[1], sys.argv[2], sys.argv[3]

    Ws = [0, 0.5, 1]
    for W in Ws:
        W_FF = W

        str_func = f'objective_{eph}_{env}_{agent}'
        func = globals()[str_func]

        study = optuna.create_study()
        study.optimize(objective_dummy_sched_dqn, n_trials=100)
