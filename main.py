from astropy.time import Time

from env_table import ObservationScheduleEnv
from gd_agent import GDAgent
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration

def main(observer=create_observer(), time=Time.now()):
    obj_dict, eph_dict = scraper(observer, time)


    config = Configuration()

    env = ObservationScheduleEnv(observer, time, obj_dict, eph_dict, config=config)
    agent = GDAgent(observer, eph_dict, time, env, config=config)
    agent.create_init_state()
    agent.gradient_descent()
    final_reward = agent.env.calculate_reward()
    print(f'Final fill factor: {final_reward}')


 

if __name__ == '__main__':
    main()