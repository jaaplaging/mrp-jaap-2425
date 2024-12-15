import numpy as np
import sys
sys.path.append('mrp-jaap-2425/')
import matplotlib.pyplot as plt
from rl_env_table import ObservationScheduleEnv
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
from astropy.time import Time


config = Configuration()
obj_dict = {'A': {},
            'B': {},
            'C': {}}
eph_dict = {'A': {},
            'B': {},
            'C': {}}
    

env = ObservationScheduleEnv(create_observer(), Time.now(), obj_dict, eph_dict, config)

env.step(2,19,0,0)
env.step(2,36,2,0)
env.step(2,86,4,0)
env.step(2,65,2,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,37,4,0)
env.step(2,99,4,0)
env.step(2,86,4,0)
env.step(2,87,4,0)
env.step(2,77,4,0)
env.step(2,96,2,0)
#env.step(2,87,4,0)
#env.step(2,29,4,0)
#env.step(2,45,4,0)
#env.step(2,37,4,0)

print(env.total_mask[2,:,4])
print(env.state)


