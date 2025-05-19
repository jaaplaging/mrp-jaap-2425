import numpy as np
import sys
sys.path.append('mrp-jaap-2425/')
import matplotlib.pyplot as plt
from rl_env_table import ObservationScheduleEnv
from mpc_scraper import scraper
from helper import create_observer
from param_config import Configuration
from astropy.time import Time
from keras.saving import load_model


config = Configuration()
obj_dict = {'A': {},
            'B': {},
            'C': {}}
eph_dict = {'A': {},
            'B': {},
            'C': {}}
    

env = ObservationScheduleEnv(create_observer(), Time.now(), obj_dict, eph_dict, config)

q_network = load_model('q_network.keras')

for i in range(1):
    prediction = q_network.predict([env.object_state.flatten().astype('float32').reshape(1,18), env.state.reshape(1,240)], verbose=0).reshape((3,240,5))
    masked = np.where(env.total_mask, prediction, np.min(prediction))
    #action = np.unravel_index(np.argmax(masked), masked.shape)
    #env.step(action[0],action[1],action[2],0)

    #print(action)

plt.imshow(masked[0,:,:], vmin=0)
plt.show()

