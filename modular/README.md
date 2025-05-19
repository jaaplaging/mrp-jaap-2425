# Modular
The modules in this directory contain classes that can all be used together for training agents and creating schedules.

## Use
Use main.py as described at the top of the module. agents.py contains the three agents classes that can be used and trained to create schedules. The three methods are DQN, PPO, and gradient descent. The environments.py module contains two environment classes, the schedule environment and the on-the-fly environment. The ephemerides.py module contains three different ephemerides classes, one for dummy NEOs, one for NEOs simulated using AMUSE and one for real ephemerides taken from the MPC NEO table. rewards.py contains a class that can be used to calculate rewards for the created schedules. 