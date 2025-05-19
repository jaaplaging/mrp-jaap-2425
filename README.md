# mrp-jaap-2425

Repository for the Master's Research Project of Jaap Laging: Telescope and Scheduling Automation with Reinforcement Learning. This code in this repo uses multiple different RL methods to create observation schedules for Near-Earth Objects. These methods include DQN and PPO. Currently, the project uses Tensorflow. 

NOTE: This project is still a work-in-progress.

## Background
The project stems from an idea of using automated algorithms to create observation schedules for Near-Earth Object. The original idea was to use a 'gradient descent'-like method to iteratively add and move observations to find an optimal schedule. This idea has been expanded upon by using RL methods such as DQN and PPO instead of random gradient descent. 

## Use
The modules under 'archive/' ('GD', 'PPO', 'RL' and 'on_the_fly') were initial experiments to attempt to solve the problem and are now mostly out of use. 'modular' contains the current version of the module that combines all different methods. Run using main.py. Instructions for use can be found in the comments at the start of the file.
