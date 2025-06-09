# Module containing all the agent classes compatible with the other classes from this directory. 

import sys
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, Flatten, CategoryEncoding
from keras import Model
from keras.models import save_model
import helper
import random
from time import perf_counter
from keras.utils import set_random_seed
from keras.losses import mean_squared_error
import copy
from environments import ScheduleEnv, OnTheFlyEnv

class PPOAgent():
    """Agent using Proximal Policy Optimization for training
    """
    def __init__(self, env, eph_class):
        """Initiates the PPO agent

        Args:
            env (environments class): Environment class from the environments.py file
            eph_class (ephemerides class): Ephemerides class from the ephemerides.py file
        """ 
        self.env = env
        self.state_space_a = (self.env.state_space_a)
        self.state_size_a = self.env.state_size_a
        self.state_type_a = self.env.state_type_a
        self.state_space_b = (self.env.state_space_b)
        self.state_size_b = self.env.state_size_b
        self.state_type_b = self.env.state_type_b
        self.action_space = (self.env.action_space)
        self.action_size = self.env.action_size
        self.learning_rate_actor = self.env.config.learning_rate
        self.learning_rate_critic = self.env.config.learning_rate
        self.discount_factor = self.env.config.discount_factor
        self.clip_ratio = self.env.config.clip_ratio
        self.batch_size = self.env.config.batch_size
        self.actor_network = self.__create_actor_network()
        self.critic_network = self.__create_critic_network()
        self.actor_network_func = tf.function(self.actor_network)
        self.critic_network_func = tf.function(self.critic_network)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_critic)
        self.eph_class = eph_class

        # Use the same seed in all modules in this directory per run
        np.random.seed(self.env.config.seed)
        set_random_seed(self.env.config.seed)

    def __create_actor_network(self):
        """Creates the policy network

        Returns:
            model (keras.Model): the policy network
        """
        input_x = Input(shape=(self.state_size_a,))
        input_y = Input(shape=(self.state_size_b,))

        x = Dense(self.env.config.layer_size, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size, activation='relu')(x)

        if self.state_type_b == 'str':
            y = CategoryEncoding(num_tokens=self.env.config.n_objects+1, output_mode='one_hot')(input_y)
            y = Flatten()(y)
        y = Dense(self.env.config.layer_size, activation='relu')(input_y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size, activation='relu')(combined)
        z = Dense(self.action_size, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    def __create_critic_network(self):
        """Creates the value network

        Returns:
            model (keras.Model): the value network        
        """
        input_x = Input(shape=(self.state_size_a,))
        input_y = Input(shape=(self.state_size_b,))

        x = Dense(self.env.config.layer_size_critic, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size_critic//4, activation='relu')(x)

        if self.state_type_b == 'str':
            y = CategoryEncoding(num_tokens=self.env.config.n_objects+1, output_mode='one_hot')(input_y)
            y = Flatten()(y)
        y = Dense(self.env.config.layer_size_critic, activation='relu')(input_y)
        y = Dense(self.env.config.layer_size_critic//4, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size_critic, activation='relu')(combined)
        z = Dense(1, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    # taken from https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a and altered
    @tf.function(reduce_retracing=True)
    def ppo_loss(self, old_logits, old_values, states, actions, returns):        
        """Function that calculates the loss of the policy and the value networks for a given episode. 
        Remove @tf.function decorator to debug properly.

        Args:
            old_logits (tf.Tensor): Logits returned from the policy network before applying gradients
            old_values (tf.Tensor): Value returned by the value network before applyting gradients
            states (tf.Tensor): States corresponding to the generated logits and values
            actions (tf.Tensor): Actions taken corresponding to the old logits and states
            returns (tf.Tensor): Rewards per taken action with the discounted sum of future rewards added.

        Returns:
            value_loss (tf.constant): Value loss of the episode
            policy_loss (tf.constant): Policy loss of the episode
        """
        @tf.function(reduce_retracing=True)
        def compute_policy_loss(logits, actions):
            """Computes the policy loss of the episode

            Args:
                logits (tf.Tensor): Logits returned from the current policy network given the episode's states
                actions (tf.Tensor): Actions taken during the episode

            Returns:
                policy_loss (tf.constant): Policy loss of the episode
                action probs (tf.Tensor): current policy value of actions taken
                old_action_probs (tf.Tensor): old policy value of actions taken
            """
            actions_onehot = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            policy = tf.nn.softmax(logits)
            action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
            old_policy = tf.nn.softmax(old_logits)
            old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

            # Policy loss
            ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Entropy bonus (optional)
            entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

            policy_loss = policy_loss - 0.01 * entropy_bonus
            return policy_loss, action_probs, old_action_probs

        @tf.function(reduce_retracing=True)
        def compute_value_loss(values, returns):
            """Computes the value loss of the given episode

            Args:
                values (tf.Tensor): Values returned by the current value network for the episode
                returns (tf.Tensor): Discounted reward sums for the episode

            Returns:
                value_loss (tf.constant): Value loss for the current value network
            """
            # Value loss
            value_loss = tf.reduce_mean(tf.square(values - returns))
            return value_loss
        
        @tf.function(reduce_retracing=True)
        def get_advantages(returns, values):
            """Calculates the normalized advantages from the returns and values.

            Args:
                returns (tf.Tensor): Discounted rewards of the episode
                values (tf.Tensor): Values of the episode

            Returns:
                tf.Tensor: Normalized advantages of the episode.
            """
            advantages = returns - values
            return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        @tf.function(reduce_retracing=True)
        def train_step(states, actions, returns, old_logits, old_values):
            """Performs a train step

            Args:
                states (tf.Tensor): States of the episode
                actions (tf.Tensor): Actions of the episode
                returns (tf.Tensor): Discounted rewards of the episode
                old_logits (tf.Tensor): Initial policy network's logit values per step
                old_values (tf.Tensor): Initial value network's values per step
            """
            with tf.GradientTape() as actor_tape:
                logits = self.actor_network(states)
                policy_loss, action_probs, old_action_probs = compute_policy_loss(logits, actions)
            with tf.GradientTape() as critic_tape:
                values = self.critic_network(states)
                value_loss = compute_value_loss(values, returns)
            policy_gradients = actor_tape.gradient(policy_loss, self.actor_network.trainable_variables)
            value_gradients = critic_tape.gradient(value_loss, self.critic_network.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(policy_gradients, self.actor_network.trainable_variables))
            self.optimizer_critic.apply_gradients(zip(value_gradients, self.critic_network.trainable_variables))
            return(policy_loss, value_loss)

        advantages = get_advantages(returns, old_values)
        # Perform n_epochs train steps
        for i in range(self.env.config.n_epochs):
            policy_loss, value_loss = train_step(states, actions, returns, old_logits, old_values)
        return(policy_loss, value_loss)
   
    def train(self):
        """Trains the network using PPO for a certain number of episodes of a certain number of steps. Keeps track of 
        rewards for later analysis. Performs an evaluation episode every evaluation_interval episodes
        to evaluate the performance of the agent. 

        Returns:
            final_rewards_eval (list): list of final rewards form the evaluation runs
            other_results (list): list of final rewards from the training runs and all terms of the 
            evaluation run rewards.
        """
        final_rewards = []
        final_rewards_eval = []

        ff_rewards = []
        time_from_peak_rewards = []
        airmass_rewards = []
        mag_rewards = []
        motion_rewards = []
        n_obs_rewards = []
        time_gap_rewards = []
        all_viewed_rewards = []

        for episode in range(self.env.config.episodes):
            
            states, actions, rewards, values, returns = [],[],[],[],[]
            state = copy.deepcopy(self.env.reset(self.eph_class))

            done = 0

            step = 0
            while not done:
                step += 1
                state = [state[0].flatten().astype('float32').reshape(1, self.state_size_a), state[1].flatten().astype('float32').reshape(1, self.state_size_b)]

                logits = self.actor_network_func(state)
                value = self.critic_network_func(state)

                # Determine the action taken using on-policy stochastic methods.
                action = tf.random.categorical(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0, 0].numpy()
                # Perform a step given the selected action
                next_state, reward, done = copy.deepcopy(self.env.step(action))

                states.append(copy.deepcopy(state))
                actions.append(action)
                rewards.append(reward)
                values.append(value.numpy()[0,0])

                state = copy.deepcopy(next_state)
                if done or step > 10 * self.env.config.steps:
                    # Calculate the discounted rewards
                    returns_batch = []
                    discounted_sum = 0
                    for r in rewards[::-1]:
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns_batch.append(discounted_sum)
                    returns_batch.reverse()
                    
                    # Convert the states, actions, values and returns to tensors
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype(self.state_type_a), np.array([states[i][1].flatten() for i in range(len(states))]).astype(self.state_type_b)]
                    actions = np.array(actions, dtype=np.int32)
                    values = tf.convert_to_tensor(values, dtype=tf.float32)
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

                    old_logits = self.actor_network_func(states) # self.actor_network.predict(states)

                    # Calculate the loss and perform train steps on the networks
                    policy_loss, value_loss = self.ppo_loss(old_logits, values, states, actions, returns_batch)
                    final_rewards.append(rewards[-1])

                    break

            # Perform an evaluation episode every evaluation_interval episodes
            if episode % self.env.config.evaluation_interval == 0:
                state = copy.deepcopy(self.env.reset(self.eph_class))

                done = 0

                step = 0
                while not done:
                    step += 1
                    state = [state[0].flatten().astype(self.state_type_a).reshape(1, self.state_size_a), state[1].flatten().astype(self.state_type_b).reshape(1, self.state_size_b)]
                    logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
                    next_state, reward, done = copy.deepcopy(self.env.step(action))

                    state = copy.deepcopy(next_state)
                    if done or step > 10 * self.env.config.steps:
                        final_rewards_eval.append(reward)
                        ff_rewards.append(self.env.reward.fill_factor_reward)
                        time_from_peak_rewards.append(self.env.reward.r_time_from_peak)
                        airmass_rewards.append(self.env.reward.r_airmass)
                        mag_rewards.append(self.env.reward.r_magnitude)
                        motion_rewards.append(self.env.reward.r_motion)
                        n_obs_rewards.append(self.env.reward.r_n_observations)
                        time_gap_rewards.append(self.env.reward.r_time_gap)
                        all_viewed_rewards.append(self.env.reward.r_all_viewed)
                        break


        other_results = [final_rewards,
                         ff_rewards,
                         time_from_peak_rewards,
                         airmass_rewards,
                         mag_rewards,
                         motion_rewards,
                         n_obs_rewards,
                         time_gap_rewards,
                         all_viewed_rewards]

        return(final_rewards_eval, other_results)
    
    def evaluate(self):
        """Runs the agent without training in order to evaluate the results
        """
        r_total = []
        r_ff = []
        r_peak = []
        r_airmass = []
        r_mag = []
        r_motion = []
        r_n_obs = []
        r_gap = []
        r_all = []
        actions_taken = []

        state = copy.deepcopy(self.env.reset(self.eph_class))
        
        done = 0

        step = 0
        while not done:
            step += 1
            state = [state[0].flatten().astype(self.state_type_a).reshape(1, self.state_size_a), state[1].flatten().astype(self.state_type_b).reshape(1, self.state_size_b)]
            logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)

            # off-policy argmax instead of on-policy stochastic
            action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
            next_state, reward, done = copy.deepcopy(self.env.step(action))

            r_total.append(reward)
            r_ff.append(self.env.reward.fill_factor_reward)
            r_peak.append(self.env.reward.r_time_from_peak)
            r_airmass.append(self.env.reward.r_airmass)
            r_mag.append(self.env.reward.r_magnitude)
            r_motion.append(self.env.reward.r_motion)
            r_n_obs.append(self.env.reward.r_n_observations)
            r_gap.append(self.env.reward.r_time_gap)
            r_all.append(self.env.reward.r_all_viewed)
            actions_taken.append(action)

            state = copy.deepcopy(next_state)
            if done or step > 10 * self.env.config.steps:
                break 
        
        rewards = [r_total,
                   r_ff,
                   r_peak,
                   r_airmass,
                   r_mag,
                   r_motion,
                   r_n_obs,
                   r_gap,
                   r_all]
        return(rewards, actions_taken, self.env.schedule)
            


class DQNAgent():
    """Agent using Deep Q-Leaning for training
    """
    def __init__(self, env, eph_class):
        """Initializes the DQN agent

        Args:
            env (environments class): Any class from the environments.py module
            eph_class (ephemerides class): Any class from the ephemerides.py module
        """
        self.env = env
        self.state_space_a = (self.env.state_space_a)
        self.state_size_a = self.env.state_size_a
        self.state_type_a = self.env.state_type_a
        self.state_space_b = (self.env.state_space_b)
        self.state_size_b = self.env.state_size_b
        self.state_type_b = self.env.state_type_b
        self.action_space = (self.env.action_space)
        self.action_size = self.env.action_size
        self.learning_rate = self.env.config.learning_rate
        self.discount_factor = self.env.config.discount_factor
        self.exploration_rate = self.env.config.exploration_rate
        self.exploration_decay = self.env.config.exploration_decay
        self.batch_size = self.env.config.batch_size
        self.target_update_freq = self.env.config.target_update_freq
        self.replay_buffer = []
        self.q_network = self.__create_q_network()
        self.q_network_func = tf.function(self.q_network)
        self.target_network = self.__create_q_network()#   .set_weights(self.q_network.get_weights())
        self.target_network.set_weights(self.q_network.get_weights())
        self.target_network_func = tf.function(self.target_network)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.eph_class = eph_class

        np.random.seed(self.env.config.seed)
        set_random_seed(self.env.config.seed)

    def __create_q_network(self):
        """Create the Q network

        Returns:
            model (keras.Model): The created Q network
        """
        input_x = Input(shape=(self.state_size_a,))
        input_y = Input(shape=(self.state_size_b,))

        x = Dense(self.env.config.layer_size, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size, activation='relu')(x)

        if self.state_type_b == 'str':
            y = CategoryEncoding(num_tokens=self.env.config.n_objects+1, output_mode='one_hot')(input_y)
            y = Flatten()(y)
        y = Dense(self.env.config.layer_size, activation='relu')(input_y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size, activation='relu')(combined)
        z = Dense(self.action_size, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)

    def __choose_action(self, logits):
        """Chooses an action given the logits returned by the Q network. Randomly chooses either a random action
        or the action corresponding to the highest logit value based on the current exploration rate.

        Args:
            logits (tf.Tensor): Logits returned by the Q network

        Returns:
            action (tf.constant): Chosen action
        """
        if np.random.uniform() < self.exploration_rate:
            unif = tf.fill(logits.shape, 0.)
            action = tf.random.categorical(tf.where(self.env.create_mask(), unif, -np.inf), 1)[0, 0].numpy()     
        else:
            action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
        return(action)

    @tf.function(reduce_retracing=True)
    def update_network(self, states, actions, rewards, dones, next_states):
        """Updates the network based on the taken actions, states and rewards from a minibatch sampled randomly
        from the replay buffer

        Args:
            states (tf.Tensor): States of the minibatch used for the network update
            actions (tf.Tensor): Actions of the minibatch used for the network update
            rewards (tf.Tensor): Rewards of the minibatch used for the network update
            dones (tf.Tensor): Indicates per minibatch state if it was a final state
            next_states (tf.Tensor): Next states returned by the environment after taking a step
        """
        actions_onehot = tf.one_hot(actions, self.action_size)
        predict_next_values = self.target_network_func(next_states)
        target_value = rewards + (1 - dones) * self.discount_factor * tf.math.reduce_max(predict_next_values, axis=1)
        with tf.GradientTape() as tape:
            predict_values = self.q_network_func(states)
            updated_q_values = predict_values * (1-actions_onehot) + tf.expand_dims(target_value, axis=-1) * actions_onehot
            loss = tf.reduce_mean(mean_squared_error(updated_q_values, predict_values))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

    def train(self):
        """Trains the network for a given number of episodes of a given number of steps. Keeps track
        of rewards. Updates the network using minibatches from a replay buffer. Performs an evaluation episode every 
        evaluation_interval episodes.

        Returns:
            final_rewards_eval (list): Final rewards of evaluation episodes. 
            other_rewards (list): List containing final rewards of the training episodes and all terms
            of the rewards from the evaluation runs.

        """
        final_rewards = []
        final_rewards_eval = []

        ff_rewards = []
        time_from_peak_rewards = []
        airmass_rewards = []
        mag_rewards = []
        motion_rewards = []
        n_obs_rewards = []
        time_gap_rewards = []
        all_viewed_rewards = []
        
        for episode in range(self.env.config.episodes):

            states, actions, rewards, next_states, dones = [],[],[],[],[]
            state = copy.deepcopy(self.env.reset(self.eph_class))

            done = 0

            step = 0
            while not done:
                step += 1
                state = [state[0].flatten().astype(self.state_type_a).reshape(1, self.state_size_a), state[1].flatten().astype(self.state_type_b).reshape(1, self.state_size_b)]
                logits = self.q_network_func(state)

                # select an action and take it
                action = self.__choose_action(logits)
                next_state, reward, done = copy.deepcopy(self.env.step(action))  # 0.00086 s, 1.2% of total

                # add the state, action, reward, next state and done status to the replay buffer
                self.replay_buffer.append([copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done])  # 0.00000062 s
                if len(self.replay_buffer) > self.env.config.memory_size:
                    self.replay_buffer.pop(0)
                if len(self.replay_buffer) > self.env.config.batch_size:
                    # Randomly sample a minibatch from the buffer, convert to tensors and update the network
                    minibatch = random.sample(self.replay_buffer, self.batch_size)  # 0.000029 s, 0.04% of total
                    states = [np.vstack([minibatch[i][0][0].flatten() for i in range(len(minibatch))]).astype(self.state_type_a), np.array([minibatch[i][0][1].flatten() for i in range(len(minibatch))]).astype(self.state_type_b)]
                    actions = tf.convert_to_tensor([minibatch[i][1] for i in range(len(minibatch))], dtype=tf.int32) 
                    rewards = tf.convert_to_tensor([minibatch[i][2] for i in range(len(minibatch))], dtype=tf.float32)
                    next_states = [np.vstack([minibatch[i][3][0].flatten() for i in range(len(minibatch))]).astype(self.state_type_a), np.array([minibatch[i][3][1].flatten() for i in range(len(minibatch))]).astype(self.state_type_b)]
                    dones = tf.convert_to_tensor([minibatch[i][4] for i in range(len(minibatch))], dtype=tf.float32)     
                    
                    self.update_network(states, actions, rewards, dones, next_states)  # 0.16 s, 222% of total
                state = copy.deepcopy(next_state)

                if done or step > 10 * self.env.config.steps:
                    final_rewards.append(reward)
                    break

            # Perform an evaluation episode
            if episode % self.env.config.evaluation_interval == 0:
                state = copy.deepcopy(self.env.reset(self.eph_class))
                
                done = 0

                step = 0
                while not done:
                    step += 1
                    state = [state[0].flatten().astype(self.state_type_a).reshape(1, self.state_size_a), state[1].flatten().astype(self.state_type_b).reshape(1, self.state_size_b)]
                    logits = self.q_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
                    next_state, reward, done = copy.deepcopy(self.env.step(action))

                    state = copy.deepcopy(next_state)
                    if done or step > 10 * self.env.config.steps:
                        final_rewards_eval.append(reward)
                        ff_rewards.append(self.env.reward.fill_factor_reward)
                        time_from_peak_rewards.append(self.env.reward.r_time_from_peak)
                        airmass_rewards.append(self.env.reward.r_airmass)
                        mag_rewards.append(self.env.reward.r_magnitude)
                        motion_rewards.append(self.env.reward.r_motion)
                        n_obs_rewards.append(self.env.reward.r_n_observations)
                        time_gap_rewards.append(self.env.reward.r_time_gap)
                        all_viewed_rewards.append(self.env.reward.r_all_viewed)
                        break 

            # Decrease the exploration rate and update the target network weights
            self.exploration_rate *= self.exploration_decay
            if episode % self.target_update_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())


        other_results = [final_rewards,
                         ff_rewards,
                         time_from_peak_rewards,
                         airmass_rewards,
                         mag_rewards,
                         motion_rewards,
                         n_obs_rewards,
                         time_gap_rewards,
                         all_viewed_rewards]

        return(final_rewards_eval, other_results)
    
    def evaluate(self):
        """Runs the agent without training in order to evaluate the results
        """
        r_total = []
        r_ff = []
        r_peak = []
        r_airmass = []
        r_mag = []
        r_motion = []
        r_n_obs = []
        r_gap = []
        r_all = []
        actions_taken = []

        state = copy.deepcopy(self.env.reset(self.eph_class))
        
        done = 0

        step = 0
        while not done:
            step += 1
            state = [state[0].flatten().astype(self.state_type_a).reshape(1, self.state_size_a), state[1].flatten().astype(self.state_type_b).reshape(1, self.state_size_b)]
            logits = self.q_network_func(state) # self.actor_network.predict(state, verbose=0)

            # off-policy argmax instead of on-policy stochastic
            action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
            next_state, reward, done = copy.deepcopy(self.env.step(action))

            r_total.append(reward)
            r_ff.append(self.env.reward.fill_factor_reward)
            r_peak.append(self.env.reward.r_time_from_peak)
            r_airmass.append(self.env.reward.r_airmass)
            r_mag.append(self.env.reward.r_magnitude)
            r_motion.append(self.env.reward.r_motion)
            r_n_obs.append(self.env.reward.r_n_observations)
            r_gap.append(self.env.reward.r_time_gap)
            r_all.append(self.env.reward.r_all_viewed)
            actions_taken.append(action)

            state = copy.deepcopy(next_state)
            if done or step > 10 * self.env.config.steps:
                break 
        
        rewards = [r_total,
                   r_ff,
                   r_peak,
                   r_airmass,
                   r_mag,
                   r_motion,
                   r_n_obs,
                   r_gap,
                   r_all]
        return(rewards, actions_taken, self.env.schedule)
    

class GDAgent():

    def __init__(self, env, eph_class, eval=False):
        """Initializes the gradient descent 'agent'

        Args:
            env (environments class): Any class from the environments.py module.
            eph_class (ephemerides class): Any class from the ephemerides.py module
        """        
        self.env = env
        self.__create_init_weights()
        self.action_weights = np.array([env.config.w_add_object,
                                        env.config.w_remove_object,
                                        env.config.w_add_obs,
                                        env.config.w_remove_obs,
                                        env.config.w_replace])
        self.eph_class = eph_class
        self.eval = eval
        np.random.seed(self.env.config.seed)


    def create_init_state(self):
        ''' Tries to insert a certain number of observations for the initial state '''
        init_weights = copy.deepcopy(self.object_weights)

        def select_object(init_weights):
            ''' Selects an object based on a weighted random choice 
            
            Args: 
                init_weights (np.ndarray): Array containing weights of the objects to be sampled

            Returns:
                obj (stint): index of the object that was sampled weighted randomly
            '''
            obj = np.random.choice(np.arange(0,len(init_weights)), p=np.array(init_weights)/np.sum(init_weights))
            return(obj)
        
        def observation_window(obj):
            ''' Finds the first and last time at which object can be observed 
            
            Args:
                obj (int): ind of object in consideration

            Returns:
                start (int): first possible index to observe the object
                end (int): last possible index to observe the object
            '''
            rise, set = self.env.object_state[obj,1], self.env.object_state[obj,2]
            start = np.max([rise, 0])
            end = np.min([set, self.env.config.state_length])
            return(start, end)
    
        def add_attempt_loop(object, start, end):
            ''' Loops until object is added or a maximum number of iterations is reached
            
            Args:
                obj (ind): index of the object to be added
                start (int): starting index of visibility of the object
                end (int): end index of visibility of the object

            Returns:
                success (bool): True if the object was successfully added
            '''
            success = False
            attempt = 0
            while not success and attempt < self.env.config.init_attempts:
                if start >= end:
                    return(success)
                add_time = np.random.randint(start, end)
                if self.env.total_mask[object, add_time, 0]:
                    self.env.add_object(object, add_time)
                    success = True
                attempt += 1
            return(success)

        attempt = 0
        while self.env.reward.fill_factor_reward < self.env.config.init_fill*2-1 and len(init_weights) > 0 and attempt < self.env.config.init_attempts:
            object = select_object(init_weights)
            start, end = observation_window(object)
            success = add_attempt_loop(object, start, end)
            attempt += 1
            if success:
                init_weights = np.delete(init_weights, object)


    def __create_init_weights(self):
        ''' Creates the initial weights for the objects '''

        self.object_weights = np.array([0. for i in range(self.env.config.n_objects)])

        # Calculate the weights of the objects based on the airmass, motion and magnitude
        for ind, obj in enumerate(self.env.object_state):
            if obj[8] == 0:
                self.object_weights[ind] = 0
            else:
                avg_airmass, avg_motion, avg_magnitude = obj[3], obj[5], obj[4]

                weight = (10 * (1/avg_airmass) * (1+np.log10(avg_motion+1)) * avg_magnitude)
                self.object_weights[ind] = weight

        self.object_weights /= np.sum(self.object_weights)
    
    def gradient_descent_schedule(self):
        ''' Fills the schedule as much as possible using gradient descent '''
        iteration = 0
        # Determine the total weight values based on the object weights and the action weights calculated before
        total_weights = np.outer(self.object_weights, self.action_weights)
        total_weights = np.tile(total_weights[:,np.newaxis,:], (1,self.env.config.state_length,1))

        if self.eval:
            self.r_total = []
            self.r_ff = []
            self.r_peak = []
            self.r_airmass = []
            self.r_mag = []
            self.r_motion = []
            self.r_n_obs = []
            self.r_gap = []
            self.r_all = []
            self.actions_taken = []

        def sample_action(next_env, total_weights):
            """Samples an action to take based on the weights and the masks of the environment

            Args:
                next_env (environments.ScheduleEnv): Environment for which to sample an action
                total_weights (np.ndarray): Combined action and object weights

            Returns:
                action (int): Sampled action
            """
            total_weights_masked = np.where(next_env.total_mask, total_weights, 0)
            total_weights_masked_norm = total_weights_masked / np.sum(total_weights_masked)
            action = np.random.choice(np.arange(next_env.action_size), p=total_weights_masked_norm.flatten())
            return(action)


        # keep performing actions until a certain number of iterations is reached or a certain fill factor is reached
        while iteration < self.env.config.max_iter: # and self.env.calculate_reward() < 0.95:
            next_env = copy.deepcopy(self.env)
            # Perform n_sub_iter sampled steps
            if self.eval:
                next_r_total, next_r_ff, next_r_peak, next_r_airmass, next_r_mag, next_r_motion, next_r_n_obs, next_r_gap, next_r_all, next_actions_taken = [],[],[],[],[],[],[],[],[],[]
            for sub_iter in range(self.env.config.n_sub_iter):
                action = sample_action(next_env, total_weights)            
                _, reward, _ = next_env.step(action)
                if self.eval:
                    next_r_total.append(reward)
                    next_r_ff.append(self.env.reward.fill_factor_reward)
                    next_r_peak.append(self.env.reward.r_time_from_peak)
                    next_r_airmass.append(self.env.reward.r_airmass)
                    next_r_mag.append(self.env.reward.r_magnitude)
                    next_r_motion.append(self.env.reward.r_motion)
                    next_r_n_obs.append(self.env.reward.r_n_observations)
                    next_r_gap.append(self.env.reward.r_time_gap)
                    next_r_all.append(self.env.reward.r_all_viewed)
                    next_actions_taken.append(action)


            # Evaluate the environment after taking n_sub_iter random steps
            if next_env.reward.get_reward(next_env,next_env.empty_flag) > self.env.reward.get_reward(self.env, self.env.empty_flag):
                self.env = copy.deepcopy(next_env)
                if self.eval:
                    self.r_total.extend(next_r_total)
                    self.r_ff.extend(next_r_ff)
                    self.r_peak.extend(next_r_peak)
                    self.r_airmass.extend(next_r_airmass)
                    self.r_mag.extend(next_r_mag)
                    self.r_motion.extend(next_r_motion)
                    self.r_n_obs.extend(next_r_n_obs)
                    self.r_gap.extend(next_r_gap)
                    self.r_all.extend(next_r_all)
                    self.actions_taken.extend(next_actions_taken)
            iteration += 1

    def gradient_descent_on_the_fly(self):
        """Fills the schedule using gradient descent for the on_the_fly environment
        """
        iteration = 0
        total_weights = np.concat([[self.env.config.w_empty_add],self.object_weights])

        if self.eval:
            self.r_total = []
            self.r_ff = []
            self.r_peak = []
            self.r_airmass = []
            self.r_mag = []
            self.r_motion = []
            self.r_n_obs = []
            self.r_gap = []
            self.r_all = []
            self.actions_taken = []

        def sample_action(next_env, total_weights):
            """Samples an action weighted randomly

            Args:
                next_env (environments.OnTheFlyEnv): Environment for which to sample an action
                total_weights (np.ndarray): Weights per action for the sampling
            """
            total_weights_masked = np.where(next_env.create_mask(), total_weights, 0)
            if np.sum(total_weights_masked) > 0:
                total_weights_masked_norm = total_weights_masked / np.sum(total_weights_masked)
                action = np.random.choice(np.arange(next_env.action_size), p=total_weights_masked_norm.flatten())
            else:
                action = -1
            return(action)


        # keep performing actions until a certain number of iterations is reached or a certain fill factor is reached
        while iteration < self.env.config.max_iter: # and self.env.calculate_reward() < 0.95:
            next_env = copy.deepcopy(self.env)
            # For n_sub_iter iterations, sample and take an action
            if self.eval:
                next_r_total, next_r_ff, next_r_peak, next_r_airmass, next_r_mag, next_r_motion, next_r_n_obs, next_r_gap, next_r_all, next_actions_taken = [],[],[],[],[],[],[],[],[],[]
            for sub_iter in range(self.env.config.n_sub_iter_otf):
                action = sample_action(next_env, total_weights)            
                _, reward, done = next_env.step(action)
                if self.eval:
                    next_r_total.append(reward)
                    next_r_ff.append(self.env.reward.fill_factor_reward)
                    next_r_peak.append(self.env.reward.r_time_from_peak)
                    next_r_airmass.append(self.env.reward.r_airmass)
                    next_r_mag.append(self.env.reward.r_magnitude)
                    next_r_motion.append(self.env.reward.r_motion)
                    next_r_n_obs.append(self.env.reward.r_n_observations)
                    next_r_gap.append(self.env.reward.r_time_gap)
                    next_r_all.append(self.env.reward.r_all_viewed)
                    next_actions_taken.append(action)
                if done:
                    break

            # Evaluate the environment after taking n_sub_iter sampled actions
            if next_env.reward.get_reward(next_env, next_env.empty_flag) > self.env.reward.get_reward(self.env, self.env.empty_flag):
                self.env = copy.deepcopy(next_env)
                if self.eval:
                    self.r_total.extend(next_r_total)
                    self.r_ff.extend(next_r_ff)
                    self.r_peak.extend(next_r_peak)
                    self.r_airmass.extend(next_r_airmass)
                    self.r_mag.extend(next_r_mag)
                    self.r_motion.extend(next_r_motion)
                    self.r_n_obs.extend(next_r_n_obs)
                    self.r_gap.extend(next_r_gap)
                    self.r_all.extend(next_r_all)
                    self.actions_taken.extend(next_actions_taken)
            iteration += 1
    
    def train(self):
        """Starts the gradient descent. Chooses which gradient descent function based on which environment is used.
        
        Returns:
            final_results_eval (list): Rewards of final states
            other_results (list): Values of all terms of the reward function
        """
        if type(self.env) == ScheduleEnv:
            self.create_init_state()
            self.gradient_descent_schedule()
        elif type(self.env) == OnTheFlyEnv:
            self.gradient_descent_on_the_fly()

        final_results_eval = self.env.reward.get_reward(self.env, self.env.empty_flag)



        other_results = [self.env.reward.get_reward(self.env, self.env.empty_flag),
                         self.env.reward.fill_factor_reward,
                         self.env.reward.r_time_from_peak,
                         self.env.reward.r_airmass,
                         self.env.reward.r_magnitude,
                         self.env.reward.r_motion,
                         self.env.reward.r_n_observations,
                         self.env.reward.r_time_gap,
                         self.env.reward.r_all_viewed]
        if not self.eval:
            return(final_results_eval, other_results)
        else:
            rewards = [self.r_total,
                       self.r_ff,
                       self.r_peak,
                       self.r_airmass,
                       self.r_mag,
                       self.r_motion,
                       self.r_n_obs,
                       self.r_gap,
                       self.r_all]
            return(rewards, self.actions_taken, self.env.schedule)
    
