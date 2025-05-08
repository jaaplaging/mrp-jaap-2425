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

    def __init__(self, env, eph_class):
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

        np.random.seed(self.env.config.seed)
        set_random_seed(self.env.config.seed)

    def __create_actor_network(self):
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
    
    # taken from https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
    @tf.function(reduce_retracing=True)
    def ppo_loss(self, old_logits, old_values, advantages, states, actions, returns):        
        @tf.function(reduce_retracing=True)
        def compute_policy_loss(logits, actions):
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
            # Value loss
            value_loss = tf.reduce_mean(tf.square(values - returns))
            return value_loss
        
        @tf.function(reduce_retracing=True)
        def get_advantages(returns, values):
            advantages = returns - values
            return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        @tf.function(reduce_retracing=True)
        def train_step(states, actions, returns, old_logits, old_values):
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
        for i in range(self.env.config.n_epochs):
            policy_loss, value_loss = train_step(states, actions, returns, old_logits, old_values)
        return(policy_loss, value_loss)
   
    def train(self):
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

                action = tf.random.categorical(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0, 0].numpy()
                next_state, reward, done = copy.deepcopy(self.env.step(action))

                states.append(copy.deepcopy(state))
                actions.append(action)
                rewards.append(reward)
                values.append(value.numpy()[0,0])

                state = copy.deepcopy(next_state)
                if done or step > 10 * self.env.config.steps:
                    returns_batch = []
                    discounted_sum = 0
                    for r in rewards[::-1]:
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns_batch.append(discounted_sum)
                    returns_batch.reverse()
                    
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype(self.state_type_a), np.array([states[i][1].flatten() for i in range(len(states))]).astype(self.state_type_b)]
                    actions = np.array(actions, dtype=np.int32)
                    values = tf.convert_to_tensor(values, dtype=tf.float32)
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

                    old_logits = self.actor_network_func(states) # self.actor_network.predict(states)

                    policy_loss, value_loss = self.ppo_loss(old_logits, values, returns_batch, states, actions, returns_batch)
                    final_rewards.append(rewards[-1])

                    break

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
            


class DQNAgent():

    def __init__(self, env, eph_class):
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
        if np.random.uniform() < self.exploration_rate:
            unif = tf.fill(logits.shape, 0.)
            action = tf.random.categorical(tf.where(self.env.create_mask(), unif, -np.inf), 1)[0, 0].numpy()     
        else:
            action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
        return(action)

    @tf.function(reduce_retracing=True)
    def update_network(self, states, actions, rewards, dones, next_states):
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

                action = self.__choose_action(logits)
                next_state, reward, done = copy.deepcopy(self.env.step(action))  # 0.00086 s, 1.2% of total

                self.replay_buffer.append([copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done])  # 0.00000062 s
                if len(self.replay_buffer) > self.env.config.memory_size:
                    self.replay_buffer.pop(0)
                if len(self.replay_buffer) > self.env.config.batch_size:
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
    

class GDAgent():

    def __init__(self, env, eph_class):
        """Initializes the gradient descent 'agent'

        Args:
            observer (astroplan.Observer): Observer representing the observatory
            eph_dict (dict): dictionary containing the ephemerides for the NEO's
            time (astropy.time.Time): date at which the observations take place
            env (env_table.ObservationScheduleEnv): schedule to be filled with observations
            config (param_config.Configuration, optional): hyperparameters to use. Defaults to Configuration().
        """        
        self.env = env
        self.__create_init_weights()
        self.action_weights = np.array([env.config.w_add_object,
                                        env.config.w_remove_object,
                                        env.config.w_add_obs,
                                        env.config.w_remove_obs,
                                        env.config.w_replace])
        self.eph_class = eph_class
        np.random.seed(self.env.config.seed)


    def create_init_state(self):
        ''' Tries to insert a certain number of observations for the initial state '''
        init_weights = copy.deepcopy(self.object_weights)

        def select_object(init_weights):
            ''' Selects an object based on a weighted random choice 
            
            Args: 
                init_weights (dict): dictionary containing weights of the objects to be sampled

            Returns:
                obj (str): key of the object that was sampled weighted randomly
            '''
            obj = np.random.choice(np.arange(0,len(init_weights)), p=np.array(init_weights)/np.sum(init_weights))
            return(obj)
        
        def observation_window(obj):
            ''' Finds the first and last time at which object can be observed 
            
            Args:
                object (int): ind of object in consideration

            Returns:
                start (astropy.time.Time): first possible time to observe the object
                end (astropy.time.TIme): last possible time to observe the object
            '''
            rise, set = self.env.object_state[obj,1], self.env.object_state[obj,2]
            start = np.max([rise, 0])
            end = np.min([set, self.env.config.state_length])
            return(start, end)
    
        def add_attempt_loop(object, start, end):
            ''' Loops until object is added or a maximum number of iterations is reached
            
            Args:
                object (str): key of object to be added
                start (astropy.time.Time): start of best airmass window
                end (astropy.time.Time): end of best airmass window

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
        total_weights = np.outer(self.object_weights, self.action_weights)
        total_weights = np.tile(total_weights[:,np.newaxis,:], (1,self.env.config.state_length,1))


        def sample_action(next_env, total_weights):
            total_weights_masked = np.where(next_env.total_mask, total_weights, 0)
            total_weights_masked_norm = total_weights_masked / np.sum(total_weights_masked)
            action = np.random.choice(np.arange(next_env.action_size), p=total_weights_masked_norm.flatten())
            return(action)


        # keep performing actions until a certain number of iterations is reached or a certain fill factor is reached
        while iteration < self.env.config.max_iter: # and self.env.calculate_reward() < 0.95:
            next_env = copy.deepcopy(self.env)
            for sub_iter in range(self.env.config.n_sub_iter):
                action = sample_action(next_env, total_weights)            
                _, _, _ = next_env.step(action)

            if next_env.reward.get_reward(next_env,next_env.empty_flag) > self.env.reward.get_reward(self.env, self.env.empty_flag):
                self.env = copy.deepcopy(next_env)
            iteration += 1

    def gradient_descent_on_the_fly(self):
        iteration = 0
        total_weights = np.concat([[self.env.config.w_empty_add],self.object_weights])


        def sample_action(next_env, total_weights):
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
            for sub_iter in range(self.env.config.n_sub_iter_otf):
                action = sample_action(next_env, total_weights)            
                _, _, done = next_env.step(action)
                if done:
                    break

            if next_env.reward.get_reward(next_env, next_env.empty_flag) > self.env.reward.get_reward(self.env, self.env.empty_flag):
                self.env = copy.deepcopy(next_env)
            iteration += 1
    
    def train(self):
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

        return(final_results_eval, other_results)