import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, Flatten, CategoryEncoding
from keras import Model
from keras.models import save_model
import helper
from param_config import Configuration
import random
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import copy

config = Configuration()

class PPOAgent():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (self.env.config.n_objects * 6)
        self.state_space_b = (self.env.config.state_length)
        self.action_space = (self.env.config.n_objects, self.env.config.state_length, 5)
        self.action_size = self.env.config.n_objects * self.env.config.state_length * 5
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


    def __create_actor_network(self):
        input_x = Input(shape=(self.env.config.n_objects * 6,))
        input_y = Input(shape=(self.env.config.state_length,))

        x = Dense(self.env.config.layer_size, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size, activation='relu')(x)

        y = CategoryEncoding(num_tokens=self.env.config.n_objects+1, output_mode='one_hot')(input_y)
        y = Flatten()(y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size, activation='relu')(combined)
        z = Dense(self.env.config.n_objects*self.env.config.state_length*5, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    def __create_critic_network(self):
        input_x = Input(shape=(self.env.config.n_objects * 6,))
        input_y = Input(shape=(self.env.config.state_length,))

        x = Dense(8, activation='relu')(input_x)
        x = Dense(4, activation='relu')(x)

        y = CategoryEncoding(num_tokens=2, output_mode='one_hot')(input_y)
        y = Flatten()(y)
        y = Dense(32, activation='relu')(y)
        y = Dense(4, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(8, activation='relu')(combined)
        z = Dense(1, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)

    # taken from https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
    @tf.function
    def ppo_loss(self, old_logits, old_values, advantages, states, actions, returns):        
        @tf.function
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

        @tf.function
        def compute_value_loss(values, returns):
            # Value loss
            value_loss = tf.reduce_mean(tf.square(values - returns))
            return value_loss
        
        @tf.function
        def get_advantages(returns, values):
            advantages = returns - values
            return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        @tf.function
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
            return(policy_loss, value_loss, action_probs, old_action_probs)

        advantages = get_advantages(returns, old_values)
        for i in range(self.env.config.n_epochs):
            policy_loss, value_loss, action_probs, old_action_probs = train_step(states, actions, returns, old_logits, old_values)
        return(policy_loss, value_loss, action_probs, old_action_probs, advantages)
    
    def train(self):
        times = [0 for i in range(20)]

        f_factors_final = []
        f_factors_max = []
        f_factors_mean = []
        rewards_mean = []
        actions_taken_total = [[],[],[],[],[]]
        actions_logits_mean = [[],[],[],[],[]]
        steps_taken = []

        policy_losses = []
        value_losses = []

        f_factors_eval = []
        
        for episode in range(self.env.config.episodes):
            
            
            try:
                f_factors_final.append(f_factors[-1])
                f_factors_max.append(np.max(f_factors))
                f_factors_mean.append(np.mean(f_factors))
                rewards_mean.append(np.mean(rewards))
                for ind, action in enumerate(self.actions_taken):
                    actions_taken_total[ind].append(action)

            except:
                pass  

            states, actions, rewards, values, returns = [],[],[],[],[]

            self.env.reset()
            state = [copy.deepcopy(self.env.object_state_norm), copy.deepcopy(self.env.state)]

            f_factors = []
            reward = -1
            reward_per_action = [0,0,0,0,0]
            taken_actions = [0,0,0,0,0]

            self.actions_taken = [0,0,0,0,0]

            for step in range(self.env.config.steps):
                state = [state[0].flatten().astype('float32').reshape(1, self.env.config.n_objects * 6), state[1].reshape(1, self.env.config.state_length)]
                logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)
                value = self.critic_network_func(state)  # self.critic_network.predict(state, verbose=0)

                action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0, 0].numpy()
                action_ind = np.unravel_index(action, (self.env.config.n_objects, self.env.config.state_length, 5))
                self.actions_taken[action_ind[2]] += 1
                previous_reward = np.sum(self.env.rewards)/len(self.env.rewards)
                next_state, reward, done = copy.deepcopy(self.env.step(action_ind[0], action_ind[1], action_ind[2], step))

                logits_reshaped = logits.numpy().reshape((self.env.config.n_objects,self.env.config.state_length,5))
                for a in range(5):
                    actions_logits_mean[a].append(np.max(logits_reshaped[:,:,a]))

                f_factors.append(np.sum(self.env.rewards)/len(self.env.rewards))
                reward_per_action[action_ind[2]] += reward
                taken_actions[action_ind[2]] += 1

                states.append(copy.deepcopy(state))
                actions.append(action)
                rewards.append(reward)
                values.append(value.numpy()[0,0])

                state = copy.deepcopy(next_state)
                if done:
                    returns_batch = []
                    discounted_sum = 0
                    for r in rewards[::-1]:
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns_batch.append(discounted_sum)
                    returns_batch.reverse()
                    
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype('float32'), np.array([states[i][1].flatten() for i in range(len(states))])]
                    actions = np.array(actions, dtype=np.int32)
                    values = tf.convert_to_tensor(values, dtype=tf.float32)
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

                    old_logits = self.actor_network_func(states) # self.actor_network.predict(states)

                    policy_loss, value_loss, action_probs, old_action_probs, advantages = self.ppo_loss(old_logits, values, returns_batch, states, actions, returns_batch)
                    print(f"Episode: {episode + 1}, Loss: {policy_loss + 0.5 * value_loss}")
                    policy_losses.append(policy_loss.numpy())
                    value_losses.append(value_loss.numpy())
                    
                    steps_taken.append(step+1)

                    break

            # Evaluation episode

            if episode % self.env.config.evaluation_interval == 0:
                self.env.reset()
                state = [copy.deepcopy(self.env.object_state_norm), copy.deepcopy(self.env.state)]
            
                for step in range(self.env.config.steps):
                    state = [state[0].flatten().astype('float32').reshape(1, self.env.config.n_objects * 6), state[1].reshape(1, self.env.config.state_length)]
                    logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0].numpy()
                    action_ind = np.unravel_index(action, (self.env.config.n_objects, self.env.config.state_length, 5))
                    next_state, reward, done = copy.deepcopy(self.env.step(action_ind[0], action_ind[1], action_ind[2], step))

                    state = copy.deepcopy(next_state)
                f_factors_eval.append(np.sum(self.env.rewards)/len(self.env.rewards))

        self.actor_network.save('ppo_actor_network.keras')
        self.critic_network.save('ppo_critic_network.keras')


        return(f_factors_max, f_factors_mean, f_factors_final, actions_taken_total, actions_logits_mean, f_factors_eval)
            

