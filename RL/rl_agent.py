import sys
sys.path.append('mrp-jaap-2425/')
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, Flatten, CategoryEncoding
from keras import Model
from keras.models import save_model
from keras.losses import mean_squared_error
import helper
from param_config import Configuration
import random
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import copy

config = Configuration()

class RLAgent():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (self.env.config.n_objects * 6)
        self.state_space_b = (self.env.config.state_length)
        self.action_space = (self.env.config.n_objects, self.env.config.state_length, 5)
        self.action_size = self.env.config.n_objects * self.env.config.state_length * 5
        self.unif = tf.fill((1, self.action_size), 0.)
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

        self.times = [0 for i in range(20)]

    def __create_q_network(self):
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

    def __choose_action(self, logits):
        if np.random.uniform() < self.exploration_rate:
            action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), self.unif, -np.inf), 1)[0, 0].numpy()
            action_ind = np.unravel_index(action, (self.env.config.n_objects, self.env.config.state_length, 5))        
        else:
            action = tf.math.argmax(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0].numpy()
            action_ind = np.unravel_index(action, (self.env.config.n_objects, self.env.config.state_length, 5))
        return(action, action_ind)

    @tf.function
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
        f_factors_final = []
        f_factors_max = []
        f_factors_mean = []
        rewards_mean = []
        actions_taken_total = [[],[],[],[],[]]
        actions_logits_mean = [[],[],[],[],[]]

        f_factors_eval = []
        

        for episode in range(self.env.config.episodes):
            print(f'Episode {episode}/{self.env.config.episodes}, exploration rate: {self.exploration_rate}')
            try:
                f_factors_final.append(f_factors[-1])
                f_factors_max.append(np.max(f_factors))
                f_factors_mean.append(np.mean(f_factors))
                rewards_mean.append(np.mean(rewards))
                for ind, action in enumerate(self.actions_taken):
                    actions_taken_total[ind].append(action)

            except:
                pass

            states, actions, rewards, next_states, dones = [],[],[],[],[]
            self.env.reset()
            state = copy.deepcopy([self.env.object_state_norm, self.env.state])

            f_factors = []
            reward = -1
            reward_per_action = [0,0,0,0,0]
            taken_actions = [0,0,0,0,0]

            self.actions_taken = [0,0,0,0,0]


            for step in range(self.env.config.steps):
                state = [state[0].flatten().astype('float32').reshape(1, self.env.config.n_objects * 6), state[1].reshape(1, self.env.config.state_length)]
                logits = self.q_network_func(state)

                action, action_ind = self.__choose_action(logits)  # 0.00012 s, 0.16 % of total
                self.actions_taken[action_ind[2]] += 1
                previous_reward = np.sum(self.env.rewards)/len(self.env.rewards)  # 0.000016 s
                next_state, reward, done = copy.deepcopy(self.env.step(action_ind[0], action_ind[1], action_ind[2], step))  # 0.00086 s, 1.2% of total

                logits_reshaped = logits.numpy().reshape((self.env.config.n_objects, self.env.config.state_length,5))
                for a in range(5):
                    actions_logits_mean[a].append(np.max(logits_reshaped[:,:,a]))

                # delta reward method
                # reward = 0.5 * reward + 2 * (reward - previous_reward)  # 0.00000061 s

                f_factors.append(np.sum(self.env.rewards)/len(self.env.rewards))  #-previous_reward)  # 0.0000036 s
                reward_per_action[action_ind[2]] += reward  #-previous_reward  # 0.00000061 s
                taken_actions[action_ind[2]] += 1  # 0.00000044 s


                self.replay_buffer.append([copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done])  # 0.00000062 s
                if len(self.replay_buffer) > self.env.config.memory_size:
                    self.replay_buffer.pop(0)
                if len(self.replay_buffer) > self.env.config.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)  # 0.000029 s, 0.04% of total
                    states = [np.vstack([minibatch[i][0][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][0][1].flatten() for i in range(len(minibatch))])]
                    actions = tf.convert_to_tensor([minibatch[i][1] for i in range(len(minibatch))], dtype=tf.int32) 
                    rewards = tf.convert_to_tensor([minibatch[i][2] for i in range(len(minibatch))], dtype=tf.float32)
                    next_states = [np.vstack([minibatch[i][3][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][3][1].flatten() for i in range(len(minibatch))])]
                    dones = tf.convert_to_tensor([minibatch[i][4] for i in range(len(minibatch))], dtype=tf.float32)     

                    time_init = perf_counter()
                    self.update_network(states, actions, rewards, dones, next_states)  # 0.16 s, 222% of total
                    self.times[0] += perf_counter() - time_init
                state = copy.deepcopy(next_state)
                self.env.taken_actions_countdown = np.where(self.env.taken_actions_countdown >= 1, self.env.taken_actions_countdown - 1, self.env.taken_actions_countdown)


            if episode % self.env.config.evaluation_interval == 0:
                self.env.reset()
                state = [copy.deepcopy(self.env.object_state_norm), copy.deepcopy(self.env.state)]
            
                for step in range(self.env.config.steps):
                    state = [state[0].flatten().astype('float32').reshape(1, self.env.config.n_objects * 6), state[1].reshape(1, self.env.config.state_length)]
                    logits = self.q_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0].numpy()
                    action_ind = np.unravel_index(action, (self.env.config.n_objects, self.env.config.state_length, 5))
                    next_state, reward, done = copy.deepcopy(self.env.step(action_ind[0], action_ind[1], action_ind[2], step))

                    state = copy.deepcopy(next_state)
                f_factors_eval.append(np.sum(self.env.rewards)/len(self.env.rewards))     

            self.exploration_rate *= self.exploration_decay
            if episode % self.target_update_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())
            
        print(self.times)
        print(np.sum(self.times))
        sys.exit()

        self.q_network.save('q_network.keras')
        return(f_factors_max, f_factors_mean, f_factors_final, actions_taken_total, actions_logits_mean, f_factors_eval)

            

