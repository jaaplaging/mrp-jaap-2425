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
from keras.losses import mean_squared_error
import copy

config = Configuration()

class OnTheFlyAgentPPO():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (10 * 9)
        self.state_space_b = (2)
        self.action_space = (11)
        self.action_size = 11
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
        input_x = Input(shape=(10 * 9,))
        input_y = Input(shape=(2,))

        x = Dense(self.env.config.layer_size, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size, activation='relu')(x)

        y = Dense(self.env.config.layer_size, activation='relu')(input_y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size, activation='relu')(combined)
        z = Dense(10 + 1, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    def __create_critic_network(self):
        input_x = Input(shape=(10 * 9,))
        input_y = Input(shape=(2,))

        x = Dense(64, activation='relu')(input_x)
        x = Dense(8, activation='relu')(x)

        y = Dense(4, activation='relu')(input_y)
        y = Dense(8, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(16, activation='relu')(combined)
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

        for episode in range(self.env.config.episodes):
            
            states, actions, rewards, values, returns = [],[],[],[],[]
            self.env.reset()
            state = copy.deepcopy([self.env.object_state_norm, np.array([-1, 1])])

            for step in range(self.env.config.steps):
                state = [state[0].flatten().astype('float32').reshape(1, 10 * 9), state[1].flatten().astype('float32').reshape(1, 2)]

                logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)
                value = self.critic_network_func(state)  # self.critic_network.predict(state, verbose=0)

                action = tf.random.categorical(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0, 0].numpy()
                next_state, reward, done = copy.deepcopy(self.env.step(action))

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
                    
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype('float32'), np.array([states[i][1].flatten() for i in range(len(states))]).astype('float32')]
                    actions = np.array(actions, dtype=np.int32)
                    values = tf.convert_to_tensor(values, dtype=tf.float32)
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

                    old_logits = self.actor_network_func(states) # self.actor_network.predict(states)

                    policy_loss, value_loss = self.ppo_loss(old_logits, values, returns_batch, states, actions, returns_batch)
                    final_rewards.append(rewards[-1])
                    print(f"Episode: {episode + 1}, Loss: {policy_loss + 0.5 * value_loss}, Final reward: {rewards[-1]}")

                    break

            if episode % self.env.config.evaluation_interval == 0:
                self.env.reset()
                state = copy.deepcopy([self.env.object_state_norm, np.array([-1, 1])])

                for step in range(self.env.config.steps):
                    state = [state[0].flatten().astype('float32').reshape(1, 10 * 9), state[1].flatten().astype('float32').reshape(1, 2)]
                    logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
                    next_state, reward, done = copy.deepcopy(self.env.step(action))

                    state = copy.deepcopy(next_state)
                    if done:
                        final_rewards_eval.append(reward)
                        print(self.env.schedule)
                        break


        self.actor_network.save('ppo_actor_network.keras')
        self.critic_network.save('ppo_critic_network.keras')

        return(final_rewards, final_rewards_eval)
            


class OnTheFlyAgentDQN():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (10 * 9)
        self.state_space_b = (2)
        self.action_space = (11)
        self.action_size = 11
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

    def __create_q_network(self):
        input_x = Input(shape=(10 * 9,))
        input_y = Input(shape=(2,))

        x = Dense(self.env.config.layer_size, activation='relu')(input_x)
        x = Dense(self.env.config.layer_size, activation='relu')(x)

        y = Dense(self.env.config.layer_size, activation='relu')(input_y)
        y = Dense(self.env.config.layer_size, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(2 * self.env.config.layer_size, activation='relu')(combined)
        z = Dense(10 + 1, activation='linear')(z)

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
        

        for episode in range(self.env.config.episodes):
            print(f'Episode {episode}/{self.env.config.episodes}, exploration rate: {self.exploration_rate}')

            states, actions, rewards, next_states, dones = [],[],[],[],[]
            self.env.reset()
            state = copy.deepcopy([self.env.object_state_norm, np.array([-1, 1])])

            for step in range(self.env.config.steps):
                state = [state[0].flatten().astype('float32').reshape(1, 10 * 9), state[1].flatten().astype('float32').reshape(1, 2)]
                logits = self.q_network_func(state)

                action = self.__choose_action(logits) 
                next_state, reward, done = copy.deepcopy(self.env.step(action))  # 0.00086 s, 1.2% of total

                self.replay_buffer.append([copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done])  # 0.00000062 s
                if len(self.replay_buffer) > self.env.config.memory_size:
                    self.replay_buffer.pop(0)
                if len(self.replay_buffer) > self.env.config.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)  # 0.000029 s, 0.04% of total
                    states = [np.vstack([minibatch[i][0][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][0][1].flatten() for i in range(len(minibatch))]).astype('float32')]
                    actions = tf.convert_to_tensor([minibatch[i][1] for i in range(len(minibatch))], dtype=tf.int32) 
                    rewards = tf.convert_to_tensor([minibatch[i][2] for i in range(len(minibatch))], dtype=tf.float32)
                    next_states = [np.vstack([minibatch[i][3][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][3][1].flatten() for i in range(len(minibatch))]).astype('float32')]
                    dones = tf.convert_to_tensor([minibatch[i][4] for i in range(len(minibatch))], dtype=tf.float32)     
                    
                    self.update_network(states, actions, rewards, dones, next_states)  # 0.16 s, 222% of total
                state = copy.deepcopy(next_state)

                if done:
                    final_rewards.append(reward)
                    break

            if episode % self.env.config.evaluation_interval == 0:
                self.env.reset()
                state = copy.deepcopy([self.env.object_state_norm, np.array([-1, 1])])
            
                for step in range(self.env.config.steps):
                    state = [state[0].flatten().astype('float32').reshape(1, 10 * 9), state[1].flatten().astype('float32').reshape(1, 2)]
                    logits = self.q_network_func(state) # self.actor_network.predict(state, verbose=0)

                    # off-policy argmax instead of on-policy stochastic
                    action = tf.math.argmax(tf.where(self.env.create_mask(), logits, -np.inf), 1)[0].numpy()
                    next_state, reward, done = copy.deepcopy(self.env.step(action))

                    state = copy.deepcopy(next_state)
                    if done:
                        final_rewards_eval.append(reward)
                        print(self.env.schedule)
                        break 

            self.exploration_rate *= self.exploration_decay
            if episode % self.target_update_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())
            

        self.q_network.save('q_network.keras')
        return(final_rewards, final_rewards_eval)