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

config = Configuration()

class PPOAgentCurriculum():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (6)
        self.state_space_b = (config.state_length)
        self.action_space = (1, config.state_length, 5)
        self.action_size = 1 * config.state_length * 5
        self.learning_rate_actor = config.learning_rate
        self.learning_rate_critic = config.learning_rate
        self.discount_factor = config.discount_factor
        self.clip_ratio = config.clip_ratio
        self.batch_size = config.batch_size
        self.actor_network = self.__create_actor_network()
        self.critic_network = self.__create_critic_network()
        self.actor_network_func = tf.function(self.actor_network, reduce_retracing=True)
        self.critic_network_func = tf.function(self.critic_network, reduce_retracing=True)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_critic)


    def __create_actor_network(self):
        input_x = Input(shape=(6,))
        input_y = Input(shape=(config.state_length,))

        x = Dense(256, activation='relu')(input_x)
        x = Dense(256, activation='relu')(x)

        y = CategoryEncoding(num_tokens=(4))(input_y)
        y = Dense(256, activation='relu')(y)
        y = Dense(256, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(512, activation='relu')(combined)
        z = Dense(1*config.state_length*5, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    def __create_critic_network(self):
        input_x = Input(shape=(6,))
        input_y = Input(shape=(config.state_length,))

        x = Dense(4, activation='relu')(input_x)
        x = Dense(4, activation='relu')(x)

        y = CategoryEncoding(num_tokens=(4))(input_y)
        y = Dense(32, activation='relu')(y)
        y = Dense(4, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(8, activation='relu')(combined)
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
            return(policy_loss, value_loss, action_probs, old_action_probs)

        advantages = get_advantages(returns, old_values)

        for i in range(config.n_epochs):
            policy_loss, value_loss, action_probs, old_action_probs = train_step(states, actions, returns, old_logits, old_values)
        return(policy_loss, value_loss, action_probs, old_action_probs, advantages)
    
    def train(self, episodes = config.episodes, steps = config.steps):
        f_factors_final = []
        f_factors_max = []
        f_factors_mean = []
        rewards_mean = []
        actions_taken_total = [[],[],[],[],[]]
        actions_logits_mean = [[],[],[],[],[]]
        steps_taken = []

        policy_losses = []
        value_losses = []
        
        for episode in range(episodes):
            
            
            try:
                #print(f'Episode number: {episode}')
                #print(f'Mean reward during episode: {np.mean(f_factors)}')
                #print(f'Max reward during episode: {np.max(f_factors)}')
                #print(f'Final reward during episode: {f_factors[-1]}')
                #print(f'Average reward per action: {[reward_per_action[i]/(taken_actions[i]+1) for i in range(5)]}')
            
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
            state = [self.env.object_state, self.env.state]

            f_factors = []
            reward = -1
            reward_per_action = [0,0,0,0,0]
            taken_actions = [0,0,0,0,0]

            self.actions_taken = [0,0,0,0,0]

            for step in range(steps):
                state = [state[0].flatten().astype('float32').reshape(1,6), state[1].reshape(1,config.state_length)]
                logits = self.actor_network_func(state) # self.actor_network.predict(state, verbose=0)
                value = self.critic_network_func(state)  # self.critic_network.predict(state, verbose=0)

                action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0, 0].numpy()
                action_ind = np.unravel_index(action, (1, config.state_length, 5))
                self.actions_taken[action_ind[2]] += 1
                previous_reward = np.sum(self.env.rewards)/len(self.env.rewards)
                next_state, reward, done = self.env.step(action_ind[0], action_ind[1], action_ind[2])

                logits_reshaped = logits.numpy().reshape((1,config.state_length,5))
                for a in range(5):
                    actions_logits_mean[a].append(np.max(logits_reshaped[:,:,a]))

                #reward = 0.5 * reward + 2 * (reward - previous_reward)
                # if reward > 0.6:
                #     reward = (reward-0.6) * 25
                # else:
                #     reward = -1
                # delta = reward - previous_reward
                # if reward > 0.4 and delta > 0:
                #     reward = (reward - 0.4) * 5/0.6 + delta * 5/0.17
                # else:
                #     reward = -1

                f_factors.append(np.sum(self.env.rewards)/len(self.env.rewards))
                reward_per_action[action_ind[2]] += reward
                taken_actions[action_ind[2]] += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.numpy()[0,0])

                state = next_state
                if done or step == config.steps-1:
                    returns_batch = []
                    discounted_sum = 0
                    for r in rewards[::-1]:
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns_batch.append(discounted_sum)
                    returns_batch.reverse()
                    
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype('float32'), np.array([states[i][1].flatten() for i in range(len(states))])]
                    actions = np.array(actions, dtype=np.int32)
                    #print(f'actions is {actions}')
                    values = tf.convert_to_tensor(values, dtype=tf.float32)
                    #print(f'values is {values}')
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)
                    #print(f'returns_batch is {returns_batch}')

                    old_logits = self.actor_network_func(states) # self.actor_network.predict(states)

                    policy_loss, value_loss, action_probs, old_action_probs, advantages = self.ppo_loss(old_logits, values, returns_batch, states, actions, returns_batch)
                    print(f"Episode: {episode + 1}, Loss: {policy_loss + 0.5 * value_loss}")
                    policy_losses.append(policy_loss.numpy())
                    value_losses.append(value_loss.numpy())

                    # for step_ind in range(step):
                    #     action_ind = np.unravel_index(actions[step_ind], (3, config.state_length, 5))
                    #     print(f'Action is {action_ind}')
                    #     print(f'Return value is {returns_batch[step_ind]}')
                    #     print(f'Advantage value is {advantages[step_ind]}')
                    #     print(f'difference in action probs is (new-old) {action_probs[step_ind]-old_action_probs[step_ind]}')
                    #     print('')

                    steps_taken.append(step+1)

                    break


        # # test run
        # self.env.reset()
        # state = [self.env.object_state, self.env.state]
        # done = 0
        # i = 0
        # while done != 1:
        #     # legal = False
        #     # while not legal:
        #     #     action = tf.random.categorical(logits, 1)[0, 0].numpy()
        #     #     action_ind = np.unravel_index(action, (3, config.state_length, 5))
        #     #     if self.env.total_mask[action_ind[0], action_ind[1], action_ind[2]]:
        #     #         legal = True
        #     action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0, 0].numpy()
        #     action_ind = np.unravel_index(action, (3, config.state_length, 5))
        #     previous_reward = reward
        #     next_state, reward, done = self.env.step(action_ind[0], action_ind[1], action_ind[2], i)
        #     print(reward)
        #     if action_ind[2] == 0:
        #         print(f'added object {action_ind[0]} at {action_ind[1]}')
        #     if action_ind[2] == 1:
        #         print(f'removed object {action_ind[0]}')
        #     if action_ind[2] == 2:
        #         print(f'added observation for object {action_ind[0]} at {action_ind[1]}')
        #     if action_ind[2] == 3:
        #         print(f'removed observation of object {action_ind[0]} at {action_ind[1]}')
        #     if action_ind[2] == 4:
        #         print(f'replaced observation of object {action_ind[0]} at {action_ind[1]}')
        #     state = next_state
        #     i += 1
        # print(self.env.state)




        self.actor_network.save('ppo_actor_network.keras')
        self.critic_network.save('ppo_critic_network.keras')

        return(rewards_mean, f_factors_mean, actions_taken_total, actions_logits_mean, steps_taken)
            

