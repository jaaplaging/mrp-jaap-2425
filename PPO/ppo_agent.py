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

class PPOAgent():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (18)
        self.state_space_b = (config.state_length)
        self.action_space = (3, config.state_length, 5)
        self.action_size = 3 * config.state_length * 5
        self.learning_rate_actor = config.learning_rate
        self.learning_rate_critic = config.learning_rate
        self.discount_factor = config.discount_factor
        self.clip_ratio = config.clip_ratio
        self.batch_size = config.batch_size
        self.actor_network = self.__create_actor_network()
        self.critic_network = self.__create_critic_network()
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_critic)


    def __create_actor_network(self):
        input_x = Input(shape=(18,))
        input_y = Input(shape=(config.state_length,))

        x = Dense(8, activation='relu')(input_x)
        x = Dense(4, activation='relu')(x)

        y = CategoryEncoding(num_tokens=(4))(input_y)
        y = Dense(64, activation='relu')(y)
        y = Dense(16, activation='relu')(y)
        y = Dense(4, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(16, activation='relu')(combined)
        z = Dense(64, activation='relu')(z)
        z = Dense(256, activation='relu')(z)
        z = Dense(3*config.state_length*5, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)
    
    def __create_critic_network(self):
        input_x = Input(shape=(18,))
        input_y = Input(shape=(config.state_length,))

        x = Dense(8, activation='relu')(input_x)
        x = Dense(4, activation='relu')(x)

        y = CategoryEncoding(num_tokens=(4))(input_y)
        y = Dense(64, activation='relu')(y)
        y = Dense(16, activation='relu')(y)
        y = Dense(4, activation='relu')(y)

        combined = Concatenate()([x, y])

        z = Dense(4, activation='relu')(combined)
        z = Dense(1, activation='linear')(z)

        model = Model(inputs=[input_x, input_y], outputs=z)

        return(model)

    # taken from https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
    def ppo_loss(self, old_logits, old_values, advantages, states, actions, returns):        
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
            return policy_loss

        def compute_value_loss(values, returns):
            # Value loss
            value_loss = tf.reduce_mean(tf.square(values - returns))
            return value_loss
        
        def get_advantages(returns, values):
            advantages = returns - values
            return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        def train_step(states, actions, returns, old_logits, old_values):
            with tf.GradientTape() as actor_tape:
                logits = self.actor_network(states)
                policy_loss = compute_policy_loss(logits, actions)
            with tf.GradientTape() as critic_tape:
                values = self.critic_network(states)
                value_loss = compute_value_loss(values, returns)
            policy_gradients = actor_tape.gradient(policy_loss, self.actor_network.trainable_variables)
            value_gradients = critic_tape.gradient(value_loss, self.critic_network.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(policy_gradients, self.actor_network.trainable_variables))
            self.optimizer_critic.apply_gradients(zip(value_gradients, self.critic_network.trainable_variables))
            return(policy_loss, value_loss)

        advantages = get_advantages(returns, old_values)
        for i in range(config.n_epochs):
            policy_loss, value_loss = train_step(states, actions, returns, old_logits, old_values)
        return(policy_loss, value_loss)
    
    def train(self, episodes = config.episodes, steps = config.steps):
        f_factors_final = []
        f_factors_max = []
        f_factors_mean = []
        actions_taken_total = [[],[],[],[],[]]
        
        
        for episode in range(episodes):
            
            
            try:
                print(f'Episode number: {episode}')
                print(f'Mean reward during episode: {np.mean(f_factors)}')
                print(f'Max reward during episode: {np.max(f_factors)}')
                print(f'Final reward during episode: {f_factors[-1]}')
                print(f'Average reward per action: {[reward_per_action[i]/(taken_actions[i]+1) for i in range(5)]}')
            
                f_factors_final.append(f_factors[-1])
                f_factors_max.append(np.max(f_factors))
                f_factors_mean.append(np.mean(f_factors))
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
                state = [state[0].flatten().astype('float32').reshape(1,18), state[1].reshape(1,config.state_length)]
                logits = self.actor_network.predict(state, verbose=0)
                value = self.critic_network.predict(state, verbose=0)

                # legal = False
                # while not legal:
                #     action = tf.random.categorical(logits, 1)[0, 0].numpy()
                #     action_ind = np.unravel_index(action, (3, config.state_length, 5))
                #     if self.env.total_mask[action_ind[0], action_ind[1], action_ind[2]]:
                #         legal = True
                action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0, 0].numpy()
                action_ind = np.unravel_index(action, (3, config.state_length, 5))
                self.actions_taken[action_ind[2]] += 1
                previous_reward = np.sum(self.env.rewards)/len(self.env.rewards)
                next_state, reward, done = self.env.step(action_ind[0], action_ind[1], action_ind[2], step)

                #reward = 0.5 * reward + 2 * (reward - previous_reward)
                # if reward > 0.6:
                #     reward = (reward-0.6) * 25
                # else:
                #     reward = -1
                delta = reward - previous_reward
                if reward > 0.4 and delta > 0:
                    reward = (reward - 0.4) * 5/0.6 + delta * 5/0.17
                else:
                    reward = -1

                f_factors.append(np.sum(self.env.rewards)/len(self.env.rewards))
                reward_per_action[action_ind[2]] += reward
                taken_actions[action_ind[2]] += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)

                state = next_state
                if done:
                    returns_batch = []
                    discounted_sum = 0
                    for r in rewards[::-1]:
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns_batch.append(discounted_sum)
                    returns_batch.reverse()
                    
                    states = [np.vstack([states[i][0].flatten() for i in range(len(states))]).astype('float32'), np.array([states[i][1].flatten() for i in range(len(states))])]
                    actions = np.array(actions, dtype=np.int32)
                    values = tf.concat(values, axis=0)
                    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

                    old_logits = self.actor_network.predict(states)

                    policy_loss, value_loss = self.ppo_loss(old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)
                    print(f"Episode: {episode + 1}, Loss: {policy_loss + 0.5 * value_loss}")

                    break

        # test run
        self.env.reset()
        state = [self.env.object_state, self.env.state]
        done = 0
        i = 0
        while done != 1:
            # legal = False
            # while not legal:
            #     action = tf.random.categorical(logits, 1)[0, 0].numpy()
            #     action_ind = np.unravel_index(action, (3, config.state_length, 5))
            #     if self.env.total_mask[action_ind[0], action_ind[1], action_ind[2]]:
            #         legal = True
            action = tf.random.categorical(tf.where(self.env.total_mask.flatten(), logits, -np.inf), 1)[0, 0].numpy()
            action_ind = np.unravel_index(action, (3, config.state_length, 5))
            previous_reward = reward
            next_state, reward, done = self.env.step(action_ind[0], action_ind[1], action_ind[2], i)
            print(reward)
            if action_ind[2] == 0:
                print(f'added object {action_ind[0]} at {action_ind[1]}')
            if action_ind[2] == 1:
                print(f'removed object {action_ind[0]}')
            if action_ind[2] == 2:
                print(f'added observation for object {action_ind[0]} at {action_ind[1]}')
            if action_ind[2] == 3:
                print(f'removed observation of object {action_ind[0]} at {action_ind[1]}')
            if action_ind[2] == 4:
                print(f'replaced observation of object {action_ind[0]} at {action_ind[1]}')
            state = next_state
            i += 1
        print(self.env.state)

        plt.plot(f_factors_max, label='Max reward',color='red')
        plt.plot(f_factors_mean, label='Mean reward',color='blue')
        plt.plot(f_factors_final, label='Final reward', color='green')
        plt.title(f'absolute rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rewards.png')
        plt.show()

        plt.plot(actions_taken_total[0], label='Add object',color='red')
        plt.plot(actions_taken_total[1], label='Remove object',color='orange')
        plt.plot(actions_taken_total[2], label='Add observation',color='green')
        plt.plot(actions_taken_total[3], label='Remove observation',color='blue')
        plt.plot(actions_taken_total[4], label='Replace observation',color='pink')
        plt.title(f'actions taken')
        plt.xlabel('Episode')
        plt.ylabel('Number of actions')
        plt.legend(bbox_to_anchor=(1.1,1))
        plt.tight_layout()
        plt.savefig('actions.png')
        plt.show()

        self.actor_network.save('ppo_actor_network.keras')
        self.critic_network.save('ppo_critic_network.keras')

            

