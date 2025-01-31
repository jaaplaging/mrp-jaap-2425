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

class RLAgent():

    def __init__(self, env):
        self.env = env
        self.state_space_a = (18)
        self.state_space_b = (config.state_length)
        self.action_space = (3, config.state_length, 5)
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.exploration_rate = config.exploration_rate
        self.exploration_decay = config.exploration_decay
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.replay_buffer = []
        self.q_network = self.__create_q_network()
        self.target_network = self.__create_q_network()#   .set_weights(self.q_network.get_weights())
        self.target_network.set_weights(self.q_network.get_weights())

    def __create_q_network(self):
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

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return(model)

    def __choose_action(self, state):
        # print(self.env.action_availability_mask[:,4])
        # print(self.env.object_unavailability_mask[0,4])
        # print(self.env.total_mask[0,:,4])
        if np.random.uniform() < self.exploration_rate:
            legal = False
            while not legal:
                action = (np.random.randint(3), np.random.randint(config.state_length), np.random.randint(5))
                if self.env.total_mask[action[0], action[1], action[2]]:
                    legal = True
            # print(action)
            return(action)
        else:
            state = [state[0].flatten().astype('float32').reshape(1,18), state[1].reshape(1,config.state_length)]
            prediction = self.q_network.predict(state, verbose=0).reshape((3,config.state_length,5))
            masked = np.where(self.env.total_mask, prediction, -np.inf)
            action = np.unravel_index(np.argmax(masked), masked.shape)
            self.expl_actions_taken[action[2]] += 1
            return(action)


    def update_network(self, minibatch):
        states = [np.vstack([minibatch[i][0][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][0][1] for i in range(len(minibatch))])]
        #  0.00015 s, 0.09% of total


        actions = [minibatch[i][1] for i in range(len(minibatch))]  # 0.000005 s
        rewards = [minibatch[i][2] for i in range(len(minibatch))]  # 0.0000042 s
        next_states = [np.vstack([minibatch[i][3][0].flatten() for i in range(len(minibatch))]).astype('float32'), np.array([minibatch[i][3][1] for i in range(len(minibatch))])]
        # 0.000098 s
        dones = [minibatch[i][4] for i in range(len(minibatch))]  # 0.0000026 s

        predict_values = self.q_network.predict(states, verbose=0).reshape(self.batch_size,3,config.state_length,5)  # 0.049 s, 31% of total
        predict_next_values = self.target_network.predict(next_states, verbose=0).reshape(self.batch_size, 3,config.state_length,5)  # 0.049 s, 31% of total


        for i in range(len(minibatch)):  # 0.00049 s, 0.3% of total
            target_value = rewards[i] + (1 - dones[i]) * self.discount_factor * np.max(predict_next_values[i])
            predict_values[i,actions[i][0],actions[i][1],actions[i][2]] = target_value

        self.q_network.fit(states, predict_values.reshape(self.batch_size, 3*config.state_length*5), verbose=0)  # 0.067 s, 42% of total


    def train(self, episodes = config.episodes):
        rewards_final = []
        rewards_max = []
        rewards_mean = []
        actions_taken_total = [[],[],[],[],[]]
        expl_actions_taken_total = [[],[],[],[],[]]
        

        for episode in range(episodes):
            self.env.reset()
            state = [self.env.object_state, self.env.state]
            done = 0

            i = 0
            try:
                print(f'Episode number: {episode}')
                print(f'Mean reward during episode: {np.mean(rewards)}')
                print(f'Max reward during episode: {np.max(rewards)}')
                print(f'Final reward during episode: {rewards[-1]}')
                print(f'Exploration rate: {self.exploration_rate}')
                print(f'Exploitation actions taken: {self.expl_actions_taken}')
                print(f'Average reward per action: {[reward_per_action[i]/(taken_actions[i]+1) for i in range(4)]}')

                rewards_final.append(rewards[-1])
                rewards_max.append(np.max(rewards))
                rewards_mean.append(np.mean(rewards))
                for ind, action in enumerate(self.actions_taken):
                    actions_taken_total[ind].append(action)
                for ind, action in enumerate(self.expl_actions_taken):
                    expl_actions_taken_total[ind].append(action)
                
            except:
                pass

            rewards = []
            reward = -1
            reward_per_action = [0,0,0,0,0]
            taken_actions = [0,0,0,0,0]

            self.actions_taken = [0,0,0,0,0]
            self.expl_actions_taken = [0,0,0,0,0]

            while done != 1:  # total 0.072 s per loop
                action = self.__choose_action(state)  # 0.00012 s, 0.16 % of total
                self.env.taken_actions_mask[:, :, action[2]] = False
                self.env.taken_actions_countdown[:, :, action[2]] = config.masking_duration
                self.actions_taken[action[2]] += 1  # 0.00000068 s
                previous_reward = np.sum(self.env.rewards)/len(self.env.rewards)  # 0.000016 s
                next_state, reward, done = self.env.step(action[0], action[1], action[2], i)  # 0.00086 s, 1.2% of total

                # delta reward method
                reward = 0.5 * reward + 2 * (reward - previous_reward)  # 0.00000061 s

                rewards.append(np.sum(self.env.rewards)/len(self.env.rewards))  #-previous_reward)  # 0.0000036 s
                reward_per_action[action[2]] += reward  #-previous_reward  # 0.00000061 s
                taken_actions[action[2]] += 1  # 0.00000044 s

                self.replay_buffer.append([state, action, reward, next_state, done])  # 0.00000062 s
                if len(self.replay_buffer) > config.memory_size:
                    self.replay_buffer.pop(0)
                if len(self.replay_buffer) > config.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)  # 0.000029 s, 0.04% of total
                    self.update_network(minibatch)  # 0.16 s, 222% of total
                state = next_state
                self.env.taken_actions_countdown = np.where(self.env.taken_actions_countdown >= 1, self.env.taken_actions_countdown - 1, self.env.taken_actions_countdown)
                i += 1

                    
                    

            self.exploration_rate *= self.exploration_decay
            if episode % self.target_update_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())
            
        # test run
        self.env.reset()
        state = [self.env.object_state, self.env.state]
        self.exploration_rate = 0.0
        done = 0
        i = 0
        while done != 1:
            action = self.__choose_action(state)
            self.env.taken_actions_mask[:, :, action[2]] = False
            self.env.taken_actions_countdown[:, :, action[2]] = config.masking_duration
            previous_reward = reward
            next_state, reward, done = self.env.step(action[0], action[1], action[2], i)
            print(reward)
            if action[2] == 0:
                print(f'added object {action[0]} at {action[1]}')
            if action[2] == 1:
                print(f'removed object {action[0]}')
            if action[2] == 2:
                print(f'added observation for object {action[0]} at {action[1]}')
            if action[2] == 3:
                print(f'removed observation of object {action[0]} at {action[1]}')
            if action[2] == 4:
                print(f'replaced observation of object {action[0]} at {action[1]}')
            state = next_state
            self.env.taken_actions_countdown = np.where(self.env.taken_actions_countdown >= 1, self.env.taken_actions_countdown - 1, self.env.taken_actions_countdown)
            i += 1
        print(self.env.state)

        plt.plot(rewards_max, label='Max reward',color='red')
        plt.plot(rewards_mean, label='Mean reward',color='blue')
        plt.plot(rewards_final, label='Final reward', color='green')
        plt.title(f'absolute rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rewards.png')
        plt.show()

        plt.plot(actions_taken_total[0], label='Add object',color='red')
        plt.plot(expl_actions_taken_total[0], label='Add object exploit', color='red', linestyle='dashed')
        plt.plot(actions_taken_total[1], label='Remove object',color='orange')
        plt.plot(expl_actions_taken_total[1], label='Remove object exploit', color='orange', linestyle='dashed')
        plt.plot(actions_taken_total[2], label='Add observation',color='green')
        plt.plot(expl_actions_taken_total[2], label='Add observation exploit', color='green', linestyle='dashed')
        plt.plot(actions_taken_total[3], label='Remove observation',color='blue')
        plt.plot(expl_actions_taken_total[3], label='Remove observation exploit', color='blue', linestyle='dashed')
        plt.plot(actions_taken_total[4], label='Replace observation',color='pink')
        plt.plot(expl_actions_taken_total[4], label='Replace observation exploit', color='pink', linestyle='dashed')
        plt.title(f'actions taken')
        plt.xlabel('Episode')
        plt.ylabel('Number of actions')
        plt.legend(bbox_to_anchor=(1.1,1))
        plt.tight_layout()
        plt.savefig('actions.png')
        plt.show()

        self.q_network.save('q_network.keras')

            

