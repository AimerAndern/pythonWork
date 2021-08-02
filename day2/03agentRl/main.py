import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow import keras

import random


#create gym from gym import wrappers
envCartPole = gym.make('CartPole-v1')
envCartPole.seed(50)

tmp_array = []

for i in range(100):
  tmp_array.append(i)
tmp_array=np.array(tmp_array)+10
print(tmp_array)

#Global Variables
EPISODES = 500
TRAIN_END = 0

#Hyper parameters
def discount_rate():
  return 0.95

def learning_rate():
  return 0.001

def batch_size():
  return 24


class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []

    def build_model(self):
        model = keras.Sequential()

        model.add(keras.layers.Dense(24, input_dim=self.nS, activation='relu'))

        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.nA, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(lr=self.alpha))
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def test_action(self, state):
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        x = []
        y = []
        np_array = np.array(minibatch)

        st = np.zeros((0, self.nS))

        nst = np.zeros((0, self.nS))

        for i in range(len(np_array)):
            st = np.append(st, np_array[i, 0], axis=0)
            nst = np.append(nst, np_array[i, 3], axis=0)

        st_predict = self.model.predict(st)
        nst_predict = self.model.predict(nst)

        index = 0

        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            nst_action_predict_model = nst_predict[index]
            if done == True:
                target = reward
            else:
                target = reward + self.gamma * np.amax(nst_action_predict_model)

            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1

        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)

        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




#create the agent
nS = envCartPole.observation_space.shape[0]
nA = envCartPole.action_space.n
dqn=DeepQNetwork(nS, nA, learning_rate(),discount_rate(),1,0.001,0.995)
batch_size=batch_size()

# training
rewards = []
epsilons = []
TEST_Episodes = 0

for e in range(EPISODES):
    state = envCartPole.reset()
    state = np.reshape(state, [1, nS])
    tot_rewards = 0

    for time in range(210):
        action = dqn.action(state)
        nstate, reward, done, _ = envCartPole.step(action)
        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(state, action, reward, nstate, done)
        state = nstate

        if done or time == 209:
            rewards.append(tot_rewards)
            epsilons.append(dqn.epsilon)
            print("episode: {}/{}, score:{}, e:{}"
                  .format(e, EPISODES, tot_rewards, dqn.epsilon))
            break

        if len(dqn.memory) > batch_size:
            dqn.experience_replay(batch_size)

    if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
        TEST_Episodes = EPISODES - e
        TRAIN_END = e
        break


#Test the agent that was trained
#In this section we ALWAYS use exploit don't train any more

for e_test in range(TEST_Episodes):
  state=envCartPole.reset()
  state = np.reshape(state, [1,nS])
  tot_rewards = 0

  for t_test in range(210):
    action = dqn.test_action(state)
    nstate, reward,done,_=envCartPole.step(action)
    nstate = np.reshape(nstate, [1,nS])
    tot_rewards +=reward
    #DONT'T STORE ANYTHING DURING TESTING
    state =nstate
    #done:CartPole fell
    #t_Test = 209: CartPole stayed upright

    if done or t_test==209:
      rewards.append(tot_rewards)
      epsilons.append(0) #We are doing full exploit
      print("episode: {}/{}, score: {}, e: {}".
            format(e_test, TEST_Episodes, tot_rewards, 0))
      break

    rolling_average = np.convolve(rewards, np.ones(100) / 100)

    plt.plot(rewards)
    plt.plot(rolling_average, color='black')

    plt.axhline(y=195, color='r', linestyle='-')

    eps_graph = [200 * x for x in epsilons]

    plt.plot(eps_graph, color='g', linestyle='-')
    plt.axvline(x=TRAIN_END, color='y', linestyle='-')
    plt.xlim((0, EPISODES))
    plt.ylim((0, 220))
    plt.show()
    envCartPole.close()