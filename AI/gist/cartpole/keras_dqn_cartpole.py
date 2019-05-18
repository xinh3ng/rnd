# |/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keras,DQN, Cart pole game

https://keon.io/deep-q-learning/ (Good)
It explains what is remember and experience replay
It explains how an action decision is made
"""
from pdb import set_trace as debug
from collections import deque
import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import backend as K

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def huber_loss(realized, prediction):
    """Huber loss: sqrt(1+error^2)-1
    
    """
    error = prediction - realized
    return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)


class DQNAgent(object):
    """Deep Q-learning Agent
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()  # NN model
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Neural net for deep Q learning
        """
        model = Sequential()
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state is state_size and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation="relu"))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def load(self, model_file):
        self.model.load_weights(model_file)

    def save(self, model_file):
        self.model.save_weights(model_file)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decide_action(self, state):
        # epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Predict the reward value based on the given state
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Pick the action based on the highest reward (predicted)

    def replay(self, batch_size):
        """Experience replay
        """
        minibatch = random.sample(self.memory, batch_size)
        # Extract informations from each memory
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                # predict the future discounted reward
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # reduce epsilon when it becomes better at the game
            logger.info("self.epsilon becomes: %.4f" % self.epsilon)
        return


#################################################################
if __name__ == "__main__":
    # Model parameters
    # episodes - a number of games we want the agent to play.
    # gamma - decay or discount rate, to calculate the future discounted reward.
    # epsilon - exploration rate
    # epsilon_decay - we want to decrease the number of explorations as it becomes good at playing
    # epsilon_min - we want the agent to explore at least this amount.
    # learning_rate - Determines how much neural net learns in each iteration

    episodes = 1000
    render = False

    # initialize gym environment and the agent
    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright until score of 500. The more time_t the more score
        for time_t in range(500):
            if render:  # turn this on if you want to render
                env.render()

            # Decide an action
            # Reward is 1 for every frame the pole survived
            action = agent.decide_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state  # make next_state the new current state for the next frame.

            if done:  # Game over
                agent.update_target_model()
                logger.info("episode: {}/{}, score: {}".format(e, episodes, time_t))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 0:
            agent.save("./save/cartpole-dqn.h5")

    logger.info("ALL DONE!\n")
