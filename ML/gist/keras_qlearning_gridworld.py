#|/usr/bin/env python
"""Keras, Q learning, Grid World

http://outlace.com/rlpart3.html
"""
from pdb import set_trace as debug
import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def rand_pair(start, end):
    return np.random.randint(start, end), np.random.randint(start, end)


def find_loc(state, obj):
    """finds an array in the "depth" dimension of the grid
    """
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i,j


def get_loc(state, level):
    """Get location from a given object
    """
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j][level] == 1):
                return i, j


def init_grid():
    """Initialize stationary grid, all items are placed deterministically

    """
    state = np.zeros((4, 4, 4)) # 4 x 4 is the grid. Last dimention of 4 is for the 4 objects
    # player
    state[0, 1] = np.array([0, 0, 0, 1])  # 3
    # wall
    state[2, 2] = np.array([0, 0, 1, 0])  # 2
    # pit
    state[1, 1] = np.array([0, 1, 0, 0])  # 1
    # goal
    state[3, 3] = np.array([1, 0, 0, 0])  # 0
    return state


def init_grid_player():
    """Initialize player in random location, but keep wall, goal and pit stationary
    """
    state = np.zeros((4, 4, 4))
    # place player
    state[rand_pair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[1, 2] = np.array([1, 0, 0, 0])

    a = find_loc(state, np.array([0, 0, 0, 1])) #find grid position of player (agent)
    w = find_loc(state, np.array([0, 0, 1, 0])) #find wall
    g = find_loc(state, np.array([1, 0, 0, 0])) #find goal
    p = find_loc(state, np.array([0, 1, 0, 0])) #find pit
    if (not a or not w or not g or not p):
        logger.info("Invalid grid. Rebuilding..")
        return init_grid_player()
    return state


def init_grid_rand():
    """Initialize grid, so that goal, pit, wall, player are all randomly placed

    """
    state = np.zeros((4, 4, 4))
    #place player
    state[rand_pair(0, 4)] = np.array([0, 0, 0, 1])
    #all
    state[rand_pair(0, 4)] = np.array([0, 0, 1, 0])
    #pit
    state[rand_pair(0, 4)] = np.array([0, 1, 0, 0])
    #place goal
    state[rand_pair(0, 4)] = np.array([1, 0, 0, 0])

    # guarantee that the objects won't overlap in position
    a = find_loc(state, np.array([0, 0, 0, 1]))
    w = find_loc(state, np.array([0, 0, 1, 0]))
    g = find_loc(state, np.array([1, 0, 0, 0]))
    p = find_loc(state, np.array([0, 1, 0, 0]))
    # If any of the "objects" are superimposed, just call the function again to replace
    if (not a or not w or not g or not p):
        logger.info("Invalid grid. Rebuilding..")
        return init_grid_rand()
    return state


def make_move(state, action):
    # need to locate player in grid
    # determine what object (if any) is in the new grid spot the player is moving to
    player_loc = find_loc(state, np.array([0, 0, 0, 1]))
    wall = find_loc(state, np.array([0, 0, 1, 0]))
    goal = find_loc(state, np.array([1, 0, 0, 0]))
    pit = find_loc(state, np.array([0, 1, 0, 0]))

    state = np.zeros((4, 4, 4))
    actions = [[-1, 0],[1, 0], [0, -1], [0, 1]]  # 4 possible actions: up down left right
                                                 # e.g. up => (player row - 1, player column + 0)

    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])  # play moves
    if (new_loc != wall):
        if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1  # 3 means the player

    new_player_loc = find_loc(state, np.array([0, 0, 0, 1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0, 0, 0, 1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1
    return state


def get_reward(state):
    """Calculate reward
    """
    player_loc = get_loc(state, 3)
    pit = get_loc(state, 1)
    goal = get_loc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1


def disp_grid(state):
    grid = np.zeros((4,4), dtype=str)
    player_loc = find_loc(state, np.array([0, 0, 0, 1]))
    wall = find_loc(state, np.array([0, 0, 1, 0]))
    goal = find_loc(state, np.array([1, 0, 0, 0]))
    pit = find_loc(state, np.array([0, 1, 0, 0]))
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i,j] = " "

    if player_loc:
        grid[player_loc] = "P" #player
    if wall:
        grid[wall] = "W" #wall
    if goal:
        grid[goal] = "+" #goal
    if pit:
        grid[pit] = "-" #pit
    print(grid)


def build_model():
    model = Sequential()
    model.add(Dense(164, kernel_initializer="lecun_uniform", input_shape=(64, )))  # 64 means all 16 cells x 4 objects
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    model.add(Dense(150, kernel_initializer="lecun_uniform"))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    model.add(Dense(4, kernel_initializer="lecun_uniform"))  # 4 meand 4 actions
    model.add(Activation("linear"))  # linear so we can have a range of real-valued outputs

    rms = RMSprop()
    model.compile(loss="mse", optimizer=rms)
    return model


def fit_model(model, state, y):
    """

    :param state: Features that are used to make a fit
    :param y: Fit values
    """
    model.fit(state.reshape(1, 64), y, batch_size=1, epochs=1, verbose=1)
    return model


def predict_model(model, state):
    """Predict q-value for 4 possible actions
    """
    return model.predict(state.reshape(1, 64), batch_size=1)


def train_model(model, epochs, gamma, epsilon):
    """

    :param epochs: Number of games to train on
    :param gamma: discount the future rewards. Since it may take several moves to goal, making gamma high
    :param epsilon: probability to "explore"
    """
    for i in range(epochs):
        state = init_grid()
        status = 1
        while(status == 1):  # While game still in progress
            # We are in state S, let's run our Q function on S to get Q values for all possible actions
            qvals = predict_model(model, state)
            if (random.random() < epsilon): # choose random action
                action = np.random.randint(0, 4)
            else: #choose best action from Q(s,a) values
                action = np.argmax(qvals)

            # Take the action, observe new state S'
            new_state = make_move(state, action)
            reward = get_reward(new_state)

            new_qvals = predict_model(model, state)
            maxq = np.max(new_qvals)
            y = np.zeros((1, 4))
            y[:] = qvals[:]
            if reward == -1:  # non-terminal state
                updated_reward = reward + gamma * maxq
            else:  # In terminal state
                updated_reward = reward

            y[0][action] = updated_reward # target output
            print("Game no.: %d" % i)
            print(y)
            model = fit_model(model, state, y)
            state = new_state
            if reward != -1:  # game is over
                status = 0

        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)  # reduce epsilon
    return model


def train_model_experience_replay(model, epochs, gamma, epsilon):
    pass


def test_algo(model, init=0):
    i = 0
    if init == 0:
        state = init_grid()
    elif init == 1:
        state = init_grid_player()
    elif init == 2:
        state = init_grid_rand()

    print("Initial State:")
    print(disp_grid(state))
    status = 1
    # while game still in progress
    while(status == 1):
        qval = predict_model(model, state)
        action = (np.argmax(qval))  # take action with highest Q-value
        print("Move #: %s; Taking action: %s" % (i, action))
        state = make_move(state, action)
        print(disp_grid(state))
        reward = get_reward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward))

        i += 1
        if (i > 20):
            print("Game lost because taking more than 20 moves.")
            break
    return

#################################################################
# Provide a usage example
#################################################################
if __name__ == "__main__":
    epochs = 2000  # number of games to train on
    gamma = 0.9 #s ince it may take several moves to goal, making gamma high
    epsilon = 1 # probability of choosing a random action

    model = build_model()
    model = train_model(model, epochs, gamma, epsilon)
    test_algo(model, init=0)
    logger.info("ALL DONE!\n")

