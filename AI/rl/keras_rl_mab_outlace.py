# |/usr/bin/env python

"""
http://outlace.com/rlpart1.html
"""
from __future__ import division
from pdb import set_trace as debug
import numpy as np
from scipy import stats
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def reward(prob, n=10):
    """

    :param prob: Probability of winning
    :param n:  No. of plays
    :return:
    """
    total = 0
    for i in range(n):
        if random.random() < prob:
            total += 1
        return total


def best_arm(av):
    """Find the best arm

    :param av: action-value array
    :return:
    """
    return np.argmax(av)


class EpsGreedyArmSelector(object):
    def __init__(self, eps=0.1):
        self.eps = eps

    def choose(self, av, probs):
        if random.random() > self.eps:  # choose to exploit
            choice = best_arm(av)
        else:  # choose to explore
            choice = np.random.choice(len(av))
        return choice

    def update(self, av):
        return None


class SoftmaxArmSelector(object):
    def __init__(self, tau=1.12):
        self.tau = tau

    def choose(self, av, probs):
        choice = np.where(arms == np.random.choice(arms, p=probs))[0][0]
        return choice

    def update(self, av):
        """Update softmax probabilities for next play

        :param av:
        :return:
        """
        n = len(av)
        denominator = np.sum(np.exp(av[:] / self.tau))
        probs = np.array([np.exp(av[i] / self.tau) / denominator for i in range(n)])
        return probs


#################
# Parameters
################
n = 10  # number of arms
arms = np.random.rand(n)  # Probability of each arm
num_rounds = 500

algo_name = "softmax"  # eps_greedy, softmax
eps = 0.1
tau = 1.12

av = np.ones(n)  # action-value array
counts = np.zeros(n)  # counts of how many times we've taken a particular action
probs = (1.0 / n) * np.ones(n)  # initialize each action to have equal probability

logger.info("%d arms are: %s" % (len(arms), arms))

if algo_name == "eps_greedy":
    arm_selector = EpsGreedyArmSelector(eps=eps)
elif algo_name == "softmax":
    arm_selector = SoftmaxArmSelector(tau=tau)
else:
    raise NotImplementedError

running_mean = np.zeros(num_rounds)
for round in range(num_rounds):
    choice = arm_selector.choose(av, probs)
    counts[choice] += 1

    rwd = reward(arms[choice])  # Calculate reward of the chosen arm
    old_avg = av[choice]
    av[choice] = old_avg + (1 / counts[choice]) * (rwd - old_avg)  # update running avg
    probs = arm_selector.update(av)

    # get a weighted average
    running_mean[round] = np.average(av, weights=np.array([counts[j] / np.sum(counts) for j in range(len(counts))]))
    plt.scatter(round, running_mean[round])

max_arm = np.argmax(av)
logger.info("Max arm is %d with av: %.4f" % (max_arm, av[max_arm]))

logger.info("running_mean is %s" % running_mean)
plt.xlabel("Round of play")
plt.ylabel("Mean reward")
plt.show()

logger.info("ALL DONE\n")
