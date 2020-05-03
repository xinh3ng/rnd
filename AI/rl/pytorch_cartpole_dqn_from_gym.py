"""

# Links

"""
from collections import namedtuple, deque
from itertools import count

import gym
import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from typing import List, Tuple

plt.ion()

Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])


class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.PReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.PReLU()
        )
        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Agent(object):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, eps: float) -> int:
        """Returns an action (𝜺-greedy)
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float):
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_q(states)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)

    def train(self, q_pred: torch.FloatTensor, q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(q_pred, q_true)
        loss.backward()
        self.optim.step()
        return loss


def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(q_pred, q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    q_predict = agent.get_q(states)
    #
    q_target = q_predict.clone().data.numpy()
    q_target[np.arange(len(q_target)), actions] = rewards + gamma * np.max(
        agent.get_q(next_states).data.numpy(), axis=1
    ) * (~done)
    q_target = agent._to_variable(q_target)
    return agent.train(q_predict, q_target)


def play_episode(
    env: gym.Env, agent: Agent, replay_memory: ReplayMemory, eps: float, batch_size: int, gamma: float
) -> int:
    """Play an episode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float):
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    state = env.reset()
    done, total_reward = False, 0
    while not done:
        a = agent.get_action(state, eps)
        state_2, reward, done, info = env.step(a)
        total_reward += reward
        if done:
            reward = -1  # Game lost, so terminal reward is -1

        replay_memory.push(state, a, reward, state_2, done)
        if len(replay_memory) > batch_size:
            minibatch = replay_memory.pop(batch_size)
            train_helper(agent=agent, minibatch=minibatch, gamma=gamma)

        state = state_2
    return total_reward


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment (CartPole-v0)
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    return input_dim, output_dim


def epsilon_annealing(episode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        episode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """
    slope = (min_eps - 1.0) / max_episode
    return max(slope * episode + 1.0, min_eps)


#############################################################################################


def main(
    env: str,
    n_episodes: int,
    batch_size: int,
    gamma: float,
    hidden_dim: int,
    capacity: int,
    max_episode: int,
    min_eps: float,
):
    env = gym.make(env)
    env = gym.wrappers.Monitor(env, directory="monitors", force=True)
    rewards = deque(maxlen=100)
    input_dim, output_dim = get_env_dim(env)
    agent = Agent(input_dim, output_dim, hidden_dim)
    replay_memory = ReplayMemory(capacity)

    for episode_idx in range(n_episodes):
        eps = epsilon_annealing(episode_idx, max_episode, min_eps)
        reward = play_episode(env, agent, replay_memory, eps, batch_size, gamma=gamma)
        print("[Episode: {:5}] Reward: {:5} epsilon-greedy: {:5.2f}".format(episode_idx + 1, reward, eps))

        rewards.append(reward)
        if len(rewards) == rewards.maxlen:
            if np.mean(rewards) >= 200:
                print("Game cleared in {} episodes with avg rewards {}".format(episode_idx + 1, np.mean(rewards)))
                break
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v0", help="Gym environment name")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount rate for q_target")
    parser.add_argument("--hidden_dim", type=int, default=12, help="Hidden dimension")
    parser.add_argument("--capacity", type=int, default=50000, help="Replay memory capacity")
    parser.add_argument(
        "--max_episode", type=int, default=100, help="e-Greedy target episode (eps will be the lowest at this episode)"
    )
    parser.add_argument("--min_eps", type=float, default=0.01, help="Min epsilon")
    args = vars(parser.parse_args())
    print("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    print("\nALL DONE!\n")
