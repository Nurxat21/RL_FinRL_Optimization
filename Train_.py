import copy
import gym
import torch
import random
import functools

import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from collections import deque

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class GradientPolicy(nn.Module):
    def __init__(self):
        """DDPG policy network initializer."""
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, 48)),
            nn.ReLU()
        )

        self.final_convolution = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1)
        )

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation .
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(device)
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action).to(device)

        last_stocks, cash_bias = self._process_last_action(last_action)

        output = self.sequential(observation) # shape [N, PORTFOLIO_SIZE + 1, 19, 1]
        output = torch.cat([output, last_stocks], dim=1) # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output) # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat([output, cash_bias], dim=2) # shape [N, 1, PORTFOLIO_SIZE + 1, 1]

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1) # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: environment observation (dictionary).
          epsilon: exploration noise to be applied.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

class PVM:
    def __init__(self, capacity):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
        """
        # initially, memory will have the same actions
        self.capacity = capacity
        self.reset()

    def reset(self):# 4 is stock_dim
        self.memory = [np.array([1] + [0] * 4, dtype=np.float32)] * (self.capacity + 1)
        self.index = 0 # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action

class ReplayBuffer:
    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
          capacity: Max capacity of buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
          Sample of batch_size size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        for experience in self.buffer.sample():
            yield experience

def polyak_average(net, target_net, tau=0.01):
    """Applies polyak average to incrementally update target net.

    Args:
    net: trained neural network.
    target_net: target neural network.
    tau: update rate.
    """
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)


class PG:
    def __init__(self,
                 env,
                 batch_size=100,
                 lr=1e-3,
                 optimizer=AdamW,
                 tau=0.05):
        """Initializes Policy Gradient for portfolio optimization.

          Args:
            env: environment.
            batch_size: batch size to train neural network.
            lr: policy neural network learning rate.
            optim: Optimizer of neural network.
            tau: update rate in Polyak averaging.
        """
        # environment
        self.env = env

        # neural networks
        self.policy = GradientPolicy().to(device)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.tau = tau

        # replay buffer and portfolio vector memory
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)
        self.pvm = PVM(self.env.episode_length)

        # dataset and dataloader
        dataset = RLDataset(self.buffer)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True)

    def train(self, episodes=100):
        """Training sequence

        Args:
            episodes: Number of episodes to simulate
        """
        for i in tqdm(range(1, episodes + 1)):
            obs = self.env.reset() # observation
            self.pvm.reset() # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = self.policy(obs_batch, last_action_batch)
                self.pvm.add(action)

                # run simulation step
                next_obs, reward, done, info = self.env.step(action)

                # add experience to replay buffer
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.buffer.append(exp)

                # update policy networks
                if len(self.buffer) == self.batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()



    def _gradient_ascent(self):
        # update target neural network
        polyak_average(self.policy, self.target_policy, tau=self.tau)

        # get batch data from dataloader
        obs, last_actions, price_variations, trf_mu = next(iter(self.dataloader))
        obs = obs.to(device)
        last_actions = last_actions.to(device)
        price_variations = price_variations.to(device)
        trf_mu = trf_mu.unsqueeze(1).to(device)

        # define policy loss (negative for gradient ascent)
        mu = self.policy.mu(obs, last_actions)
        policy_loss = - torch.mean(torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)))

        # update policy network
        self.policy.zero_grad()
        policy_loss.backward()
        self.optimizer.step()