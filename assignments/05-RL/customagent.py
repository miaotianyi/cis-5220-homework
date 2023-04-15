import gymnasium as gym

import math
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from contextlib import contextmanager


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    is_train = net.training
    try:
        net.eval()
        yield net
    finally:
        if is_train:
            net.train()


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size], dtype=np.float32)
        self.rewards_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr = 0
        self.size = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):  # -> Dict[str, np.ndarray]:
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idx],
            next_obs=self.next_obs_buf[idx],
            actions=self.actions_buf[idx],
            rewards=self.rewards_buf[idx],
            done=self.done_buf[idx],
        )

    def __len__(self) -> int:
        return self.size


class SimpleNet(nn.Module):
    def __init__(self, n_observations, n_actions, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Preprocessor:
    """
    A preprocessor for LunarLander-v2
    """

    def __init__(self, observation_space: gym.spaces.Box):
        """
        The preprocessor will compress all numerical values
        between 0 and 1.

        The first 6 features are continuous,
        the last 2 features are boolean.

        Parameters
        ----------
        observation_space: gym.spaces.Box
            The observation space for LunarLander-v2
        """
        self.h = 1
        self.low = observation_space.low[:6]
        self.range = (observation_space.high - observation_space.low)[:6]

    def __call__(self, observation: gym.spaces.Box):
        # return np.array(observation)
        return np.array(observation)
        # x = (observation[:6] - self.low) / self.range
        # w = np.geomspace(0.99 * math.tau, 1e5 * math.tau, num=self.h // 2)
        # x = x[..., None] * w
        # return np.concatenate(
        #     [np.sin(x).flatten(), np.cos(x).flatten(), observation[6:]]
        # )

    @property
    def n_features(self):
        return 4
        # return self.h * 6 + 2


class Agent:
    """
    My own agent for LunarLander-v2 reinforcement learning
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize the agent based on action space and observation space.

        Parameters
        ----------
        action_space: gym.spaces.Discrete

        observation_space: gym.spaces.Box
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.prep = Preprocessor(observation_space=observation_space)

        # observation size
        self.n_observations = self.prep.n_features
        self.n_actions = action_space.n

        # hyperparameters
        self.gamma = 0.99
        self.target_update = 5  # steps before updating target network
        self.batch_size = 32

        # replay buffer
        self.memory = ReplayBuffer(
            obs_dim=self.n_observations, size=10000, batch_size=self.batch_size
        )

        # network architecture
        self.h = 128
        self.learning_rate = 1e-3
        self.policy_net = SimpleNet(self.n_observations, self.n_actions, self.h)
        # initialize target network
        self.target_net = SimpleNet(self.n_observations, self.n_actions, self.h)
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # self.policy_net = DQN()
        # self.target_net = DQN(self.n_observations, self.n_actions, self.h)

        # global count tracker
        self.step_count = 0
        self.episode_count = 0

        # cache current state/action
        self.transition = []

    def epsilon(self):
        max_epsilon = 0.9
        min_epsilon = 0.01
        eps_decay = 100
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -1.0 * self.step_count / eps_decay
        )

    def _compute_dqn_loss(self, samples) -> torch.Tensor:
        """Return dqn loss."""
        state = torch.FloatTensor(samples["obs"])
        next_state = torch.FloatTensor(samples["next_obs"])
        action = torch.LongTensor(samples["actions"].reshape(-1, 1))
        reward = torch.FloatTensor(samples["rewards"].reshape(-1, 1))
        done = torch.FloatTensor(samples["done"].reshape(-1, 1))

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q_value = (
                self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
            )
            mask = 1 - done
            target = reward + self.gamma * next_q_value * mask

        # calculate dqn loss
        loss = F.mse_loss(curr_q_value, target)
        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Take an action (no grad) given an observation.

        Parameters
        ----------
        observation: gym.spaces.Box

        Returns
        -------
        int
        """
        self.step_count += 1
        obs = self.prep(observation)
        if random.random() > self.epsilon():
            with torch.no_grad(), evaluating(self.policy_net):
                obs = torch.tensor(obs, dtype=torch.float32)
                action = self.policy_net(obs.unsqueeze(0)).max(dim=1)[1].item()
        else:
            action = self.action_space.sample()
        self.transition = [obs, action]
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learn from one experience.

        Parameters
        ----------
        observation
        reward
        terminated
        truncated

        Returns
        -------

        """
        if truncated or terminated:
            self.episode_count += 1
        obs = self.prep(observation)
        # only terminated count as done; for truncated, the next obs is still useful
        self.transition += [reward, obs, terminated or truncated]
        self.memory.store(*self.transition)
        if len(self.memory) < self.batch_size:
            return

        # start learning
        if self.step_count % self.target_update == 0:
            samples = self.memory.sample_batch()
            loss = self._compute_dqn_loss(samples)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.step_count % self.target_update == 0:
            self._target_hard_update()


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    print(observation)
    prep = Preprocessor(env.observation_space)
    print(prep(observation))


if __name__ == "__main__":
    main()
