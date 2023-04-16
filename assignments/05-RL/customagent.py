import gymnasium as gym

import math
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from contextlib import contextmanager

import operator
from typing import Callable, Union, Sequence


class SegmentTree:
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4):  # -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.actions_buf[indices]
        rews = self.rewards_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            actions=acts,
            rewards=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):  # -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


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
        self.original_num_features = observation_space.shape[0]
        self.h = 8
        # observation_space.high and low give wrong
        self.high = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5])
        self.low = -self.high
        self.range = self.high * 2

    def __call__(self, observation: gym.spaces.Box):
        # return np.array(observation)
        obs = np.array(observation)
        # obs[:6] = obs[:6] / self.high
        return obs
        x = observation[:6] / self.high
        w = np.geomspace(math.tau / 0.001, math.tau / 2.1, num=self.h // 2)
        x = x[..., None] * w
        x = np.concatenate([np.sin(x).flatten(), np.cos(x).flatten(), observation[6:]])
        return x

    @property
    def n_features(self):
        return self.original_num_features
        # return self.h * 6 + 2


class DQNAgent:
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
        self.batch_size = 64

        # replay buffer
        self.alpha: float = 0.2
        self.prior_eps: float = 1e-6
        # self.memory = ReplayBuffer(
        #     obs_dim=self.n_observations, size=10000, batch_size=self.batch_size
        # )
        self.memory = PrioritizedReplayBuffer(
            obs_dim=self.n_observations,
            size=10000,
            batch_size=self.batch_size,
            alpha=self.alpha,
        )

        # network architecture
        self.h = 128
        self.learning_rate = 1e-4
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

    def beta(self):
        beta_start = 0.6
        max_steps = 50_000
        return beta_start + (1.0 - beta_start) * (self.step_count / max_steps)

    def epsilon(self):
        max_epsilon = 0.7
        min_epsilon = 0.001
        eps_decay = 10000
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

        with torch.no_grad(), evaluating(self.policy_net):
            # next_q_value = (
            #     self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
            # )
            # Double DQN
            next_q_value = (
                self.target_net(next_state)
                .gather(1, self.policy_net(next_state).argmax(dim=1, keepdim=True))
                .detach()
            )
            mask = 1 - done
            target = reward + self.gamma * next_q_value * mask

        # calculate dqn loss (element-wise)
        elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")
        return elementwise_loss

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
            samples = self.memory.sample_batch(beta=self.beta())
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1))
            elementwise_loss = self._compute_dqn_loss(samples)
            loss = torch.mean(elementwise_loss * weights)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # update priorities
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(samples["indices"], new_priorities)

        if self.step_count % self.target_update == 0:
            self._target_hard_update()


# separation: PPO


# 1 torch.Tensor or a list of torch.Tensor's
# Useful for passing them into neural networks
SomeTensor = Union[torch.Tensor, Sequence[torch.Tensor]]


def stack_tensors(tensors: Sequence[SomeTensor]):
    """
    input is a list of tensors [x, x, x, ...] (each has the same shape)
    or a list of tuples of tensors [(x1, x2, x3, ...), (x1, x2, x3, ...), ...]
    each x or each xi (x1, x2, x3) has the same Tensor shape
    output is a Tensor y or a tuple (y1, y2, y3, ...)
    where yi is obtained by stacking all xi along a new batch dimension using `torch.stack`
    so each xi should not have the extra batch dimension like the 1 in [1, C, H, W]
    """
    if isinstance(tensors[0], torch.Tensor):
        # if input is a list of tensors, stack them along a new batch dimension
        return torch.stack(tensors)
    else:
        # if input is a list of tuples, stack each element of the tuples
        return tuple(torch.stack(t) for t in zip(*tensors))


def cat_tensors(tensors: Sequence[SomeTensor]):
    # same as above, except cat is used instead of stack (no new dimension)
    if isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors)
    else:
        # if input is a list of tuples, stack each element of the tuples
        return tuple(torch.cat(t) for t in zip(*tensors))


def move_tensors(tensors: SomeTensor, device=None, dtype=None):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device, dtype=dtype)
    else:
        return tuple(t.to(device=device, dtype=dtype) for t in tensors)


def select_tensors(tensors: SomeTensor, idx):
    # tensors is a tensor or a tuple of tensors
    if isinstance(tensors, torch.Tensor):
        return tensors[idx]
    else:
        return tuple(t[idx] for t in tensors)


def monte_carlo_returns(rewards: Sequence, gamma: float = 1.0) -> np.ndarray:
    """
    Without gradients, calculate the return at each timestep
    as the discounted sum of rewards.

    This is a more "vanilla" version of estimating returns from a trajectory,
    using Monte Carlo estimation alone.
    So it might suffer from high variance

    Parameters
    ----------
    rewards : list of float
        The scalar reward collected at each timestep.

    gamma : float, default = 1
        Discount rate

    Returns
    -------
    returns : list of float
        With the same length as rewards
    """
    if gamma == 1:  # faster?
        returns = np.cumsum(rewards[::-1])[::-1]
    else:
        n_steps = len(rewards)  # number of time steps
        returns = np.zeros(n_steps)
        returns[-1] = rewards[-1]
        for t in reversed(range(n_steps - 1)):  # time step within trajectory/episode
            returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def gae_advantages(
    rewards: torch.Tensor, critic_values: torch.Tensor, gamma: float, gae_lambda: float
):
    # Single trajectory generalized advantage estimation
    # Without gradients, compute a list of advantage based on
    # estimated values and rewards per timestep.
    # rewards, critic_values should have the same length n (the last terminal state is excluded)
    # it's better to calculate the entire thing in CPU
    with torch.no_grad():
        n_steps = len(rewards)
        old_device = rewards.device
        # rewards = torch.tensor(rewards, device="cpu")
        rewards = rewards.detach().clone().cpu()
        critic_values = critic_values.detach().clone().cpu()
        advantages = torch.zeros(n_steps, device="cpu")
        advantages[-1] = rewards[-1] - critic_values[-1]
        for t in reversed(range(n_steps - 1)):  # from n_steps-2 to 0 inclusive
            delta = rewards[t] + gamma * critic_values[t + 1] - critic_values[t]
            advantages[t] = delta + gamma * gae_lambda * advantages[t + 1]
        return advantages.to(device=old_device)


def unclipped_critic_loss(old_values: torch.Tensor, critic_values: torch.Tensor):
    # use advantage + critic_values instead of vanilla trajectory returns (discounted sum of rewards)
    # choosing lambda=1 for gae still has higher bias/lower variance than vanilla returns
    # critic_values have gradients, advantages/returns don't
    # with torch.no_grad():
    #     advantages = torch.tensor(advantages, dtype=critic_values.dtype, device=critic_values.device)
    #     returns = advantages + critic_values.detach()
    vf_loss = F.mse_loss(critic_values, old_values)
    return 0.5 * vf_loss


def clipped_critic_loss(
    critic_values: torch.Tensor,
    old_values: torch.Tensor,
    old_returns: torch.Tensor,
    clip: float,
):
    """
    Clipped value function loss.

    central idea: old advantages/old values are based on the current (old) network
    it's only computed once for all current trajectories in this agent-epoch
    the old network is copied from the new network at the end of each agent-epoch
    The critic values change per PPO-epoch.

    In an agent-epoch, the agent interacts with the env to obtain trajectories,
    then updates its internal parameters.
    In a PPO-epoch, the parameters of the network are updated once,
    by iterating over mini-batches of current trajectories


    Parameters
    ----------
    critic_values: torch.Tensor
        The only input argument with gradients tracked.

        Has length n, where n is the number of time steps in the trajectory.
        (not including the terminal state)

    old_values: torch.Tensor
        The value estimation by old critic.
        Has length n, where n is the number of time steps in the trajectory.

    old_returns: torch.Tensor
        The returns calculated by the old critic (value function)
        from previous agent-epoch.

        Precomputed as `old_returns = old_values + old_advantages`.

        Has length n, where n is the number of time steps in the trajectory.

    clip: float
        The positive clip hyperparameter

    Returns
    -------
    scalar
    """

    clipped_values = old_values + (critic_values - old_values).clamp(
        min=-clip, max=clip
    )
    vf_loss = torch.maximum(
        (critic_values - old_returns) ** 2, (clipped_values - old_returns) ** 2
    )
    return 0.5 * vf_loss.mean()


def clipped_ppo_loss(actor_log_probs, old_log_probs, old_advantages, clip):
    """

    Parameters
    ----------
    actor_log_probs: torch.Tensor
        1D tensor of log probability of taking the action at each time step;
        tracks gradient.

    old_log_probs: torch.Tensor
        Action log probability scored by the old network.

    old_advantages: torch.Tensor
        Action advantages scored by the old network

    clip: float

    Returns
    -------
    scalar
    """
    ratio = torch.exp(actor_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
    # objective: bigger is better!
    policy_objective = torch.minimum(
        ratio * old_advantages, clipped_ratio * old_advantages
    )
    return -policy_objective.mean()


def fuse_trajectories(trajectories, old_agent, gamma, gae_lambda, device=None):
    # concatenate multiple trajectories using GAE (for PPO)
    # old agent is the behavioral policy (that goes out and explore in the wild)
    # use dictionary to prevent argument order errors
    # `states` is either:
    # a n_steps-length tensor
    # a tuple of n_steps-length tensors (e.g. CNN input)
    # a list of tensors (e.g. transformer inputs)
    # a list of tuples of tensors
    sample = {
        "states": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "log_probs": [],
        "advantages": [],
        "returns": [],
    }
    with torch.no_grad(), evaluating(old_agent):
        old_agent.to(device=device)
        for i, (states, actions, rewards) in enumerate(trajectories):
            # `states` is a list of states, each state is a SomeTensor
            # `actions` is a list of actions, each action is a SomeTensor
            # `rewards` is a list of rewards, each reward is a torch.Tensor scalar
            n_steps = len(actions)
            states = states[:n_steps]  # remove the terminal state
            # output states is a list of (N, ...) tensors (where the sample axis is axis 0)
            # this assumes that the agent can take batched (N, ...), (N, ...) states and actions
            states = move_tensors(stack_tensors(states), device=device)
            actions = move_tensors(stack_tensors(actions), device=device)
            rewards = torch.tensor(
                rewards, device=device, dtype=torch.float32
            )  # same device as input tensors
            values, log_probs, _ = old_agent.score(states, actions)  # no entropy
            values = values.flatten()
            log_probs = log_probs.flatten()
            advantages = gae_advantages(
                rewards=rewards,
                critic_values=values,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            returns = values + advantages
            # append results for this trajectory
            sample["states"].append(states)
            sample["actions"].append(actions)
            sample["rewards"].append(rewards)
            sample["values"].append(values)
            sample["log_probs"].append(log_probs)
            sample["advantages"].append(advantages)
            sample["returns"].append(returns)

    # final cleanup
    sample["states"] = cat_tensors(sample["states"])
    sample["actions"] = cat_tensors(sample["actions"])

    # scalar per time step, concat
    sample["rewards"] = torch.cat(sample["rewards"])
    sample["values"] = torch.cat(sample["values"])
    sample["log_probs"] = torch.cat(sample["log_probs"])
    sample["advantages"] = torch.cat(sample["advantages"])
    sample["returns"] = torch.cat(sample["returns"])
    return sample


def batch_ppo_iter(
    trajectories,
    agent,
    old_agent,
    optimizer,
    gamma,
    gae_lambda,
    ppo_epochs,
    batch_size,
    vf_coef,
    ent_coef,
    vf_clip,
    ppo_clip,
    device=None,
):
    # PPO iteration that supports multiple trajectories/episodes
    # and arbitrary state/agent shape (e.g. tensor, tuple of tensors)
    sample = fuse_trajectories(
        trajectories,
        old_agent=old_agent,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    # prepare mini-batches for gradient descent
    n_total_steps = len(sample["rewards"])

    agent.to(device=device)
    agent.train()

    for epoch in range(ppo_epochs):
        batch_indices = torch.split(
            torch.randperm(n_total_steps, device=device),
            split_size_or_sections=batch_size,
        )
        for batch_idx in batch_indices:
            # states have the same shape; all actions have the same shape
            # b_ prefix means "batch"
            b_states = select_tensors(sample["states"], batch_idx)
            b_actions = select_tensors(sample["actions"], batch_idx)
            b_old_log_probs = sample["log_probs"][batch_idx]
            b_old_advantages = sample["advantages"][batch_idx]
            b_old_values = sample["values"][batch_idx]
            b_old_returns = sample["returns"][batch_idx]
            b_new_values, b_new_log_probs, b_new_entropy = agent.score(
                b_states, b_actions
            )  # tracks gradient
            vf_loss = clipped_critic_loss(
                critic_values=b_new_values,
                old_values=b_old_values,
                old_returns=b_old_returns,
                clip=vf_clip,
            )
            ppo_loss = clipped_ppo_loss(
                actor_log_probs=b_new_log_probs,
                old_log_probs=b_old_log_probs,
                old_advantages=b_old_advantages,
                clip=ppo_clip,
            )
            entropy_loss = (
                -b_new_entropy.mean()
            )  # bigger entropy, better (more explore)
            loss = ppo_loss + vf_coef * vf_loss + ent_coef * entropy_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class MulticlassMLPAgent(nn.Module):
    def __init__(self, n_features, n_actions, d=64):
        """
        A simple MLP backbone for a PPO agent,
        where the input features are all numerical,
        and the output is the choice within n actions.
        """
        super().__init__()
        self.d = d
        self.n_features = n_features
        self.n_actions = n_actions
        self.extractor = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def sample(self, state):
        state = self.extractor(state)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def score(self, state, action):
        state = self.extractor(state)

        logits = self.actor(state)

        dist = torch.distributions.Categorical(logits=logits)
        estimated_value = self.critic(state).flatten()
        action_log_prob = dist.log_prob(action.flatten())
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy


class PPOAgent:
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
        self.gamma = 0.999
        self.gae_lambda = 0.9
        self.ppo_epochs = 6
        self.vf_coef = 0.001
        self.ent_coef = 1.0
        self.vf_clip = 10.0
        self.ppo_clip = 0.1

        self.batch_size = 64
        self.episode_epochs = 5  # number of exploratory episodes before PPO update

        # network architecture
        self.h = 64
        self.agent = MulticlassMLPAgent(self.n_observations, self.n_actions, d=self.h)
        # old agent has the same architecture
        self.old_agent = MulticlassMLPAgent(
            self.n_observations, self.n_actions, d=self.h
        )
        self.old_agent.load_state_dict(self.agent.state_dict())
        for p in self.old_agent.parameters():
            p.requires_grad_(False)  # doesn't require gradient in old agent
        self.learning_rate = 3e-4
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate)

        # global count tracker
        self.step_count = 0
        self.episode_count = 0

        self.taus = []
        self.reset_trajectory_buffer()

    def reset_trajectory_buffer(self):
        # cache current state/action
        self.taus = [([], [], []) for _ in range(self.episode_epochs)]

    def update_sa(self, obs, action):
        tau_id = self.episode_count % self.episode_epochs
        self.taus[tau_id][0].append(obs)
        self.taus[tau_id][1].append(torch.tensor(action))

    def update_r(self, reward):
        tau_id = self.episode_count % self.episode_epochs
        self.taus[tau_id][2].append(reward)

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
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad(), evaluating(self.old_agent):
            action = self.old_agent.sample(obs).item()
        self.update_sa(obs, action)
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
        self.update_r(reward)
        if truncated or terminated:
            self.episode_count += 1

            if self.episode_count % self.episode_epochs == 0:
                batch_ppo_iter(
                    self.taus,
                    agent=self.agent,
                    old_agent=self.old_agent,
                    optimizer=self.optimizer,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    ppo_epochs=self.ppo_epochs,
                    batch_size=self.batch_size,
                    vf_coef=self.vf_coef,
                    ent_coef=self.ent_coef,
                    vf_clip=self.vf_clip,
                    ppo_clip=self.ppo_clip,
                )
                self.reset_trajectory_buffer()
                self.old_agent.load_state_dict(self.agent.state_dict())


class Agent(PPOAgent):
    pass


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    print(observation)
    prep = Preprocessor(env.observation_space)
    print(prep(observation))


if __name__ == "__main__":
    main()
