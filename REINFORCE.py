""""This code closely follows the Tensorflow Quantum tutorial notebook
"Parametrized Quantum Circuits for Reinforcement Learning". Credits therefore are due to the developers of Tensorflow Quantum
"""

import tensorflow as tf

# import tensorflow_quantum as tfq

# import cirq, sympy
# import gymnasium as gym
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# from cirq.contrib.svg import SVGCircuit

from fox_in_a_hole_gym import FIAH, Givens


class reinforce_agent:
    def __init__(self, env_name, batch_size, max_steps, len_state, transfer_matrices):
        self.env_name = env_name
        self.max_steps = max_steps
        self.len_state = len_state
        self.transfer_matrices = transfer_matrices
        self.batch_size = batch_size
        self.envs = self.create_environments(batch_size)
        pass

    def create_environments(self, batch_size):
        if self.env_name.lower() == "fiah":
            envs = [
                FIAH(self.max_steps, self.len_state, self.transfer_matrices)
                for _ in range(batch_size)
            ]

        elif self.env_name.lower() == "givens":
            envs = [
                Givens(self.max_steps, self.len_state, self.transfer_matrices)
                for _ in range(batch_size)
            ]

        elif self.env_name.lower() == "qfiah":
            raise ValueError("QFIAH has been disabled to to work in progress")
            envs = [
                QFIAH(self.max_steps, self.len_state, self.transfer_matrices)
                for _ in range(batch_size)
            ]

        else:
            raise KeyError

        return envs

    def gather_episodes_training(self, n_actions, model):
        """Interact with environment in batched fashion to gather training episodes."""

        batch_size = self.batch_size
        model_returns_nan = False

        trajectories = [defaultdict(list) for _ in range(batch_size)]

        envs = self.envs

        done = [False for _ in range(batch_size)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(batch_size) if not done[i]]
            states = [s for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, states):
                trajectories[i]["states"].append(state)

            states = tf.convert_to_tensor(states)
            # Compute policy for all unfinished envs in parallel
            action_probs = model(states)
            # Store action and transition all environments to the next state
            states = [None for i in range(batch_size)]
            if np.isnan(np.sum(action_probs.numpy())):
                model_returns_nan = True
                return {}, model_returns_nan

            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(n_actions, p=policy)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]["actions"].append(action)
                trajectories[i]["rewards"].append(reward)

        return trajectories, model_returns_nan

    def gather_episodes_greedy(self, model, batch_size):
        """Interact with environment in batched fashion to gather evaluation episodes."""

        model_returns_nan = False

        trajectories = [defaultdict(list) for _ in range(batch_size)]

        envs = self.create_environments(batch_size)

        done = [False for _ in range(batch_size)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(batch_size) if not done[i]]
            states = [s for i, s in enumerate(states) if not done[i]]
            for i, state in zip(unfinished_ids, states):
                trajectories[i]["states"].append(state)

            states = tf.convert_to_tensor(states)
            # Compute policy for all unfinished envs in parallel
            action_probs = model(states)
            # Store action and transition all environments to the next state
            states = [None for i in range(batch_size)]
            if np.isnan(np.sum(action_probs.numpy())):
                model_returns_nan = True
                return {}, model_returns_nan

            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.argmax(policy)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]["actions"].append(action)
                trajectories[i]["rewards"].append(reward)

        return trajectories, model_returns_nan

    def compute_returns(self, rewards_history, gamma):
        """Compute discounted returns with discount factor `gamma`."""
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize them for faster and more stable learning
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        return returns
