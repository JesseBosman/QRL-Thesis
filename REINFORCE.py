""""This code closely follows the Tensorflow Quantum tutorial notebook
"Parametrized Quantum Circuits for Reinforcement Learning". Credits therefore are due to the developers of Tensorflow Quantum
"""

import tensorflow as tf
# import tensorflow_quantum as tfq

# import gym, cirq, sympy
import gymnasium as gym
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
# from cirq.contrib.svg import SVGCircuit

class reinforce_agent():
  def __init__(self, batch_size):
    self.batch_size=batch_size
    self.brain = None
    pass

# dit moet in de PCQ class denk ik
  # @tf.function
  # def update(states, actions, returns, model):
  #   # need to be able to specify which optimizer and which model
  #   states = tf.convert_to_tensor(states)
  #   actions = tf.convert_to_tensor(actions)
  #   returns = tf.convert_to_tensor(returns)

  #   with tf.GradientTape() as tape:
  #       tape.watch(model.trainable_variables)
  #       logits = model(states)
  #       p_actions = tf.gather_nd(logits, actions)
  #       log_probs = tf.math.log(p_actions)
  #       loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
  #   grads = tape.gradient(loss, model.trainable_variables)
  #   for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
  #       optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

  def gather_episodes(self,state_bounds, n_actions, model, batch_size, env_name):
      """Interact with environment in batched fashion."""

      trajectories = [defaultdict(list) for _ in range(batch_size)]
      envs = [gym.make(env_name) for _ in range(batch_size)]

      done = [False for _ in range(batch_size)]
      states = [e.reset()[0] for e in envs]

      while not all(done):
          unfinished_ids = [i for i in range(batch_size) if not done[i]]
          # normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]
          normalized_states = [s for i, s in enumerate(states) if not done[i]]

          for i, state in zip(unfinished_ids, normalized_states):
              trajectories[i]['states'].append(state)

          # Compute policy for all unfinished envs in parallel
          states = tf.convert_to_tensor(normalized_states)
          action_probs = model(states)

          # Store action and transition all environments to the next state
          states = [None for i in range(batch_size)]
          for i, policy in zip(unfinished_ids, action_probs.numpy()):
              action = np.random.choice(n_actions, p=policy)
              states[i], reward, done[i], _ = envs[i].step(action)[:-1] # this slicing is added because otherwise tensorflow adds a {} to the list.
              trajectories[i]['actions'].append(action)
              trajectories[i]['rewards'].append(reward)

      # print('action_probs')
      # print(action_probs)
      # print('trajectories')
      # print(trajectories)
      
      return trajectories

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
