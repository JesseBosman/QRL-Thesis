""""This code closely follows the Tensorflow Quantum tutorial notebook
"Parametrized Quantum Circuits for Reinforcement Learning". Credits therefore are due to the developers of Tensorflow Quantum
"""

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

