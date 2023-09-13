from REINFORCE import reinforce_agent
from NN import PolicyModel
import numpy as np

env_name = "FoxInAHole"
n_episodes = 5000
batch_size=100
n_actions = 5
state_bounds = 1
gamma = 1
input_dim = 5
learning_rate = 0.01

averaging_window = 1000

agent = reinforce_agent(batch_size=batch_size)

# Start training the agent
model = PolicyModel(n_hidden_layers=2, n_nodes_per_layer=5, input_dim= input_dim, output_dim=n_actions, learning_rate= learning_rate)
episode_reward_history = []
for batch in range(n_episodes // batch_size):
    # Gather episodes
    
    episodes = agent.gather_episodes(state_bounds, input_dim, n_actions, model, batch_size, env_name)

    # Group states, actions and returns in numpy arrays
    states = np.concatenate([ep['states'] for ep in episodes])
    actions = np.concatenate([ep['actions'] for ep in episodes])
    rewards = [ep['rewards'] for ep in episodes]
    returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

    returns = np.array(returns, dtype=np.float32)
    id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

    # Update model parameters.
    model.update_reinforce(states, id_action_pairs, returns, batch_size=batch_size)

    # Store collected rewards
    for ep_rwds in rewards:
        episode_reward_history.append(np.sum(ep_rwds))

    avg_rewards = np.mean(episode_reward_history[-averaging_window:])

    print('Finished episode', (batch + 1) * batch_size,
          'Average rewards: ', avg_rewards)

    if avg_rewards >= 500.0:
        break