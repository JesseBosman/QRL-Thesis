from REINFORCE import reinforce_agent
from NN import PolicyModel
import numpy as np

env_name = "CartPole-v1"
n_episodes = 10000
batch_size=10
n_actions = 2
state_bounds = np.array([4.8000002, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38])
gamma = 1
input_dim = 4

agent = reinforce_agent(batch_size=batch_size)

# Start training the agent
episode_reward_history = []
for batch in range(n_episodes // batch_size):
    # Gather episodes
    model = PolicyModel(input_dim=input_dim, n_hidden_layers=2, n_nodes_per_layer=5)
    episodes = agent.gather_episodes(state_bounds, n_actions, model, batch_size, env_name)

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

    avg_rewards = np.mean(episode_reward_history[-10:])

    print('Finished episode', (batch + 1) * batch_size,
          'Average rewards: ', avg_rewards)

    if avg_rewards >= 500.0:
        break