from REINFORCE import reinforce_agent
from NN import PolicyModel
import numpy as np
from plot import plot
from tqdm import tqdm

env_name = "FoxInAHole"
n_episodes = 200000
n_holes = 10
batch_size=1000
n_actions = n_holes
state_bounds = 1
gamma = 1
input_dim = 2*n_holes -2
learning_rate = 0.001

plotting = True
print_avg = False

averaging_window = 5000

agent = reinforce_agent(batch_size=batch_size)

# Start training the agent
model = PolicyModel(n_hidden_layers=2, n_nodes_per_layer=20, input_dim= input_dim, output_dim=n_actions, learning_rate= learning_rate)
episode_reward_history = []
for batch in tqdm(range(n_episodes // batch_size)):
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

    
    if print_avg:
        avg_rewards = np.mean(episode_reward_history[-averaging_window:])
        print('Finished episode', (batch + 1) * batch_size,
          'Average rewards: ', avg_rewards)

if plotting:
    plot(episode_reward_history, "NN", averaging_window)

