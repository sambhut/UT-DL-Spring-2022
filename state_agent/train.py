import numpy as np
from torch.distributions import Bernoulli

from player_draft import Player
from state_agent.planner import network_features, Planner
from utils import show_agent, rollout_many
import torch
import copy

if __name__ == "__main__":

    device = None
    if torch.backends.mps.is_available():
        print("MPS device  found.")
        device = torch.device("mps")
        x = torch.ones(8, device=device)
        print(x)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    action_net = Planner(17, 32, 1)
    actor = Player(action_net)
    many_action_nets = [Planner(17, 32, 1) for i in range(10)]
    data = rollout_many([Player(action_net) for action_net in many_action_nets], n_steps=600)
    good_initialization = many_action_nets[np.argmax([d[-1]['overall_distance'] for d in data])]
    show_agent(actor, n_steps=600)

    n_epochs = 20
    n_trajectories = 10
    n_iterations = 50
    batch_size = 128
    T = 20

    action_net = copy.deepcopy(good_initialization)

    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        eps = 1e-2

        # Roll out the policy, compute the Expectation
        trajectories = rollout_many([Player(action_net)] * n_trajectories, n_steps=600)

        print('epoch = %d   best_dist = ' % epoch,
              np.max([t[-1]['overall_distance'] for t in trajectories]))

        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                # Compute the returns
                returns.append(trajectory[i]['reward_state'])
                # Compute the features
                features.append(torch.as_tensor(
                    network_features(trajectory[i]['player_state'], trajectory[i]['opponent_state'],
                                     trajectory[i]['soccer_state']), dtype=torch.float32).cuda().view(-1))
                # Store the action that we took
                actions.append(trajectory[i]['action'].steer > 0)

        # Upload everything to the GPU
        returns = torch.as_tensor(returns, dtype=torch.float32).cuda()
        actions = torch.as_tensor(actions, dtype=torch.float32).cuda()
        features = torch.stack(features).cuda()

        returns = (((returns - returns.mean()) / returns.std()) + 1e-8)

        action_net.train().cuda()
        avg_expected_log_return = []
        for it in range(n_iterations):
            batch_ids = torch.randint(0, len(returns), (batch_size,), device=device)
            batch_returns = returns[batch_ids]
            batch_actions = actions[batch_ids]
            batch_features = features[batch_ids]

            output = action_net(batch_features)
            pi = Bernoulli(logits=output[:, 0])

            expected_log_return = (pi.log_prob(batch_actions) * batch_returns).mean()
            optim.zero_grad()
            (-expected_log_return).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return))

        best_performance, current_performance = rollout_many([Player(best_action_net), Player(action_net)],
                                                             n_steps=600)
        if best_performance[-1]['overall_distance'] < current_performance[-1][
            'overall_distance']:
            best_action_net = copy.deepcopy(action_net)
    # %%
    show_agent(Player(best_action_net))