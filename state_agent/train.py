import numpy as np
from torch.distributions import Bernoulli

from player_draft import Player
from state_agent.planner import network_features, Planner
from utils import show_agent, rollout_many,show_viz_rolloutagent
import torch
import copy
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    device = None
    if torch.backends.mps.is_available():
        print("MPS device  found.")
        device = torch.device("mps")
        x = torch.ones(8, device=device)
        print(x)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    action_net = Planner(17, 32, 3)
    actor = Player(action_net)
    many_action_nets = [Planner(17, 32, 3) for i in range(10)]
    data = rollout_many([Player(action_net) for action_net in many_action_nets], n_steps=600)
    good_initialization = many_action_nets[np.argmax([d[-1]['overall_distance'] for d in data])]
    viz_rollout = show_agent(actor, n_steps=600)

    show_viz_rolloutagent(actor,viz_rollout, n_steps=600)

    n_epochs = 10
    n_trajectories = 10
    n_iterations = 50
    batch_size = 128

    action_net = copy.deepcopy(good_initialization)

    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)

    for epoch in range(n_epochs):

        # Roll out the policy, compute the Expectation
        trajectories = rollout_many([Player(action_net)] * n_trajectories, n_steps=600)
        rewards = [t[-1]['reward_state'] for t in trajectories]

        print(f"Epoch = {epoch}")
        print(f"Best distance = {np.max([t[-1]['overall_distance'] for t in trajectories])}")
        print(f"Average reward: {np.mean(rewards)}")
        print(f"Min reward: {np.min(rewards)}")
        print(f"Max reward: {np.max(rewards)}")

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
            acceleration = torch.sigmoid(output[:, 0])
            steering = torch.tanh(output[:, 1])
            braking = torch.sigmoid(output[:, 2])

            log_prob_acceleration = (batch_actions * torch.log(acceleration) + (1 - batch_actions) * torch.log(
                1 - acceleration)).sum()
            log_prob_steering = torch.log(1 - torch.abs(batch_actions - steering)).sum()
            log_prob_braking = (batch_actions * torch.log(braking) + (1 - batch_actions) * torch.log(1 - braking)).sum()

            expected_log_return = (log_prob_acceleration + log_prob_steering + log_prob_braking) * batch_returns
            optim.zero_grad()
            (-expected_log_return.mean()).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return.mean()))

    show_viz_rolloutagent(Player(action_net), viz_rollout)