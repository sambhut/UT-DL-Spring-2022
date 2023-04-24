import numpy as np
import torch
import warnings
import copy

from state_agent import Team
from state_agent.critic import ValueNetwork
from state_agent.custom_runner import record_manystate
from state_agent.planner import network_features, Planner, network_features_v2
import os

warnings.filterwarnings("ignore", category=UserWarning)



def save_model(model, filename):
    from torch import save
    from os import path
    if isinstance(model, Planner) or isinstance(model, ValueNetwork):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), filename))
    raise ValueError("model type '%s' not supported!" % str(type(model)))



if __name__ == "__main__":
    device = None
    if torch.backends.mps.is_available():
        print("MPS device found.")
        device = torch.device("mps")
        x = torch.ones(8, device=device)
        print(x)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    player1 = Planner(17, 32, 3).to(device)
    player2 = Planner(17, 32, 3).to(device)
    value_net1 = ValueNetwork(17, 32).to(device)
    value_net2 = ValueNetwork(17, 32).to(device)
    actor = Team(player1,player2)
    many_action_nets = [Team(Planner(17, 32, 3),Planner(17, 32, 3)) for i in range(10)]
    data = record_manystate(many_action_nets)
    good_initialization = many_action_nets[np.argmax([d[-1]['team1']['highest_distance'] for d in data])]

#####################################################
    ##  HYPER PARAMS ##
    n_epochs = 100
    n_trajectories = 10
    n_iterations = 50
    batch_size = 128
    ppo_eps = 0.2
######################################################

    player1_net = copy.deepcopy(good_initialization.model0)
    player2_net = copy.deepcopy(good_initialization.model1)

    best_player1_net = copy.deepcopy(player1_net)
    best_player2_net = copy.deepcopy(player2_net)

    player1_model_filepath = "player1_action_model.pt"
    player2_model_filepath = "player2_action_model.pt"
    value1_model_filepath = "player1_value_model.pt"
    value2_model_filepath = "player2_value_model.pt"


    action_optim1 = torch.optim.Adam(player1_net.parameters(), lr=0.005 , weight_decay=1e-5)
    action_optim2 = torch.optim.Adam(player2_net.parameters(), lr=0.005, weight_decay=1e-5)
    value_optim1 = torch.optim.Adam(value_net1.parameters(), lr=0.005, weight_decay=1e-5)
    value_optim2 = torch.optim.Adam(value_net2.parameters(), lr=0.005, weight_decay=1e-5)

    team_rewards = []
    best_team_reward = -np.inf

    for epoch in range(n_epochs):
        trajectories = record_manystate([Team(player1_net,player2_net)] * n_trajectories)
        rewards = [step_data['reward_state'] for step_tuple in trajectories for step_data in step_tuple[0]]

        team_reward = np.mean([step_data['reward_state'] for step_tuple in trajectories for step_data in step_tuple[0]])
        team_rewards.append(team_reward)

        print(f"Epoch = {epoch}")
        #print(f"Best distance = {np.max([t[-1]['overall_distance'] for t in trajectories])}")
        print(f"Average reward: {np.mean(rewards)}")
        print(f"Min reward: {np.min(rewards)}")
        print(f"Max reward: {np.max(rewards)}")
        print(f"Team reward: {team_reward}")
        print(f"Best Team reward: {best_team_reward}")

        features1 = []
        features2 = []
        returns1 = []
        returns2 = []
        actions1 = []
        actions2 = []
        log_probs_old1 = []
        log_probs_old2 = []
        for trajectory in trajectories:
            trajectory = trajectory[0]
            for i in range(len(trajectory)):
                returns1.append(trajectory[i]['reward_state'])
                returns2.append(trajectory[i]['reward_state'])


                state_features1 = network_features_v2(trajectory[i]['team1_state'][0]['kart'], trajectory[i]['team2_state'],
                                                  trajectory[i]['soccer_state'])
                state_features2 = network_features_v2(trajectory[i]['team1_state'][1]['kart'], trajectory[i]['team2_state'],
                                                  trajectory[i]['soccer_state'])
                features1.append(torch.as_tensor(state_features1, dtype=torch.float32).cuda().view(-1))
                features2.append(torch.as_tensor(state_features2, dtype=torch.float32).cuda().view(-1))

                actions1.append({
                    'steer': trajectory[i]['actions'][0]['steer'],
                    'acceleration': trajectory[i]['actions'][0]['acceleration'],
                    'brake': trajectory[i]['actions'][0]['brake']
                })
                actions2.append({
                    'steer': trajectory[i]['actions'][2]['steer'],
                    'acceleration': trajectory[i]['actions'][2]['acceleration'],
                    'brake': trajectory[i]['actions'][2]['brake']
                })

                with torch.no_grad():
                    state_features_tensor1 = torch.as_tensor(state_features1, dtype=torch.float32).cuda().view(1, -1)
                    state_features_tensor2 = torch.as_tensor(state_features2, dtype=torch.float32).cuda().view(1, -1)
                    output_old1 = player1_net(state_features_tensor1)
                    output_old2 = player2_net(state_features_tensor2)

                    action_distribution_old1 = {
                        'steer': torch.distributions.normal.Normal(output_old1[:, 2], 1),
                        'acceleration': torch.distributions.normal.Normal(output_old1[:, 1], 1),
                        'brake': torch.distributions.normal.Normal(output_old1[:, 0], 1),
                    }
                    action_distribution_old2 = {
                        'steer': torch.distributions.normal.Normal(output_old2[:, 2], 1),
                        'acceleration': torch.distributions.normal.Normal(output_old2[:, 1], 1),
                        'brake': torch.distributions.normal.Normal(output_old2[:, 0], 1),
                    }

                    log_prob_old1 = torch.stack([
                        action_distribution_old1['steer'].log_prob(
                            torch.tensor(trajectory[i]['actions'][0]['steer'], dtype=torch.float32).cuda()),
                        action_distribution_old1['acceleration'].log_prob(
                            torch.tensor(trajectory[i]['actions'][0]['acceleration'], dtype=torch.float32).cuda()),
                        action_distribution_old1['brake'].log_prob(
                            torch.tensor(trajectory[i]['actions'][0]['brake'], dtype=torch.float32).cuda())
                    ]).sum()

                log_prob_old2 = torch.stack([
                    action_distribution_old2['steer'].log_prob(
                        torch.tensor(trajectory[i]['actions'][2]['steer'], dtype=torch.float32).cuda()),
                    action_distribution_old2['acceleration'].log_prob(
                        torch.tensor(trajectory[i]['actions'][2]['acceleration'], dtype=torch.float32).cuda()),
                    action_distribution_old2['brake'].log_prob(
                        torch.tensor(trajectory[i]['actions'][2]['brake'], dtype=torch.float32).cuda())
                ]).sum()

                log_probs_old1.append(log_prob_old1.unsqueeze(0))
                log_probs_old2.append(log_prob_old2.unsqueeze(0))

        log_probs_old1 = torch.cat(log_probs_old1).view(-1).cuda()
        log_probs_old2 = torch.cat(log_probs_old2).view(-1).cuda()
        returns1 = torch.as_tensor(returns1, dtype=torch.float32).cuda()
        returns2 = torch.as_tensor(returns2, dtype=torch.float32).cuda()

        ## lILIANS BLGO - NORMALIZE THE RETURNS PLZZZZ

        #returns1 = ((returns1 - returns1.mean()) / returns1.std())
        #returns2 = ((returns2 - returns2.mean()) / returns2.std())

        #### handle nans
        returns1 = ((returns1 - returns1.mean()) / returns1.std()) + 1e-8
        returns2 = ((returns2 - returns2.mean()) / returns2.std()) + 1e-8

        actions1 = {
            'steer': torch.tensor([a['steer'] for a in actions1], dtype=torch.float32).cuda(),
            'acceleration': torch.tensor([a['acceleration'] for a in actions1], dtype=torch.float32).cuda(),
            'brake': torch.tensor([a['brake'] for a in actions1], dtype=torch.float32).cuda()
        }
        actions2 = {
            'steer': torch.tensor([a['steer'] for a in actions2], dtype=torch.float32).cuda(),
            'acceleration': torch.tensor([a['acceleration'] for a in actions2], dtype=torch.float32).cuda(),
            'brake': torch.tensor([a['brake'] for a in actions2], dtype=torch.float32).cuda()
        }

        features1 = torch.stack(features1).cuda()
        features2 = torch.stack(features2).cuda()

        with torch.no_grad():
            state_values1 = value_net1(features1).squeeze()
            state_values2 = value_net2(features2).squeeze()

        advantages1 = returns1 - state_values1
        advantages2 = returns2 - state_values2

        ## CALCULATE ADVANTAGE -   TODO CHECK LILIANS BLOG  TODO - NORMALIZE
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-8)
        advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-8)

        player1_net.train()
        player2_net.train()
        value_net1.train()
        value_net2.train()

        for it in range(n_iterations):

            for player_net, features, actions, returns, advantages, log_probs_old, value_net, value_optim, action_optim in [
                (player1_net, features1, actions1, returns1, advantages1, log_probs_old1, value_net1, value_optim1, action_optim1),
                (player2_net, features2, actions2, returns2, advantages2, log_probs_old2, value_net2, value_optim2, action_optim2)
            ]:


                batch_ids = torch.randint(0, len(returns), (batch_size,))
                batch_features = features[batch_ids]
                batch_actions = {k: v[batch_ids] for k, v in actions.items()}
                batch_returns = returns[batch_ids]
                batch_advantages = advantages[batch_ids]
                batch_log_probs_old = log_probs_old[batch_ids]

                state_values = value_net(batch_features).squeeze()
                value_loss = (batch_returns - state_values).pow(2).mean()
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()


                output = player_net(batch_features)
                action_distribution = {
                    'steer': torch.distributions.normal.Normal(output[:, 2], 1),
                    'acceleration': torch.distributions.normal.Normal(output[:, 1], 1),
                    'brake': torch.distributions.normal.Normal(output[:, 0], 1),
                }
                log_probs = torch.stack([
                    action_distribution['steer'].log_prob(batch_actions['steer']),
                    action_distribution['acceleration'].log_prob(batch_actions['acceleration']),
                    action_distribution['brake'].log_prob(batch_actions['brake'])
                ]).sum(dim=0)

                ratio = (log_probs - batch_log_probs_old).exp()
                surr1 = ratio * batch_advantages

                ## double check the forumule from LLIANS BLOG -   TODO

                surr2 = torch.clamp(ratio, 1.0 - ppo_eps, 1.0 + ppo_eps) * batch_advantages
                action_loss = -torch.min(surr1, surr2).mean()
                action_optim.zero_grad()
                action_loss.backward()
                action_optim.step()

        if team_reward > best_team_reward:
            best_player1_net = copy.deepcopy(player1_net)
            best_player2_net = copy.deepcopy(player2_net)
            best_team_reward = team_reward

            save_model(best_player1_net, player1_model_filepath)
            save_model(best_player2_net, player2_model_filepath)
            save_model(value_net1, value1_model_filepath)
            save_model(value_net2, value2_model_filepath)




