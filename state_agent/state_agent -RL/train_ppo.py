import numpy as np
import torch
import warnings
import copy

from state_agent.player import Team
from state_agent.critic import ValueNetwork
from state_agent.custom_runner import record_manystate
from state_agent.planner import network_features, Planner, network_features_v2
from state_agent.player import ACTION_SPACE
import os

warnings.filterwarnings("ignore", category=UserWarning)

def save_model(model, filename):
    from torch import save
    from os import path
    if isinstance(model, Planner) or isinstance(model, ValueNetwork):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), filename))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


"""def compute_gae1(rewards, state_values, gamma, lambda_):
    gae = 0
    returns = []
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * state_values[t + 1] - state_values[t]
        gae = delta + gamma * lambda_ * gae
        returns.insert(0, gae + state_values[t])
    # Handle the last step
    t = len(rewards) - 1
    delta = rewards[t] - state_values[t]
    gae = delta + gamma * lambda_ * gae
    returns.insert(0, gae + state_values[t])
    return torch.tensor(returns, dtype=torch.float32).cuda()"""

def compute_gae(next_value, rewards, dones, values, gamma=0.99, tau=0.95):
    masks = [1 - done for done in dones]
    values = values.tolist() + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def calculate_discounted_rewards(rewards, dones, next_value, gamma=0.99):
    returns = torch.tensor([0]*len(rewards))
    #nextnonterminal = 1 # for now # TODO: use 'dones' (terminal state flags)
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
        else:
            #nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1]
        returns[t] = rewards[t] + gamma * nextnonterminal * next_return
    return returns

if __name__ == "__main__":
    device = None
    if torch.backends.mps.is_available():
        print("MPS device found.")
        device = torch.device("mps")
        x = torch.ones(8, device=device)
        print(x)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    value_net = ValueNetwork(21, 32).to(device)

    # PPO training hyperparams -- vary and experiment.
    n_epochs = 10
    n_trajectories = 1  # so, no. of timesteps in each epoch = MAX_FRAMES_TRAIN * n_trajectories
    batch_size = 128  # maybe keep this constant
    n_iterations = 16  # vary this to match: 2*n_trajectories * MAX_FRAMES_TRAIN = batch_size * n_iterations
    ppo_eps = 0.2

    # Below feature size is made 21 after including puck velocity. Find 'puck_velocity' in state_agent/player.py: act()
    # Action space has 6 cardinality now, so output_size is made 6. Find ACTION_SPACE in state_agent/player.py
    player_net = Planner(21, 32, 6).to(device)

    player_model_filepath = "player1_action_model.pt"
    value_model_filepath = "player1_value_model.pt"

    optimizer = torch.optim.Adam(list(player_net.parameters()) + list(value_net.parameters()), lr=0.0001) # weight_decay=1e-5

    team_rewards = []
    best_team_reward = -np.inf

    for epoch in range(n_epochs): #main PPO loop
            trajectories = record_manystate([Team(player_net,player_net)] * n_trajectories)

            rewards = [step_data['reward_state'] for step_tuple in trajectories for step_data in step_tuple[0]]

            team_reward = np.mean([step_data['reward_state'] for step_tuple in trajectories for step_data in step_tuple[0]])
            team_rewards.append(team_reward)

            print(f"Epoch = {epoch}")
            print(f"Average reward: {np.mean(rewards)}")
            print(f"Min reward: {np.min(rewards)}")
            print(f"Max reward: {np.max(rewards)}")
            print(f"Team reward: {team_reward}")
            print(f"Best Team reward: {best_team_reward}")

            features1 = []
            features2 = []
            rewards1 = []
            rewards2 = []
            action_ids1 = []
            action_ids2 = []
            log_probs_old1 = []
            log_probs_old2 = []
            dones1 = []
            dones2 = []
            for trajectory in trajectories: #for every episode trajectory in the batch
                trajectory = trajectory[0]
                for i in range(len(trajectory)): #for every time step
                    rewards1.append(trajectory[i]['reward_state'][0])
                    rewards2.append(trajectory[i]['reward_state'][1])
                    dones1.append(trajectory[i]['done'])
                    dones2.append(trajectory[i]['done'])

                    # state s_i
                    state_features1 = network_features_v2(trajectory[i]['team1_state'][0]['kart'], trajectory[i]['team2_state'],
                                                      trajectory[i]['soccer_state'])
                    state_features2 = network_features_v2(trajectory[i]['team1_state'][1]['kart'], trajectory[i]['team2_state'],
                                                      trajectory[i]['soccer_state'])

                    # Note: Appended below puck velocity into state features
                    state_features1 = torch.cat([state_features1, trajectory[i]['puck_velocity']])
                    state_features2 = torch.cat([state_features2, trajectory[i]['puck_velocity']])

                    features1.append(torch.as_tensor(state_features1, dtype=torch.float32).cuda().view(-1))
                    features2.append(torch.as_tensor(state_features2, dtype=torch.float32).cuda().view(-1))

                    #action a_i (store just ids, not actual actions)
                    action_ids1.append(trajectory[i]['action_ids'][0])
                    action_ids2.append(trajectory[i]['action_ids'][1])

                    #probs p_i
                    log_probs_old1.append(trajectory[i]['logprobs'][0])
                    log_probs_old2.append(trajectory[i]['logprobs'][1])

            #store as tensors: states(features), actions, logprobs, returns collected across all trajectories

            #logprobs
            log_probs_old = log_probs_old1 + log_probs_old2 # all logprobs in one list
            log_probs_old = torch.as_tensor(log_probs_old, dtype=torch.float32).detach().cuda()

            #returns
            rewards = rewards1 + rewards2
            rewards = torch.as_tensor(rewards, dtype=torch.float32).detach().cuda()

            #actions
            action_ids = action_ids1 + action_ids2
            action_ids = torch.tensor(action_ids, dtype=torch.float32).cuda()

            #state features
            features1 = torch.stack(features1).cuda()
            features2 = torch.stack(features2).cuda()
            features = torch.cat([features1, features2]).cuda()

            values1 = value_net(features1).squeeze().detach().cuda()
            values2 = value_net(features2).squeeze().detach().cuda()  #value model (i.e. [state => value] mathematical function) could just be same for both players
            values = torch.cat([values1, values2])

            #returns
            next_value1 = value_net(features1[-1])
            next_value2 = value_net(features2[-1])
            returns1 = compute_gae(next_value1, rewards1, dones1, values1)  #calculate_discounted_rewards(rewards1, dones1, next_value1)
            returns2 = compute_gae(next_value2, rewards2, dones2, values2)  #calculate_discounted_rewards(rewards2, dones2, next_value2)
            returns = torch.cat([torch.tensor(returns1), torch.tensor(returns2)]).cuda()

            # advantages
            advantages = returns - values

            player_net.train()
            value_net.train()

            #PPO loss and gradient updates loop (batched)
            for it in range(n_iterations):
                for just_once in range(1):
                    batch_ids = torch.randint(0, len(returns), (batch_size,))
                    batch_features = features[batch_ids]
                    batch_action_ids = action_ids[batch_ids]
                    batch_returns = returns[batch_ids]
                    batch_advantages = advantages[batch_ids]
                    batch_log_probs_old = log_probs_old[batch_ids]

                    #get new logprobs
                    current_policy_dist = player_net(batch_features)
                    action_indexes = current_policy_dist.sample()
                    batch_log_probs_new = current_policy_dist.log_prob(action_indexes)

                    # Entropy term
                    entropy = current_policy_dist.entropy().mean()

                    #Clipped ratio terms
                    ratio = (batch_log_probs_new - batch_log_probs_old).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - ppo_eps, 1.0 + ppo_eps) * batch_advantages

                    #losses (policy loss, value loss, entropy term and finally ppo loss)
                    actor_loss = -torch.min(surr1, surr2).mean()
                    state_values = value_net(batch_features).squeeze() #TODO: check dimensions
                    value_loss = (batch_returns - state_values).pow(2).mean()

                    ppo_loss = 0.5 * value_loss + actor_loss - 0.001 * entropy
                    print("PPO loss: ", actor_loss)

                    #gradient step
                    optimizer.zero_grad()
                    ppo_loss.backward()
                    optimizer.step()


            best_player_net = copy.deepcopy(player_net)
            if team_reward > best_team_reward:
                best_team_reward = team_reward
                save_model(best_player_net, player_model_filepath)
                save_model(value_net, value_model_filepath)
