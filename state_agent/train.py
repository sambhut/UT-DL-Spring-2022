import numpy as np
import torch
import warnings
import copy

from state_agent import Team
from state_agent.critic import ValueNetwork
from state_agent.custom_runner import record_manystate
from state_agent.expert_player import extract_features
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


def compute_gae(rewards, state_values, gamma, lambda_):
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
    return torch.tensor(returns, dtype=torch.float32).cuda()



if __name__ == "__main__":
    device = None
    if torch.backends.mps.is_available():
        print("MPS device found.")
        device = torch.device("mps")
        x = torch.ones(8, device=device)
        print(x)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # TODO:check.  Below feature size as 21 after including puck velocity. Find 'puck_velocity' in state_agent/player.py: act()
    value_net1 = ValueNetwork(21, 32).to(device)
    value_net2 = ValueNetwork(21, 32).to(device)
    """many_action_nets = [Team() for i in range(10)]
    data = record_manystate(many_action_nets)
    good_initialization = many_action_nets[np.argmax([d[-1]['team1']['highest_distance'] for d in data])]"""

    # PPO training hyperparams -- vary and experiment.
    n_epochs = 20
    n_trajectories = 10  # so, no. of timesteps in each epoch = MAX_FRAMES_TRAIN*10  (which is 100*10=1000 for now)
    batch_size = 128  # maybe keep this constant
    n_iterations = 8  # vary this to match: n_trajectories * MAX_FRAMES_TRAIN = batch_size * n_iterations
                      # currently: 100*10 ~ 128*8
    ppo_eps = 0.2


    """expert1_net = copy.deepcopy(good_initialization.model0)
    expert2_net = copy.deepcopy(good_initialization.model1)"""

    # TODO:check.  Below feature size is made 21 after including puck velocity. Find 'puck_velocity' in state_agent/player.py: act()
    # TODO:check. Action space has 6 cardinality now, so output_size is made 6. Find ACTION_SPACE in state_agent/player.py
    player1_net = Planner(21, 32, 6).to(device)
    player2_net = Planner(21, 32, 6).to(device)
    """dic1 = torch.load('player1_action_model.pt')
    dic2 = torch.load('player2_action_model.pt')
    player1_net.load_state_dict(dic1)
    player2_net.load_state_dict(dic2)


    best_player1_net = copy.deepcopy(expert1_net)
    best_player2_net = copy.deepcopy(expert2_net)"""

    player1_model_filepath = "player1_action_model.pt"
    player2_model_filepath = "player2_action_model.pt"
    value1_model_filepath = "player1_value_model.pt"
    value2_model_filepath = "player2_value_model.pt"


    action_optim1 = torch.optim.Adam(player1_net.parameters(), lr=0.001 , weight_decay=1e-5)
    action_optim2 = torch.optim.Adam(player2_net.parameters(), lr=0.001, weight_decay=1e-5)
    value_optim1 = torch.optim.Adam(value_net1.parameters(), lr=0.001, weight_decay=1e-5)
    value_optim2 = torch.optim.Adam(value_net2.parameters(), lr=0.001, weight_decay=1e-5)

    team_rewards = []
    best_team_reward = -np.inf

    for epoch in range(n_epochs): #main PPO loop
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
            action_ids1 = []
            action_ids2 = []
            log_probs_old1 = []
            log_probs_old2 = []
            for trajectory in trajectories: #for every episode trajectory in the batch
                trajectory = trajectory[0]
                for i in range(len(trajectory)): #for every time step
                    returns1.append(trajectory[i]['reward_state'])
                    returns2.append(trajectory[i]['reward_state'])

                    # state s_i
                    state_features1 = network_features_v2(trajectory[i]['team1_state'][0]['kart'], trajectory[i]['team2_state'],
                                                      trajectory[i]['soccer_state'])
                    state_features2 = network_features_v2(trajectory[i]['team1_state'][1]['kart'], trajectory[i]['team2_state'],
                                                      trajectory[i]['soccer_state'])

                    # TODO (check). appended below puck velocity into state features
                    state_features1 = torch.cat([state_features1, trajectory[i]['puck_velocity']])
                    state_features2 = torch.cat([state_features2, trajectory[i]['puck_velocity']])

                    features1.append(torch.as_tensor(state_features1, dtype=torch.float32).cuda().view(-1))
                    features2.append(torch.as_tensor(state_features2, dtype=torch.float32).cuda().view(-1))

                    #action a_i (store just ids, not actual actions)
                    action_ids1.append(trajectory[i]['action_ids'][0])
                    action_ids2.append(trajectory[i]['action_ids'][1])

                    #probs p_i
                    log_probs_old1.append(torch.tensor(trajectory[i]['logprobs'][0]).unsqueeze(0))
                    log_probs_old2.append(torch.tensor(trajectory[i]['logprobs'][1]).unsqueeze(0))

            log_probs_old1 = torch.cat(log_probs_old1).view(-1).cuda()
            log_probs_old2 = torch.cat(log_probs_old2).view(-1).cuda()
            returns1 = torch.as_tensor(returns1, dtype=torch.float32).cuda()
            returns2 = torch.as_tensor(returns2, dtype=torch.float32).cuda()

            returns1 = ((returns1 - returns1.mean()) / returns1.std()) + 1e-8
            returns2 = ((returns2 - returns2.mean()) / returns2.std()) + 1e-8

            action_ids1 = torch.tensor(action_ids1, dtype=torch.float32).cuda()
            action_ids2 = torch.tensor(action_ids2, dtype=torch.float32).cuda()

            features1 = torch.stack(features1).cuda()
            features2 = torch.stack(features2).cuda()

            player1_net.train()
            player2_net.train()
            value_net1.train()
            value_net2.train()

            #PPO loss and gradient updates loop (batched)
            for it in range(n_iterations):

                for player_net, features, action_ids, returns, log_probs_old, value_net, value_optim, action_optim in [
                    (player1_net, features1, action_ids1, returns1, log_probs_old1, value_net1, value_optim1, action_optim1),
                    (player2_net, features2, action_ids2, returns2, log_probs_old2, value_net2, value_optim2, action_optim2)
                ]:
                    with torch.no_grad():
                        state_values = value_net(features).squeeze()
                        state_values_next = torch.cat((state_values[1:], torch.tensor([0], device=device)))

                    advantages = compute_gae(returns, state_values_next, 0.99, 0.95)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    #TODO: you are doing only for 1 mini-batch. Need to loop for all mini-batches.
                    #TODO: check advantages and returns; they are returning nans.
                    batch_ids = torch.randint(0, len(returns), (batch_size,))
                    batch_features = features[batch_ids]
                    batch_action_ids = action_ids[batch_ids]
                    batch_returns = returns[batch_ids]
                    batch_advantages = advantages[batch_ids]
                    batch_log_probs_old = log_probs_old[batch_ids]

                    state_values = value_net(batch_features).squeeze()
                    value_loss = (batch_returns - state_values).pow(2).mean()
                    value_optim.zero_grad()
                    value_loss.backward()
                    value_optim.step()


                    current_policy_dist = player_net(batch_features)
                    action_indexes = current_policy_dist.sample()
                    batch_log_probs_new = current_policy_dist.log_prob(action_indexes)

                    # Entropy term
                    entropy = current_policy_dist.entropy().mean()

                    ratio = (batch_log_probs_new - batch_log_probs_old).exp()
                    surr1 = ratio * batch_advantages

                    ## double check the forumule from LLIANS BLOG -   TODO

                    surr2 = torch.clamp(ratio, 1.0 - ppo_eps, 1.0 + ppo_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    #TODO: 1. Combine into single loss and train value+policy together
                    # ppo_loss = 0.5 * value_loss + actor_loss - 0.001 * entropy
                    #TODO: 2. Learn one policy network for both karts

                    print("PPO loss: ", actor_loss)
                    action_optim.zero_grad()
                    actor_loss.backward()
                    action_optim.step()


            best_player1_net = copy.deepcopy(player1_net)
            best_player2_net = copy.deepcopy(player2_net)
            if team_reward > best_team_reward:
                best_team_reward = team_reward
                save_model(best_player1_net, player1_model_filepath)
                save_model(best_player2_net, player2_model_filepath)
                save_model(value_net1, value1_model_filepath)
                save_model(value_net2, value2_model_filepath)

    #TODO:
    #1. Review rewards: sparse reward system + (small -ve reward) for every time step
    #2. Fix the batching in PPO
    #3. Review log_probs_old
    #4. NaN issues in rewards and advantages
    #5. Add terminal state logic
    #6. Train across multiple environments. (opponent team = AI / Jurgen / etc.). PPO opt should consider many trajectories
    #7. Test for long epochs/iterations/whatever. May fail.
    #8. Train value and policy together (entropy term already added)
    #9. Try only 1 network for both players (karts)
    #10. Change layers in model (can add BNs, etc.) (Can also try CNNs - last resort)
    #11. Clean up the code. Should be submission ready. Things are messy now.
    #12. Inference code should be particularly neat, separate and ready irrespective of training/experiments. Need to run grader and check for any crashes.
    #13. Review features (velocities already added now)
    #14. Try more complex action space for acceleration values. Don't change steer and brake.
    #15. Initialization. Start from imitation/dagger learnt model and try.
    #16. Normalizations wherever appropriate
    #17. Review GAE



