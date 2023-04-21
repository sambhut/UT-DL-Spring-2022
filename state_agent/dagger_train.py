import numpy as np
from torch.distributions import Bernoulli

from state_agent.player_draft import Player
from state_agent.planner import network_features, Planner
from state_agent.utils import show_agent, rollout_many,show_viz_rolloutagent
from jurgen_agent.player import Team as Jur
from yann_agent.player import Team as Yan
from geoffrey_agent.player import Team as Geo
import copy
import torch
import warnings
import torch.utils.tensorboard as tb
import pystk
warnings.filterwarnings("ignore", category=UserWarning)

class Actor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()

    def __call__(self, player_state, opponent_state, soccer_state, **kwargs):
        f = network_features(player_state, opponent_state, soccer_state)
        input_tensor = torch.as_tensor(f).view(1, -1)
        output = self.action_net.forward(input_tensor)[0]

        action = pystk.Action()

        brake_threshold = 0.2
        if torch.sigmoid(output[2]).item() > brake_threshold:
            action.brake = True
            action.acceleration = 0.0
        else:
            action.brake = False
            action.acceleration = torch.sigmoid(output[0]).item()

        steering_gain = 0.3
        steering_gain = torch.tanh(output[1]).item() * steering_gain
        action.steer = np.clip(steering_gain, -1, 1)
        return action

def rolloutstate_to_args(train_data):
    player_state_d = [{}]
    ps = train_data['player_state']
    player_state_d[0]['kart'] = {'front' : ps.front, 'location' : ps.location, 'rotation' : ps.rotation, 'size' : ps.size, 'velocity' : ps.velocity}

    opponent_state_d = [{},{}]
    os = train_data['opponent_state']
    opponent_state_d[0]['kart'] = {'front' : os.front, 'location' : os.location, 'rotation' : os.rotation, 'size' : os.size, 'velocity' : os.velocity}
    opponent_state_d[1]['kart'] = {'front' : os.front, 'location' : os.location, 'rotation' : os.rotation, 'size' : os.size, 'velocity' : os.velocity}

    ss = train_data['soccer_state']
    soccer_state_d = {'ball' : {'location' : ss.ball.location}, 'goal_line' : ss.goal_line}

    return player_state_d, opponent_state_d, soccer_state_d


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    action_net = Planner(17, 32, 3).to(device)
    actor = Actor(action_net)

    expert_player = Jur()
    #expert_player = Yan()
    #expert_player = Geo()

    expert_player.new_match(0, 1)

    n_epochs = 10
    n_trajectories = 10
    batch_size = 128

    # Create the optimizer
    optimizer = torch.optim.Adam(action_net.parameters())

    # Create the loss
    loss = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        trajectories = rollout_many([Actor(action_net)] * n_trajectories)

        train_features = []
        train_actions = []

        for trajectory in trajectories:
            for i in range(len(trajectory)):
                train_features.append(torch.as_tensor(
                    network_features(trajectory[i]['player_state'], trajectory[i]['opponent_state'],
                                     trajectory[i]['soccer_state']), dtype=torch.float32).cuda().view(-1))

                player_state, opponent_state, soccer_state = rolloutstate_to_args(trajectory[i])
                actions = expert_player.act(player_state, opponent_state, soccer_state)
                #print("actions returned is ", actions)
                train_actions.append(torch.cat((actions[0]['acceleration'], actions[0]['steer'], actions[0]['brake'])))

        print(train_actions[0])

        train_features = torch.stack(train_features).cuda()
        train_actions = torch.stack(train_actions).cuda()

        print("length of train_features is ", train_features.size())

        action_net.train().cuda()
        for iteration in range(0, len(train_features), batch_size):
            batch_ids = torch.randint(0, len(train_features), (batch_size,), device=device)
            batch_images = train_features[batch_ids]
            batch_labels = train_actions[batch_ids]
            output = action_net(batch_images)
            loss_val = loss(output, batch_labels)
            print("loss in iteration %d, epoch %d is %f" % (iteration/batch_size, epoch, loss_val))

            optimizer.zero_grad()
            loss_val.backward(retain_graph=True)
            optimizer.step()

    show_agent(Actor(action_net), n_steps=600)