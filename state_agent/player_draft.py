import torch
import pystk
from torch.distributions import Bernoulli
from state_agent.planner import network_features
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Player:
    def __init__(self, action_net):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.action_net = action_net.to(self.device)


    """
    This will be replaced with act method
    """
    def __call__(self, player_state, opponent_state, soccer_state, **kwargs):

        f = network_features(player_state, opponent_state, soccer_state)
        input_tensor = torch.tensor(torch.as_tensor(f).view(1, -1), dtype=torch.float).cuda()
        output = self.action_net.forward(input_tensor)[0]

        action = pystk.Action()

        # Normalize brake and acceleration values
        brake = torch.sigmoid(output[0]).item()
        acceleration = torch.sigmoid(output[1]).item()
        total = brake + acceleration

        brake = brake / total
        acceleration = acceleration / total

        if brake > acceleration:
            action.brake = 1
            action.acceleration = 0
        else:
           action.brake = 0
           action.acceleration = acceleration

           # Use continuous steering value
        steering_gain = torch.tanh(output[2]).item()
        action.steer = np.clip(steering_gain, -1, 1)

        return action

