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
        self.action_net = action_net.cpu().eval()

    """
    This will be replaced with act method
    """
    def __call__(self, player_state, opponent_state, soccer_state, **kwargs):
        f = network_features(player_state, opponent_state, soccer_state)
        input_tensor = torch.tensor(torch.as_tensor(f).view(1, -1), dtype=torch.float)
        output = self.action_net.forward(input_tensor)[0]

        action = pystk.Action()

        brake_threshold = 0.2
        if torch.sigmoid(output[2]).item() > brake_threshold:
            action.brake = True
            action.acceleration = 0.0
            action.brake = False
            action.acceleration = torch.sigmoid(output[0]).item()

        steering_gain = 0.3
        steering_gain = torch.tanh(output[1]).item() * steering_gain
        action.steer = np.clip(steering_gain, -1, 1)
        return action

