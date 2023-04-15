from torch.distributions import Bernoulli

from utils import show_agent
from state_agent.planner import Planner
import torch
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

        # Unpack relevant features
        kart_to_puck_angle_difference = f[13]
        kart_to_goal_line_angle_difference = f[16]

        action = pystk.Action()
        # Full acceleration

        brake_threshold = 0.2
        if abs(kart_to_puck_angle_difference) > brake_threshold:
            action.brake = True
            action.acceleration = 0.0  # No acceleration when braking
        else:
            action.brake = False
            action.acceleration = 1.0  # Full acceleration

        # Steering based on angle difference to puck
        action.steer = kart_to_puck_angle_difference * 1.8
        return action

if __name__ == "__main__":
    action_net = Planner(17, 32, 3)
    agent = Player(action_net)
    show_agent(agent, n_steps=600)






