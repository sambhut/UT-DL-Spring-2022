from torch.distributions import Bernoulli

from utils import show_agent
import torch
import torch
import pystk
from torch.distributions import Bernoulli
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Player:

    """
    This will be replaced with act method
    """
    def __call__(self, player_state, opponent_state, soccer_state, **kwargs):


        action = pystk.Action()
        action.acceleration = 1


        return action

if __name__ == "__main__":
    agent = Player()
    show_agent(agent, n_steps=600)






