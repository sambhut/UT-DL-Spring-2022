from torch.distributions import Bernoulli

from state_agent import Team
from state_agent.custom_runner import record_video, record_state, record_manystate
from state_agent.planner import Planner
import torch
import torch
import pystk
from torch.distributions import Bernoulli
from state_agent.planner import network_features
import torch
import numpy as np
from os import path
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    player1_net = Planner(21, 32, 6).to(device)
    player2_net = Planner(21, 32, 6).to(device)
    dic1 = torch.load('player1_action_model.pt')
    dic2 = torch.load('player2_action_model.pt')
    player1_net.load_state_dict(dic1)
    player2_net.load_state_dict(dic2)
    action_net = Team(player1_net,player2_net)
    record_video(action_net)






