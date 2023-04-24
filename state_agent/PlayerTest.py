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
    action_net = Team()
    record_video(action_net)






