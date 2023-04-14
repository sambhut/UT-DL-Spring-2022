from torch.distributions import Bernoulli

from state_agent.player_draft import Player
from utils import show_agent
from state_agent.planner import Planner
import torch

if __name__ == "__main__":
    action_net = Planner(17, 32, 1)
    agent = Player(action_net)
    show_agent(agent, n_steps=600)

