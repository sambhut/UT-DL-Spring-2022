from torch.distributions import Bernoulli

from utils import show_agent
from player_draft import GreedyActor
from state_agent.planner import planner
import torch

if __name__ == "__main__":
    action_net = planner()
    greedy_agent = GreedyActor(action_net)
    show_agent(greedy_agent, n_steps=600)

