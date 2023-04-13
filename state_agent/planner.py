import numpy as np
import torch


def planner():
    return torch.nn.Linear(8, 1, bias=False)


def network_features(player_pos, opponent_pos, ball_pos):

    p = np.array(player_pos.location)[[0, 2]].astype(np.float32)
    o = np.array(opponent_pos.location)[[0, 2]].astype(np.float32)
    b = np.array(ball_pos)[[0, 2]].astype(np.float32)
    t = np.array(player_pos.front)[[0, 2]].astype(np.float32)
    d = (p - t) / np.linalg.norm(p - t)
    d_o = np.array([-d[1], d[0]], dtype=np.float32)
    ball_rel = b - p
    dist_to_ball = np.linalg.norm(p - b)
    dist_to_opp = np.linalg.norm(p - o)
    features = np.concatenate([d, d_o, ball_rel, [dist_to_ball, dist_to_opp]])

    return features
