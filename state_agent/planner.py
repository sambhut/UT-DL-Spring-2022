import numpy as np
import torch

class Planner(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Planner, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.BN1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, 64)
        self.BN2 = torch.nn.BatchNorm1d(64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.BN3 = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, output_size, bias=False)


    def forward(self, x):
        out = x
        if out.dim() == 1:
            out = out.unsqueeze(0)
        out = self.fc1(out)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.BN3(out)
        out = self.relu(out)
        out = self.fc4(out)
        if out.dim() != x.dim():
            out = out.squeeze(0)
        return out


def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def network_features(player_pos, opponent_pos, ball_pos):

    kart_front = torch.tensor(player_pos.front, dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(player_pos.location, dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    puck_center = torch.tensor(ball_pos.ball.location, dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of opponents
    opponent_center0 = torch.tensor(opponent_pos.location, dtype=torch.float32)[[0, 2]]
    opponent_center1 = torch.tensor(opponent_pos.location, dtype=torch.float32)[[0, 2]]

    kart_to_opponent0 = (opponent_center0 - kart_center) / torch.norm(opponent_center0-kart_center)
    kart_to_opponent1 = (opponent_center1 - kart_center) / torch.norm(opponent_center1-kart_center)

    kart_to_opponent0_angle = torch.atan2(kart_to_opponent0[1], kart_to_opponent0[0])
    kart_to_opponent1_angle = torch.atan2(kart_to_opponent1[1], kart_to_opponent1[0])

    kart_to_opponent0_angle_difference = limit_period((kart_angle - kart_to_opponent0_angle)/np.pi)
    kart_to_opponent1_angle_difference = limit_period((kart_angle - kart_to_opponent1_angle)/np.pi)

    # features of score-line
    goal_line_center = torch.tensor(ball_pos.goal_line[0], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0])
    kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle)/np.pi)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, opponent_center0[0],
        opponent_center0[1], opponent_center1[0], opponent_center1[1], kart_to_opponent0_angle, kart_to_opponent1_angle,
        goal_line_center[0], goal_line_center[1], puck_to_goal_line_angle, kart_to_puck_angle_difference,
        kart_to_opponent0_angle_difference, kart_to_opponent1_angle_difference,
        kart_to_goal_line_angle_difference], dtype=torch.float32)

    return features