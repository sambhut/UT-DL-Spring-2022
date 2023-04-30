from os import path
import numpy as np
import torch
from torch.distributions import Categorical

from state_agent.planner import Planner

# Defining our action space in the environment
ACTION_SPACE = [(1,0,-1), (1,0,0), (1,0,1), (0,1,-1), (0,1,0), (0,1,1)] # all possible (brake, acc, steer) tuples


def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


def extract_features(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
    kart_velocity = torch.tensor(pstate['kart']['velocity'], dtype=torch.float32)[[0, 2]]

    # features of soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)

    # features of opponents
    opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]]
    opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]]

    kart_to_opponent0 = (opponent_center0 - kart_center) / torch.norm(opponent_center0 - kart_center)
    kart_to_opponent1 = (opponent_center1 - kart_center) / torch.norm(opponent_center1 - kart_center)

    kart_to_opponent0_angle = torch.atan2(kart_to_opponent0[1], kart_to_opponent0[0])
    kart_to_opponent1_angle = torch.atan2(kart_to_opponent1[1], kart_to_opponent1[0])

    kart_to_opponent0_angle_difference = limit_period((kart_angle - kart_to_opponent0_angle) / np.pi)
    kart_to_opponent1_angle_difference = limit_period((kart_angle - kart_to_opponent1_angle) / np.pi)

    # features of score-line
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0])
    kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle) / np.pi)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, opponent_center0[0],
                             opponent_center0[1], opponent_center1[0], opponent_center1[1], kart_to_opponent0_angle,
                             kart_to_opponent1_angle,
                             goal_line_center[0], goal_line_center[1], puck_to_goal_line_angle,
                             kart_to_puck_angle_difference,
                             kart_to_opponent0_angle_difference, kart_to_opponent1_angle_difference,
                             kart_to_goal_line_angle_difference,
                             kart_velocity[0], kart_velocity[1]], dtype=torch.float32)

    return features


class Team:
    agent_type = 'state'

    def __init__(self, player1=None, player2=None):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.team = None
        self.num_players = None
        self.old_puck_center = None
        if player1 is None or player2 is None:
            self.model0 = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'geoffrey_agent0.pt'))
            self.model1 = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'geoffrey_agent1.pt'))
        else:
            self.model0 = player1
            self.model1 = player2
        self.model0.eval()
        self.model1.eval()
        self.model0.to(device)
        self.model1.to(device)

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        self.old_puck_center = torch.Tensor([0, 0])
        return ['tux'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight

        #compute puck velocity (will be part of feature space)
        current_puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        puck_velocity = current_puck_center - self.old_puck_center

        actions = []
        action_ids = []
        logprobs = []
        for player_id, pstate in enumerate(player_state):
            features = extract_features(pstate, soccer_state, opponent_state, 1)
            features = torch.cat([features, puck_velocity]) #TODO: need to incorporate this somehow  both here and in custom runner!!
            input_tensor = features.cuda()
            if player_id % 2 == 0:
                output_dist = self.model0(input_tensor)
            else:
                output_dist = self.model1(input_tensor)

            # Sample the action from output probabilities
            action_id = output_dist.sample()          #tensor with 1 value representing index in action space
            action_tuple = ACTION_SPACE[action_id]    #tuple of 3 values
            actions.append(dict(acceleration=action_tuple[1], steer=action_tuple[2], brake=action_tuple[0]))
            action_ids.append(action_id)

            # prob required for PPO (only for training)
            logprob = output_dist.log_prob(action_id).item()
            logprobs.append(logprob)

        #print("act(): logprobs, action_ids", logprobs, action_ids)
        return actions, logprobs, action_ids
