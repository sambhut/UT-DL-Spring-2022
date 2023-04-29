
import torch
import numpy as np
from os import path
from torch.distributions import Categorical

from state_agent.planner import Planner, load_model

ACTION_SPACE = [(1,0,-1), (1,0,0), (1,0,1), (0,1,-1), (0,1,0), (0,1,1)] # all possible (brake, acc, steer) tuples, i.e. our action space


def action_to_actionspace(brake, acceleration, steer):

    if brake == 0:
        b_tuple = 0
        a_tuple = 1

    else:
        b_tuple = 1
        a_tuple = 0

    s_tuple = steer

    tup = (b_tuple, a_tuple, s_tuple)

    for i in range(0, len(ACTION_SPACE)):
        if tup == ACTION_SPACE[i]:
            #print("got values %f, %f, %f, returning index %d" % (brake, acceleration, steer, i))
            return int(i)

    print("did not get valid tuple %f, %f, %f" %(brake, acceleration, steer))
    return -1

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2
def network_features(player_pos, opponent_pos, ball_pos, team_id):

    # features of ego-vehicle
    kart_front = torch.tensor(player_pos['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(player_pos['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer
    puck_center = torch.tensor(ball_pos['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line
    """
    print("print1 is ", ball_pos['goal_line'][0])
    print("print2 is ", torch.tensor(ball_pos['goal_line'][0], dtype=torch.float32)[:, [0, 2]])
    print("print3 is ", torch.tensor(ball_pos['goal_line'][0], dtype=torch.float32)[:, [0, 2]][0])
    print("print3 is ", torch.tensor(ball_pos['goal_line'][0], dtype=torch.float32)[:, [0, 2]][1])

    goal_line_center = torch.tensor(ball_pos['goal_line'][0], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle,
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference,
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)
    """

    goal0_posts = torch.tensor(ball_pos['goal_line'][(team_id + 1) % 2], dtype=torch.float32)[:, [0, 2]]
    puck_to_goal0_post0 = (goal0_posts[0]-puck_center) / torch.norm(goal0_posts[0]-puck_center)
    puck_to_goal0_post1 = (goal0_posts[1]-puck_center) / torch.norm(goal0_posts[1]-puck_center)

    puck_to_goal0_post0_angle = torch.atan2(puck_to_goal0_post0[1], puck_to_goal0_post0[0])
    puck_to_goal0_post1_angle = torch.atan2(puck_to_goal0_post1[1], puck_to_goal0_post1[0])

    goal1_posts = torch.tensor(ball_pos['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]]
    puck_to_goal1_post0 = (goal1_posts[0]-puck_center) / torch.norm(goal1_posts[0]-puck_center)
    puck_to_goal1_post1 = (goal1_posts[1]-puck_center) / torch.norm(goal1_posts[1]-puck_center)

    puck_to_goal1_post0_angle = torch.atan2(puck_to_goal1_post0[1], puck_to_goal1_post0[0])
    puck_to_goal1_post1_angle = torch.atan2(puck_to_goal1_post1[1], puck_to_goal1_post1[0])

    #print("goal_posts is ", goal_posts)
    #print("goal_posts[0] is ", goal_posts[0])
    #print("goal_posts[1] is ", goal_posts[1])
    #print("puck_to_goal_post0 is ", puck_to_goal_post0)
    #print("puck_to_goal_post1 is ", puck_to_goal_post1)

    """
    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle,
        kart_to_puck_angle_difference, puck_center[0], puck_center[1], puck_to_goal0_post0[0],
        puck_to_goal0_post0[1], puck_to_goal0_post1[0], puck_to_goal0_post1[1], puck_to_goal1_post0[0],
        puck_to_goal1_post0[1], puck_to_goal1_post1[0], puck_to_goal1_post1[1]], dtype=torch.float32)
    """

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle,
        kart_to_puck_angle_difference, puck_center[0], puck_center[1], puck_to_goal1_post0[0],
        puck_to_goal1_post0[1], puck_to_goal1_post1[0], puck_to_goal1_post1[1], puck_to_goal1_post0_angle,
        puck_to_goal1_post1_angle], dtype=torch.float32)

    #print("size of features is ", features.size())

    return features
class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.old_puck_center = None
        self.kart1_center = None
        self.kart2_center = None
        action_net = Planner(17, 128, 6)
        model_filepath = "state_agent.pt"
        # Create the network
        self.model = load_model(action_net, model_filepath)
        self.model.eval()

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
        self.kart1_center = torch.Tensor([0, 0])
        self.kart2_center = torch.Tensor([0, 0])
        #return ['sara_the_racer'] * num_players
        return ['tux'] * num_players
        #return ['pidgin'] * num_players

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

        #print("player_state in state_agent is ", player_state)
        #print("opponent_state in state_agent is ", opponent_state)
        #print("soccer_state in state_agent is ", soccer_state)

        # compute puck and kart velocity
        current_puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        current_kart_center1 = torch.tensor(player_state[0]["kart"]["location"], dtype=torch.float32)[[0, 2]]
        current_kart_center2 = torch.tensor(player_state[1]["kart"]["location"], dtype=torch.float32)[[0, 2]]
        puck_velocity = current_puck_center - self.old_puck_center
        kart1_velocity = current_kart_center1 - self.kart1_center
        kart2_velocity = current_kart_center2 - self.kart2_center

        actions = []
        for player_id, pstate in enumerate(player_state):
            features = network_features(pstate, opponent_state, soccer_state, self.team)
            features = torch.cat([features, puck_velocity, kart1_velocity if player_id == 0 else kart2_velocity])

            """
            acceleration, steer, brake = self.model(features)

            brake_threshold = 0.7
            if brake > brake_threshold:
                brake = True
                acceleration = 0.0
            else:
                brake = False
            """

            logits = self.model(features)
            #print("logits returned are ", logits)
            softmax_output = torch.nn.functional.softmax(logits, dim=0)
            #print("softmax_output returned is ", softmax_output)
            action_id = torch.argmax(softmax_output)
            #print("predicted label is ", action_id)

            #print("output of model is ", logits)

            #logits = Categorical(logits)

            #print(logits)
            #print("outout.categorical.probs is ", logits.probs)

            #action_id = logits.sample()

            #print("action_id is ", action_id)

            action_tuple = ACTION_SPACE[action_id]
            #action_tuple = ACTION_SPACE[-100]
            #actions.append(dict(acceleration=acceleration, steer=steer, brake=brake, nitro=True))  # drift=True))
            actions.append(dict(acceleration=action_tuple[1], steer=action_tuple[2], brake=action_tuple[0] ,fire = True))
            # update puck center
            self.old_puck_center = current_puck_center
            self.kart1_center = current_kart_center1
            self.kart2_center = current_kart_center2
        return actions
