import logging
import math

import numpy as np
from collections import namedtuple
import torch

from state_agent.player import Team
import pickle
from pathlib import Path
from os import environ

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1200
MAX_FRAMES_TRAIN = 600 # tune this for PPO training purpose
NUM_PLAYER = 2

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)


class AIRunner:
    agent_type = 'state'
    is_ai = True

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team):
        self._error = None
        self._team = None
        try:
            self._team = team
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r, lp, ids = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r, lp, ids
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)


class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2


class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    def __init__(self, use_graphics=False, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        from tournament.remote import ray
        if ray is not None and isinstance(f, (ray.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def assign_reward(self, playerid, puck_pos, goal_pos, team1_state, team2_state, old_score, new_score):
        print("playerid, puck_pos, goal_pos : ", playerid, ",")
        print("\t", puck_pos)
        print("\t", goal_pos)
        goal = 2
        default = -0.05
        puck_reward = 0.5
        angle1_reward = 0.2
        angle2_reward = 0.2
        angle3_reward = 0.2
        dis_threshold = 10

        # default
        reward = default

        # Scoring goals
        if old_score[0] < new_score[0]:
            reward += goal
        elif old_score[1] > new_score[1]:
            reward -= goal

        #player distances to puck
        player_pos = torch.tensor(team1_state[playerid]['kart']['location'], dtype=torch.float32)[[0, 2]]

        kart_front = torch.tensor(team1_state[playerid]['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_center = torch.tensor(team1_state[playerid]['kart']['location'], dtype=torch.float32)[[0, 2]]
        player_dir = (kart_front - kart_center) / torch.norm(kart_front - kart_center)

        # Distance between player and puck
        dist_to_puck = torch.norm(player_pos - puck_pos)
        print("dist_to_puck: ", dist_to_puck)

        # Compute angle between player and goal post
        player_goal_dir = (goal_pos - player_pos) / torch.norm(goal_pos - player_pos)
        player_puck_dir = (puck_pos - player_pos) / torch.norm(puck_pos - player_pos)
        angle1 = torch.acos(torch.dot(player_goal_dir, player_puck_dir))
        print("player_goal_dir, player_puck_dir: ", player_goal_dir, player_puck_dir)

        if dist_to_puck > dis_threshold:
            reward += puck_reward * (1 / dist_to_puck)
        else:
            reward += angle1_reward * (torch.cos(angle1))  #reward for facing the goal post

        # Compute angle of kart's direction with kart-puck
        angle2 = torch.acos(torch.dot(player_dir, player_puck_dir))
        # Compute angle of kart's direction with puck-goal
        puck_goal_dir = (goal_pos - puck_pos) / torch.norm(goal_pos - puck_pos)
        angle3 = torch.acos(torch.dot(player_dir, puck_goal_dir))

        print("angles 1,2,3: ", angle1, angle2, angle3)

        if dist_to_puck > dis_threshold:
            reward += angle2_reward * (torch.cos(angle2))
        else:
            reward += angle3_reward * (torch.cos(angle3))
        # Opponents:

        return reward.item()

    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1000000000,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], verbose=False):
        RaceConfig = self._pystk.RaceConfig

        logging.info('Creating teams')

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        # Deal with crashes
        #t1_can_act, t2_can_act = self._check(team1, team2, 'new_match', 0, timeout)
        t1_can_act, t2_can_act = True, True

        # Setup the race config
        logging.info('Setting up race')

        race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(self._make_config(0, hasattr(team1, 'is_ai') and team1.is_ai, t1_cars[i % len(t1_cars)]))
            race_config.players.append(self._make_config(1, hasattr(team2, 'is_ai') and team2.is_ai, t2_cars[i % len(t2_cars)]))

        # Start the match
        logging.info('Starting race')
        race = self._pystk.Race(race_config)
        race.start()
        race.step()

        state = self._pystk.WorldState()
        state.update()

        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))
        data = []
        highest_distance = -np.inf
        payload = {
            'team1': {
                'highest_distance': None
            }
        }

        threshold_goal_distance = 50
        reward_puck_kart_threshold = 5
        total_rewards = 0

        old_puck_center = torch.Tensor([initial_ball_location[0], initial_ball_location[1]])
        for it in range(max_frames):
            print("FRAME ", it, ":")
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)

            # Have each team produce actions (in parallel)
            if t1_can_act:
                print("t1 can act")
                team1_actions_delayed, logprobs, team1_action_ids = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_can_act:
                team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None
            team2_actions = self._g(team2_actions_delayed) if t2_can_act else None

            # Assemble the actions
            actions = []
            action_ids = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a1_id = team1_action_ids[i] if team1_action_ids is not None and i < len(team1_action_ids) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                action_ids.append(a1_id)
                actions.append(a2)

            # save velocity and score for next frame
            soccer_ball_loc = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            puck_velocity = soccer_ball_loc - old_puck_center

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=None, team2_images=None)

            old_score = soccer_state['score']
            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            # Take a step in the environment
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break
            new_score = soccer_state['score']
            done = (new_score != old_score) # terminal state flag

            # Rewards
            puck_pos = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            goal_pos1 = torch.tensor(soccer_state['goal_line'][0], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            goal_pos2 = torch.tensor(soccer_state['goal_line'][1], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            reward1 = self.assign_reward(0, puck_pos, goal_pos1, team1_state, team2_state, old_score, new_score)
            reward2 = self.assign_reward(1, puck_pos, goal_pos2, team1_state, team2_state, old_score, new_score)
            reward_state = [reward1, reward2]

            # Save this step's data
            data_temp = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state,
                             action_ids=action_ids, reward_state=reward_state, logprobs=logprobs,
                             puck_velocity=puck_velocity, done=done)
            print("Reward (data_temp): ", reward_state)
            data.append(data_temp)

            #update old values with new after taking step
            old_puck_center = soccer_ball_loc

        race.stop()
        del race

        return data, payload

    def wait(self, x):
        return x

def load_recorded_state(file_path):
    with open(file_path, 'rb') as f:
        recorded_state = pickle.load(f)
    return recorded_state


def record_state(team):
    from . import  utils
    #team1 = TeamRunner(team)
    team2 = AIRunner()
    state_file_path = 'recorded_state'

    match = Match(use_graphics=False)
    result =None
    try:
        result = match.run(team, team2, 2, 1200, 3)
    except MatchException as e:
        print('Match failed', e.score)
        print(' T1:', e.msg1)
        print(' T2:', e.msg2)

    return result


def record_video(team):
    from . import utils
    #team1 = TeamRunner(team)
    team2 = AIRunner()
    video_file_path = 'my_video.mp4'
    recorder = utils.VideoRecorder(video_file_path)
    match = Match(use_graphics=False)
    try:
        result = match.run(team, team2, 2, 1200, 3,record_fn=recorder)
    except MatchException as e:
        print('Match failed', e.score)
        print(' T1:', e.msg1)
        print(' T2:', e.msg2)



def record_manystate(many_agents,parallel=10):
    #from . import remote
    #import ray
    team2 = AIRunner()
    results = []
    remote_calls = []
    match = Match(use_graphics=False)

    for agent in many_agents:
        remote_calls.append(match.run(agent, team2, NUM_PLAYER, MAX_FRAMES_TRAIN, 3))

    return remote_calls
