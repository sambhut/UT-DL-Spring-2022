import logging
import numpy as np
from collections import namedtuple
import torch

from geoffrey_agent import Team
import pickle
from argparse import ArgumentParser
from pathlib import Path
from os import environ



TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

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
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
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

    def _check(self, team1, team2, where, n_iter, timeout):
        _, error, t1 = self._g(self._r(team1.info)())
        if error:
            raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))

        _, error, t2 = self._g(self._r(team2.info)())
        if error:
            raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')

        logging.debug('timeout {} <? {} {}'.format(timeout, t1, t2))
        return t1 < timeout, t2 < timeout

    def euclidean_distance(self,point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1000000000,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], verbose=False):
        RaceConfig = self._pystk.RaceConfig

        logging.info('Creating teams')

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        t1_type, *_ = self._g(self._r(team1.info)())
        t2_type, *_ = self._g(self._r(team2.info)())

        if t1_type == 'image' or t2_type == 'image':
            assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        t1_can_act, t2_can_act = self._check(team1, team2, 'new_match', 0, timeout)

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

        threshold_goal_distance = 2
        reward_puck_kart_threshold = 1.0
        total_rewards = 0
        for it in range(max_frames):
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            team1_images = team2_images = None
            if self._use_graphics:
                team1_images = [np.array(race.render_data[i].image) for i in range(0, len(race.render_data), 2)]
                team2_images = [np.array(race.render_data[i].image) for i in range(1, len(race.render_data), 2)]

            # Have each team produce actions (in parallel)
            if t1_can_act:
                if t1_type == 'image':
                    team1_actions_delayed = self._r(team1.act)(team1_state, team1_images)
                else:
                    team1_actions_delayed = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_can_act:
                if t2_type == 'image':
                    team2_actions_delayed = self._r(team2.act)(team2_state, team2_images)
                else:
                    team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None
            team2_actions = self._g(team2_actions_delayed) if t2_can_act else None

            new_t1_can_act, new_t2_can_act = self._check(team1, team2, 'act', it, timeout)
            if not new_t1_can_act and t1_can_act and verbose:
                print('Team 1 timed out')
            if not new_t2_can_act and t2_can_act and verbose:
                print('Team 2 timed out')

            t1_can_act, t2_can_act = new_t1_can_act, new_t2_can_act


            # Assemble the actions
            actions = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                actions.append(a2)

            current_distance = soccer_state["ball"]["location"][0]
            if current_distance > highest_distance:
                highest_distance = current_distance
                payload = {
                    'team1': {
                        'highest_distance': highest_distance
                    }
                }
            #Rewards

            #Soccer ball and goal distance (Dense reward setting)
            soccer_ball_loc = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            goal_line_center = torch.tensor(soccer_state['goal_line'][1], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            #goal_location = torch.tensor([0, -64.5], dtype=torch.float32)
            current_distance = self.euclidean_distance(soccer_ball_loc, goal_line_center)
            if current_distance < threshold_goal_distance:
                puck_goal_distance_reward = 1
            else:
                # No reward
                puck_goal_distance_reward = 0


            # rewards towards puck - distance

            for player_info in team1_state:
                distance = self.euclidean_distance(
                    torch.tensor(player_info['kart']['location'], dtype=torch.float32)[[0, 2]], soccer_ball_loc)
                reward = 1 if distance < reward_puck_kart_threshold else 0
                total_rewards += reward

            average_reward = total_rewards / 2
            reward_towards_puck = average_reward


            reward_weight_puck_goal = 2
            reward_weight_towards_puck = 3.5
            reward_weight_puck_direction = 2.5

            reward_state = (
                    (reward_weight_puck_goal * puck_goal_distance_reward ) +
                    (reward_weight_towards_puck * reward_towards_puck)
            )

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=team1_images, team2_images=team2_images)
            data_temp = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions,reward_state=reward_state)


            print(f"Rewards towards puck : {reward_towards_puck}")
            print(f"Puck-goal distance reward : {puck_goal_distance_reward}")
            print(f"Total Reward state: {reward_state}")

            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break
            data.append(data_temp)

        race.stop()
        del race

        return data,payload

    def wait(self, x):
        return x

def load_recorded_state(file_path):
    with open(file_path, 'rb') as f:
        recorded_state = pickle.load(f)
    return recorded_state


def record_state(team):
    from . import  utils
    team1 = TeamRunner(team)
    team2 = AIRunner()
    state_file_path = 'recorded_state'

    match = Match(use_graphics=False)
    result =None
    try:
        result = match.run(team1, team2, 2, 1200, 3)
    except MatchException as e:
        print('Match failed', e.score)
        print(' T1:', e.msg1)
        print(' T2:', e.msg2)

    return result


def record_video(team):
    from . import utils
    team1 = TeamRunner(team)
    team2 = AIRunner()
    video_file_path = 'my_video.mp4'
    recorder = utils.VideoRecorder(video_file_path)
    match = Match(use_graphics=False)
    try:
        result = match.run(team1, team2, 2, 1200, 3,record_fn=recorder)
    except MatchException as e:
        print('Match failed', e.score)
        print(' T1:', e.msg1)
        print(' T2:', e.msg2)



def record_manystate(many_agents,parallel=10):
    from . import remote
    import ray
    team2 = AIRunner()
    results = []
    remote_calls = []
    match = Match(use_graphics=False)
    remote_calls.append(match.run(TeamRunner(many_agents[0]), team2, 2, 1200, 3))

    return remote_calls



















