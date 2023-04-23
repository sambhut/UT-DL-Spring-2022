import torch
import pystk
import ray
import numpy as np
from jurgen_agent.player import Team as Jurgen
from geoffrey_agent.player import Team as Geoffrey
from tournament.utils import VideoRecorder
#more imports..

#WARNING! == Won't compile. Work under progress

MAX_FRAMES = 1000

ray_init_done = 0

pystk_init_done = False

class AIRunner:
    #agent_type = 'state'
    #is_ai = True
    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    #def info(self):
    #    return RunnerInfo('state', None, 0)

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

def init_ray():
    ray.init() #??
    #TODO: need to complete ray support. Some functions are added for now (_r and _g below)

class Rollout_new:
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
        import ray
        if ray is not None:
            return ray.get(f)
        return f

    def __init__(self, team0, team1=AIRunner(), num_player=2, use_ray=False):

        global pystk_init_done
        # fire up pystk
        graphics_config = pystk.GraphicsConfig.none()

        if pystk_init_done == False:
            pystk_init_done = True
            pystk.init(graphics_config)

        self.num_player = num_player

        # set teams for a new match
        #team0_cars = self._g(self._r(team0.new_match)(0, num_player))
        #team1_cars = self._g(self._r(team1.new_match)(1, num_player))
        team0_cars = team0.new_match(0, num_player) or ['tux']
        team1_cars = team1.new_match(1, num_player) or ['sara_the_racer']

        # set race config and players config
        RaceConfig = pystk.RaceConfig
        race_config = RaceConfig(track="icy_soccer_field", mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)

        PlayerConfig = pystk.PlayerConfig
        controller0 = PlayerConfig.Controller.PLAYER_CONTROL
        controller1 = PlayerConfig.Controller.AI_CONTROL
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(PlayerConfig(controller=controller0, team=0, kart=team0_cars[i % len(team0_cars)]))
            race_config.players.append(PlayerConfig(controller=controller1, team=1, kart=team1_cars[i % len(team1_cars)]))

        #Start the match
        self.race = pystk.Race(race_config)
        self.race.start()
        #self.race.step()

        self.team0 = team0
        self.team1 = team1

    def __call__(self, initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], max_frames=MAX_FRAMES, use_ray=False, record_fn=None):
        global pystk_init_done
        global print_flag

        data = []
        state = pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))

        if record_fn:
            print("record_fn is not None")
        else:
            print("record_fn is None")

        for it in range(max_frames):
            state.update()

            # Get the state
            team0_state = [to_native(p) for p in state.players[0::2]]
            team1_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            agent_data = {'player_state': team0_state, 'opponent_state': team1_state, 'soccer_state': soccer_state}

            # print some data every 100 frames
            if (it%100) == 0:
                print("iteration {%d} / {%d}" % (it, max_frames))
                print("team0 state is ", team0_state[0]["kart"]["location"])
                print("team1 state is ", team1_state[0]["kart"]["location"])
                print("soccer state is ", soccer_state['ball']['location'])

            # Have each team produce actions (in parallel)
            t0_can_act = True  # True for now, or we can use _check function in runner
            t1_can_act = True
            if t0_can_act:
                team0_actions_delayed = self._r(self.team0.act)(team0_state, team1_state, soccer_state)

            if t1_can_act:
                team1_actions_delayed = self._r(self.team1.act)(team1_state, team0_state, soccer_state)

            #Wait for actions to finish
            #team0_actions = self._g(team0_actions_delayed) if t0_can_act else None
            #team1_actions = self._g(team1_actions_delayed) if t1_can_act else None

            team0_actions = team0_actions_delayed if t0_can_act else None
            team1_actions = team1_actions_delayed if t1_can_act else None

            """
            CHECK: do we need this really??
            new_t0_can_act, new_t1_can_act = self._check(team0, team1, 'act', it, timeout)
            if not new_t0_can_act and t0_can_act and verbose:
                print('Team 0 timed out')
            if not new_t1_can_act and t1_can_act and verbose:
                print('Team 1 timed out')

            t0_can_act, t1_can_act = new_t0_can_act, new_t1_can_act
            """

            # Assemble the actions
            actions = []
            for i in range(self.num_player):
                a0 = team0_actions[i] if team0_actions is not None and i < len(team0_actions) else {}
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                actions.append(a0)
                actions.append(a1)

            if record_fn:
                self._r(record_fn)(team0_state, team1_state, soccer_state=soccer_state, actions=actions)

            agent_data['action'] = team0_actions[0]
            self.race.step([pystk.Action(**a) for a in actions])

            # Save all relevant data
            data.append(agent_data)


        self.race.stop()
        del self.race
        pystk.clean()
        pystk_init_done = False

        return data

def rollout_many(many_agents, **kwargs):
    data = []
    for i, agent in enumerate(many_agents):
         rollout = Rollout_new(many_agents[i], **kwargs)
         data.append(rollout.__call__(**kwargs))
    return data

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    team0 = Jurgen()
    team1 = AIRunner()
    num_player = 2
    use_ray = False
    record_video = True
    video_name = "rollout.mp4"

    recorder = None
    if record_video:
        recorder = recorder & VideoRecorder(video_name)

    rollout = Rollout_new(team0=team0, team1=team1, num_player=num_player, use_ray=use_ray)

    rollout.__call__(use_ray=use_ray, record_fn=recorder)

