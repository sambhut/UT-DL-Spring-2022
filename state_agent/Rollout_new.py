import torch
import pystk
import numpy as np
from geoffrey_agent.player import Team as Geoffrey
from image_jurgen_agent.player import Team as ImJurgen
from jurgen_agent.player import Team as Jurgen
from yann_agent.player import Team as Yann
from yoshua_agent.player import Team as Yoshua
from tournament.utils import VideoRecorder
import random

MAX_FRAMES = 1200

ray_init_done = 0

pystk_init_done = False

opponents_str = ["Geoffrey", "ImJurgen()", "Jurgen()", "Yann()", "Yoshua()", "AIRunner()"]

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



class Rollout_new:
    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    def __init__(self, team0, opponent_id=-1, team1=AIRunner(), num_player=2, use_ray=False, train=False):

        global pystk_init_done
        # fire up pystk
        graphics_config = pystk.GraphicsConfig.none()

        if pystk_init_done == False:
            pystk_init_done = True
            pystk.init(graphics_config)

        opponents = [Geoffrey(), ImJurgen(), Jurgen(), Yann(), Yoshua(), AIRunner()]

        self.num_player = num_player

        opponent_no = opponent_id % len(opponents)

        AIteam = -1

        self.player_team = 0

        if train is True:

            if opponent_id < len(opponents):
                team0_updated = team0
                team1_updated = opponents[opponent_no]
                AIteam = 1 if opponent_no == (len(opponents)-1) else -1
                self.player_team = 0

            else:
                team0_updated = opponents[opponent_no]
                team1_updated = team0
                AIteam = 0 if opponent_no == (len(opponents)-1) else -1
                self.player_team = 1

        else:
            team0_updated = team0
            team1_updated = team1
            opponent_no = len(opponents) - 1
            AIteam = 1
            self.player_team = 0

        print("opponent_no is ", opponent_no)
        print("AIteam is ", AIteam)
        print("team0 is ", team0_updated)
        print("team1 is ", team1_updated)
        print("player team is ", self.player_team)

        # set teams for a new match
        #team0_cars = self._g(self._r(team0.new_match)(0, num_player))
        #team1_cars = self._g(self._r(team1.new_match)(1, num_player))
        team0_cars = team0_updated.new_match(0, num_player) or ['tux']
        team1_cars = team1_updated.new_match(1, num_player) or ['tux']

        # set race config and players config
        RaceConfig = pystk.RaceConfig
        race_config = RaceConfig(track="icy_soccer_field", mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)

        PlayerConfig = pystk.PlayerConfig

        if AIteam == -1:
            controller0 = PlayerConfig.Controller.PLAYER_CONTROL
            controller1 = PlayerConfig.Controller.PLAYER_CONTROL

        elif AIteam == 0:
            controller0 = PlayerConfig.Controller.AI_CONTROL
            controller1 = PlayerConfig.Controller.PLAYER_CONTROL
        else:
            controller0 = PlayerConfig.Controller.PLAYER_CONTROL
            controller1 = PlayerConfig.Controller.AI_CONTROL

        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(PlayerConfig(controller=controller0, team=0, kart=team0_cars[i % len(team0_cars)]))
            race_config.players.append(PlayerConfig(controller=controller1, team=1, kart=team1_cars[i % len(team1_cars)]))

        #Start the match
        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.step()

        self.team0 = team0
        self.team1 = team1_updated

    def __call__(self, initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], initial_angular_velocity = [0, 0] ,max_frames=MAX_FRAMES, use_ray=False, record_fn=None, train=False):
        global pystk_init_done

        data = []
        state = pystk.WorldState()
        state.update()

        # Add some randomness to the starting location
        if train is True :
            rand3 = random.uniform(-10, 10)
            rand4 = random.uniform(-10, 10)
            soccer_state = to_native(state.soccer)
            print("Before- soccer ball location is ", soccer_state['ball']['location'] ,rand3,rand4)
            state.set_ball_location([initial_ball_location[0] + rand3, 1, initial_ball_location[1]+ rand4],
                                [initial_ball_velocity[0], 0, initial_ball_velocity[1]],[initial_angular_velocity[0] , 0, initial_angular_velocity[1]])
            state.update()

        old_puck_center = torch.Tensor([0, 0])
        old_kart_center1 = torch.Tensor([0, 0])
        old_kart_center2 = torch.Tensor([0, 0])

        for it in range(max_frames):
            #print("soccer ball location before update is ", to_native(state.soccer)['ball']['location'])
            state.update()

            # Get the state
            team0_state = [to_native(p) for p in state.players[0::2]]
            team1_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            if self.player_team == 0:
                team_state = team0_state
                opp_state = team1_state
            else:
                team_state = team1_state
                opp_state = team0_state

            agent_data = {'player_state': team_state, 'opponent_state': opp_state, 'soccer_state': soccer_state}

            # print some data every 100 frames
            if it<10 and train is True:
                print("train is ", train)
                print("iteration {%d} / {%d}" % (it, max_frames))
                print("player kart0 location is ", team_state[0]["kart"]["location"])
                print("player kart1 location is ", team_state[1]["kart"]["location"])
                print("opponent kart0 is ", opp_state[0]["kart"]["location"])
                print("opponent kart1 is ", opp_state[1]["kart"]["location"])
                print("After soccer ball location is ", soccer_state['ball']['location'])

            # Have each team produce actions (in parallel)
            team_can_act = True  # True for now, or we can use _check function in runner
            opp_can_act = True
            if team_can_act:
                team_actions_delayed = self._r(self.team0.act)(team_state, opp_state, soccer_state)

            if opp_can_act:
                opp_actions_delayed = self._r(self.team1.act)(opp_state, team_state, soccer_state)

            team_actions = team_actions_delayed if team_can_act else None
            opp_actions = opp_actions_delayed if opp_can_act else None

            # Assemble the actions
            actions = []
            for i in range(self.num_player):
                a0 = team_actions[i] if team_actions is not None and i < len(team_actions) else {}
                a1 = opp_actions[i] if opp_actions is not None and i < len(opp_actions) else {}
                actions.append(a0)
                actions.append(a1)

            if record_fn:
                self._r(record_fn)(team_state, opp_state, soccer_state=soccer_state, actions=actions)


            agent_data['action0'] = team_actions[0]
            agent_data['action1'] = team_actions[1]

            self.race.step([pystk.Action(**a) for a in actions])

            # Gather velocity of puck and karts
            current_puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            current_kart_center1 = torch.tensor(team_state[0]["kart"]["location"], dtype=torch.float32)[[0, 2]]
            current_kart_center2 = torch.tensor(team_state[1]["kart"]["location"], dtype=torch.float32)[[0, 2]]
            agent_data['ball_velocity'] = current_puck_center - old_puck_center
            agent_data['kart_velocity'] = []
            agent_data['kart_velocity'].append(current_kart_center1 - old_kart_center1)
            agent_data['kart_velocity'].append(current_kart_center2 - old_kart_center2)

            # Save all relevant data
            data.append(agent_data)
            old_puck_center = current_puck_center
            old_kart_center1 = current_kart_center1
            old_kart_center2 = current_kart_center2

        self.race.stop()
        del self.race
        pystk.clean()
        pystk_init_done = False

        return data

def rollout_many(many_agents, **kwargs):
    data = []
    for i, agent in enumerate(many_agents):
        print("performing rollout number ", i)
        rollout = Rollout_new(many_agents[i], i, train=True, **kwargs)
        data.append(rollout.__call__(train=True, **kwargs))
    return data

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    team0 = Jurgen()
    #team1 = AIRunner()
    team1 = Geoffrey()
    num_player = 2
    use_ray = False
    record_video = True
    video_name = "rollout.mp4"
    rand = True

    recorder = None
    if record_video:
        recorder = recorder & VideoRecorder(video_name)

    rollout = Rollout_new(team0=team0, team1=team1, num_player=num_player, use_ray=use_ray, train=False)

    rollout.__call__(use_ray=use_ray, record_fn=recorder)

