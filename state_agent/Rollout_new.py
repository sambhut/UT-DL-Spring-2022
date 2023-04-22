import torch
import pystk
import ray
#more imports..

#WARNING! == Won't compile. Work under progress

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
        from .remote import ray
        if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def __init__(self, team0, team1, num_player=1):
        # fire up pystk
        graphics_config = pystk.GraphicsConfig.none()
        pystk.init(graphics_config)

        # set teams for a new match
        team0_cars = self._g(self._r(team0.new_match)(0, num_player))
        team1_cars = self._g(self._r(team1.new_match)(1, num_player))

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
        race = pystk.Race(race_config)
        race.start()
        race.step()

        self.team0 = team0
        self.team1 = team1

    def __call__(self, initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], max_frames=1000):
        data = []
        state = pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))

        for it in range(max_frames):
            state.update()

            # Get the state
            team0_state = [to_native(p) for p in state.players[0::2]]
            team1_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            agent_data = {'player_state': team0_state, 'opponent_state': team1_state, 'soccer_state': soccer_state}

            # Have each team produce actions (in parallel)
            t0_can_act = True  # True for now, or we can use _check function in runner
            t1_can_act = True
            if t0_can_act:
                team0_actions_delayed = self._r(team0.act)(team0_state, team1_state, soccer_state)

            if t1_can_act:
                team1_actions_delayed = self._r(team1.act)(team1_state, team0_state, soccer_state)

            #Wait for actions to finish
            team0_actions = self._g(team0_actions_delayed) if t0_can_act else None
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None

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
            for i in range(num_player):
                a0 = team0_actions[i] if team0_actions is not None and i < len(team0_actions) else {}
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                actions.append(a0)
                actions.append(a1)

            agent_data['action'] = team0_actions[0]
            race.step([pystk.Action(**a) for a in actions])

            # Save all relevant data
            data.append(agent_data)
        return data
