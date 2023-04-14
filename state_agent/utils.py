import torch
import pystk
import ray
import numpy as np
from config import device
from IPython.display import Video, display
import imageio

@ray.remote
class Rollout:

    def euclidean_distance(self,pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def __init__(self, screen_width, screen_height, hd=True, track='icy_soccer_field', render=True, frame_skip=1):

        config = pystk.GraphicsConfig.hd()
        config.screen_width = screen_width
        config.screen_height = screen_height
        pystk.init(config)

        self.frame_skip = frame_skip
        self.render = render
        config = pystk.RaceConfig()
        config.mode = config.RaceMode.SOCCER
        config.track = "icy_soccer_field"
        config.step_size = 0.1
        config.num_kart = 2
        config.players[0].kart = "wilber"
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        config.players[0].team = 0
        config.players.append(
        pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1))

        self.race = pystk.Race(config)
        self.race.start()

    def __call__(self, agent, n_steps=200):
        torch.set_num_threads(1)
        self.race.restart()
        self.race.step()
        data = []
        world_info = pystk.WorldState()
        world_info.update()
        total_distance = 0
        prev_position = world_info.players[0].kart.location
        for i in range(n_steps // self.frame_skip):
            world_info = pystk.WorldState()

            world_info.update()

            player_info = world_info.players[0].kart
            opponent_info = world_info.players[1].kart
            soccer_ball = world_info.soccer.ball
            soccer_state = world_info.soccer
            soccer_ball_loc = world_info.soccer.ball.location
            goal_location = soccer_state.goal_line[1]
            goal_ball_distance = np.array(soccer_ball_loc) - np.array(goal_location[1])
            puck_goal_distance = np.linalg.norm(goal_ball_distance)
            reward_state = 1/(puck_goal_distance +1)

            current_position = np.array(player_info.location)
            total_distance += self.euclidean_distance(prev_position, current_position)
            prev_position = current_position

            agent_data = { 'player_state': player_info,'opponent_state':opponent_info,'soccer_state':soccer_state,'overall_distance':total_distance,"reward_state":reward_state}
            if self.render:
                agent_data['image'] = np.array(self.race.render_data[0].image)

            action = agent(**agent_data)
            agent_data['action'] = action

            for it in range(self.frame_skip):
                self.race.step(action)

            data.append(agent_data)
        return data


def show_video(frames, fps=30):
    imageio.mimwrite('/tmp/test.mp4', frames, fps=fps, bitrate=1000000)
    display(Video('/tmp/test.mp4', width=800, height=600, embed=True))


viz_rollout = Rollout.remote(400, 300)

rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5) for i in range(10)]
def rollout_many(many_agents, **kwargs):
    ray_data = []
    for i, agent in enumerate(many_agents):
         ray_data.append( rollouts[i % len(rollouts)].__call__.remote(agent, **kwargs) )
    return ray.get(ray_data)


def show_agent(agent, n_steps=600):
    data = ray.get(viz_rollout.__call__.remote(agent, n_steps=n_steps))
    show_video([d['image'] for d in data])



def dummy_agent(**kwargs):
    action = pystk.Action()
    action.acceleration = 1
    return action
