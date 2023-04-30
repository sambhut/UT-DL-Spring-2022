from time import time

import pystk
import numpy as np

from argparse import ArgumentParser

from state_agent.dummy import gui

import random


class Player:

    def __init__(self, team_orientation_multiplier, player_id):
        self.team_orientation_multiplier = team_orientation_multiplier
        self.player_id = player_id
        self.goal = np.array([0.0, 64.5])
        self.last_world_pos = np.array([0.0, -64.5])


    def act(self, action, player_info, puck_location=None, last_seen_side=None, testing=False):
        if (puck_location is not None):
            puck_location *= self.team_orientation_multiplier
        kart = player_info

        pos_me = to_numpy(kart.location) * self.team_orientation_multiplier
        front_me = to_numpy(kart.front) * self.team_orientation_multiplier
        ori_me = get_vector_from_this_to_that(pos_me, front_me)


        kart_vel = np.dot(to_numpy(kart.velocity) * self.team_orientation_multiplier, ori_me)


        if (kart_vel == 0 and abs(np.linalg.norm(self.last_world_pos - pos_me)) > 5):
            Player.goalieID = 0.

        self.last_world_pos = pos_me

        ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location, normalize=False)
        ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
        ori_puck_to_goal_n = get_vector_from_this_to_that(puck_location, self.goal, normalize=True)

        to_puck_mag = np.linalg.norm(ori_to_puck)

        if (to_puck_mag > 20):
            action["acceleration"] = .8
            if (to_puck_mag > 80):
                action["acceleration"] = .5
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
            if turn_mag > 1e-25:
                action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me)) * turn_mag * 5000
        else:
            action["acceleration"] = .8
            if (to_puck_mag > 10):  # really close
                action["acceleration"] = .5
            pos_hit_loc = puck_location - 1.3 * ori_puck_to_goal_n
            ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
            if turn_mag > 1e-25:
                action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me)) * turn_mag * 5000

        return action


def is_location_almost_equal(point1, point2, tolerance=1e-6):
    return all(abs(coord1 - coord2) <= tolerance for coord1, coord2 in zip(point1, point2))



def to_numpy(location):
    return np.float32([location[0], location[2]])


def get_vector_from_this_to_that(me, obj, normalize=True):
    vector = obj - me
    if normalize:
        return vector / np.linalg.norm(vector)
    return vector


if __name__ == "__main__":
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300

    pystk.init(config)

    config = pystk.RaceConfig()
    config.track = "icy_soccer_field"
    config.mode = config.RaceMode.SOCCER
    config.step_size = 0.1
    config.num_kart = 2
    config.players[0].kart = "tux"
    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    config.players[0].team = 0
    config.players.append(pystk.PlayerConfig("", pystk.PlayerConfig.Controller.PLAYER_CONTROL, 0))
    config.difficulty = 2
    race = pystk.Race(config)
    race.start()

    uis = [gui.UI([gui.VT['IMAGE']])]

    state = pystk.WorldState()
    t0 = time()
    n = 0


    goal_line = np.array([[[-10.449999809265137, 0.07000000029802322, -64.5], [10.449999809265137, 0.07000000029802322, -64.5]], [[10.460000038146973, 0.07000000029802322, 64.5], [-10.510000228881836, 0.07000000029802322, 64.5]]])
    
    
    team_orientaion_multiplier = -2*(config.players[0].team %2)+1
    ctrl0 = Player(team_orientaion_multiplier, 0)
    ctrl1 = Player(team_orientaion_multiplier, 1)
    last_seen_side0 = None
    last_seen_side1 = None
    goal = np.array([0.0,64.5])
    max_frames = 12000000000
    initial_ball_location = [0.0, 1.0]
    initial_ball_velocity = [0, 0]
    initial_angular_velocity = [0, 0]
    target_location = np.array([-0.47587478160858154, 1.4951531887054443, 0.5443593263626099])

    race.step(uis[0].current_action)
    state.update()
    for it in range(2):
        rand3 = random.uniform(-10, 10)
        rand4 = random.uniform(-10, 10)
        state.set_ball_location((initial_ball_location[0] + rand3, 0, initial_ball_location[1] + rand4),
                                (initial_ball_velocity[0] + rand3, 0, initial_ball_velocity[1] + rand4))


        state.update()

        print("soccer ball location before playing game is ", state.soccer.ball.location)

        for it in range(1200):

            race.step(uis[0].current_action)
            state.update()

            target_location = np.array([-0.47587478160858154, 1.4951531887054443, 0.5443593263626099])
            if is_location_almost_equal(state.soccer.ball.location, target_location):
                state.set_ball_location((initial_ball_location[0] + rand3, 0, initial_ball_location[1] + rand4),
                                        (initial_ball_velocity[0] + rand3, 0, initial_ball_velocity[1] + rand4))
                state.update()



            print("soccer ball location after start game is ", state.soccer.ball.location)

            puck_location = to_numpy(state.soccer.ball.location)
            if (len(state.karts)==1):
                pos_ai =np.array([0.0,0.0])
            else:
                pos_ai = to_numpy(state.karts[1].location)
            pos_me = to_numpy(state.karts[0].location)

            puck_location*=team_orientaion_multiplier
            pos_ai*=team_orientaion_multiplier
            pos_me*=team_orientaion_multiplier


            closest_item_distance = np.linalg.norm(
                        get_vector_from_this_to_that(pos_me, puck_location, normalize=False))

            front_me = to_numpy(state.karts[0].front)*team_orientaion_multiplier
            ori_me = get_vector_from_this_to_that(pos_me, front_me)
            ori_to_ai = get_vector_from_this_to_that(pos_me, pos_ai)
            ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location)
            ori_puck_to_goal = get_vector_from_this_to_that(puck_location, goal)

            action0 = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0, 'fire': False}
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck))


            action0 = ctrl0.act(action0, state.karts[0], puck_location=puck_location,last_seen_side=last_seen_side0,testing=True)

            uis[0].current_action.steer = action0["steer"]
            uis[0].current_action.acceleration = action0["acceleration"]
            uis[0].current_action.brake = action0["brake"]
            uis[0].current_action.fire = action0["fire"]
            uis[0].current_action.rescue = action0["rescue"]
            print(uis[0].current_action)

            if (len(state.karts)>=3 and len(uis)>=3):
                pos_me1 = to_numpy(state.karts[0].location)*team_orientaion_multiplier
                front_me1 = to_numpy(state.karts[2].front)*team_orientaion_multiplier
                ori_me1 = get_vector_from_this_to_that(pos_me1, front_me1)
                ori_to_puck1 = get_vector_from_this_to_that(pos_me1, puck_location)
                action1 = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0, 'fire': False}
                turn_mag1 = abs(1 - np.dot(ori_me1, ori_to_puck1))
                if (turn_mag1 >.4):
                    action1 = ctrl1.act(action1, state.karts[2],puck_location=None,last_seen_side=last_seen_side1, testing=True)
                else:
                    print("WE CAN SEE")
                    last_seen_side1 = np.sign(np.cross(ori_to_puck, ori_me))
                    action1 = ctrl1.act(action1, state.karts[2], puck_location=puck_location,last_seen_side=last_seen_side1,testing=True)

                uis[2].current_action.steer = action1["steer"]
                uis[2].current_action.acceleration = action1["acceleration"]
                uis[2].current_action.brake = action1["brake"]
                uis[2].current_action.fire = action1["fire"]

            print(len(race.render_data))
            if (len(uis)>=3):
                #for ui, d in zip(uis, race.render_data):
                #    ui.show(d)
                for ui, d in zip([uis[1],uis[2]], [race.render_data[1],race.render_data[2]]):
                    ui.show(d)
            else:
                for ui, d in zip(uis, race.render_data):
                    ui.show(d)

            n += 1
            delta_d = n * config.step_size - (time() - t0)

    race.stop()
    del race
    pystk.clean()
