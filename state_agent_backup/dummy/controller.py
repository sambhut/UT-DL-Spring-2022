import numpy as np
import torch
DEBUG = False


def to_numpy(location):
    return np.float32([location[0], location[2]])


def get_vector_from_this_to_that(me, obj, normalize=True):
    vector = obj - me
    if normalize:
        return vector / np.linalg.norm(vector)
    return vector

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Controller1:
    goalieID=0.
    

    def __init__(self, team_orientaion_multiplier, player_id):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        self.team_orientaion_multiplier = team_orientaion_multiplier
        self.player_id = player_id
        self.goal = np.array([0.0,64.5]) #const
        self.goalKeepLoc = np.array([0.0,-66]) #const
        self.attempted_to_fire = False
        self.last_world_pos = np.array([0.0,-64.5])

        self.backupCounter = -1
     
    def act(self, action, player_info, puck_location=None, last_seen_side=None, testing=False):
        # Fire every other frame
        action["fire"]= self.attempted_to_fire 
        self.attempted_to_fire = not self.attempted_to_fire

        if testing:
            # Get world positions
            if (puck_location is not None):
                #  Standardizing direction 2 elements
                # [0] is negitive when facing left side of court (left of your goal), positive when right
                # [1] is positive towards enemy goal, negitive when facing your goal
                puck_location*=self.team_orientaion_multiplier
            kart = player_info
        else:
            kart = player_info.kart
        pos_me = to_numpy(kart.location)*self.team_orientaion_multiplier

        # Get kart vector
        front_me = to_numpy(kart.front)*self.team_orientaion_multiplier
        ori_me = get_vector_from_this_to_that(pos_me, front_me)

        # Determine we are moving backwards
        backing_turn_multiplier = 1.
        kart_vel = np.dot(to_numpy(kart.velocity)*self.team_orientaion_multiplier,ori_me)
        if kart_vel < 0:
            backing_turn_multiplier = -1.
        if DEBUG:
            print("kart_vel",kart_vel)

        # determine if we are  in a new round
        if (kart_vel == 0 and abs(np.linalg.norm(self.last_world_pos-pos_me))>5):
            Controller1.goalieID=0.

        self.last_world_pos = pos_me


        self.backupCounter = -1
        ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
        ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
        ori_puck_to_goal = get_vector_from_this_to_that(puck_location, self.goal,normalize=False)
        ori_puck_to_goal_n = get_vector_from_this_to_that(puck_location, self.goal,normalize=True)

        to_puck_mag = np.linalg.norm(ori_to_puck)
        #if (pos_me[1]>24.5): #there third
        #elif (pos_me[1]<-24.5): #our third

        if (to_puck_mag>20): # not close to puck
            action["acceleration"] = .8
            if (to_puck_mag>80):# really far
                action["acceleration"] = .5
            if DEBUG:
                print("not close to puck")
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
            #print(turn_mag)
            if turn_mag > 1e-25:
                action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000*backing_turn_multiplier
        else: # close to puck
            if DEBUG:
                print("close to puck")
            #ab_player_puck = angle_between(ori_to_puck,ori_me)

            action["acceleration"] = .8
            if (to_puck_mag>10):# really close
                action["acceleration"] = .5
            pos_hit_loc = puck_location-1.3*ori_puck_to_goal_n
            ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
            if turn_mag > 1e-25:
                action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000*backing_turn_multiplier




        return action

