import random
import datetime
import pygame
import numpy as np
import math
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from robot import Robot

class IntuitiveAgent:

    def __init__(self,
                 robot: Robot):
        self.robot = robot
        self.actions = robot.actions

    def select_action(self, x, y, target_x, target_y, distances):
        action_list = []
        
        # First check whether to turn
        direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])

        angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360

        angle_diff = (angle_to_target - self.robot.orientation + 540) % 360 - 180
        
        if distances[0] <= 10 or distances [1] <= 30:
            action_list.append('turn_right')
        elif angle_diff <= 0:
            action_list.append('turn_left')

        if distances[4] <= 10 or distances [3] <= 30:
            action_list.append('turn_left')
        elif angle_diff >= 0:
            action_list.append('turn_right')

        # Limit speed
        if distances[2] >= 50:
            action_list.append('accelerate')
        
        elif distances[2] <= 50:
            action_list.append('break')

        elif distances[2] <= 25:
            action_list.append('break_hard')
        print(action_list)
        return action_list
        

