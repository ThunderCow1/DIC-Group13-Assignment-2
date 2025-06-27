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

    def select_action(self, x, y, robot_orientation,
                      robot_speed, target_x, target_y,
                      angle_diff, distances:  list):
        
        action_list = []
        
        if distances[0] <= 25 or distances [1] <= 50:
            action_list.append('turn_right')
        elif angle_diff <= 0:
            action_list.append('turn_left')

        if distances[4] <= 25 or distances [3] <= 50:
            action_list.append('turn_left')
        elif angle_diff >= 0:
            action_list.append('turn_right')

        # Limit speed
        if distances[2] >= 100:
            action_list.append('accelerate')
        
        elif distances[2] <= 70:
            action_list.append('break')

        elif distances[2] <= 30:
            action_list.append('break_hard')

        return action_list
    
    def _update(self, reward, old_pos, action_list):
        pass
        

