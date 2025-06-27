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
from environment.robot import Robot

class IntuitiveAgent:

    def __init__(self,
                 robot: Robot):
        self.robot = robot
        self.actions = robot.actions

    def select_action(self, x_diff, y_dff, orientation, speed, angle_diff, distances):
        
        action_list = []
        
        if distances[0] <= 25 or distances [1] <= 50:
            action_list.append('turn_right')
        elif angle_diff <= 0:
            action_list.append('turn_left')

        if distances[-1] <= 25 or distances [-2] <= 50:
            action_list.append('turn_left')
        elif angle_diff >= 0:
            action_list.append('turn_right')

        # Limit speed
        if distances[math.floor(len(distances)/2)] >= 100:
            action_list.append('accelerate')
        
        elif distances[math.floor(len(distances)/2)] <= 70:
            action_list.append('break')

        elif distances[math.floor(len(distances)/2)] <= 30:
            action_list.append('break_hard')

        return action_list
    
    def _update(self, reward, old_pos, action_list):
        pass
        

