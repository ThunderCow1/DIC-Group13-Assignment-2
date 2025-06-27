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
from environment.robot import Robot

class HumanAgent:
    def __init__(self, robot):
        self.robot = robot
        if type(robot) == Robot:
            self.key_action_map = {
                pygame.K_RIGHT: 'turn_right',
                pygame.K_LEFT: 'turn_left',
                pygame.K_UP: 'accelerate',
                pygame.K_DOWN: 'break',
                pygame.K_SPACE: 'break_hard'
            }

    def select_action(self, x_diff, y_dff, orientation, speed, angle_diff, distances):
        pygame.event.pump()  # Process event queue
        keys = pygame.key.get_pressed()

        action_list = [
            action for key, action in self.key_action_map.items() if keys[key]
        ]

        return action_list
    
    def _update(self, state, action, reward, next_state, done):
        pass