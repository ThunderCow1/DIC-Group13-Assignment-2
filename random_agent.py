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

class RandomAgent:

    def __init__(self,
                 robot: Robot):
        self.robot = robot
        self.actions = robot.actions

    def select_action(self, environment):
        action_list = []
        for action in self.actions:
            if random.random() > 0.5:
                action_list.append(action)

        return action_list