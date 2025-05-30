import random
import datetime
import pygame
import numpy as np
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from map import Map
from robot import Robot
from random_agent import RandomAgent
from human_agent import HumanAgent
from intuitive_agent import IntuitiveAgent

class Environment:
    def __init__(self,
                 map_fp: Path,
                 no_gui: bool = False,
                 agent_start_pos: tuple[float, float] = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = random.random(),
                 draw = False):
        
        random.seed(random_seed)

        # Initialize Grid
        self.map = Map(map_fp = map_fp)
        self.obstacle_mask = self.map.create_obstacle_mask()

        self.draw = draw

        if agent_start_pos == None:
            print('No start position, initialising random position')
            self.agent_pos = self.map.random_pos()
        else:
            self.agent_pos = agent_start_pos

        self.robot = Robot(self.agent_pos)

        self.screen = pygame.display.set_mode(self.map.map_size)
        self.mask_surface = self.obstacle_mask.to_surface(self.screen, setcolor=(200,200,200))



    def _update(self, agent):
        distances = self.robot.gain_sensor_output(self.obstacle_mask, get_directions=False)

        action_list = agent.select_action(self.robot.position[0], 
                                          self.robot.position[1], 
                                          self.map.current_target[0], 
                                          self.map.current_target[1],
                                          distances)

        self.robot.take_action(action_list)
        old_pos = self.robot.position
        new_pos = self.robot._update()

        collision = self.check_collision(new_pos, self.robot.size, self.obstacle_mask)

        target_reached = self.check_target(new_pos, self.robot.size)

        if target_reached:
            self.map.update_target()

        #reward = self.reward_function(new_pos, old_pos, collision, target_reached)

        if collision:
            self.robot.speed = 0
        else:
            self.robot.position = new_pos

        if self.draw:
            self._draw()

    def check_target(self, new_pos, robot_radius):
        dist = np.linalg.norm(np.array(self.map.current_target) - np.array(new_pos))

        if dist <= 30.0:
            return True

    def check_collision(self, new_pos, robot_radius, object_mask):
        diameter = robot_radius * 2
        robot_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(robot_surf, (255, 255, 255), (robot_radius, robot_radius), robot_radius)

        robot_mask = pygame.mask.from_surface(robot_surf)

        top_left = (int(new_pos[0] - robot_radius), int(new_pos[1] - robot_radius))
        top_left_x = top_left[0]
        top_left_y = top_left[1]


        mask_width, mask_height = object_mask.get_size()
        if (top_left_x < 0 or top_left_y < 0 or
        top_left_x + diameter > mask_width or
        top_left_y + diameter > mask_height):
            return True  # robot would be partially outside the environment
        
        offset = top_left

        return object_mask.overlap(robot_mask, offset) is not None


    def _draw(self):
        self.screen.fill((255, 255, 255))

        mask_surface = self.obstacle_mask.to_surface(setcolor=(150, 150, 150), unsetcolor=(255, 255, 255))
        self.screen.blit(mask_surface, (0, 0))

        if len(self.map.targets) > 0:
            pygame.draw.circle(self.screen, (0,255,0), self.map.current_target, 15)

        x, y = map(int, self.robot.position)

        distances, directions = self.robot.gain_sensor_output(self.obstacle_mask, get_directions=True)
        self.robot.draw_sensors(self.screen, distances, directions, self.screen.get_size())
        self.robot._draw(self.screen)
        #self.robot.display_sensor_values(self.screen, distances, directions, self.screen.get_size())
        pygame.display.flip()
        
if __name__ == "__main__":
    pygame.init()
    env = Environment("map1.json", agent_start_pos = (50,50), draw = True)
    typerun = True

    agent = IntuitiveAgent(env.robot)
    clock = pygame.time.Clock()

    while typerun == True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                typerun = False
        
        env._update(agent=agent)
        
        clock.tick()

