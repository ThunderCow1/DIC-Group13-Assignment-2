import random
import datetime
import math
import pygame
import numpy as np
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from map import Map
from robot2 import Robot
from random_agent import RandomAgent
from human_agent import HumanAgent
from intuitive_agent import IntuitiveAgent
from helper_functions import check_collision
from npc import NPC

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
        self.npcs = []

        self.screen = pygame.display.set_mode(self.map.map_size)
        self.mask_surface = self.obstacle_mask.to_surface(self.screen, setcolor=(200,200,200))

        self.cum_reward = 0

    def _update(self, agent):
        # Create a temporary mask for this update
        # temp_mask = self.obstacle_mask.copy()
        # temp_screen = pygame.Surface(self.map.map_size, pygame.SRCALPHA)

        # for npc in self.npcs:
        #     npc._draw(temp_screen)

        # temp_mask.draw(pygame.mask.from_surface(temp_screen), offset=(0,0))

        # Get distances from sensors
        distances = self.robot.gain_sensor_output(self.obstacle_mask, get_directions=False)
        
        # Get location of self and target
        target_x, target_y = self.map.current_target
        x, y = self.robot.position

        # Compute angle diff for input vector
        direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])
        angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
        angle_diff = (angle_to_target - self.robot.orientation + 540) % 360 - 180
        state =[x, 
                y,
                target_x,
                target_y]
        state.extend(distances)
        
        action_list = agent.select_action(x, 
                                          y, 
                                          self.robot.orientation,
                                          self.robot.speed,
                                          target_x, 
                                          target_y,
                                          angle_diff,
                                          distances)
        old_pos = self.robot.position
        self.robot.take_action(action_list)
        new_pos = self.robot.position

        collision = check_collision(new_pos, self.robot.size, self.obstacle_mask)
        target_reached = self.check_target(new_pos, self.robot.size)

        self.collision = collision
        self.target_reached = target_reached
        
        if target_reached:
            self.map.update_target()

        if collision:
            self.robot.speed = 0
        else:
            self.robot.position = new_pos

        for npc in self.npcs:
            new_pos = npc._update()
            collision = check_collision(new_pos, npc.size, self.obstacle_mask)
            if collision:
                npc.orientation = None
            else:
                npc.position = new_pos

        reward = self.reward_function(collision, target_reached, old_pos, new_pos)

        self.cum_reward += reward

        agent._update(reward, old_pos, action_list)

        if self.draw:
            self._draw()

        return state, action_list, reward

    def reward_function(self, collision, target_reached, old_pos, new_pos):
        r = 0
        if not collision and not target_reached:
            r -= 0.01
        elif collision:
            r -= 20
        elif target_reached:
            r += 100

        r += 0.01 * (np.linalg.norm(np.array(self.map.current_target) - np.array(old_pos))
                      - np.linalg.norm(np.array(self.map.current_target) - np.array(new_pos)))
        return r

    def check_target(self, new_pos, robot_radius):
        dist = np.linalg.norm(np.array(self.map.current_target) - np.array(new_pos))

        if dist <= 30.0:
            return True
        else:
            return False

    def _draw(self):
        self.screen.fill((255, 255, 255))

        mask_surface = self.obstacle_mask.to_surface(setcolor=(150, 150, 150), unsetcolor=(255, 255, 255))
        self.screen.blit(mask_surface, (0, 0))

        if len(self.map.targets) > 0:
            pygame.draw.circle(self.screen, (0,255,0), self.map.current_target, 15)

        distances, directions = self.robot.gain_sensor_output(self.obstacle_mask, get_directions=True)
        self.robot.draw_sensors(self.screen, distances, directions, self.screen.get_size())
        self.robot._draw(self.screen)

        for npc in self.npcs:
            npc._draw(self.screen)
        #self.robot.display_sensor_values(self.screen, distances, directions, self.screen.get_size())

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
        cum_reward = font.render(f"{round(self.cum_reward,2)}", True, (0,0,0))
        self.screen.blit(cum_reward, [self.map.map_size[0]-50, 10])

        pygame.display.flip()

    def reset(self, agent_start_pos: tuple[float, float] = None):
        if agent_start_pos == None:
            print('No start position, initialising random position')
            self.agent_pos = self.map.random_pos()
        else:
            self.agent_pos = agent_start_pos

        self.robot.reset(self.agent_pos)
        self.cum_reward = 0
        
        if self.draw:
            self._draw()
        

if __name__ == "__main__":
    pygame.init()
    env = Environment("map1.json", agent_start_pos = (50,50), draw = True)
    # env.npcs.append(NPC((700,500),3,15,0.01))
    fpss = []
    typerun = True

    agent = IntuitiveAgent(env.robot)
    clock = pygame.time.Clock()
    i = 0
    while typerun == True:
        i += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                print(np.mean(fpss))
                typerun = False
        
        env._update(agent=agent)
        if i % 1000 == 0:
            env.reset(agent_start_pos=(50,50))
        clock.tick()
        fpss.append(clock.get_fps())

