import random
import datetime
import math
import pygame
import numpy as np
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from environment.map import Map
from environment.robot import Robot
from environment.helper_functions import check_collision
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

    def get_state(self):
        # Get distances from sensors
        distances = self.robot.gain_sensor_output(self.obstacle_mask, get_directions=False)
        
        # Get location of self and target
        target_x, target_y = self.map.current_target
        x, y = self.robot.position

        # Compute angle diff for input vector
        direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])
        angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
        angle_diff = (angle_to_target - self.robot.orientation + 540) % 360 - 180

        state =[x - target_x, 
                y - target_y, 
                self.robot.orientation,
                self.robot.speed,
                angle_diff]
        
        state.extend(distances)

        return state
    
    def _update(self, agent):
        state = self.get_state()

        action = agent.select_action(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5:]
        )

        # Updates robot position
        self.robot.take_action(action)
        
        old_pos = self.robot.position
        new_pos = self.robot._update()

        # Check different collisions
        collision = check_collision(new_pos, self.robot.size, self.obstacle_mask)
        target_reached = self.check_target(new_pos, self.robot.size)

        self.collision = collision
        self.target_reached = target_reached
        
        # Update map for reaching target
        if target_reached:
            self.map.update_target()

        # Update robot based on collision
        if collision:
            self.robot.speed = 0
            new_pos = old_pos
        else:
            self.robot.position = new_pos

        # Calculate and update rewards
        reward = self.reward_function(collision, target_reached, old_pos, new_pos, distances=state[5:])
        self.cum_reward += reward

        # All states have now been updated, compute updated state
        new_state = self.get_state()

        agent._update(state, action, reward, new_state, target_reached)

        if self.draw:
            self._draw()

        return target_reached

    def reward_function(self, collision, target_reached, old_pos, new_pos, distances = None, treshold = 128, offset = 3):
        offset = min(offset, math.floor(len(self.robot.sensors)/2))
        r = -0.02
        if not collision and not target_reached:
            pass
        elif target_reached:
            r += 100

        moved_closer = (np.linalg.norm(np.array(self.map.current_target) - np.array(old_pos))
                      - np.linalg.norm(np.array(self.map.current_target) - np.array(new_pos)))


        if old_pos == new_pos:
            r += -0.1

        r += 0.2 * moved_closer
        
        return r

    def check_target(self, new_pos, robot_radius):
        dist = np.linalg.norm(np.array(self.map.current_target) - np.array(new_pos))

        if dist <= 15 + robot_radius:
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

