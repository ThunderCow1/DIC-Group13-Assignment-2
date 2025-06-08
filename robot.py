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

class Robot:

    def __init__(self,
                 position: tuple[float, float] = None,
                 orientation: float = 0,
                 speed: float = 0,
                 top_speed = 10,
                 acceleration=0.1,
                 turn_speed = 3,
                 size = 15):
        
        self.position = position
        self.start_position = np.copy(position)
        self.orientation = orientation
        self.speed = speed
        self.acceleration = acceleration
        self.top_speed = top_speed
        self.turn_speed = turn_speed
        self.size = size

        self.actions = ['turn_right',
                        'turn_left',
                        'accelerate',
                        'break',
                        'break_hard']
        
    def take_action(self, actions):

        for action in actions:
            match action:
                case 'turn_right':
                    self.orientation += self.turn_speed
                case 'turn_left':
                    self.orientation -= self.turn_speed
                case 'accelerate':
                    self.speed += self.acceleration*(self.top_speed - self.speed)
                case 'break':
                    self.speed -= 0.2
                    self.speed = max(self.speed, 0)
                case 'break_hard':
                    self.speed -= 1
                    self.speed = max(self.speed,0)

    def _update(self):
        angle_rad = math.radians(self.orientation)
        dx = math.cos(angle_rad) * self.speed
        dy = math.sin(angle_rad) * self.speed

        x, y = self.position
        return (x+dx, y+dy)

    def _draw(self, surface):
        x, y = self.position
        x = int(round(x))
        y = int(round(y))

        angle_rad = math.radians(self.orientation)
        dx = math.cos(angle_rad) * self.size/2
        dy = math.sin(angle_rad) * self.size/2
        
        pygame.draw.circle(surface, (0,0,255), (x,y),self.size)
        pygame.draw.circle(surface, (255,255,255), (x +dx , y + dy), self.size/3)
        pygame.draw.circle(surface, (0,0,0), (x +1.1*dx , y + 1.1*dy), self.size/6)

    def gain_sensor_output(self, mask, sensors = [-90, -45, 0, 45, 90], max_distance = 256, get_directions = False):
        
        outputs = []
        directions = []

        for sensor_angle in sensors:
            angle_deg = self.orientation + sensor_angle
            angle_rad = math.radians(angle_deg)

            direction = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad))

            directions.append(direction)

            for dist in range(1, max_distance + 1):
                point = self.position + direction * dist
                x, y = int(point.x), int(point.y)

                if not (0 <= x < mask.get_size()[0] and 0 <= y < mask.get_size()[1]):
                    outputs.append(dist)
                    break

                if mask.get_at((x, y)):
                    outputs.append(dist)
                    break
            else:
                outputs.append(max_distance)

        if not get_directions:
            return outputs
        else:
            return outputs, directions
        
    def draw_sensors(self, screen, distances, directions, screen_size, max_distance = 256):

        screen_width, screen_height = screen_size

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)

        def clip_to_bounds(point):
            x = max(0, min(screen_width - 1, int(point.x)))
            y = max(0, min(screen_height - 1, int(point.y)))
            return pygame.Vector2(x, y)

        for dist, direction_vec in zip(distances, directions):

            # Calculate hit and end points
            hit_point = self.position + direction_vec * dist
            end_point = self.position + direction_vec * max_distance

            # Clip both points
            hit_point = clip_to_bounds(hit_point)
            end_point = clip_to_bounds(end_point)

            label = font.render(f"{dist}", True, (0, 0, 0))
            screen.blit(label, hit_point)

            # Draw segments
            pygame.draw.line(screen, (0, 255, 0), self.position, hit_point, 2)
            pygame.draw.line(screen, (255, 0, 0), hit_point, end_point, 1)

    def reset(self, position=None, orientation=None, speed=None):
        if position is not None:
            self.position = position
        else:
            self.position = self.start_position
        if orientation is not None:
            self.orientation = orientation
        else:
            self.orientation = 0
        if speed is not None:
            self.speed = speed
        else:
            self.speed = 0



