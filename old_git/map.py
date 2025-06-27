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
import json
from helper_functions import check_collision

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)


class Target:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)


class Map:
    def __init__(self,
                 map_fp=None,
                 map_size=(1024, 768)):
        
        self.map_name = "unnamed"
        self.map_size = map_size
        self.obstacles = []
        self.targets = []

        if map_fp:
            self.load_from_file(map_fp)
            self.current_target = self.targets[random.randint(0,len(self.targets)-1)]

        if len(self.targets) == 0:
            raise ValueError("No Targets Defined")
        
    def update_target(self):
        idx = self.targets.index(self.current_target)
        options = list(range(len(self.targets)))
        options.pop(idx)

        self.current_target = self.targets[np.random.choice(options)]

    def load_from_file(self, map_fp):

        map_fp = Path(map_fp)

        if not map_fp.exists():
            raise FileNotFoundError(f"Map file {map_fp} not found")

        with open(map_fp, 'r') as f:
            data = json.load(f)

        self.map_name = data.get("map_name", "unnamed")
        self.map_size = tuple(data.get("map_size", self.map_size))

        for obstacle in data.get("obstacles", []):
            self.create_obstacle(obstacle)

        for target in data.get("targets", []):
            self.targets.append(target)

    def create_obstacle(self, obstacle: dict):
        x = obstacle["x"]
        y = obstacle["y"]
        width = obstacle["width"]
        height = obstacle["height"]
        self.obstacles.append(Obstacle(x, y, width, height))

    def create_random_obstacle(self, min_size=30, max_size=100):
        max_width, max_height = self.map_size

        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)

        x = random.randint(0, max_width - width)
        y = random.randint(0, max_height - height)

        self.obstacles.append(Obstacle(x, y, width, height))

    def create_obstacle_mask(self):
        """
        Creates a mask with all obstacles (white = obstacle).
        """
        surface = pygame.Surface(self.map_size, pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))  # Fully transparent

        for obstacle in self.obstacles:
            pygame.draw.rect(surface, (255, 255, 255), obstacle.rect)  # White = solid

        return pygame.mask.from_surface(surface)
    
    def random_pos(self):
        
        mask = self.create_obstacle_mask()
        i = 0
        margin=15 # Robot size buffer
        while i < 1000:
            i += 1
            x, y = random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1])
            if not check_collision([x, y], 15, mask):
                return (x,y)
            
        print('Could not find suitable start position')


    

if __name__ == "__main__":
    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Random Map Test")

    # Create a map and add random obstacles
    test_map = Map(map_size=screen_size)
    for _ in range(3):
        test_map.create_random_obstacle()

    # Draw the map
    screen.fill((30, 30, 30))  # Background color

    for obstacle in test_map.obstacles:
        pygame.draw.rect(screen, (200, 0, 0), obstacle.rect)

    pygame.display.flip()

    # Keep the window open for 5 seconds
    pygame.time.wait(5000)
    pygame.quit()