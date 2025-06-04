import random
import pygame
import numpy as np


class NPC:

    def __init__(self,
                 position: tuple[float, float] = None,
                 speed: float = 3,
                 size = 15,
                 erratic = 0.01):
        
        self.position = position
        self.orientation_map = [(1,0),
                                (0,1),
                                (-1,0),
                                (0,-1)]
        
        self.orientation = random.choice(self.orientation_map)
        self.speed = speed
        self.size = size
        self.erratic = erratic

    def _update(self):
        x, y = self.position
        if self.orientation:
            dx, dy = self.orientation
            
        else:
            dx, dy = 0, 0
            if random.uniform(0,1) <= self.erratic:
                self.orientation = random.choice(self.orientation_map)

        return (x+dx, y+dy)

    def _draw(self, surface):
        x, y = self.position
        x = int(round(x))
        y = int(round(y))
        
        pygame.draw.circle(surface, (100,100,200), (x,y), self.size)