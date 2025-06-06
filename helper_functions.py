import pygame
import numpy as np

def check_collision(new_pos, robot_radius, object_mask):
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

def action_to_index(action):
    action_map = {
        "turn_right": 0,
        "turn_left": 1,
        "accelerate": 2,
        "break": 3,
        "break_hard": 4
    }
    return action_map.get(action, -1)

def index_to_action(index):
    action_map = {
        0: "turn_right",
        1: "turn_left",
        2: "accelerate",
        3: "break",
        4: "break_hard"
    }
    return action_map.get(index, None)