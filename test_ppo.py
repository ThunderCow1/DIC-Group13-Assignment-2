import pygame
import numpy as np
import math
from environment import Environment
from PPO_agent import PPOAgent
from helper_functions import check_collision
import time


def test_trained_agent():
    pygame.init()

    # Create environment and agent
    env = Environment("map1.json", agent_start_pos=(50, 50), draw=True)
    agent = PPOAgent(env.robot)

    dummy_x, dummy_y = 0, 0
    dummy_orientation = 0
    dummy_speed = 0
    dummy_target_x, dummy_target_y = 100, 100
    dummy_angle_diff = 0
    dummy_distances = [100] * 5

    # This initializes state_dim and builds networks
    agent.select_action(dummy_x, dummy_y, dummy_orientation, dummy_speed,
                        dummy_target_x, dummy_target_y, dummy_angle_diff, dummy_distances)

    # Load the trained models
    actor_path = "actor_1_collision.pth"
    critic_path = "critic_1_collision.pth"
    agent.load_models(actor_path, critic_path)
    print("Loaded trained models.")

    num_test_episodes = 50
    max_steps = 500

    for episode in range(num_test_episodes):
        # Reset environment
        env.robot.position = env.map.random_pos()
        env.robot.orientation = np.random.uniform(0, 360)
        env.robot.speed = 0
        env.cum_reward = 0
        total_reward = 0
        done = False

        print(f"\nStarting Test Episode {episode + 1}")

        for step in range(max_steps):
            env._draw() 

            for event in pygame.event.get():  
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            x, y = env.robot.position
            target_x, target_y = env.map.current_target

            direction_vec = pygame.Vector2(target_x, target_y) - pygame.Vector2(x, y)
            angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
            angle_diff = (angle_to_target - env.robot.orientation + 540) % 360 - 180

            distances = env.robot.gain_sensor_output(env.obstacle_mask, get_directions=False)

            action_list = agent.select_action(
                x, y, env.robot.orientation, env.robot.speed,
                target_x, target_y, angle_diff, distances
            )

            env.robot.take_action(action_list)
            new_pos = env.robot._update()

            collision = check_collision(new_pos, env.robot.size, env.obstacle_mask)
            target_reached = env.check_target(new_pos, env.robot.size)

            if collision:
                print(f"[Step {step}] Collision detected!")
                break

            if target_reached:
                print(f"[Step {step}] Target reached!")
                env.map.update_target()

            env.robot.position = new_pos
            total_reward += 1

            time.sleep(0.01)  # Slow down the loop for better visualization

        print(f"Episode {episode + 1} finished in {step + 1} steps.")

    pygame.quit()
    print("Test completed and pygame closed.")


if __name__ == "__main__":
    test_trained_agent()
