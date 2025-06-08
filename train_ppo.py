import pygame
import numpy as np
import math
from environment import Environment
from PPO_agent import PPOAgent
from helper_functions import check_collision
import os

def train():

    pygame.init()

    config = {
        'num_episodes': 1000,
        'max_steps_per_episode': 200,
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_eps': 0.2,
        'entropy_coef': 0.02,
        'train_epochs': 4
    }
    
    # Create environment and agent
    env = Environment("map2.json", agent_start_pos=(50, 50), draw=False)
    agent = PPOAgent(env.robot, 
                     lr_actor=config['lr_actor'], 
                     lr_critic=config['lr_critic'],
                     gamma=config['gamma'],
                     lam=config['lam'],
                     clip_eps=config['clip_eps'],
                     entropy_coef=config['entropy_coef'])
    
    episode_rewards = []
    episode_lengths = []
    targets_reached = []
    total_targets = 0
    
    for key, value in config.items():
        print(f"  {key}: {value}")

    for episode in range(config['num_episodes']):
        env.robot.position = env.map.random_pos()
        env.robot.orientation = np.random.uniform(0,360)
        env.robot.speed = 0
        env.cum_reward = 0
        
        episode_reward = 0
        episode_length = 0
        targets_hit = 0
        collision_count = 0

        if episode % 10 == 0:
            print(f"Starting episode {episode+1}/{config['num_episodes']}")
        
        for step in range(config['max_steps_per_episode']):
            old_pos = env.robot.position
            old_target = env.map.current_target

            distances = env.robot.gain_sensor_output(env.obstacle_mask, get_directions=False)
            
            target_x, target_y = env.map.current_target
            x, y = env.robot.position
            direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x, y])
            angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
            angle_diff = (angle_to_target - env.robot.orientation + 540) % 360 - 180

            action_list = agent.select_action(x, y, env.robot.orientation, env.robot.speed,
                                            target_x, target_y, angle_diff, distances)
            
            env.robot.take_action(action_list)
            new_pos = env.robot._update()

            collision = check_collision(new_pos, env.robot.size, env.obstacle_mask)
            target_reached = env.check_target(new_pos, env.robot.size)
            
            reward = 0
            done = False
            
            if collision:
                reward = -50.0
                env.robot.speed = 0
                collision_count += 1
                done = True
            elif target_reached:
                reward = 500.0
                targets_hit += 1
                total_targets += 1
                env.map.update_target()
            else:
                old_dist = np.linalg.norm(np.array(old_target) - np.array(old_pos))
                new_dist = np.linalg.norm(np.array(env.map.current_target) - np.array(new_pos))
                
                distance_improvement = old_dist - new_dist
                if distance_improvement > 2.0:
                    reward += distance_improvement * 1.0
                
                movement_dist = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                if movement_dist < 1.0:
                    reward -= 5.0
                
                reward -= 0.1
                
                env.robot.position = new_pos
            
            agent.store_reward(reward, done)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        targets_reached.append(targets_hit)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_targets = np.mean(targets_reached[-10:])
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Avg Targets: {avg_targets:.1f} | "
                  f"Length: {episode_length:3d} | "
                  f"Collisions: {collision_count}")
        
        agent.train(epochs=config['train_epochs'])
        
    
    print(f"Total targets reached: {total_targets}")
    
    pygame.quit()

if __name__ == "__main__":
    train()