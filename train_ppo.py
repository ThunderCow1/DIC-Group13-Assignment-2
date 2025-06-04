import pygame
import numpy as np
from environment import Environment
from PPO_agent import PPOAgent
from helper_functions import check_collision

def train():
    print("starting training...")
    pygame.init()
    print("pygame initialized")
    # Create environment with drawing disabled for faster training
    env = Environment("map1.json", agent_start_pos=None, draw=False)  # Use random start positions
    agent = PPOAgent(env.robot)
    
    num_episodes = 1000
    steps_per_episode = 1000
    save_interval = 50
    
    print("Starting PPO training...")
    total_targets = 0
    
    for episode in range(num_episodes):
        # Reset environment
        env.robot.position = env.map.random_pos()
        env.robot.orientation = 0
        env.robot.speed = 0
        
        episode_reward = 0
        targets_hit = 0
        print("resetting environment")
        
        # Run episode
        for step in range(steps_per_episode):
            print("runnging step", step)
            # Get old state data
            old_pos = env.robot.position
            old_target = env.map.current_target
            
            # Update environment (this calls select_action)
            env._update(agent)
            
            # Calculate reward
            reward = 0
            
            # Check if target reached
            target_reached = old_target != env.map.current_target
            
            # Check for collision
            collision = check_collision(env.robot.position, env.robot.size, env.obstacle_mask)
            
            if target_reached:
                reward += 100.0
                targets_hit += 1
                total_targets += 1
            
            if collision:
                reward -= 10.0
                done = True
            else:
                done = False
                
                # Distance-based reward shaping
                prev_dist = np.linalg.norm(np.array(old_target) - np.array(old_pos))
                curr_dist = np.linalg.norm(np.array(env.map.current_target) - np.array(env.robot.position))
                reward += (prev_dist - curr_dist) * 0.5
                
                # Small penalty for time
                reward -= 0.1
            
            # Store reward
            agent.store_reward(reward, done)
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.1f} | Targets: {targets_hit}")
        
        # Train agent
        agent.train()
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model(f"ppo_model_ep{episode+1}.npy")
            print(f"Model saved at episode {episode+1}")
    
    # Save final model
    agent.save_model("ppo_model_final.npy")
    print(f"Training complete! Total targets reached: {total_targets}")
    pygame.quit()

if __name__ == "__main__":
    train()