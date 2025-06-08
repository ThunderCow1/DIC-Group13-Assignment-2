import torch
from PPO_agent import PPOAgent
from environment import Environment 
import pygame

def main():
    # Hyperparameters
    num_episodes = 1000
    max_steps_per_episode = 2048
    start_pos = (50, 50)

    # Environment and agent
    pygame.init()
    env = Environment("map1.json", agent_start_pos=start_pos, draw=True)
    agent = PPOAgent(env.robot)

    for episode in range(num_episodes):
        env.reset(agent_start_pos=start_pos)
        total_reward = 0
        num_steps = 0

        done = False
        while not done and num_steps < max_steps_per_episode:
            env._update(agent)  
            total_reward = env.cum_reward
            num_steps += 1

            # End episode if target is reached or a condition is met
            if env.map.current_target is None:
                break

        print(f"[Episode {episode}] Total reward: {total_reward:.2f}")

        # Train PPO agent
        agent.train()

    # Save model
    # torch.save(agent.actor.state_dict(), "ppo_actor.pth")
    # torch.save(agent.critic.state_dict(), "ppo_critic.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()
