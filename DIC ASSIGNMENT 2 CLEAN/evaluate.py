import numpy as np
from argparse import ArgumentParser
import json
import pygame
import time
import random
import torch

# Agents
#from agents.DQN_torch import DQN as DQNAgent
from agents.DQN_torch2 import DQN as DQNAgent

from agents.PPO_agent2 import PPOAgent
from agents.human_agent import HumanAgent
from agents.intuitive_agent import IntuitiveAgent


# Environment
from environment.environment import Environment

def parse_args():
    p = ArgumentParser(description="DIC Continuous Environment")
    p.add_argument("MAP", type=str, default='maps/map1.json')
    p.add_argument("--agent", type=str, default="human", choices=["human", "intuitive", "DQN", "PPO"],
                   help="Agent type to train: human, intuitive, DQN, PPO")
    p.add_argument("--train", type=int, default=2500,
                   help="Number of iterations to train.")
    p.add_argument("--evaluation", type=int, default=10,
                   help="Number of iterations to evaluate.")
    p.add_argument("--Steps", type=int, default=1000,
                   help="Steps agent takes in environment")
    p.add_argument("--Draw", type=bool, default=True,
                   help="Draw environment?")
    p.add_argument("--val_times", type=int, default=50,
                   help="val_times?")
    return p.parse_args()

def train_agent_DQN(agent, epochs_per_train):
    env.draw = False
    
    for n in range(epochs_per_train):
        agent.losses = []
        env.reset()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        font = pygame.font.SysFont(None, 48)
        epoch_counter = font.render(f"Step in Epoch: {n}/{epochs_per_train}", True, (255,255,255))
        
        env.robot.orientation = random.randint(0,360)
        env.robot.speed = 0

        done = False
        steps = 0
        printpred = "No prediction made yet"
        while not done and steps <= max_steps:
            steps += 1
            env.screen.fill((0,0,0))
            # Write epoch counter
            env.screen.blit(epoch_counter, [10, 10])
            stop_game_break()
            done = env._update(agent)

            if steps == 1 and agent.last_prediction != None:
                printpred = agent.last_prediction
            
            # Write cumulative reward and epsilon
            cum_font = font.render(f"Cumulative Reward: {env.cum_reward}", True, (255,255,255))
            env.screen.blit(cum_font, [10, 50])
            eps_font = font.render(f"Epsilon: {agent.epsilon}", True, (255,255,255))
            env.screen.blit(eps_font, [10, 90])
            eps_font = font.render(f"First Predicted Q-values: {printpred}", True, (255,255,255))
            env.screen.blit(eps_font, [10, 130])
            pygame.display.flip()

        print(np.mean(agent.losses))

def train_agent_PPO(agent, epochs_per_train):
    env.draw = False
    ml = 'not calculated yet'
    for n in range(epochs_per_train):
        env.reset()
        # agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        font = pygame.font.SysFont(None, 48)
        epoch_counter = font.render(f"Step in Epoch: {n}/{epochs_per_train}", True, (255,255,255))
        
        env.robot.orientation = random.randint(0,360)
        env.robot.speed = 0

        done = False
        steps = 0
        printpred = "No prediction made yet"
        
        while not done and steps <= max_steps:
            steps += 1
            env.screen.fill((0,0,0))
            # Write epoch counter
            env.screen.blit(epoch_counter, [10, 10])

            stop_game_break()

            done = env._update(agent)
            
            # Write cumulative reward and epsilon
            cum_font = font.render(f"Cumulative Reward: {env.cum_reward}", True, (255,255,255))
            env.screen.blit(cum_font, [10, 50])
            eps_font = font.render(f"Mean_loss: {ml}", True, (255,255,255))
            env.screen.blit(eps_font, [10, 90])

            
            # eps_font = font.render(f"First Predicted Q-values: {printpred}", True, (255,255,255))
            # env.screen.blit(eps_font, [10, 130])
            pygame.display.flip()
        agent.train_()



def evaluate_agent(agent):
    agent.train = False
    env.draw = True
    cum_rewards = []
    for n in range(validation_epochs):
        env.reset()
        env.robot.orientation = random.randint(0,360)
        env.robot.speed = random.uniform(0, env.robot.top_speed)

        done = False
        steps = 0
        while not done and steps <= max_steps:
            stop_game_break()
            steps += 1
            done = env._update(agent)

        cum_rewards.append(env.cum_reward)

    agent.train = True

    return np.mean(cum_rewards)

def stop_game_break():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            with open(f"results_{agent_type}.json", 'w') as fp:
                json.dump(epoch_results, fp)
            raise ValueError("Pygame Quit")
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pygame.init()
    pygame.font.init()
    args = parse_args()
    agent_type = args.agent
    val_times = args.val_times
    validation_epochs = args.evaluation
    max_steps = args.Steps

    epochs_per_train = int(args.train/val_times)

    m = args.MAP
    env = Environment(m, random_seed=42, draw=args.Draw)
    num_sensors = 7

    # Create agent and set agent-specific values
    match agent_type:
        case 'human':
            agent = HumanAgent(env.robot)
            train = False
        case 'DQN':
            agent = DQNAgent(1.0, input_size = 5 + num_sensors)
            agent.load('networks_great_success (somewhat)\epoch19_main_network_dqn.pkl', 'networks_great_success (somewhat)\epoch19_target_network_dqn.pkl')
            train = train_agent_DQN
        case 'PPO':
            agent = PPOAgent(env.robot,
                             lr_actor = 3e-4,
                             lr_critic = 1e-3,
                             gamma = 0.99,
                             lam = 0.95,
                             clip_eps = 0.1,
                             entropy_coef = 0.1,
                             )
            train = train_agent_PPO
        case 'intuitive':
            agent = IntuitiveAgent(env.robot)
            train = False

    # Training and validation per epoch
    if train != False:
        epoch_results = {}
        for n in range(20, 20+val_times):
            print(f"Starting loop {n}/{val_times}")
            train(agent, epochs_per_train)
            print(f"Saving results")
            agent.save(f"networks/epoch{n}")

            results = evaluate_agent(agent)
            epoch_results[n] = {'mean_cum_reward' : results}

        with open(f"results_{agent_type}.json", 'w') as fp:
            json.dump(epoch_results, fp)

    # Loop to see final agent:
    active = True
    
    env.draw = True
    i = 0
    while active:
        env._update(agent=agent)
        i += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                active = False
        
        if i % 1000 == 0:
            env.reset()

        time.sleep(0.02)


    

    

    
    
