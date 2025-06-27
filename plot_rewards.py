import json
import matplotlib.pyplot as plt
import numpy as np
def plot_rewards(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    epochs = list(map(int, data.keys()))
    mean_cum_rewards = [data[str(epoch)]['mean_cum_reward'] for epoch in epochs]

    # Only plot first 20 iterations
    epochs = epochs
    mean_cum_rewards = mean_cum_rewards

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_cum_rewards, marker='o', linestyle='-', color='b')
    plt.title('Mean Cumulative Rewards per Epoch (PPO)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Cumulative Reward')
    plt.grid()
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 5))
    plt.savefig('mean_cum_rewards_PPO.png')
    plt.show()

if __name__ == "__main__":
    file_path = 'results_PPO.json' 
    plot_rewards(file_path)