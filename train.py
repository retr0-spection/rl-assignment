import gym as old_gym
import stable_baselines3
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import random
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.stats import gmean
from dqn import CNN_DQN
import os
import getpass

username = getpass.getuser()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure the directory exists
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
args = parser.parse_args()

register(id='CrafterNoReward-v1',entry_point=crafter.Env)

env = old_gym.make('CrafterNoReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './logs',
  save_stats=True,
  save_video=False,
  save_episode=False,
)
env = GymV21CompatibilityV0(env=env)
num_actions = env.action_space.n


def train_base_dqn():
    os.makedirs(f"/datasets/{username}/rl/checkpoints/base_dqn", exist_ok=True)
    os.makedirs(f"/datasets/{username}/rl/checkpoints/base_dqn/plots", exist_ok=True)

    model = CNN_DQN(3, num_actions).to(device)
    target_model = CNN_DQN(3, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    memory = deque(maxlen=100000)

    gamma = 0.99
    batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    update_target_every = 1000
    steps_done = 0
    start_episode = 0

    cumulative_rewards = []
    survival_times = []
    achievement_unlocks = []

    # --- Resume from checkpoint ---
    checkpoint_files = sorted([f for f in os.listdir(f"/datasets/{username}/rl/checkpoints/base_dqn") if f.endswith(".pth")])
    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        print(f"Resuming from checkpoint: {last_checkpoint}")
        checkpoint = torch.load(f"/datasets/{username}/rl/checkpoints/base_dqn/{last_checkpoint}", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        target_model.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        steps_done = checkpoint['steps_done']
        memory = checkpoint.get('memory', memory)
        cumulative_rewards = checkpoint.get('cumulative_rewards', cumulative_rewards)
        survival_times = checkpoint.get('survival_times', survival_times)
        achievement_unlocks = checkpoint.get('achievement_unlocks', achievement_unlocks)
        start_episode = int(last_checkpoint.split("ep")[-1].split(".pth")[0]) + 1

    def select_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            # Keep your original forward pass
            q_values = model(state)
            return torch.argmax(q_values).item()

    def replay():
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)

        # Keep your original forward pass exactly
        q_values = model(states).gather(1, actions.unsqueeze(1))
        next_q_values = target_model(next_states).max(1)[0].detach()
        targets = rewards + gamma * (1 - dones) * next_q_values

        loss = F.smooth_l1_loss(q_values.squeeze(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    episodes = 1000
    for ep in range(start_episode, episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        timestep = 0

        episode_achievements = []
        while not done:
            action = select_action(state)
            next_state, reward, terminated, trunc, info = env.step(action)
            done = terminated or trunc

            if 'achievement' in info:
                episode_achievements.append(info['achievement'])

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            timestep += 1

            replay()
            steps_done += 1
            if steps_done % update_target_every == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        cumulative_rewards.append(total_reward)
        survival_times.append(timestep)
        achievement_unlocks.append(np.array(episode_achievements).sum(axis=0) if episode_achievements else np.array([]))

        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Survival: {timestep}")

        if (ep + 1) % 100 == 0:
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'steps_done': steps_done,
                'memory': memory,
                'cumulative_rewards': cumulative_rewards,
                'survival_times': survival_times,
                'achievement_unlocks': achievement_unlocks
            }, f"/datasets/{username}/rl/checkpoints/base_dqn/crafter_dqn_ep{ep}.pth")

            # Plot metrics
            plt.figure()
            plt.plot(cumulative_rewards, label='Cumulative Reward')
            plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
            plt.savefig(f"/datasets/{username}/rl/checkpoints/base_dqn/plots/cumulative_reward_ep{ep}.png"); plt.close()

            plt.figure()
            plt.plot(survival_times, label='Survival Time')
            plt.xlabel('Episode'); plt.ylabel('Timesteps'); plt.legend()
            plt.savefig(f"/datasets/{username}/rl/checkpoints/base_dqn/plots/survival_time_ep{ep}.png"); plt.close()

            if achievement_unlocks and len(achievement_unlocks[0]) > 0:
                achievement_array = np.array(achievement_unlocks)
                unlock_rates = achievement_array.mean(axis=0)
                geo_mean = gmean(np.maximum(unlock_rates, 1e-6))
                plt.figure()
                plt.bar(range(len(unlock_rates)), unlock_rates)
                plt.xlabel('Achievement'); plt.ylabel('Unlock Rate')
                plt.title(f'Achievement Unlock Rates (GeoMean={geo_mean:.3f})')
                plt.savefig(f"/datasets/{username}/rl/checkpoints/base_dqn/plots/achievement_rates_ep{ep}.png")
                plt.close()


if __name__ == '__main__':
    print("Starting base DQN training procedure")
    train_base_dqn()
    print("Training finished")
