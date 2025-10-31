import gym
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.preprocessing import is_image_space
import crafter
import torch
import numpy as np
import os
import argparse
import csv
import matplotlib.pyplot as plt
from collections import deque
import json
import numpy as np
from collections import defaultdict

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_ppo_frame_stack')
parser.add_argument('--steps', type=int, default=500_000)
parser.add_argument('--eval_episodes', type=int, default=50)
parser.add_argument('--frame_stack', type=int, default=4, help='Number of frames to stack')
parser.add_argument('--checkpoint_interval', type=int, default=100_000, help='Save checkpoint every N timesteps')
parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- FRAME STACKING WRAPPER ---
class FrameStackWrapper(gym.Wrapper):
    """
    Frame stacking wrapper that stacks multiple consecutive frames
    to provide temporal context to the agent.
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        # Update observation space to account for stacked frames
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(old_space.shape[2] * k, old_space.shape[0], old_space.shape[1]),
            dtype=old_space.dtype
        )
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_stacked_frames()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_frames(), reward, done, info
    
    def _get_stacked_frames(self):
        assert len(self.frames) == self.k
        # Stack frames along channel dimension: (k*C, H, W)
        return np.concatenate(list(self.frames), axis=2).transpose(2, 0, 1)

# --- ENV CREATION WITH FRAME STACKING ---
def make_env(record_dir=None):
    env = crafter.Env()
    if record_dir:
        env = crafter.Recorder(
            env,
            directory=record_dir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
    # Apply frame stacking
    env = FrameStackWrapper(env, k=args.frame_stack)
    return env

# --- CUSTOM CALLBACK FOR CHECKPOINTING ---
class CheckpointCallback(sb3.common.callbacks.BaseCallback):
    """
    Callback for saving the model every checkpoint_interval timesteps
    """
    def __init__(self, save_path, checkpoint_interval, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                self.save_path, 
                f"checkpoint_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose >= 1:
                print(f"Saved checkpoint to {checkpoint_path}")
                
            # save training progress info
            progress_path = os.path.join(self.save_path, "training_progress.txt")
            with open(progress_path, "w") as f:
                f.write(f"Current timesteps: {self.num_timesteps}\n")
                f.write(f"Checkpoint saved at: {checkpoint_path}\n")
                f.write(f"Last checkpoint: {self.num_timesteps}/{self.model._total_timesteps}\n")
        
        return True

# --- FIND LATEST CHECKPOINT ---
def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the directory based on timesteps
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith("_steps.zip"):
            try:
                # Extract timesteps from filename: checkpoint_100000_steps.zip
                timesteps = int(file.split("_")[1])
                checkpoints.append((timesteps, os.path.join(checkpoint_dir, file)))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
        
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    print(f"Found latest checkpoint: {latest_checkpoint[1]} with {latest_checkpoint[0]} steps")
    return latest_checkpoint[1]

# --- TRAINING WITH FRAME STACKING AND CHECKPOINTING ---
def train_ppo_frame_stack():
    env = make_env(args.outdir)
    print(f"Training PPO with Frame Stacking (k={args.frame_stack}) on device: {device}")
    
    # Update policy kwargs to handle stacked frames
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
    )

    start_timesteps = 0
    model = None
    
    if args.resume_from:
        # Load from specified checkpoint
        if os.path.exists(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            model = sb3.PPO.load(args.resume_from, env=env, device=device)
            # Extract timesteps from filename
            try:
                start_timesteps = int(args.resume_from.split("_")[1])
                print(f"Resumed from {start_timesteps} steps")
            except (ValueError, IndexError):
                print("Could not determine starting timesteps from checkpoint filename")
                start_timesteps = 0
        else:
            print(f"Warning: Specified checkpoint {args.resume_from} not found. Starting from scratch.")
    
    # If no checkpoint specified, try to find the latest one automatically
    elif args.resume_from is None:
        latest_checkpoint = find_latest_checkpoint(args.outdir)
        if latest_checkpoint:
            response = input(f"Found existing checkpoint {latest_checkpoint}. Resume from it? (y/n): ")
            if response.lower() in ['y', 'yes']:
                model = sb3.PPO.load(latest_checkpoint, env=env, device=device)
                try:
                    start_timesteps = int(latest_checkpoint.split("_")[1])
                    print(f"Resumed from {start_timesteps} steps")
                except (ValueError, IndexError):
                    start_timesteps = 0

    # Create new model if not resuming
    if model is None:
        model = sb3.PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,           # Number of steps to run for each environment per update
            batch_size=64,          # Minibatch size
            n_epochs=10,            # Number of epoch when optimizing the surrogate loss
            learning_rate=3e-4,     # Learning rate
            clip_range=0.2,         # Clipping parameter
            gamma=0.99,             # Discount factor
            gae_lambda=0.95,        # Factor for trade-off of bias vs variance for GAE
            ent_coef=0.01,          # Entropy coefficient for exploration
            vf_coef=0.5,            # Value function coefficient
            max_grad_norm=0.5,      # Max gradient norm for gradient clipping
            device=device,
            tensorboard_log=args.outdir,
        )

    print(f"Observation space with frame stacking: {env.observation_space.shape}")
    print(f"Starting from {start_timesteps} steps, training for {args.steps} total steps")
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_path=args.outdir,
        checkpoint_interval=args.checkpoint_interval,
        verbose=1
    )
    
    # Calculate remaining timesteps
    remaining_timesteps = max(0, args.steps - start_timesteps)
    
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps, 
            log_interval=1,
            callback=checkpoint_callback,
            reset_num_timesteps=False  # Don't reset timestep counter when resuming
        )
        # Save final model
        model.save(os.path.join(args.outdir, "crafter_ppo_frame_stack_final"))
    else:
        print("Training already completed! Using existing model.")
    
    print("Training complete!")
    env.close()
    return model

# --- EVALUATION WITH FRAME STACKING ---
def evaluate_agent_frame_stack(model=None, episodes=50):
    eval_dir = os.path.join(args.outdir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print(f"Loading latest checkpoint for evaluation: {latest_checkpoint}")
        model = sb3.PPO.load(latest_checkpoint, device=device)
    else:
        print("No checkpoint found. Using current model.")

    eval_env = make_env(record_dir=eval_dir)
    all_rewards, all_lengths = [], []
    
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    eval_env.close()

    # Crafter built-in evaluation
    stats = evaluate_crafter(eval_dir)
    print("\n=== Evaluation Metrics (with Frame Stacking) ===")
    print(f"Geometric Mean Score: {stats['score']:.4f}")
    print(f"Average Survival Time: {stats['length_mean']:.2f}")
    print(f"Average Cumulative Reward: {stats['reward_mean']:.2f}")
    print(f"Custom Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Custom Average Length: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print("\nAchievement Unlock Rates:")
    for key, value in sorted(stats['achievements'].items()):
        print(f"  {key}: {value*100:.2f}%")

    # --- SAVE TO CSV ---
    csv_path = os.path.join(eval_dir, "evaluation_metrics_frame_stack.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Geometric Mean Score", stats['score']])
        writer.writerow(["Average Survival Time", stats['length_mean']])
        writer.writerow(["Average Cumulative Reward", stats['reward_mean']])
        writer.writerow(["Custom Average Reward", np.mean(all_rewards)])
        writer.writerow(["Custom Average Reward Std", np.std(all_rewards)])
        writer.writerow(["Custom Average Length", np.mean(all_lengths)])
        writer.writerow(["Custom Average Length Std", np.std(all_lengths)])
        writer.writerow(["Frame Stack Size", args.frame_stack])
        writer.writerow([])
        writer.writerow(["Achievement", "Unlock Rate (%)"])
        for k, v in sorted(stats["achievements"].items()):
            writer.writerow([k, v * 100])
    print(f"\nSaved metrics to {csv_path}")

    # --- PLOTS ---
    plot_metrics_frame_stack(stats, eval_dir, all_rewards, all_lengths)

    return stats

# --- PLOTTING FOR FRAME STACKING ---
def plot_metrics_frame_stack(stats, outdir, rewards, lengths):
    # Achievement plot
    plt.figure(figsize=(10, 6))
    achievements = list(stats['achievements'].keys())
    rates = [v * 100 for v in stats['achievements'].values()]
    plt.barh(achievements, rates, color='lightcoral')
    plt.xlabel('Unlock Rate (%)')
    plt.title(f'Crafter Achievement Unlock Rates (PPO + Frame Stack k={args.frame_stack})')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'achievements_frame_stack.png'))
    plt.close()

    # Summary plot
    plt.figure(figsize=(8, 5))
    metrics = ['Geometric Mean', 'Survival Time', 'Cumulative Reward']
    values = [stats['score'], stats['length_mean'], stats['reward_mean']]
    colors = ['orange', 'green', 'blue']
    
    plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title(f'Evaluation Summary (PPO + Frame Stack k={args.frame_stack})')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'summary_frame_stack.png'))
    plt.close()
    
    # Reward distribution
    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'Reward Distribution (PPO + Frame Stack k={args.frame_stack})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'reward_distribution_frame_stack.png'))
    plt.close()
    
    print(f"Saved plots to {outdir}")
    
def evaluate_crafter(eval_dir):
    stats_path = os.path.join(eval_dir, "stats.jsonl")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"No stats.jsonl file found in {eval_dir}. Did you enable recording?")

    # Load episode stats
    episodes = []
    with open(stats_path, "r") as f:
        for line in f:
            episodes.append(json.loads(line))

    # Compute aggregate metrics
    rewards = [ep.get("reward", 0) for ep in episodes]
    lengths = [ep.get("length", 0) for ep in episodes]

    reward_mean = float(np.mean(rewards))
    length_mean = float(np.mean(lengths))

    # Aggregate achievements
    achievements = defaultdict(float)
    for ep in episodes:
        for k, v in ep.get("achievements", {}).items():
            achievements[k] += v
    for k in achievements:
        achievements[k] /= len(episodes)

    # Geometric mean score
    eps = 1e-8
    score = float(np.exp(np.mean([np.log(max(eps, v)) for v in achievements.values()]))) if achievements else 0.0

    return {
        "reward_mean": reward_mean,
        "length_mean": length_mean,
        "score": score,
        "achievements": dict(achievements),
    }

if __name__ == '__main__':
    """ print("=== PPO with Frame Stacking Improvement ===")
    print(f"Frame stack size: {args.frame_stack}")
    print(f"Output directory: {args.outdir}")
    print(f"Checkpoint interval: {args.checkpoint_interval} steps")
    
    model = train_ppo_frame_stack() """
    print("Finished training, starting evaluation")
    stats = evaluate_agent_frame_stack(episodes=args.eval_episodes)
    
    # Save configuration
    config_path = os.path.join(args.outdir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Algorithm: PPO with Frame Stacking\n")
        f.write(f"Frame stack size: {args.frame_stack}\n")
        f.write(f"Total timesteps: {args.steps}\n")
        f.write(f"Evaluation episodes: {args.eval_episodes}\n")
        f.write(f"Checkpoint interval: {args.checkpoint_interval}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Final Geometric Mean Score: {stats['score']:.4f}\n")
    
    print("Evaluation complete.")
    print(f"Configuration saved to {config_path}")