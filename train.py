import gym
import stable_baselines3 as sb3
import crafter
import torch
import os
import argparse
import csv
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_dqn')
parser.add_argument('--steps', type=int, default=500_000)
parser.add_argument('--eval_episodes', type=int, default=50)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- ENV CREATION ---
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
    return env


# --- TRAINING ---
def train_base_dqn():
    env = make_env(args.outdir)
    print(f"Training on device: {device}")

    model = sb3.DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=50_000,
        learning_starts=1_000,
        device=device,
        tensorboard_log=args.outdir,
    )

    model.learn(total_timesteps=args.steps, log_interval=4)
    model.save(os.path.join(args.outdir, "crafter_cnn_dqn_base"))
    print("Training complete!")
    env.close()
    return model


# --- EVALUATION ---
def evaluate_agent(model, episodes=50):
    eval_dir = os.path.join(args.outdir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_env = make_env(record_dir=eval_dir)

    print(f"Evaluating for {episodes} episodes...")
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
    eval_env.close()

    # Crafter built-in evaluation
    stats = crafter.evaluate(eval_dir)
    print("\n=== Evaluation Metrics ===")
    print(f"Geometric Mean Score: {stats['score']:.4f}")
    print(f"Average Survival Time: {stats['length_mean']:.2f}")
    print(f"Average Cumulative Reward: {stats['reward_mean']:.2f}")
    print("\nAchievement Unlock Rates:")
    for key, value in sorted(stats['achievements'].items()):
        print(f"  {key}: {value*100:.2f}%")

    # --- SAVE TO CSV ---
    csv_path = os.path.join(eval_dir, "evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Geometric Mean Score", stats['score']])
        writer.writerow(["Average Survival Time", stats['length_mean']])
        writer.writerow(["Average Cumulative Reward", stats['reward_mean']])
        writer.writerow([])
        writer.writerow(["Achievement", "Unlock Rate (%)"])
        for k, v in sorted(stats["achievements"].items()):
            writer.writerow([k, v * 100])
    print(f"\nSaved metrics to {csv_path}")

    # --- PLOTS ---
    plot_metrics(stats, eval_dir)

    return stats


# --- PLOTTING ---
def plot_metrics(stats, outdir):
    plt.figure(figsize=(8, 4))
    achievements = list(stats['achievements'].keys())
    rates = [v * 100 for v in stats['achievements'].values()]
    plt.barh(achievements, rates, color='skyblue')
    plt.xlabel('Unlock Rate (%)')
    plt.title('Crafter Achievement Unlock Rates')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'achievements.png'))
    plt.close()

    # Summary plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Geometric Mean', 'Survival Time', 'Cumulative Reward'],
            [stats['score'], stats['length_mean'], stats['reward_mean']],
            color=['orange', 'green', 'blue'])
    plt.title('Evaluation Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'summary.png'))
    plt.close()
    print(f"Saved plots to {outdir}")


if __name__ == '__main__':
    print("Starting DQN training procedure")
    model = train_base_dqn()
    print("Finished training, starting evaluation")
    evaluate_agent(model, episodes=args.eval_episodes)
    print("Evaluation complete.")
