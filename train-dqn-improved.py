import os
import argparse
import csv
import torch
import matplotlib.pyplot as plt
from cnn import CNNAttentionExtractor
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
import crafter

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_dqn_improved')
parser.add_argument('--steps', type=int, default=500_000)
parser.add_argument('--eval_episodes', type=int, default=50)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------------------
# ENVIRONMENT HELPERS
# ---------------------

def make_train_env():
    env = crafter.Env()
    env = Monitor(env)
    return env

def make_eval_env(record_dir=None):
    env = crafter.Env()
    if record_dir:
        env = crafter.Recorder(
            env,
            directory=record_dir,
            save_stats=True,
            save_video=False,
            save_episode=False
        )
    env = Monitor(env)
    return env

# ---------------------
# TRAINING
# ---------------------

def train_super_qrdqn():
    train_env = make_train_env()
    print(f"Training Super QRDQN on device: {device}")

    policy_kwargs = dict(
        features_extractor_class=CNNAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )


    model = QRDQN(
        "CnnPolicy",
        train_env,
        buffer_size=200_000,
        learning_starts=50_000,
        batch_size=32,
        learning_rate=1e-4,
        target_update_interval=500,
        train_freq=4,
        exploration_fraction=0.6,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.outdir,
    )

    model.learn(total_timesteps=args.steps, log_interval=4)
    model.save(os.path.join(args.outdir, "crafter_cnn_dqn_base"))
    print("Training complete!")
    train_env.close()
    return model

# ---------------------
# EVALUATION
# ---------------------

def evaluate_agent(model, episodes=50):
    eval_dir = os.path.join(args.outdir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_env = make_eval_env(record_dir=eval_dir)
    print(f"Evaluating for {episodes} episodes...")

    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
    eval_env.close()

    stats = crafter.evaluate(eval_dir)
    print("\n=== Evaluation Metrics ===")
    print(f"Geometric Mean Score: {stats['score']:.4f}")
    print(f"Average Survival Time: {stats['length_mean']:.2f}")
    print(f"Average Cumulative Reward: {stats['reward_mean']:.2f}")
    print("\nAchievement Unlock Rates:")
    for key, value in sorted(stats['achievements'].items()):
        print(f"  {key}: {value*100:.2f}%")

    # Save metrics to CSV
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
    print(f"Saved metrics to {csv_path}")

    # Plot metrics
    plot_metrics(stats, eval_dir)

    return stats

# ---------------------
# PLOTTING
# ---------------------

def plot_metrics(stats, outdir):
    # Achievement bar plot
    plt.figure(figsize=(8, 4))
    achievements = list(stats['achievements'].keys())
    rates = [v * 100 for v in stats['achievements'].values()]
    plt.barh(achievements, rates, color='skyblue')
    plt.xlabel('Unlock Rate (%)')
    plt.title('Crafter Achievement Unlock Rates')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'achievements.png'))
    plt.close()

    # Summary metrics plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Geometric Mean', 'Survival Time', 'Cumulative Reward'],
            [stats['score'], stats['length_mean'], stats['reward_mean']],
            color=['orange', 'green', 'blue'])
    plt.title('Evaluation Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'summary.png'))
    plt.close()
    print(f"Saved plots to {outdir}")

# ---------------------
# MAIN
# ---------------------

if __name__ == '__main__':
    print("Starting Super QRDQN training procedure")
    model = train_super_qrdqn()
    print("Finished training, starting evaluation")
    evaluate_agent(model, episodes=args.eval_episodes)
    print("Evaluation complete.")
