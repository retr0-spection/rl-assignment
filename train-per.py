import os
import argparse

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN
import crafter

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_qrdqn', help="Directory to save logs and model")
parser.add_argument('--steps', type=int, default=500_000, help="Total training timesteps")
parser.add_argument('--eval_episodes', type=int, default=50, help="Number of episodes for evaluation")
parser.add_argument('--checkpoint_interval', type=int, default=100_000, help="Timesteps between saving checkpoints")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------------------
# ENVIRONMENT HELPERS
# ---------------------

def make_train_env():
    """Return a monitored training env"""
    env = crafter.Env()
    env = Monitor(env)
    return env

def make_eval_env(record_dir=None):
    """Return a Recorder-wrapped environment for evaluation"""
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

def train_qrdqn():
    train_env = make_train_env()
    print(f"Training QR-DQN on device: {device}")

    model = QRDQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=10_000,
        device=device,
        tensorboard_log=args.outdir,
    )

    timesteps_done = 0
    while timesteps_done < args.steps:
        remaining = args.steps - timesteps_done
        step_chunk = min(args.checkpoint_interval, remaining)

        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False)
        timesteps_done += step_chunk

        checkpoint_path = os.path.join(args.outdir, f"qrdqn_{timesteps_done}.zip")
        model.save(checkpoint_path)
        print(f"[Checkpoint] Saved model at {timesteps_done} timesteps → {checkpoint_path}")

    model.save(os.path.join(args.outdir, "qrdqn_final"))
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


# ---------------------
# MAIN
# ---------------------

if __name__ == '__main__':
    print("Starting QR-DQN training procedure")
    model = train_qrdqn()
    print("Finished training, starting evaluation")
    evaluate_agent(model, episodes=args.eval_episodes)
    print("Evaluation complete.")
