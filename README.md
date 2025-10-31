COMPARATIVE EVALUATION OF DQN AND PPO AGENTS IN THE CRAFTER ENVIRONMENT
=======================================================================

Python: 3.10+
Frameworks: Stable Baselines3, SB3-Contrib
License: MIT
Crafter Benchmark: arXiv:2109.06780

A reinforcement learning project evaluating Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents in the Crafter environment.
Conducted as part of the Wits University COMS4061A / COMS7071A Reinforcement Learning Assignment.

-----------------------------------------------------------------------
OVERVIEW
-----------------------------------------------------------------------

This repository contains code, results, and evaluation plots for training and benchmarking different RL agents in the Crafter environment.
Implemented models:

 - Base DQN: Classic Deep Q-Learning with CNN policy
 - QRDQN: Quantile Regression DQN (distributional RL)
 - DQN + CNN + Attention: DQN variant with spatial attention (in progress)
 - PPO and PPO + Attention (planned)

Agents were trained for 5 million timesteps, evaluated over 50 episodes, and assessed using geometric mean score, average survival time, cumulative reward, and achievement unlock rates.

-----------------------------------------------------------------------
PROJECT STRUCTURE
-----------------------------------------------------------------------

rl-assignment/
│
├── base_dqn.py                  -> Baseline DQN training and evaluation
├── qrdqn_variant.py             -> QRDQN implementation (sb3-contrib)
├── cnn_attention_extractor.py   -> Custom CNN + Attention feature extractor
│
├── utils/
│   ├── evaluate_crafter.py      -> Evaluation logic and plotting helpers
│   └── plot_metrics.py          -> Achievement + summary plot generation
│
├── logdir/                      -> TensorBoard logs and trained models
│   ├── crafter_dqn/
│   ├── crafter_qrdqn/
│   └── crafter_attention/
│
├── results/
│   ├── achievements_dqn_base.png
│   ├── summary_dqn.png
│   ├── achievements_qrdqn.png
│   ├── summary_qrdqn.png
│   ├── achievements_attention.png
│   └── summary_attention.png
│
├── report/
│   └── Reinforcement_Learning_Report.tex
│
├── requirements.txt
└── README.txt

-----------------------------------------------------------------------
INSTALLATION
-----------------------------------------------------------------------

1. Clone the repository
   git clone https://github.com/retr0-spection/rl-assignment.git
   cd rl-assignment

2. Create and activate a virtual environment
   python3 -m venv venv
   source venv/bin/activate  (Windows: venv\Scripts\activate)

3. Install dependencies
   pip install -r requirements.txt

4. Install Crafter
   pip install crafter

-----------------------------------------------------------------------
TRAINING AND EVALUATION
-----------------------------------------------------------------------

Base DQN:
   python base_dqn.py --steps 5000000 --eval_episodes 50

QRDQN (Distributional DQN):
   python qrdqn_variant.py --steps 5000000 --eval_episodes 50

DQN + CNN + Attention:
   python dqn_attention_variant.py --steps 5000000 --eval_episodes 50

Each script saves logs and results in: logdir/<variant>/eval/

-----------------------------------------------------------------------
RESULTS
-----------------------------------------------------------------------

Model Variant           | Geometric Mean | Avg. Survival | Avg. Reward
------------------------|----------------|----------------|-------------
Base DQN                | 0.0035          | 214.23         | 6.78
QRDQN                   | 0.0033          | 244.02         | 9.24
DQN + CNN + Attention   | (in progress)   | 210.00         | 5.32

QRDQN shows the best overall performance.
The attention-based model is still training but shows stable learning dynamics.

Figures:
 - summary_dqn.png
 - summary_qrdqn.png

-----------------------------------------------------------------------
EVALUATION METRICS
-----------------------------------------------------------------------

 - Geometric Mean Score: Aggregated achievement score (Crafter metric)
 - Average Survival Time: Mean timesteps survived per episode
 - Average Cumulative Reward: Mean total reward per episode
 - Achievement Unlock Rate: % of total achievements completed

All metrics are computed via evaluate_crafter() and saved to evaluation_metrics.csv.

-----------------------------------------------------------------------
REFERENCES
-----------------------------------------------------------------------

1. D. Hafner, "Benchmarking the Spectrum of Agent Capabilities in the Crafter Environment," arXiv:2109.06780, 2021.
2. A. Raffin et al., "Stable Baselines3: Reliable Reinforcement Learning Implementations," JMLR, 2021.
3. W. Dabney et al., "Distributional Reinforcement Learning with Quantile Regression," AAAI, 2018.
4. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

-----------------------------------------------------------------------
AUTHORS
-----------------------------------------------------------------------

Teddy Mngwenya (2582286)   - Model Development and Evaluation
Oratile Nailana (2327853)   - Analysis and Documentation
Nthabiseng Mabetlela (1828559) - Training, Visualization, and Report Writing

-----------------------------------------------------------------------
ACKNOWLEDGMENT
-----------------------------------------------------------------------

This project was conducted under the School of Computer Science and Applied Mathematics,
University of the Witwatersrand, Johannesburg.
Computation was supported by the Mathematical Sciences HPC Cluster.

-----------------------------------------------------------------------
LICENSE
-----------------------------------------------------------------------

This repository is licensed under the MIT License.

-----------------------------------------------------------------------
CITATION
-----------------------------------------------------------------------

If you use this code or results, please cite:

Mngwenya, T., Nailana, O., and Mabetlela, N. (2025).
"Comparative Evaluation of Deep Q-Learning and Proximal Policy Optimization Agents in the Crafter Environment."
University of the Witwatersrand. GitHub: https://github.com/retr0-spection/rl-assignment
