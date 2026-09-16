"""
Generates all three paper graphs:
  1. Box plot of episode rewards (all four policies)
  2. Mean backlog trajectory over episode steps (all four policies)
  3. Mean crew utilization over episode steps (all four policies)

Saves each as a PDF ready to drop into LaTeX.
Run from your project root.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.city import CityServiceEnv
from src.environment.travel import TRAVEL_MATRIX
from src.baselines.greedy2 import greedy_policy
from src.baselines.baseline_policies import reactive_fcfs_policy, closest_or_oldest_policy

# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

NUM_ZONES = 22
NUM_CREWS = 11
FORECAST_HORIZON = 5
N_EVAL_EPISODES = 100
EPISODE_LENGTH = 30

EVAL_DATE_INDICES = np.load("results/eval_date_indices_scorevec.npy")

POLICY_COLORS = {
    "PPO (Proposed)":    "#2563EB",
    "Oldest-or-Closest": "#16A34A",
    "Magnitude-Greedy":  "#D97706",
    "Reactive FCFS":     "#DC2626",
}

# ----------------------------------------------------------------
# Data collection helpers
# ----------------------------------------------------------------

def collect_ppo_data():
    """Run PPO on held-out episodes, collect per-step and total data."""
    def make_eval_env():
        return CityServiceEnv(
            date_indices=EVAL_DATE_INDICES,
            forecast_horizon=FORECAST_HORIZON
        )

    raw_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load(
        "results/models/vecnormalize_scorevec_v2_final.pkl", raw_env
    )
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(
        "results/models/best_scorevec_v2/best_model", env=eval_env
    )

    episode_rewards = []
    all_backlogs = []       # shape: (n_episodes, episode_length)
    all_utilizations = []   # shape: (n_episodes, episode_length)

    for _ in range(N_EVAL_EPISODES):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        ep_backlog = []
        ep_util = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            ep_backlog.append(info[0]["total_backlog"])
            ep_util.append(info[0]["crew_utilization"])

        episode_rewards.append(total_reward)
        all_backlogs.append(ep_backlog)
        all_utilizations.append(ep_util)

    return (
        np.array(episode_rewards),
        np.array(all_backlogs),
        np.array(all_utilizations),
    )


def collect_baseline_data(policy_fn):
    """Run a baseline policy, collect per-step and total data."""
    episode_rewards = []
    all_backlogs = []
    all_utilizations = []

    for start_idx in EVAL_DATE_INDICES[:N_EVAL_EPISODES]:
        env = CityServiceEnv(
            date_indices=np.array([start_idx]),
            forecast_horizon=FORECAST_HORIZON
        )
        obs, _ = env.reset()
        total_reward = 0.0
        ep_backlog = []
        ep_util = []
        terminated = truncated = False

        while not (terminated or truncated):
            zone_list = policy_fn(obs, env.crew_locations)
            obs, reward, terminated, truncated, info = env.step_with_zones(zone_list)
            total_reward += reward
            ep_backlog.append(info["total_backlog"])
            ep_util.append(info["crew_utilization"])

        episode_rewards.append(total_reward)
        all_backlogs.append(ep_backlog)
        all_utilizations.append(ep_util)

    return (
        np.array(episode_rewards),
        np.array(all_backlogs),
        np.array(all_utilizations),
    )


# ----------------------------------------------------------------
# Collect data for all four policies
# ----------------------------------------------------------------
print("Collecting PPO data...")
ppo_rewards, ppo_backlogs, ppo_utils = collect_ppo_data()

print("Collecting Magnitude-Greedy data...")
greedy_rewards, greedy_backlogs, greedy_utils = collect_baseline_data(
    lambda obs, crew_loc: greedy_policy(
        obs, NUM_CREWS, NUM_ZONES,
        crew_capacity=30, forecast_horizon=FORECAST_HORIZON
    )
)

print("Collecting Reactive FCFS data...")
fcfs_rewards, fcfs_backlogs, fcfs_utils = collect_baseline_data(
    lambda obs, crew_loc: reactive_fcfs_policy(
        obs, NUM_CREWS, NUM_ZONES, crew_loc,
        forecast_horizon=FORECAST_HORIZON
    )
)

print("Collecting Oldest-or-Closest data...")
closest_rewards, closest_backlogs, closest_utils = collect_baseline_data(
    lambda obs, crew_loc: closest_or_oldest_policy(
        obs, NUM_CREWS, NUM_ZONES, crew_loc,
        TRAVEL_MATRIX, forecast_horizon=FORECAST_HORIZON
    )
)

print("Data collection complete. Generating graphs...")

# ----------------------------------------------------------------
# GRAPH 1: Box plot of episode rewards
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

policy_names = [
    "PPO\n(Proposed)",
    "Oldest-or-\nClosest",
    "Magnitude-\nGreedy",
    "Reactive\nFCFS",
]
reward_data = [ppo_rewards, closest_rewards, greedy_rewards, fcfs_rewards]
colors = [
    POLICY_COLORS["PPO (Proposed)"],
    POLICY_COLORS["Oldest-or-Closest"],
    POLICY_COLORS["Magnitude-Greedy"],
    POLICY_COLORS["Reactive FCFS"],
]

bp = ax.boxplot(
    reward_data,
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, alpha=0.5),
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

for flier, color in zip(bp["fliers"], colors):
    flier.set(markerfacecolor=color, markeredgecolor=color)

ax.set_xticks(range(1, len(policy_names) + 1))
ax.set_xticklabels(policy_names)
ax.set_ylabel("Episode Reward")
ax.set_title("Distribution of Episode Rewards by Policy")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("results/graphs/boxplot_rewards.pdf", bbox_inches="tight")
plt.savefig("results/graphs/boxplot_rewards.png", bbox_inches="tight")
print("Saved: boxplot_rewards.pdf")
plt.close()

# ----------------------------------------------------------------
# GRAPH 2: Mean backlog trajectory over episode steps
# ----------------------------------------------------------------
days = np.arange(1, EPISODE_LENGTH + 1)

fig, ax = plt.subplots(figsize=(7, 4.5))

for name, backlogs, color, ls in [
    ("PPO (Proposed)",    ppo_backlogs,     "#2563EB", "-"),
    ("Oldest-or-Closest", closest_backlogs, "#16A34A", "--"),
    ("Magnitude-Greedy",  greedy_backlogs,  "#D97706", "-."),
    ("Reactive FCFS",     fcfs_backlogs,    "#DC2626", "-"),
]:
   
    mean_bl = np.mean(backlogs, axis=0)
    std_bl  = np.std(backlogs, axis=0)
    ax.plot(days, mean_bl, label=name, color=color, 
            linewidth=2, linestyle=ls)
    #ax.fill_between(
        #days,
      #  mean_bl - std_bl,
        #mean_bl + std_bl,
       # color=color, alpha=0.12
   # )#

ax.set_xlabel("Episode Day")
ax.set_ylabel("Total Backlog (requests)")
ax.set_title("Mean Total Backlog Over Episode")
ax.legend(loc="upper left", framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(1, EPISODE_LENGTH)

plt.tight_layout()
plt.savefig("results/graphs/backlog_trajectory.pdf", bbox_inches="tight")
plt.savefig("results/graphs/backlog_trajectory.png", bbox_inches="tight")
print("Saved: backlog_trajectory.pdf")
plt.close()

# ----------------------------------------------------------------
# GRAPH 3: Mean crew utilization over episode steps
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

for name, utils, color in [
    ("PPO (Proposed)",    ppo_utils,     POLICY_COLORS["PPO (Proposed)"]),
    ("Oldest-or-Closest", closest_utils, POLICY_COLORS["Oldest-or-Closest"]),
    ("Magnitude-Greedy",  greedy_utils,  POLICY_COLORS["Magnitude-Greedy"]),
    ("Reactive FCFS",     fcfs_utils,    POLICY_COLORS["Reactive FCFS"]),
]:
    mean_util = np.mean(utils, axis=0)
    std_util  = np.std(utils, axis=0)
    ax.plot(days, mean_util, label=name, color=color, linewidth=2)
    

ax.set_xlabel("Episode Day")
ax.set_ylabel("Crew Utilization")
ax.set_title("Mean Crew Utilization Over Episode")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(1, EPISODE_LENGTH)

plt.tight_layout()
plt.savefig("results/graphs/utilization_trajectory.pdf", bbox_inches="tight")
plt.savefig("results/graphs/utilization_trajectory.png", bbox_inches="tight")
print("Saved: utilization_trajectory.pdf")
plt.close()

print("\nAll graphs saved to results/graphs/")
print("Files: boxplot_rewards.pdf, backlog_trajectory.pdf, utilization_trajectory.pdf")