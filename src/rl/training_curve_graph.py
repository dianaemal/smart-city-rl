import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

df = pd.read_csv("/Users/dianaemal/LLM_Research/src/training_curve.csv")

# TensorBoard CSV columns are: Wall time, Step, Value
steps = df["Step"]
rewards = df["Value"]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(steps, rewards, color="#2563EB", linewidth=1.2, alpha=0.4,
        label="Raw")

# smoothed line (rolling average) so the trend is clear
smoothed = rewards.rolling(window=10, min_periods=1).mean()
ax.plot(steps, smoothed, color="#2563EB", linewidth=2,
        label="Smoothed (10-step avg)")

ax.set_xlabel("Training Timesteps")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("PPO Training Curve")
ax.legend(framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# format x-axis as e.g. 200k, 400k
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k"
    )
)

plt.tight_layout()
plt.savefig("results/graphs/training_curve.pdf", bbox_inches="tight")
plt.savefig("results/graphs/training_curve.png", bbox_inches="tight")
print("Saved: training_curve.pdf")