import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "lines.linewidth": 1.2,
    "lines.markersize": 5, "figure.dpi": 300,
})

# ── config ────────────────────────────────────────────────────────────────
RESULTS_FILE = "results_UNCERTAINTY_AWARE_keepout_noise_high_temp.csv"
N_SEEDS      = 10   # runs per beta value, in order of appearance
BETAS        = [0.5, 1.0, 1.282, 1.645, 2.0, 2.576]
CURRENT_BETA = 1.645

# ── load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_FILE)

# Assign beta by block of N_SEEDS rows in order
assert len(df) == len(BETAS) * N_SEEDS, \
    f"Expected {len(BETAS)*N_SEEDS} rows, got {len(df)}. Check N_SEEDS and BETAS."

df["beta"] = np.repeat(BETAS, N_SEEDS)

# ── aggregate ─────────────────────────────────────────────────────────────
agg = df.groupby("beta").agg(
    success_mean    = ("success",      "mean"),
    success_se      = ("success",      lambda x: x.std() / np.sqrt(len(x))),
    violations_mean = ("n_violations", "mean"),
    violations_se   = ("n_violations", lambda x: x.std() / np.sqrt(len(x))),
).reset_index()

betas      = agg["beta"].values
success    = agg["success_mean"].values
success_se = agg["success_se"].values
viol       = agg["violations_mean"].values
viol_se    = agg["violations_se"].values

# ── plot ──────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

color_s = "C0"
color_v = "C1"

ax1.errorbar(betas, success, yerr=success_se, marker="o",
             color=color_s, capsize=3, label="Success rate")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylabel("Success rate", color=color_s)
ax1.tick_params(axis="y", labelcolor=color_s)
ax1.set_ylim(0, 1.05)
ax1.spines[["top"]].set_visible(False)

ax2 = ax1.twinx()
ax2.errorbar(betas, viol, yerr=viol_se, marker="^",
             color=color_v, ls="--", capsize=3, label="Violations / ep")
ax2.set_ylabel("Mean violations / episode", color=color_v)
ax2.tick_params(axis="y", labelcolor=color_v)
ax2.set_ylim(bottom=0)
ax2.spines[["top"]].set_visible(False)

ax1.axvline(x=CURRENT_BETA, color="grey", lw=2.0, ls=(0, (1, 2)), alpha=0.9)
ax1.text(CURRENT_BETA + 0.02, 0.16, r"current $\beta$", fontsize=6, color="grey")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

ax1.set_title(r"CC-MPC: success rate and violations vs. $\beta$")
ax1.grid(True, lw=0.3, alpha=0.5, axis="y")

fig.savefig("fig_beta_sweep.pdf", bbox_inches="tight")
print("saved fig_beta_sweep.pdf")
plt.show()
