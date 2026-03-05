"""
Create a beautiful bar chart showing TTE loss per 10% SOC interval.
Visualizes the "fastest drain" perception analysis from Section 6.1.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load actual Video Streaming scenario data from CSV
csv_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "tte_loss_video_streaming_detailed.csv"
)
df = pd.read_csv(csv_path)

# Extract data from CSV
soc_intervals = df["SOC_Interval"].tolist()
tte_loss_hours = df["TTE_Loss (h)"].tolist()

# Calculate percentage of total drain for each interval
total_drain = sum(tte_loss_hours)
tte_loss_pct = [(loss / total_drain) * 100 for loss in tte_loss_hours]

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Color gradient from red (high drain) to blue (lower drain)
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(tte_loss_hours)))

# ========== Left plot: Absolute TTE loss (hours) ==========
bars1 = ax1.bar(
    range(len(soc_intervals)),
    tte_loss_hours,
    color=colors,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.85,
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, tte_loss_hours)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{val:.2f}h",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax1.set_xlabel("SOC Interval", fontsize=12, fontweight="bold")
ax1.set_ylabel("TTE Loss (hours)", fontsize=12, fontweight="bold")
ax1.set_title(
    "Time-to-Empty Loss per 10% SOC Drop\nVideo Streaming (Screen 80%, SoC 20%, Radio 90%)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax1.set_xticks(range(len(soc_intervals)))
ax1.set_xticklabels(soc_intervals, rotation=45, ha="right", fontsize=10)
ax1.set_ylim(0, 1.1)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.axhline(
    y=np.mean(tte_loss_hours),
    color="red",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Average: {np.mean(tte_loss_hours):.2f}h",
)
ax1.legend(fontsize=10, loc="upper right")

# Add annotation for lowest SOC (fastest perceived drain)
min_idx = tte_loss_hours.index(min(tte_loss_hours))
ax1.annotate(
    "Fastest perceived drain\n(Low remaining energy)",
    xy=(min_idx, tte_loss_hours[min_idx]),
    xytext=(7.5, 0.20),
    arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
    fontsize=9,
    color="darkred",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.8),
)

# ========== Right plot: Percentage of total drain ==========
bars2 = ax2.bar(
    range(len(soc_intervals)),
    tte_loss_pct,
    color=colors,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.85,
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, tte_loss_pct)):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.1,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax2.set_xlabel("SOC Interval", fontsize=12, fontweight="bold")
ax2.set_ylabel("Percentage of Total TTE (%)", fontsize=12, fontweight="bold")
ax2.set_title(
    "Relative Contribution to Total Discharge Time\nVideo Streaming (Total TTE = {:.2f}h, Power = 2.598W)".format(
        total_drain
    ),
    fontsize=11,
    fontweight="bold",
    pad=15,
)
ax2.set_xticks(range(len(soc_intervals)))
ax2.set_xticklabels(soc_intervals, rotation=45, ha="right", fontsize=10)
ax2.set_ylim(0, 13)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.axhline(
    y=100 / 9,
    color="green",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Equal share: {100/9:.1f}%",
)
ax2.legend(fontsize=10, loc="upper right")

# Add text box with key finding
textstr = (
    f"Key Finding:\n"
    f"• Nearly uniform drain rate (0.84-0.99 h per 10% SOC)\n"
    f"• Slight decrease at lower SOC due to nonlinear voltage dynamics\n"
    f'• User perception: "Battery drains faster when LOW"\n'
    f"  - 100%→90%: {tte_loss_hours[0]:.2f}h absolute loss (but plenty of energy left)\n"
    f"  - 20%→10%: {tte_loss_hours[-1]:.2f}h absolute loss (but critical low energy!)\n"
    f"  → Low SOC feels faster due to anxiety about remaining capacity"
)

props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
fig.text(0.5, -0.12, textstr, fontsize=10, ha="center", bbox=props, family="monospace")

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# Save figure
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(results_dir, exist_ok=True)
fig_path = os.path.join(results_dir, "fig_tte_drain_intervals.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {fig_path}")

# Also create a simplified single-panel version
fig2, ax = plt.subplots(figsize=(10, 6))

# Create gradient bars
bars = ax.bar(
    range(len(soc_intervals)),
    tte_loss_hours,
    color=colors,
    edgecolor="black",
    linewidth=2,
    alpha=0.9,
)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, tte_loss_hours)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{val:.2f}h\n({tte_loss_pct[i]:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xlabel("SOC Interval (10% drop)", fontsize=13, fontweight="bold")
ax.set_ylabel("Time-to-Empty Loss (hours)", fontsize=13, fontweight="bold")
ax.set_title(
    "Battery Drain Rate Analysis: TTE Loss per 10% SOC Decrease\n"
    + "Video Streaming (Screen 80%, SoC 20%, Radio 90%) | Q=5.0Ah | Numerical Integration",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
ax.set_xticks(range(len(soc_intervals)))
ax.set_xticklabels(soc_intervals, rotation=45, ha="right", fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1)

# Add horizontal line for average
avg_line = ax.axhline(
    y=np.mean(tte_loss_hours),
    color="red",
    linestyle="--",
    linewidth=2.5,
    alpha=0.8,
    label=f"Average: {np.mean(tte_loss_hours):.2f}h",
)

# Add shading for "low SOC critical zone"
ax.axvspan(
    6.5,
    8.5,
    alpha=0.2,
    color="red",
    label="Low-SOC critical zone (fastest perceived drain)",
)

# Legend - moved to upper right to avoid overlap with bars
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

# Add annotation for LOW SOC (20%→10%)
min_idx = tte_loss_hours.index(min(tte_loss_hours))
ax.annotate(
    'Fastest perceived drain:\n"Every 10% feels like it\ngoes in seconds!"\n(Low remaining energy)',
    xy=(min_idx, tte_loss_hours[min_idx]),
    xytext=(min_idx - 0.5, 0.30),
    arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5),
    fontsize=11,
    color="darkred",
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.7",
        facecolor="orange",
        edgecolor="darkred",
        linewidth=2,
        alpha=0.9,
    ),
)

# Add statistics box to lower left corner inside plot
stats_text = (
    f"Statistics (9 intervals):\n"
    f"  Max:    {max(tte_loss_hours):.2f} h (100%→90%)\n"
    f"  Min:    {min(tte_loss_hours):.2f} h (20%→10%)\n"
    f"  Mean:   {np.mean(tte_loss_hours):.2f} h\n"
    f"  Std:    {np.std(tte_loss_hours):.3f} h\n"
    f"  Range:  {max(tte_loss_hours)-min(tte_loss_hours):.2f} h\n"
    f"  CV:     {(np.std(tte_loss_hours)/np.mean(tte_loss_hours)*100):.1f}%"
)

props = dict(
    boxstyle="round", facecolor="lightblue", alpha=0.85, edgecolor="navy", linewidth=1.5
)
ax.text(
    0.02,
    0.02,
    stats_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    horizontalalignment="left",
    bbox=props,
    family="monospace",
    fontweight="bold",
)

plt.tight_layout()

# Save simplified version
fig_path2 = os.path.join(results_dir, "fig_tte_drain_intervals_simple.png")
plt.savefig(fig_path2, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {fig_path2}")

plt.show()

print("\n" + "=" * 70)
print("TTE DRAIN INTERVAL ANALYSIS")
print("=" * 70)
print(
    f"\n{'SOC Interval':<15} {'TTE Loss (h)':<15} {'% of Total':<15} {'Cum. Loss (h)':<15}"
)
print("-" * 70)
cumulative = 0
for interval, loss, pct in zip(soc_intervals, tte_loss_hours, tte_loss_pct):
    cumulative += loss
    print(
        f"{interval:<15} {loss:>8.2f} h      {pct:>8.1f} %      {cumulative:>10.2f} h"
    )
print("-" * 70)
print(f"{'TOTAL':<15} {sum(tte_loss_hours):>8.2f} h      {sum(tte_loss_pct):>8.1f} %")
print("=" * 70)
print(
    f"\nKey Insight: Variation of only {(max(tte_loss_hours)-min(tte_loss_hours)):.2f}h"
)
print(
    f"({(max(tte_loss_hours)-min(tte_loss_hours))/np.mean(tte_loss_hours)*100:.1f}% of mean)"
)
print("confirms approximately linear discharge behavior.")
print("=" * 70)
