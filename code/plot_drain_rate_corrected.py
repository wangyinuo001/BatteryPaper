"""
Correct analysis: Time per 1% SOC (how fast each percentage drains)
Shows that lower SOC percentages drain FASTER in absolute time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Raw data from TTE vs SOC
soc_points = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
tte_remaining = [8.498, 7.512, 6.559, 5.631, 4.726, 3.838, 2.964, 2.103, 1.253, 0.414]

# Calculate time per 1% SOC for each interval
intervals = []
time_per_1pct = []

for i in range(len(soc_points) - 1):
    soc_high = soc_points[i]
    soc_low = soc_points[i + 1]
    tte_high = tte_remaining[i]
    tte_low = tte_remaining[i + 1]

    delta_soc = soc_high - soc_low  # Should be 10%
    delta_time = tte_high - tte_low  # Hours consumed

    # Time per 1% SOC
    time_per_pct = delta_time / delta_soc * 60  # Convert to minutes

    intervals.append(f"{soc_high}%→{soc_low}%")
    time_per_1pct.append(time_per_pct)

print("=" * 70)
print("TIME PER 1% SOC ANALYSIS (Video Streaming)")
print("=" * 70)
print(f"\n{'Interval':<15} {'Time/1% SOC':<20} {'Interpretation'}")
print("-" * 70)
for interval, time_min in zip(intervals, time_per_1pct):
    if time_min == max(time_per_1pct):
        marker = " ← SLOWEST drain"
    elif time_min == min(time_per_1pct):
        marker = " ← FASTEST drain"
    else:
        marker = ""
    print(f"{interval:<15} {time_min:>6.2f} minutes/1%{marker}")
print("-" * 70)
print(f"\nKey finding: Time per 1% SOC is nearly constant (5.0-5.9 min/1%)")
print(f"Variation: {max(time_per_1pct) - min(time_per_1pct):.2f} min/1%")
print(
    f"Coefficient of variation: {np.std(time_per_1pct)/np.mean(time_per_1pct)*100:.1f}%"
)
print("=" * 70)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Color map
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(time_per_1pct)))

# ========== Plot 1: Time per 1% SOC ==========
ax1 = axes[0]
bars1 = ax1.bar(
    range(len(intervals)),
    time_per_1pct,
    color=colors,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.85,
)

for i, (bar, val) in enumerate(zip(bars1, time_per_1pct)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{val:.2f}\nmin",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax1.set_xlabel("SOC Interval", fontsize=11, fontweight="bold")
ax1.set_ylabel("Minutes per 1% SOC", fontsize=11, fontweight="bold")
ax1.set_title(
    "Discharge Rate: Time Required per 1% SOC Drop\n(Higher = Slower Drain)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax1.set_xticks(range(len(intervals)))
ax1.set_xticklabels(intervals, rotation=45, ha="right", fontsize=9)
ax1.set_ylim(0, max(time_per_1pct) * 1.15)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.axhline(
    y=np.mean(time_per_1pct),
    color="blue",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Average: {np.mean(time_per_1pct):.2f} min/1%",
)
ax1.legend(fontsize=9)

# ========== Plot 2: Total time loss (original) ==========
ax2 = axes[1]
total_time_loss = [t * 10 / 60 for t in time_per_1pct]  # Back to hours for 10% interval
bars2 = ax2.bar(
    range(len(intervals)),
    total_time_loss,
    color=colors[::-1],
    edgecolor="black",
    linewidth=1.5,
    alpha=0.85,
)

for i, (bar, val) in enumerate(zip(bars2, total_time_loss)):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{val:.2f}h",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax2.set_xlabel("SOC Interval", fontsize=11, fontweight="bold")
ax2.set_ylabel("Total Time Loss (hours)", fontsize=11, fontweight="bold")
ax2.set_title(
    "Total TTE Loss per 10% SOC Interval\n(User Perception View)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax2.set_xticks(range(len(intervals)))
ax2.set_xticklabels(intervals, rotation=45, ha="right", fontsize=9)
ax2.set_ylim(0, max(total_time_loss) * 1.15)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# ========== Plot 3: Remaining TTE curve ==========
ax3 = axes[2]
ax3.plot(
    soc_points,
    tte_remaining,
    "o-",
    linewidth=3,
    markersize=8,
    color="navy",
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=1.5,
)

for soc, tte in zip(soc_points, tte_remaining):
    ax3.text(soc, tte + 0.2, f"{tte:.2f}h", ha="center", fontsize=9, fontweight="bold")

ax3.set_xlabel("State of Charge (%)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Remaining TTE (hours)", fontsize=11, fontweight="bold")
ax3.set_title(
    "Remaining Battery Life vs SOC\n(Nearly Linear Relationship)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.invert_xaxis()
ax3.set_xlim(105, 5)
ax3.set_ylim(0, 9)

# Add linear fit line
from numpy.polynomial import polynomial as P

coeffs = P.polyfit(soc_points, tte_remaining, 1)
fit_line = P.polyval(soc_points, coeffs)
ax3.plot(
    soc_points,
    fit_line,
    "--",
    color="red",
    linewidth=2,
    alpha=0.7,
    label=f"Linear fit: R²={1-np.sum((np.array(tte_remaining)-fit_line)**2)/np.sum((np.array(tte_remaining)-np.mean(tte_remaining))**2):.4f}",
)
ax3.legend(fontsize=9)

# Add explanation text
explanation = (
    "Key Insights:\n"
    "• Nearly uniform drain rate: 5.0-5.9 min per 1% SOC\n"
    "• High SOC (100%→90%): each 1% costs ~5.9 min (slower drain)\n"
    "• Low SOC (20%→10%): each 1% costs ~5.0 min (faster drain)\n"
    "• But difference is small (0.9 min), confirming linear behavior\n"
    "• User perception of 'faster drain' at high SOC is because\n"
    "  10% interval at high SOC = 0.99h absolute time loss"
)

fig.text(
    0.5,
    -0.08,
    explanation,
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    family="monospace",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
fig_path = os.path.join(results_dir, "fig_drain_rate_corrected.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {fig_path}")

plt.show()
