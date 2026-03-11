"""
Publication-quality matplotlib style for Energy Conversion and Management.

Elsevier single-column width : 90 mm ≈ 3.54 in
Elsevier double-column width: 190 mm ≈ 7.48 in

Design principles
-----------------
* Times New Roman / STIX maths   (Elsevier standard)
* pdf.fonttype 42                 (editable text in PDF)
* 600 DPI raster / vector PDF     (press-ready)
* Colourblind-safe Wong palette   (≤8 categories)
* Minor ticks on, grid subtle     (ECM house style)
* Panel labels via `label_panel`  (a), (b), …
"""

from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# ═══════════════════════════════════════════════════════════════
# Colour palette — Wong (2011) colourblind-safe, print-friendly
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "shepherd": "#D62728",  # vermillion-red  (proposed model)
    "nbm": "#1F77B4",  # blue
    "rint": "#FF7F0E",  # orange
    "thev1": "#2CA02C",  # green
    "thev2": "#9467BD",  # purple
    "exp": "#1A1A1A",  # near-black (experimental data)
    "accent": "#17BECF",  # cyan
    "gray": "#7F7F7F",  # neutral
}

MODEL_COLORS = {
    "Shepherd (Proposed)": COLORS["shepherd"],
    "Shepherd": COLORS["shepherd"],
    "NBM": COLORS["nbm"],
    "Rint": COLORS["rint"],
    "Thevenin-1RC": COLORS["thev1"],
    "Thévenin-1RC": COLORS["thev1"],
    "Thevenin-2RC": COLORS["thev2"],
    "Thévenin-2RC": COLORS["thev2"],
    "Experimental": COLORS["exp"],
}

MODEL_MARKERS = {
    "Shepherd (Proposed)": "s",
    "NBM": "^",
    "Rint": "D",
    "Thevenin-1RC": "o",
    "Thevenin-2RC": "v",
}

MODEL_LINESTYLES = {
    "Shepherd (Proposed)": "-",
    "NBM": "--",
    "Rint": "-.",
    "Thevenin-1RC": ":",
    "Thevenin-2RC": (0, (3, 1, 1, 1)),
}

# Sub-component colours (power breakdown / pie charts)
COMP_COLORS = {
    "Screen": "#E53935",
    "SoC": "#1E88E5",
    "Radio": "#43A047",
    "GPS": "#FFB300",
    "Base": "#757575",
}

# Phase colours for dynamic scenario
PHASE_COLORS = {
    "Gaming": "#E53935",
    "Video Streaming": "#1E88E5",
    "Reading": "#43A047",
    "Standby": "#FFB300",
    "Navigation": "#9467BD",
}

# ═══════════════════════════════════════════════════════════════
# Figure sizes (Elsevier guidelines)
# ═══════════════════════════════════════════════════════════════
SINGLE_COL = (3.54, 2.75)
SINGLE_COL_TALL = (3.54, 3.54)
DOUBLE_COL = (7.48, 3.2)
DOUBLE_COL_TALL = (7.48, 5.0)
DOUBLE_COL_SQ = (7.48, 7.0)


# ═══════════════════════════════════════════════════════════════
# rcParams
# ═══════════════════════════════════════════════════════════════
def apply_style():
    """Set matplotlib rcParams for Elsevier submission."""
    mpl.rcParams.update(
        {
            # ── Font ──────────────────────────────────────────────
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7.5,
            # ── Lines ─────────────────────────────────────────────
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # ── Axes ──────────────────────────────────────────────
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,  # grid behind data
            "grid.alpha": 0.20,
            "grid.linewidth": 0.4,
            "grid.linestyle": "--",
            # ── Ticks ─────────────────────────────────────────────
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.35,
            "ytick.minor.width": 0.35,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            # ── Legend ────────────────────────────────────────────
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.80",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.6,
            # ── Save ──────────────────────────────────────────────
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 150,
            "figure.constrained_layout.use": False,
            # ── PDF ───────────────────────────────────────────────
            "pdf.fonttype": 42,  # TrueType → editable text
            "ps.fonttype": 42,
            # ── Math ──────────────────────────────────────────────
            "mathtext.fontset": "stix",
        }
    )


# ═══════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════
def label_panel(ax, label, x=-0.12, y=1.06, **kwargs):
    """Add (a)/(b)/… panel label outside the axes (ECM convention)."""
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
        **kwargs,
    )


def embed_metric(ax, text, x=0.97, y=0.05, **kwargs):
    """Embed RMSE / R² annotation inside the plot area."""
    defaults = dict(
        fontsize=7,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


def set_minor_ticks(ax, x_minor=None, y_minor=None):
    """Force AutoMinorLocator on specified axes."""
    if x_minor is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y_minor is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def save_fig(fig, name, results_dir, formats=("pdf", "png")):
    """Save figure in multiple formats (press-ready 600 DPI)."""
    os.makedirs(results_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(results_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  ✓ Saved: {name}.{'/'.join(formats)}")
