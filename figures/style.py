"""Shared matplotlib style for the 4-page workshop paper."""

from __future__ import annotations

import matplotlib as mpl

# Warm, low-saturation palette. Shapes, fills, and line styles carry meaning
# independently of hue so the figures remain legible in grayscale.
INK = "#3D352E"
GREY = "#655F59"
TAUPE = "#8B7355"
DARK_BEIGE = "#A58A6A"
BEIGE = "#D8C3A5"
SAND = "#E9DCC9"
CREAM = "#F7F2EA"
CLAY = "#B56F5A"
SAGE = "#7E8A6A"
DUSTY_ROSE = "#B99086"
LIGHT = "#E6DED3"

D1_COLOR = SAGE
D2_COLOR = CLAY
C2_COLOR = CLAY
C2B_COLOR = DARK_BEIGE
SECONDARY_COLOR = GREY
PERSIST = TAUPE
SPLIT = BEIGE
RECOVER = SAGE
OPENAI = TAUPE
ANTHROPIC = CLAY
CROSS = SAGE
C0_COLOR = TAUPE

JUDGE_ORDER = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)

SHORT_NAMES = {
    "gpt-5.6-sol": "Sol",
    "gpt-5.6-terra": "Terra",
    "gpt-5.6-luna": "Luna",
    "claude-opus-5": "Opus",
    "claude-sonnet-5": "Sonnet",
    "claude-haiku-4-5-20251001": "Haiku",
}

# Full, reader-friendly model names for axis labels.
FULL_NAMES = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "claude-opus-5": "Claude Opus-5",
    "claude-sonnet-5": "Claude Sonnet-5",
    "claude-haiku-4-5-20251001": "Claude Haiku-4.5",
}

# Two-line variants for tight grids (e.g. heatmap axes).
FULL_NAMES_2L = {
    "gpt-5.6-sol": "GPT-5.6\nSol",
    "gpt-5.6-terra": "GPT-5.6\nTerra",
    "gpt-5.6-luna": "GPT-5.6\nLuna",
    "claude-opus-5": "Claude\nOpus-5",
    "claude-sonnet-5": "Claude\nSonnet-5",
    "claude-haiku-4-5-20251001": "Claude\nHaiku-4.5",
}

PREGISTERED_DELTA = 0.5
COL_WIDTH = 3.25
FULL_WIDTH = 5.5


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "regular",
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
