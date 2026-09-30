#!/usr/bin/env python3
"""Build paper figures and LaTeX tables from frozen D1/D2 results.

Run from the workshop root:

    python figures/build.py

Outputs PDF/PNG under paper/figures/ and booktabs snippets under paper/tables/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from style import (  # noqa: E402
    BEIGE,
    C0_COLOR,
    C2_COLOR,
    C2B_COLOR,
    CREAM,
    D1_COLOR,
    D2_COLOR,
    DARK_BEIGE,
    FULL_NAMES,
    FULL_NAMES_2L,
    FULL_WIDTH,
    GREY,
    INK,
    JUDGE_ORDER,
    PERSIST,
    PREGISTERED_DELTA,
    RECOVER,
    SAND,
    SECONDARY_COLOR,
    SHORT_NAMES,
    SPLIT,
    apply_style,
)

from behavioural_test.analysis import (  # noqa: E402
    behavioral_invariance,
    effective_sample_sizes,
    phi_matrix,
    records_to_matrices,
)
from behavioural_test.data import read_jsonl  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def short_judge(name: str) -> str:
    model = name.split("#")[0]
    return SHORT_NAMES.get(model, model)


def full_judge(name: str) -> str:
    model = name.split("#")[0]
    return FULL_NAMES.get(model, model)


def full_judge_2l(name: str) -> str:
    model = name.split("#")[0]
    return FULL_NAMES_2L.get(model, model)


def to_full(labels: list[str], two_line: bool = False) -> list[str]:
    """Map short judge labels back to full model names for display."""
    inverse = {v: k for k, v in SHORT_NAMES.items()}
    table = FULL_NAMES_2L if two_line else FULL_NAMES
    out = []
    for label in labels:
        model = inverse.get(label, label)
        out.append(table.get(model, label))
    return out


def reorder_panel(
    judges: list[str], predictions: np.ndarray, errors: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    models = [judge.split("#")[0] for judge in judges]
    indices = []
    for model in JUDGE_ORDER:
        indices.extend(i for i, name in enumerate(models) if name == model)
    missing = [i for i in range(len(judges)) if i not in indices]
    indices.extend(missing)
    labels = [short_judge(judges[i]) for i in indices]
    return labels, predictions[indices], errors[indices]


def tercile_labels(values: list[float], item_ids: list[str]) -> dict[str, str]:
    ordered = sorted(zip(values, item_ids, strict=True))
    n = len(ordered)
    cuts = [n // 3, (2 * n) // 3]
    labels = {}
    for index, (_, item_id) in enumerate(ordered):
        if index < cuts[0]:
            labels[item_id] = "low"
        elif index < cuts[1]:
            labels[item_id] = "medium"
        else:
            labels[item_id] = "high"
    return labels


def subset_neff(errors: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 8:
        return None
    try:
        _, kish, _ = effective_sample_sizes(phi_matrix(errors[:, mask]))
    except ValueError:
        return None
    return float(kish)


def unanimous_fate(
    base_errors: np.ndarray, int_errors: np.ndarray
) -> dict[str, int]:
    wrong = np.all(base_errors == 1, axis=0)
    persist = int(np.sum(wrong & np.all(int_errors == 1, axis=0)))
    recover = int(np.sum(wrong & np.all(int_errors == 0, axis=0)))
    split = int(np.sum(wrong) - persist - recover)
    return {
        "unanimous_wrong": int(wrong.sum()),
        "persist": persist,
        "split": split,
        "recover": recover,
    }


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"wrote {pdf.relative_to(ROOT)}")


def plot_delta_neff(comparisons: dict[str, dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    # (dataset tag, tag colour, condition text, dataset key, condition key, marker colour)
    rows = [
        ("D1", D1_COLOR, "C2 per-judge views", "d1", "c2", C2_COLOR),
        ("D1", D1_COLOR, "C2b shared paraphrase", "d1", "c2b", C2B_COLOR),
        ("D2", D2_COLOR, "C2 per-judge views", "d2", "c2", C2_COLOR),
        ("D2", D2_COLOR, "C2b shared paraphrase", "d2", "c2b", C2B_COLOR),
        ("D1", D1_COLOR, "C1 distinct rubrics", "d1", "c1", SECONDARY_COLOR),
        ("D1", D1_COLOR, "C3 distinct 4-shot sets", "d1", "c3", SECONDARY_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.35))
    y = np.arange(len(rows))[::-1]

    # Shade the primary (C2 / C2b) block to separate it from the secondary rows.
    ax.axhspan(1.5, 5.5, color=CREAM, zorder=0)
    ax.axhline(1.5, color=BEIGE, lw=0.8, zorder=1)

    for ypos, (_tag, _tagc, _cond, dataset, cond, color) in zip(y, rows, strict=True):
        payload = comparisons[dataset][cond]
        delta = payload["delta_kish_n_eff"]
        lo, hi = payload["delta_kish_95_ci"]
        marker = "o" if cond == "c2" else ("D" if cond == "c2b" else "s")
        ax.hlines(ypos, lo, hi, color=color, lw=2.0, zorder=2)
        ax.plot(
            [lo, hi],
            [ypos, ypos],
            "|",
            color=color,
            ms=5.0,
            markeredgewidth=1.3,
            zorder=2,
        )
        ax.plot(
            delta,
            ypos,
            marker,
            color=color,
            ms=6.6,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.text(
            hi + 0.03,
            ypos,
            f"{delta:+.2f}",
            va="center",
            ha="left",
            fontsize=7,
            color=color,
        )

    # Null line and the +0.5 target.
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.axvline(PREGISTERED_DELTA, color=DARK_BEIGE, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(
        PREGISTERED_DELTA,
        1.015,
        r"$+0.5$",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=DARK_BEIGE,
        transform=ax.get_xaxis_transform(),
    )

    # Clean, left-aligned row labels: coloured dataset tag + condition text.
    ax.set_yticks(y)
    ax.set_yticklabels([])
    tag_tf = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    for ypos, (tag, tagc, cond, *_rest) in zip(y, rows, strict=True):
        ax.text(0.015, ypos, tag, transform=tag_tf, ha="left", va="center",
                fontsize=7.6, color=tagc, fontweight="bold")
        ax.text(0.062, ypos, cond, transform=tag_tf, ha="left", va="center",
                fontsize=7.6, color=INK)

    ax.set_xlim(-0.55, 0.65)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xlabel(r"Change in effective panel size $\Delta n_{\mathrm{eff}}$ vs. C0")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.20)
    save(fig, "fig_delta_neff")
    plt.close(fig)


def plot_invariance(
    d1_fate: dict[str, int],
    d2_fate: dict[str, int],
    d1_stability: float,
    d2_stability: float,
) -> None:
    """Of the items every judge got wrong at C0, how many stay unanimous errors
    once each judge sees a different rewrite? Counts, not shares."""
    import matplotlib.pyplot as plt

    specs = [
        (0, d1_fate, d1_stability, "D1", "ChaosNLI", D1_COLOR),
        (1, d2_fate, d2_stability, "D2", "RewardBench", D2_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.55))
    bar_w = 0.52
    max_total = max(d1_fate["unanimous_wrong"], d2_fate["unanimous_wrong"], 1)

    for x, fate, stability, tag, dsname, tagc in specs:
        total = fate["unanimous_wrong"]
        persist = fate["persist"]
        split = fate["split"]
        recover = fate["recover"]

        ax.bar(x, persist, bar_w, color=PERSIST, edgecolor="white", linewidth=1.0,
               zorder=3, label="Stayed a unanimous error" if x == 0 else None)
        if split:
            ax.bar(x, split, bar_w, bottom=persist, color=SPLIT, edgecolor="white",
                   linewidth=1.0, zorder=3,
                   label="Panel no longer unanimous" if x == 0 else None)
        if recover:
            ax.bar(x, recover, bar_w, bottom=persist + split, color=RECOVER,
                   edgecolor="white", linewidth=1.0, zorder=3)

        # Segment counts.
        ax.text(x, persist / 2, str(persist), ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
        if split:
            ax.text(x, persist + split / 2, str(split), ha="center", va="center",
                    color=INK, fontsize=9, zorder=4)

        # Headline above the bar: persistence rate + none recovered.
        rate = persist / total if total else 0.0
        ax.text(x, total + max_total * 0.055,
                f"{persist} of {total} stay ({rate:.0%})",
                ha="center", va="bottom", fontsize=8, color=tagc, fontweight="bold")
        ax.text(x, total + max_total * 0.005, "0 recovered",
                ha="center", va="bottom", fontsize=6.8, color=GREY, style="italic")

        # Dataset label + label stability beneath the axis.
        ax.text(x, -max_total * 0.055, tag, ha="center", va="top",
                fontsize=9, color=tagc, fontweight="bold")
        ax.text(x, -max_total * 0.115, dsname, ha="center", va="top",
                fontsize=7.5, color=INK)
        ax.text(x, -max_total * 0.175, f"label stability {stability:.0%}",
                ha="center", va="top", fontsize=6.8, color=GREY)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max_total * 1.18)
    ax.set_xticks([])
    ax.set_ylabel("Items all six judges got wrong at C0")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              handlelength=1.1, borderaxespad=0.4)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.16)
    save(fig, "fig_invariance")
    plt.close(fig)


def plot_provider_phi(summaries: dict[str, dict[str, dict[str, float]]]) -> None:
    import matplotlib.pyplot as plt

    groups = ["Within OpenAI", "Within Anthropic", "Cross-provider"]
    keys = ["within_openai_mean_phi", "within_anthropic_mean_phi", "cross_provider_mean_phi"]
    x = np.arange(len(groups))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3), sharey=True)
    for ax, dataset, title in (
        (axes[0], "d1", "D1  ChaosNLI"),
        (axes[1], "d2", "D2  RewardBench Chat"),
    ):
        c0 = [summaries[dataset]["c0"][key] for key in keys]
        c2 = [summaries[dataset]["c2"][key] for key in keys]
        ax.bar(
            x - width / 2,
            c0,
            width,
            color=BEIGE,
            edgecolor=C0_COLOR,
            linewidth=0.8,
            hatch="//",
            label="C0 identical input",
        )
        ax.bar(
            x + width / 2,
            c2,
            width,
            color=C2_COLOR,
            edgecolor=C2_COLOR,
            linewidth=0.8,
            label="C2 per-judge views",
        )
        ax.set_xticks(x, groups, fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 0.75)
        ax.axhline(0, color=INK, lw=0.5)
    axes[0].set_ylabel(r"Mean error $\phi$")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_provider_phi")
    plt.close(fig)


def plot_phi_heatmaps(
    panels: dict[str, dict[str, tuple[list[str], np.ndarray]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.65))
    norm = Normalize(vmin=0.15, vmax=0.75)
    cmap = LinearSegmentedColormap.from_list(
        "earth",
        ["#F8F3EB", "#E8D8C3", "#C9AB86", "#9B735A", "#5C4436"],
    )
    layout = [
        (0, 0, "d1", "c0", "D1 C0", D1_COLOR),
        (0, 1, "d1", "c2", "D1 C2", D1_COLOR),
        (1, 0, "d2", "c0", "D2 C0", D2_COLOR),
        (1, 1, "d2", "c2", "D2 C2", D2_COLOR),
    ]
    image = None
    for row, col, dataset, cond, title, title_color in layout:
        ax = axes[row, col]
        labels, corr = panels[dataset][cond]
        full = to_full(labels, two_line=True)
        image = ax.imshow(corr, cmap=cmap, norm=norm)
        ax.set_xticks(range(len(labels)), full, rotation=0, ha="center", fontsize=4.7)
        ax.set_yticks(range(len(labels)), full, va="center", fontsize=4.9)
        ax.set_title(title, loc="left", color=title_color, fontweight="bold")
        ax.tick_params(length=0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = corr[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value > 0.48 else INK,
                )
    fig.subplots_adjust(left=0.12, right=0.86, top=0.95, bottom=0.10, wspace=0.42, hspace=0.42)
    cax = fig.add_axes([0.89, 0.24, 0.018, 0.52])
    fig.colorbar(image, cax=cax, label=r"Error $\phi$")
    save(fig, "fig_phi_heatmaps")
    plt.close(fig)


def plot_noise_twin(d1_c2: dict[str, Any], d1_summary: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    twin = d1_c2["noise_matched_control"]
    labels = ["C0 panel", "C2 panel", "Noise-matched\ntwin of C2", "Best C0 judge"]
    values = [
        d1_c2["baseline_panel_accuracy"],
        d1_c2["intervention_panel_accuracy"],
        twin["panel_accuracy_mean"],
        d1_summary["best_individual_accuracy"],
    ]
    lo = [None, None, twin["panel_accuracy_95_interval"][0], None]
    hi = [None, None, twin["panel_accuracy_95_interval"][1], None]
    colors = [BEIGE, C2_COLOR, DARK_BEIGE, SAND]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 2.35))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62, edgecolor=INK, linewidth=0.5)
    if lo[2] is not None and hi[2] is not None:
        ax.errorbar(2, values[2], yerr=[[values[2] - lo[2]], [hi[2] - values[2]]], fmt="none", ecolor=GREY, capsize=2, lw=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Majority-vote accuracy")
    ax.set_ylim(0.68, 0.76)
    fig.tight_layout()
    save(fig, "fig_noise_twin")
    plt.close(fig)


def plot_entropy(
    item_ids: list[str],
    c0_errors: np.ndarray,
    c2_errors: np.ndarray,
    entropy_by_item: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    bins = tercile_labels([entropy_by_item[i] for i in item_ids], item_ids)
    names = ["low", "medium", "high"]
    x = np.arange(len(names))
    c0_vals = []
    c2_vals = []
    for name in names:
        mask = np.array([bins[i] == name for i in item_ids])
        c0_vals.append(subset_neff(c0_errors, mask))
        c2_vals.append(subset_neff(c2_errors, mask))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 2.25))
    width = 0.34
    ax.bar(
        x - width / 2,
        c0_vals,
        width,
        color=BEIGE,
        edgecolor=C0_COLOR,
        linewidth=0.8,
        hatch="//",
        label="C0",
    )
    ax.bar(
        x + width / 2,
        c2_vals,
        width,
        color=C2_COLOR,
        edgecolor=C2_COLOR,
        linewidth=0.8,
        label="C2",
    )
    ax.set_xticks(x, ["Low entropy", "Medium", "High entropy"])
    ax.set_ylabel(r"$n_{\mathrm{eff}}$")
    ax.set_ylim(1.2, 2.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_entropy")
    plt.close(fig)


def plot_flips(item_ids: list[str], base_pred: np.ndarray, int_pred: np.ndarray, title: str, stem: str) -> None:
    import matplotlib.pyplot as plt

    flips = np.sum(base_pred != int_pred, axis=0)
    counts = np.bincount(flips.astype(int), minlength=7)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 2.15))
    ax.bar(np.arange(7), counts, color=SAND, edgecolor=C0_COLOR, linewidth=0.7, width=0.72)
    ax.set_xlabel("Judges whose label changes from C0 to C2")
    ax.set_ylabel("Items")
    ax.set_xticks(range(7))
    ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def plot_model_accuracy(
    d1_c0: tuple[list[str], list[str], np.ndarray, np.ndarray],
    d1_c2: tuple[list[str], list[str], np.ndarray, np.ndarray],
    d2_c0: tuple[list[str], list[str], np.ndarray, np.ndarray],
    d2_c2: tuple[list[str], list[str], np.ndarray, np.ndarray],
) -> None:
    """Dumbbell plot of per-judge accuracy under shared and private views."""
    import matplotlib.pyplot as plt

    labels = to_full(d1_c0[1])
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4), sharey=True)
    specs = [
        (
            axes[0],
            1 - d1_c0[3].mean(axis=1),
            1 - d1_c2[3].mean(axis=1),
            "D1  ChaosNLI",
            D1_COLOR,
            (0.63, 0.74),
            0.728,
            0.728,
        ),
        (
            axes[1],
            1 - d2_c0[3].mean(axis=1),
            1 - d2_c2[3].mean(axis=1),
            "D2  RewardBench",
            D2_COLOR,
            (0.79, 0.91),
            0.897,
            0.903,
        ),
    ]
    for ax, c0, c2, title, title_color, xlim, panel_c0, panel_c2 in specs:
        for row in range(len(labels)):
            if row % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color=CREAM, zorder=0)
        ax.axhline(2.5, color=BEIGE, lw=0.8, zorder=1)
        for ypos, baseline, intervention in zip(y, c0, c2, strict=True):
            ax.plot(
                [baseline, intervention],
                [ypos, ypos],
                color=DARK_BEIGE,
                lw=1.5,
                alpha=0.78,
                zorder=2,
            )
        ax.scatter(
            c0,
            y,
            s=28,
            marker="s",
            facecolor="white",
            edgecolor=C0_COLOR,
            linewidth=1.1,
            label="C0 shared input",
            zorder=3,
        )
        ax.scatter(
            c2,
            y,
            s=30,
            marker="o",
            facecolor=C2_COLOR,
            edgecolor="white",
            linewidth=0.7,
            label="C2 per-judge views",
            zorder=4,
        )
        ax.axvline(panel_c0, color=C0_COLOR, lw=0.8, ls=(0, (4, 2)), alpha=0.75)
        if abs(panel_c2 - panel_c0) > 0.0005:
            ax.axvline(panel_c2, color=C2_COLOR, lw=0.9, ls=(0, (1.5, 1.5)), alpha=0.8)
        ax.text(
            0.02,
            0.98,
            f"Panel  {panel_c0:.3f} → {panel_c2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=GREY,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, 5.6)
        ax.set_title(title, loc="left", color=title_color, fontweight="bold")
        ax.set_xlabel("Accuracy")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y, labels)
    axes[1].tick_params(axis="y", labelleft=False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.28, wspace=0.16)
    save(fig, "fig_model_accuracy")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_delta(value: float, ci: list[float]) -> str:
    lo, hi = ci
    sign = "+" if value >= 0 else ""
    return rf"{sign}{value:.3f} [{lo:.3f}, {hi:.3f}]"


def beats_twin(payload: dict[str, Any]) -> str:
    e2 = payload.get("e2_vs_noise_twin")
    if e2 is None:
        return "---"
    lower = e2["one_sided_95_lower_bound"]
    return "yes" if lower > 0 else "no"


def write_main_table(
    summaries: dict[str, dict[str, dict[str, float]]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    spec = [
        ("D1", "C0 shared input", "d1", "c0", None),
        ("", "C2 per-judge views", "d1", "c2", "c2"),
        ("", "C2b shared paraphrase", "d1", "c2b", "c2b"),
        ("D2", "C0 shared input", "d2", "c0", None),
        ("", "C2 per-judge views", "d2", "c2", "c2"),
        ("", "C2b shared paraphrase", "d2", "c2b", "c2b"),
    ]
    for dataset_name, cond_name, dataset, cond, compare_key in spec:
        s = summaries[dataset][cond]
        if compare_key is None:
            delta = "---"
        else:
            payload = comparisons[dataset][compare_key]
            delta = fmt_delta(payload["delta_kish_n_eff"], payload["delta_kish_95_ci"])
        rows.append(
            " & ".join(
                [
                    dataset_name,
                    cond_name,
                    fmt(s["kish_n_eff"]),
                    fmt(s["mean_phi"]),
                    fmt(s["panel_accuracy"]),
                    delta,
                ]
            )
            + r" \\"
        )
        if cond_name == "C2b shared paraphrase" and dataset == "d1":
            rows.append(r"\midrule")
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Primary intervention results. $\Delta n_{{\text{{eff}}}}$ is relative to C0;
brackets give paired-bootstrap 95\% confidence intervals. Per-judge views do
not improve effective panel size on either dataset.}}
\label{{tab:main}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{llcccc}}
\toprule
Dataset & Condition & $n_{{\text{{eff}}}}$ & $\bar\phi$ & Panel acc. & $\Delta n_{{\text{{eff}}}}$ [95\% CI] \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path = TAB_DIR / "tab_main.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def write_endpoint_table(endpoints: dict[str, Any]) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for dataset, label in (("d1", "D1 ChaosNLI"), ("d2", "D2 RewardBench")):
        row = endpoints[dataset]
        e2 = row["e2_holm_p"]
        e2_s = "---" if e2 is None else f"{e2:.3f}"
        lines.append(
            " & ".join(
                [
                    label,
                    f"{row['e1_holm_p']:.3f}",
                    "yes" if row["e1_supports_surface_form"] else "no",
                    e2_s,
                    "yes" if row["e2_supports_useful_diversity"] else "no",
                ]
            )
            + r" \\"
        )
    tex = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Primary endpoints, Holm-corrected across D1 and D2. Neither dataset supports H1.}}
\label{{tab:endpoints}}
\begin{{tabular}}{{lcccc}}
\toprule
Dataset & E1 Holm $p$ & Supports H1? & E2 Holm $p$ & Useful diversity? \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path = TAB_DIR / "tab_endpoints.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def write_invariance_table(
    d1: dict[str, Any], d2: dict[str, Any], d1_fate: dict[str, int], d2_fate: dict[str, int]
) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    def row(name: str, inv: dict[str, Any], fate: dict[str, int]) -> str:
        persist = fate["persist"] / fate["unanimous_wrong"] if fate["unanimous_wrong"] else 0.0
        return " & ".join(
            [
                name,
                f"{inv['label_stability_rate']:.3f}",
                str(inv["unanimous_wrong_count"]),
                f"{persist:.3f}",
                str(inv["unanimous_correct_count"]),
                f"{inv['unanimous_correct_panel_flip_rate']:.3f}",
            ]
        ) + r" \\"

    tex = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Behavioral invariance from C0 to C2. Persistence is the share of C0 unanimous errors that remain unanimous errors.}}
\label{{tab:invariance}}
\begin{{tabular}}{{lccccc}}
\toprule
Dataset & Label stability & Unan.\ wrong & Persist & Unan.\ correct & Panel flip \\
\midrule
{row("D1 ChaosNLI", d1, d1_fate)}
{row("D2 RewardBench", d2, d2_fate)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path = TAB_DIR / "tab_invariance.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def write_judge_table(
    d1_c0: tuple[list[str], np.ndarray, np.ndarray],
    d1_c2: tuple[list[str], np.ndarray, np.ndarray],
    d2_c0: tuple[list[str], np.ndarray, np.ndarray],
    d2_c2: tuple[list[str], np.ndarray, np.ndarray],
) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    labels = d1_c0[1]
    lines = []
    for i, name in enumerate(labels):
        cells = [
            name,
            fmt(1 - d1_c0[3][i].mean()),
            fmt(1 - d1_c2[3][i].mean()),
            fmt(1 - d2_c0[3][i].mean()),
            fmt(1 - d2_c2[3][i].mean()),
        ]
        lines.append(" & ".join(cells) + r" \\")
    tex = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-judge accuracy. Individual accuracy is essentially unchanged by C2; the panel does not gain a new expert.}}
\label{{tab:judges}}
\begin{{tabular}}{{lcccc}}
\toprule
Judge & D1 C0 & D1 C2 & D2 C0 & D2 C2 \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path = TAB_DIR / "tab_judges.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def aligned(records: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    item_ids, judges, predictions, _gold, errors = records_to_matrices(records)
    labels, predictions, errors = reorder_panel(judges, predictions, errors)
    return item_ids, labels, predictions, errors


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {
        "d1": {
            "c0": load_json(RESULTS / "c0_summary.json"),
            "c1": load_json(RESULTS / "c1_summary.json"),
            "c2": load_json(RESULTS / "c2_summary.json"),
            "c2b": load_json(RESULTS / "c2b_summary.json"),
            "c3": load_json(RESULTS / "c3_summary.json"),
        },
        "d2": {
            "c0": load_json(RESULTS / "d2_c0_summary.json"),
            "c2": load_json(RESULTS / "d2_c2_summary.json"),
            "c2b": load_json(RESULTS / "d2_c2b_summary.json"),
        },
    }
    comparisons = {
        "d1": {
            "c1": load_json(RESULTS / "c1_vs_c0.json"),
            "c2": load_json(RESULTS / "c2_vs_c0.json"),
            "c2b": load_json(RESULTS / "c2b_vs_c0.json"),
            "c3": load_json(RESULTS / "c3_vs_c0.json"),
        },
        "d2": {
            "c2": load_json(RESULTS / "d2_c2_vs_c0.json"),
            "c2b": load_json(RESULTS / "d2_c2b_vs_c0.json"),
        },
    }
    endpoints = load_json(RESULTS / "endpoints.json")

    d1_c0 = aligned(read_jsonl(RESULTS / "c0.jsonl"))
    d1_c2 = aligned(read_jsonl(RESULTS / "c2.jsonl"))
    d2_c0 = aligned(read_jsonl(RESULTS / "d2_c0.jsonl"))
    d2_c2 = aligned(read_jsonl(RESULTS / "d2_c2.jsonl"))

    d1_fate = unanimous_fate(d1_c0[3], d1_c2[3])
    d2_fate = unanimous_fate(d2_c0[3], d2_c2[3])
    d1_inv = behavioral_invariance(read_jsonl(RESULTS / "c0.jsonl"), read_jsonl(RESULTS / "c2.jsonl"))
    d2_inv = behavioral_invariance(read_jsonl(RESULTS / "d2_c0.jsonl"), read_jsonl(RESULTS / "d2_c2.jsonl"))

    plot_delta_neff(comparisons)
    plot_invariance(d1_fate, d2_fate, d1_inv.label_stability_rate, d2_inv.label_stability_rate)
    plot_model_accuracy(d1_c0, d1_c2, d2_c0, d2_c2)
    plot_provider_phi(summaries)

    heatmaps = {
        "d1": {
            "c0": (d1_c0[1], phi_matrix(d1_c0[3])),
            "c2": (d1_c2[1], phi_matrix(d1_c2[3])),
        },
        "d2": {
            "c0": (d2_c0[1], phi_matrix(d2_c0[3])),
            "c2": (d2_c2[1], phi_matrix(d2_c2[3])),
        },
    }
    plot_phi_heatmaps(heatmaps)
    plot_noise_twin(comparisons["d1"]["c2"], summaries["d1"]["c0"])

    chaos = {item["item_id"]: item["human_entropy"] for item in read_jsonl(DATA / "chaos_mnli_500.jsonl")}
    plot_entropy(d1_c0[0], d1_c0[3], d1_c2[3], chaos)
    plot_flips(d1_c0[0], d1_c0[2], d1_c2[2], "D1  ChaosNLI", "fig_flips_d1")
    plot_flips(d2_c0[0], d2_c0[2], d2_c2[2], "D2  RewardBench", "fig_flips_d2")

    write_main_table(summaries, comparisons)
    write_endpoint_table(endpoints["primary_endpoints"])
    write_invariance_table(d1_inv.to_dict(), d2_inv.to_dict(), d1_fate, d2_fate)
    write_judge_table(d1_c0, d1_c2, d2_c0, d2_c2)
    print("done")


if __name__ == "__main__":
    main()
