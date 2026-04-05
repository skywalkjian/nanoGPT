#!/usr/bin/env python3
"""Export 3500-step OWT comparison figures from TensorBoard event files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


TIMELINE_STEPS = [0, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 3500]
STEADY_STATE_START = 3000
STEADY_STATE_END = 3490
SCALAR_TAGS = [
    "eval/train_loss",
    "eval/val_loss",
    "eval/best_val_loss",
    "perf/iter_ms",
    "perf/mfu",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, help="Directory containing baseline TensorBoard events")
    parser.add_argument("--bar_dir", required=True, help="Directory containing BAR TensorBoard events")
    parser.add_argument("--output_dir", required=True, help="Directory to write figure assets into")
    return parser.parse_args()


def find_event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("events.out.tfevents*"))
    if not candidates:
        raise FileNotFoundError(f"no TensorBoard event file found under {run_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_scalars(event_file: Path, tags: list[str]) -> dict[str, dict[int, float]]:
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()
    scalars = {}
    for tag in tags:
        scalars[tag] = {event.step: event.value for event in acc.Scalars(tag)}
    return scalars


def summarize_run(run_scalars: dict[str, dict[int, float]]) -> dict[str, float | int]:
    train = run_scalars["eval/train_loss"]
    val = run_scalars["eval/val_loss"]
    final_step = max(val)
    iter_window = [value for step, value in run_scalars["perf/iter_ms"].items() if STEADY_STATE_START <= step <= STEADY_STATE_END]
    mfu_window = [value for step, value in run_scalars["perf/mfu"].items() if STEADY_STATE_START <= step <= STEADY_STATE_END]
    best_val_step = min(val, key=val.get)
    return {
        "final_step": final_step,
        "final_eval_train_loss": train[final_step],
        "final_eval_val_loss": val[final_step],
        "best_val_step": best_val_step,
        "best_val_loss": val[best_val_step],
        "steady_state_iter_ms_median": median(iter_window),
        "steady_state_iter_ms_mean": sum(iter_window) / len(iter_window),
        "steady_state_mfu_median": median(mfu_window),
        "steady_state_mfu_mean": sum(mfu_window) / len(mfu_window),
    }


def build_summary(baseline_scalars: dict[str, dict[int, float]], bar_scalars: dict[str, dict[int, float]]):
    base_train = baseline_scalars["eval/train_loss"]
    base_val = baseline_scalars["eval/val_loss"]
    bar_train = bar_scalars["eval/train_loss"]
    bar_val = bar_scalars["eval/val_loss"]

    shared_eval_steps = sorted(set(base_val) & set(bar_val))
    if not shared_eval_steps:
        raise ValueError("baseline and BAR event files do not share eval steps")

    peak_step = max(shared_eval_steps, key=lambda step: base_val[step] - bar_val[step])
    sustained_flip_step = None
    for step in shared_eval_steps:
        if step == 0:
            continue
        if all((base_val[s] - bar_val[s]) < 0 for s in shared_eval_steps if s >= step):
            sustained_flip_step = step
            break

    summary = {
        "steady_state_window": {"start": STEADY_STATE_START, "end": STEADY_STATE_END},
        "baseline": summarize_run(baseline_scalars),
        "bar": summarize_run(bar_scalars),
        "peak_step": peak_step,
        "peak_val_advantage": base_val[peak_step] - bar_val[peak_step],
        "peak_train_advantage": base_train[peak_step] - bar_train[peak_step],
        "sustained_flip_step": sustained_flip_step,
        "final_step": shared_eval_steps[-1],
        "final_train_gap": base_train[shared_eval_steps[-1]] - bar_train[shared_eval_steps[-1]],
        "final_val_gap": base_val[shared_eval_steps[-1]] - bar_val[shared_eval_steps[-1]],
        "timeline": [],
    }

    for step in TIMELINE_STEPS:
        if step not in base_val or step not in bar_val:
            continue
        summary["timeline"].append(
            {
                "step": step,
                "baseline_eval_train": base_train[step],
                "bar_eval_train": bar_train[step],
                "baseline_eval_val": base_val[step],
                "bar_eval_val": bar_val[step],
                "val_base_minus_bar": base_val[step] - bar_val[step],
            }
        )

    return summary


def save_train_val_curves(output_path: Path, baseline_scalars, bar_scalars, summary):
    base_train = baseline_scalars["eval/train_loss"]
    base_val = baseline_scalars["eval/val_loss"]
    bar_train = bar_scalars["eval/train_loss"]
    bar_val = bar_scalars["eval/val_loss"]
    shared_steps = sorted(set(base_val) & set(bar_val))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    plots = [
        (axes[0], "Eval Train Loss", base_train, bar_train),
        (axes[1], "Eval Val Loss", base_val, bar_val),
    ]
    for ax, title, baseline, bar in plots:
        ax.plot(shared_steps, [baseline[step] for step in shared_steps], label="Baseline", linewidth=2, color="#1f77b4")
        ax.plot(shared_steps, [bar[step] for step in shared_steps], label="BAR", linewidth=2, color="#d62728")
        ax.axvline(summary["peak_step"], linestyle="--", color="#2ca02c", alpha=0.75, linewidth=1.5)
        ax.axvline(summary["final_step"], linestyle=":", color="#444444", alpha=0.8, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[1].text(
        0.03,
        0.03,
        f"Peak BAR val advantage: {summary['peak_val_advantage']:.4f} @ step {summary['peak_step']}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_val_gap_curve(output_path: Path, baseline_scalars, bar_scalars, summary):
    base_val = baseline_scalars["eval/val_loss"]
    bar_val = bar_scalars["eval/val_loss"]
    shared_steps = sorted(set(base_val) & set(bar_val))
    gap_values = [base_val[step] - bar_val[step] for step in shared_steps]

    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    ax.plot(shared_steps, gap_values, linewidth=2.2, color="#0b7285")
    ax.axhline(0.0, linestyle="--", color="black", alpha=0.6)
    ax.axvline(summary["peak_step"], linestyle="--", color="#2ca02c", alpha=0.75, linewidth=1.5)
    ax.axvline(summary["final_step"], linestyle=":", color="#444444", alpha=0.8, linewidth=1.5)
    ax.scatter([summary["peak_step"]], [summary["peak_val_advantage"]], color="#2ca02c", zorder=3)
    ax.scatter([summary["final_step"]], [summary["final_val_gap"]], color="#444444", zorder=3)
    ax.set_title("Validation Gap: Baseline Val Loss - BAR Val Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Positive means BAR is better")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.05,
        f"Final gap @ {summary['final_step']}: {summary['final_val_gap']:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_efficiency_summary(output_path: Path, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    labels = ["Baseline", "BAR"]
    colors = ["#1f77b4", "#d62728"]

    iter_values = [
        summary["baseline"]["steady_state_iter_ms_median"],
        summary["bar"]["steady_state_iter_ms_median"],
    ]
    mfu_values = [
        summary["baseline"]["steady_state_mfu_median"],
        summary["bar"]["steady_state_mfu_median"],
    ]

    axes[0].bar(labels, iter_values, color=colors, width=0.6)
    axes[1].bar(labels, mfu_values, color=colors, width=0.6)

    axes[0].set_title("Steady-State Iteration Time")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("Steady-State MFU")
    axes[1].set_ylabel("MFU")

    for ax, values, fmt in [
        (axes[0], iter_values, "{:.2f}"),
        (axes[1], mfu_values, "{:.2f}"),
    ]:
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Steady-state window: steps {STEADY_STATE_START}..{STEADY_STATE_END}", fontsize=11)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    bar_dir = Path(args.bar_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_event = find_event_file(baseline_dir)
    bar_event = find_event_file(bar_dir)

    baseline_scalars = load_scalars(baseline_event, SCALAR_TAGS)
    bar_scalars = load_scalars(bar_event, SCALAR_TAGS)
    summary = build_summary(baseline_scalars, bar_scalars)
    summary["baseline_event_file"] = str(baseline_event)
    summary["bar_event_file"] = str(bar_event)

    save_train_val_curves(output_dir / "train_val_curves.png", baseline_scalars, bar_scalars, summary)
    save_val_gap_curve(output_dir / "val_gap_curve.png", baseline_scalars, bar_scalars, summary)
    save_efficiency_summary(output_dir / "efficiency_summary.png", summary)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported 3500-step OWT figures to {output_dir}")


if __name__ == "__main__":
    main()
