#!/usr/bin/env python3
"""Export stable showcase figures from TensorBoard events and BAR analysis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


TIMELINE_STEPS = [0, 600, 1000, 2000, 2400, 2600, 3000, 4000, 5000]
KEY_RESIDUAL_LAYERS = [7, 8, 9]
WEIGHT_LAYERS = [7, 8]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, help="Directory containing baseline TensorBoard events")
    parser.add_argument("--bar_dir", required=True, help="Directory containing BAR TensorBoard events")
    parser.add_argument("--bar_analysis_dir", required=True, help="Directory containing BAR analysis artifacts")
    parser.add_argument("--output_dir", required=True, help="Directory to write showcase assets into")
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


def make_metrics_summary(baseline_scalars, bar_scalars, bar_analysis_dir: Path):
    base_train = baseline_scalars["eval/train_loss"]
    base_val = baseline_scalars["eval/val_loss"]
    bar_train = bar_scalars["eval/train_loss"]
    bar_val = bar_scalars["eval/val_loss"]

    shared_steps = sorted(set(base_val) & set(bar_val))
    if not shared_steps:
        raise ValueError("baseline and BAR event files do not share eval steps")

    peak_step = max(shared_steps, key=lambda step: base_val[step] - bar_val[step])
    sustained_flip_step = None
    for step in shared_steps:
        if step == 0:
            continue
        if all((base_val[s] - bar_val[s]) < 0 for s in shared_steps if s >= step):
            sustained_flip_step = step
            break

    final_step = shared_steps[-1]
    summary = {
        "peak_step": peak_step,
        "peak_val_advantage": base_val[peak_step] - bar_val[peak_step],
        "peak_train_advantage": base_train[peak_step] - bar_train[peak_step],
        "sustained_flip_step": sustained_flip_step,
        "final_step": final_step,
        "final_val_gap": base_val[final_step] - bar_val[final_step],
        "final_train_gap": base_train[final_step] - bar_train[final_step],
        "bar_best_val_step": min(bar_val, key=bar_val.get),
        "bar_best_val": min(bar_val.values()),
        "baseline_best_val_step": min(base_val, key=base_val.get),
        "baseline_best_val": min(base_val.values()),
        "timeline": [],
    }

    for step in TIMELINE_STEPS:
        if step not in base_val or step not in bar_val:
            continue
        summary["timeline"].append(
            {
                "step": step,
                "baseline_train": base_train[step],
                "bar_train": bar_train[step],
                "baseline_val": base_val[step],
                "bar_val": bar_val[step],
                "val_base_minus_bar": base_val[step] - bar_val[step],
            }
        )

    diagnosis = json.loads((bar_analysis_dir / "diagnosis_summary.json").read_text())
    loss_modes = json.loads((bar_analysis_dir / "loss_mode_comparison.json").read_text())
    summary["bar_analysis"] = {
        "classification": diagnosis["classification"],
        "message": diagnosis["message"],
        "losses": diagnosis["losses"],
        "loss_gaps": diagnosis["loss_gaps"],
        "weight_metrics": diagnosis["weight_metrics"],
        "score_metrics": diagnosis["score_metrics"],
        "loss_mode_rows": loss_modes,
    }
    return summary


def save_train_val_curves(output_path: Path, baseline_scalars, bar_scalars, summary):
    base_train = baseline_scalars["eval/train_loss"]
    base_val = baseline_scalars["eval/val_loss"]
    bar_train = bar_scalars["eval/train_loss"]
    bar_val = bar_scalars["eval/val_loss"]
    shared_steps = sorted(set(base_val) & set(bar_val))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    plots = [
        (axes[0], "Train Loss", base_train, bar_train),
        (axes[1], "Val Loss", base_val, bar_val),
    ]
    for ax, title, baseline, bar in plots:
        baseline_values = [baseline[step] for step in shared_steps]
        bar_values = [bar[step] for step in shared_steps]
        ax.plot(shared_steps, baseline_values, label="Baseline", linewidth=2)
        ax.plot(shared_steps, bar_values, label="BAR", linewidth=2)
        ax.axvline(summary["peak_step"], linestyle="--", alpha=0.5, color="tab:green")
        if summary["sustained_flip_step"] is not None:
            ax.axvline(summary["sustained_flip_step"], linestyle="--", alpha=0.5, color="tab:red")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].text(
        0.02,
        0.02,
        f"Peak BAR advantage @ {summary['peak_step']} steps",
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    if summary["sustained_flip_step"] is not None:
        axes[1].text(
            0.02,
            0.02,
            f"Sustained flip @ {summary['sustained_flip_step']} steps",
            transform=axes[1].transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_val_gap_curve(output_path: Path, baseline_scalars, bar_scalars, summary):
    base_val = baseline_scalars["eval/val_loss"]
    bar_val = bar_scalars["eval/val_loss"]
    shared_steps = sorted(set(base_val) & set(bar_val))
    gap_values = [base_val[step] - bar_val[step] for step in shared_steps]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(shared_steps, gap_values, linewidth=2, color="tab:purple")
    ax.axhline(0.0, linestyle="--", color="black", alpha=0.6)
    ax.axvline(summary["peak_step"], linestyle="--", alpha=0.5, color="tab:green")
    if summary["sustained_flip_step"] is not None:
        ax.axvline(summary["sustained_flip_step"], linestyle="--", alpha=0.5, color="tab:red")
    ax.set_title("Validation Gap: Baseline Val Loss - BAR Val Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Positive means BAR is better")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_residual_dynamics(output_path: Path, bar_scalars):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    shared_steps = sorted(bar_scalars["residual_stats/layer_7/attn_history_share"])

    for layer in KEY_RESIDUAL_LAYERS:
        axes[0].plot(
            shared_steps,
            [bar_scalars[f"residual_stats/layer_{layer}/attn_history_share"][step] for step in shared_steps],
            label=f"layer {layer} attn_history",
            linewidth=2,
        )
        axes[0].plot(
            shared_steps,
            [bar_scalars[f"residual_stats/layer_{layer}/mlp_history_share"][step] for step in shared_steps],
            label=f"layer {layer} mlp_history",
            linewidth=2,
            linestyle="--",
        )

    for layer in WEIGHT_LAYERS:
        axes[1].plot(
            shared_steps,
            [bar_scalars[f"residual_weights/layer_{layer}/attn_res_agg_l2"][step] for step in shared_steps],
            label=f"layer {layer} attn_l2",
            linewidth=2,
        )
        axes[1].plot(
            shared_steps,
            [bar_scalars[f"residual_weights/layer_{layer}/mlp_res_agg_l2"][step] for step in shared_steps],
            label=f"layer {layer} mlp_l2",
            linewidth=2,
            linestyle="--",
        )

    for layer in [7, 8]:
        axes[2].plot(
            shared_steps,
            [bar_scalars[f"residual_stats/layer_{layer}/output_norm"][step] for step in shared_steps],
            label=f"layer {layer} output_norm",
            linewidth=2,
        )

    axes[0].set_title("History Share Dynamics")
    axes[1].set_title("Residual Weight Norms")
    axes[2].set_title("Deep Output Norms")

    for ax in axes:
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Share")
    axes[1].set_ylabel("L2 norm")
    axes[2].set_ylabel("Mean L2 norm")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def copy_static_figures(bar_analysis_dir: Path, output_dir: Path):
    static_files = {
        bar_analysis_dir / "bar_scores_heatmap.png": output_dir / "bar_scores_heatmap.png",
        bar_analysis_dir / "bar_hidden_norms.png": output_dir / "bar_hidden_norms.png",
    }
    for source, target in static_files.items():
        shutil.copy2(source, target)


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    bar_dir = Path(args.bar_dir)
    bar_analysis_dir = Path(args.bar_analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_event = find_event_file(baseline_dir)
    bar_event = find_event_file(bar_dir)

    baseline_scalars = load_scalars(baseline_event, ["eval/train_loss", "eval/val_loss"])
    bar_scalars = load_scalars(
        bar_event,
        [
            "eval/train_loss",
            "eval/val_loss",
            "residual_stats/layer_7/attn_history_share",
            "residual_stats/layer_7/mlp_history_share",
            "residual_stats/layer_8/attn_history_share",
            "residual_stats/layer_8/mlp_history_share",
            "residual_stats/layer_9/attn_history_share",
            "residual_stats/layer_9/mlp_history_share",
            "residual_stats/layer_7/output_norm",
            "residual_stats/layer_8/output_norm",
            "residual_weights/layer_7/attn_res_agg_l2",
            "residual_weights/layer_7/mlp_res_agg_l2",
            "residual_weights/layer_8/attn_res_agg_l2",
            "residual_weights/layer_8/mlp_res_agg_l2",
        ],
    )

    summary = make_metrics_summary(baseline_scalars, bar_scalars, bar_analysis_dir)
    save_train_val_curves(output_dir / "owt_train_val_curves.png", baseline_scalars, bar_scalars, summary)
    save_val_gap_curve(output_dir / "owt_val_gap_curve.png", baseline_scalars, bar_scalars, summary)
    save_residual_dynamics(output_dir / "bar_residual_dynamics.png", bar_scalars)
    copy_static_figures(bar_analysis_dir, output_dir)

    summary["baseline_event_file"] = str(baseline_event)
    summary["bar_event_file"] = str(bar_event)
    summary["bar_analysis_dir"] = str(bar_analysis_dir)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported showcase assets to {output_dir}")


if __name__ == "__main__":
    main()
