"""
Residual-attention analysis utility for nanoGPT checkpoints.

The entrypoint name stays as analyze_bar.py for compatibility, but it now
supports baseline, BAR, and FAR checkpoints.
"""

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPTConfig, GPT

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for analyze_bar.py. Install it with `pip install matplotlib`.") from exc

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


EVAL_MODES = ('learned', 'uniform', 'current_only')
MODE_PREFIX = {
    'baseline': 'baseline',
    'bar': 'bar',
    'full_attention_residuals': 'far',
}
MODE_LABEL = {
    'baseline': 'Baseline',
    'bar': 'BAR',
    'full_attention_residuals': 'FAR',
}
WEIGHT_FIELDS = ['layer_idx', 'module', 'mean', 'std', 'min', 'max', 'l2_norm']
LOSS_FIELDS = ['evaluation_mode', 'mean_loss', 'delta_vs_learned']


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze residual-attention checkpoints.")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory that contains ckpt.pt')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name under data/. Defaults to checkpoint config.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split to analyze.')
    parser.add_argument('--num_batches', type=int, default=4, help='Number of batches to average for analysis metrics.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size used during analysis.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for analysis artifacts.')
    parser.add_argument('--use_tqdm', action=argparse.BooleanOptionalAction, default=True, help='Show tqdm progress bars if tqdm is installed.')
    return parser.parse_args()


def setup_ctx(device, dtype):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return device_type, ctx


def strip_unwanted_prefix(state_dict, prefix='_orig_mod.'):
    for key in list(state_dict.keys()):
        if key.startswith(prefix):
            state_dict[key[len(prefix):]] = state_dict.pop(key)
    return state_dict


def load_checkpoint(out_dir, device):
    ckpt_path = Path(out_dir) / 'ckpt.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device):
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = strip_unwanted_prefix(checkpoint['model'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def load_split_memmap(dataset, split):
    data_path = Path('data') / dataset / f'{split}.bin'
    if not data_path.exists():
        raise FileNotFoundError(f"dataset split not found: {data_path}")
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def make_rng(seed):
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    return rng


def get_batch(data, block_size, batch_size, device, device_type, rng):
    upper = len(data) - block_size - 1
    if upper <= 0:
        raise ValueError(f"dataset split is too short for block_size={block_size}")
    ix = torch.randint(upper, (batch_size,), generator=rng)
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix.tolist()])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def mode_prefix(mode):
    return MODE_PREFIX.get(mode, mode)


def mode_label(mode):
    return MODE_LABEL.get(mode, mode.replace('_', ' ').title())


def save_json_csv(rows, output_dir, stem, fieldnames):
    json_path = output_dir / f'{stem}.json'
    csv_path = output_dir / f'{stem}.csv'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_weights(model):
    rows = []
    for layer_idx, block in enumerate(model.transformer.h):
        if not hasattr(block, 'attn_res_agg'):
            continue
        for name, weight in [
            ('attn_res_agg', block.attn_res_agg.weight),
            ('mlp_res_agg', block.mlp_res_agg.weight),
        ]:
            data = weight.detach().float().cpu()
            rows.append({
                'layer_idx': layer_idx,
                'module': name,
                'mean': float(data.mean().item()),
                'std': float(data.std(unbiased=False).item()),
                'min': float(data.min().item()),
                'max': float(data.max().item()),
                'l2_norm': float(data.norm().item()),
            })
    return rows


def scores_to_heatmap(block_stats, score_key):
    rows = []
    max_depth = 1
    for stats in block_stats:
        scores = stats.get(score_key)
        if scores is None:
            row = np.full(1, np.nan, dtype=np.float32)
        else:
            row = scores.mean(dim=(1, 2)).numpy().astype(np.float32)
            max_depth = max(max_depth, len(row))
        rows.append(row)
    heatmap = np.full((len(rows), max_depth), np.nan, dtype=np.float32)
    for layer_idx, row in enumerate(rows):
        heatmap[layer_idx, :len(row)] = row
    return heatmap


def save_heatmap(attn_heatmap, mlp_heatmap, output_dir, prefix, checkpoint_mode):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    title_prefix = mode_label(checkpoint_mode)
    for ax, heatmap, title in [
        (axes[0], attn_heatmap, 'Attention Scores'),
        (axes[1], mlp_heatmap, 'MLP Scores'),
    ]:
        masked = np.ma.masked_invalid(heatmap)
        image = ax.imshow(masked, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set_title(f'{title_prefix} {title}')
        ax.set_xlabel('Depth candidate')
        ax.set_ylabel('Layer index')
        if masked.mask.all():
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / f'{prefix}_scores_heatmap.png', dpi=200)
    plt.close(fig)


def aggregate_norm_stats(all_block_stats):
    norm_keys = ['input_norm', 'attn_agg_norm', 'post_attn_norm', 'mlp_agg_norm', 'output_norm']
    per_layer = {}
    for batch_stats in all_block_stats:
        for stats in batch_stats:
            layer_idx = stats['layer_idx']
            per_layer.setdefault(layer_idx, {key: [] for key in norm_keys})
            for key in norm_keys:
                if key in stats:
                    per_layer[layer_idx][key].append(stats[key])

    rows = []
    for layer_idx in sorted(per_layer):
        row = {'layer_idx': layer_idx}
        for key in norm_keys:
            values = per_layer[layer_idx][key]
            row[key] = float(np.mean(values)) if values else float('nan')
        rows.append(row)
    return rows


def save_hidden_norms(rows, output_dir, prefix, checkpoint_mode):
    fieldnames = ['layer_idx', 'input_norm', 'attn_agg_norm', 'post_attn_norm', 'mlp_agg_norm', 'output_norm']
    csv_path = output_dir / f'{prefix}_hidden_norms.csv'
    png_path = output_dir / f'{prefix}_hidden_norms.png'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    layer_indices = [row['layer_idx'] for row in rows]
    for key in fieldnames[1:]:
        values = [row[key] for row in rows]
        ax.plot(layer_indices, values, marker='o', label=key)
    ax.set_title(f'{mode_label(checkpoint_mode)} Hidden Norm Statistics')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean L2 norm')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def score_entropy(scores):
    if scores.size(0) <= 1:
        return 0.0
    probs = scores.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=0)
    return float(entropy.mean().item())


def normalized_score_entropy(scores):
    if scores.size(0) <= 1:
        return 0.0
    return score_entropy(scores) / float(np.log(scores.size(0)))


def summarize_score_behavior(all_block_stats):
    metric_keys = [
        'attn_entropy',
        'attn_normalized_entropy',
        'attn_current_share',
        'attn_history_share',
        'mlp_entropy',
        'mlp_normalized_entropy',
        'mlp_current_share',
        'mlp_history_share',
    ]
    per_layer = {}
    for batch_stats in all_block_stats:
        for stats in batch_stats:
            layer_idx = stats['layer_idx']
            layer_metrics = per_layer.setdefault(layer_idx, {key: [] for key in metric_keys})
            for score_key, prefix in [('attn_scores', 'attn'), ('mlp_scores', 'mlp')]:
                scores = stats.get(score_key)
                if scores is None:
                    continue
                layer_metrics[f'{prefix}_entropy'].append(score_entropy(scores))
                layer_metrics[f'{prefix}_normalized_entropy'].append(normalized_score_entropy(scores))
                layer_metrics[f'{prefix}_current_share'].append(float(scores[-1].mean().item()))
                history_share = float(scores[:-1].sum(dim=0).mean().item()) if scores.size(0) > 1 else 0.0
                layer_metrics[f'{prefix}_history_share'].append(history_share)

    rows = []
    for layer_idx in sorted(per_layer):
        row = {'layer_idx': layer_idx}
        for key in metric_keys:
            values = per_layer[layer_idx][key]
            row[key] = float(np.mean(values)) if values else float('nan')
        rows.append(row)
    return rows


def average_metric(rows, key):
    values = [row[key] for row in rows if not np.isnan(row[key])]
    return float(np.mean(values)) if values else None


def run_analysis_pass(model, data, args, device_type, ctx, evaluation_mode, collect_stats=False):
    rng = make_rng(args.seed)
    losses = []
    all_block_stats = []
    first_batch_stats = None
    aux_mode = None
    batch_pbar = None
    if args.use_tqdm and tqdm is not None:
        batch_pbar = tqdm(total=args.num_batches, desc=f'analyze:{evaluation_mode}', dynamic_ncols=True, leave=False)
    elif args.use_tqdm and tqdm is None:
        print("WARNING: tqdm is not installed; continuing without progress bar. Install it with `pip install tqdm`.")

    with model.use_residual_analysis_mode(evaluation_mode):
        with torch.no_grad():
            for batch_idx in range(args.num_batches):
                x, y = get_batch(data, model.config.block_size, args.batch_size, args.device, device_type, rng)
                with ctx:
                    if collect_stats:
                        _, loss, aux = model(x, y, return_bar_stats=True)
                        all_block_stats.append(aux['block_stats'])
                        aux_mode = aux['mode']
                        if first_batch_stats is None:
                            first_batch_stats = aux['block_stats']
                    else:
                        _, loss = model(x, y)
                losses.append(float(loss.item()))
                if batch_pbar is not None:
                    batch_pbar.update(1)
                    batch_pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    if batch_pbar is not None:
        batch_pbar.close()

    return {
        'evaluation_mode': evaluation_mode,
        'mean_loss': float(np.mean(losses)),
        'first_batch_stats': first_batch_stats,
        'all_block_stats': all_block_stats,
        'checkpoint_mode': aux_mode,
    }


def save_loss_mode_comparison(rows, output_dir):
    save_json_csv(rows, output_dir, 'loss_mode_comparison', LOSS_FIELDS)


def build_diagnosis_summary(checkpoint_mode, weight_rows, score_rows, loss_rows):
    losses = {row['evaluation_mode']: row['mean_loss'] for row in loss_rows}
    learned_loss = losses['learned']
    uniform_gap = losses['uniform'] - learned_loss
    current_only_gap = losses['current_only'] - learned_loss
    has_residual_aggregators = len(weight_rows) > 0
    max_weight_norm = max((row['l2_norm'] for row in weight_rows), default=0.0)
    mean_normalized_entropy = average_metric(score_rows, 'attn_normalized_entropy')
    if mean_normalized_entropy is None:
        mean_normalized_entropy = average_metric(score_rows, 'mlp_normalized_entropy')
    mean_current_share = average_metric(score_rows, 'attn_current_share')
    mean_history_share = average_metric(score_rows, 'attn_history_share')
    if mean_current_share is None:
        mean_current_share = average_metric(score_rows, 'mlp_current_share')
    if mean_history_share is None:
        mean_history_share = average_metric(score_rows, 'mlp_history_share')

    loss_eps = 5e-4
    near_zero_norm = 1e-2
    nearly_uniform_entropy = 0.98

    if not has_residual_aggregators:
        label = 'baseline_reference'
        message = 'Baseline checkpoint has no residual aggregators, so learned/uniform/current_only are identical by construction.'
    elif (
        max_weight_norm < near_zero_norm
        and mean_normalized_entropy is not None
        and mean_normalized_entropy >= nearly_uniform_entropy
        and abs(uniform_gap) <= loss_eps
        and abs(current_only_gap) <= loss_eps
    ):
        label = 'aggregator_near_init'
        message = 'Residual aggregators stay close to initialization and behave almost like uniform/current-only mixing.'
    elif abs(current_only_gap) <= loss_eps and (max_weight_norm >= near_zero_norm or (mean_normalized_entropy is not None and mean_normalized_entropy < nearly_uniform_entropy)):
        label = 'history_not_helping'
        message = 'Residual aggregators changed, but removing history barely changes loss, so historical states may not help yet.'
    elif abs(uniform_gap) > loss_eps or abs(current_only_gap) > loss_eps:
        label = 'aggregator_affects_loss'
        message = 'Residual aggregators materially change loss relative to at least one ablation mode.'
    else:
        label = 'inconclusive'
        message = 'Residual diagnostics are mixed; run more batches or seeds before drawing a conclusion.'

    return {
        'checkpoint_mode': checkpoint_mode,
        'mode_label': mode_label(checkpoint_mode),
        'classification': label,
        'message': message,
        'losses': losses,
        'loss_gaps': {
            'uniform_minus_learned': uniform_gap,
            'current_only_minus_learned': current_only_gap,
        },
        'weight_metrics': {
            'has_residual_aggregators': has_residual_aggregators,
            'max_l2_norm': max_weight_norm,
        },
        'score_metrics': {
            'mean_normalized_entropy': mean_normalized_entropy,
            'mean_current_share': mean_current_share,
            'mean_history_share': mean_history_share,
        },
        'thresholds': {
            'loss_epsilon': loss_eps,
            'near_zero_l2_norm': near_zero_norm,
            'nearly_uniform_entropy': nearly_uniform_entropy,
        },
    }


def main():
    args = parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num_batches must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    output_dir = Path(args.output_dir or Path(args.out_dir) / 'bar_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device_type, ctx = setup_ctx(args.device, args.dtype)
    checkpoint = load_checkpoint(args.out_dir, args.device)
    dataset = args.dataset or checkpoint.get('config', {}).get('dataset')
    if dataset is None:
        raise ValueError("dataset is not specified and cannot be inferred from checkpoint config")

    model = build_model(checkpoint, args.device)
    data = load_split_memmap(dataset, args.split)

    learned_pass = run_analysis_pass(model, data, args, device_type, ctx, 'learned', collect_stats=True)
    checkpoint_mode = learned_pass['checkpoint_mode'] or 'baseline'
    prefix = mode_prefix(checkpoint_mode)

    weight_rows = summarize_weights(model)
    save_json_csv(weight_rows, output_dir, f'{prefix}_weight_summary', WEIGHT_FIELDS)

    attn_heatmap = scores_to_heatmap(learned_pass['first_batch_stats'], 'attn_scores')
    mlp_heatmap = scores_to_heatmap(learned_pass['first_batch_stats'], 'mlp_scores')
    save_heatmap(attn_heatmap, mlp_heatmap, output_dir, prefix, checkpoint_mode)

    hidden_norm_rows = aggregate_norm_stats(learned_pass['all_block_stats'])
    save_hidden_norms(hidden_norm_rows, output_dir, prefix, checkpoint_mode)

    loss_rows = [{'evaluation_mode': 'learned', 'mean_loss': learned_pass['mean_loss'], 'delta_vs_learned': 0.0}]
    for evaluation_mode in EVAL_MODES[1:]:
        result = run_analysis_pass(model, data, args, device_type, ctx, evaluation_mode, collect_stats=False)
        loss_rows.append({
            'evaluation_mode': evaluation_mode,
            'mean_loss': result['mean_loss'],
            'delta_vs_learned': result['mean_loss'] - learned_pass['mean_loss'],
        })
    save_loss_mode_comparison(loss_rows, output_dir)

    score_rows = summarize_score_behavior(learned_pass['all_block_stats'])
    diagnosis_summary = build_diagnosis_summary(checkpoint_mode, weight_rows, score_rows, loss_rows)
    with open(output_dir / 'diagnosis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(diagnosis_summary, f, indent=2, ensure_ascii=False)

    meta = {
        'out_dir': args.out_dir,
        'dataset': dataset,
        'split': args.split,
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'checkpoint_mode': checkpoint_mode,
        'checkpoint_mode_label': mode_label(checkpoint_mode),
        'evaluation_modes': list(EVAL_MODES),
        'mean_loss': learned_pass['mean_loss'],
    }
    if checkpoint_mode == 'bar':
        meta['layers_per_block'] = model.config.n_layer // model.config.attn_res_num_blocks
        meta['attn_res_num_blocks'] = model.config.attn_res_num_blocks
    with open(output_dir / 'analysis_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {mode_label(checkpoint_mode)} analysis artifacts to {output_dir}")


if __name__ == '__main__':
    main()
