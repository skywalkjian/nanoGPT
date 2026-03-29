"""
Minimal analysis utility for nanoGPT Block Attention Residuals checkpoints.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Block Attention Residual checkpoints.")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory that contains ckpt.pt')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name under data/. Defaults to checkpoint config.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split to analyze.')
    parser.add_argument('--num_batches', type=int, default=4, help='Number of batches to average for norm statistics.')
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


def save_weight_summary(rows, output_dir):
    json_path = output_dir / 'bar_weight_summary.json'
    csv_path = output_dir / 'bar_weight_summary.csv'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer_idx', 'module', 'mean', 'std', 'min', 'max', 'l2_norm'])
        writer.writeheader()
        writer.writerows(rows)


def scores_to_heatmap(block_stats, score_key):
    rows = []
    for stats in block_stats:
        scores = stats.get(score_key)
        if scores is None:
            rows.append(np.zeros(1, dtype=np.float32))
            continue
        row = scores.mean(dim=(1, 2)).numpy()
        rows.append(row.astype(np.float32))
    max_depth = max(len(row) for row in rows)
    heatmap = np.full((len(rows), max_depth), np.nan, dtype=np.float32)
    for layer_idx, row in enumerate(rows):
        heatmap[layer_idx, :len(row)] = row
    return heatmap


def save_heatmap(attn_heatmap, mlp_heatmap, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, heatmap, title in [
        (axes[0], attn_heatmap, 'Attention BAR Scores'),
        (axes[1], mlp_heatmap, 'MLP BAR Scores'),
    ]:
        image = ax.imshow(heatmap, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Depth candidate')
        ax.set_ylabel('Layer index')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / 'bar_scores_heatmap.png', dpi=200)
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


def save_hidden_norms(rows, output_dir):
    csv_path = output_dir / 'hidden_norms.csv'
    png_path = output_dir / 'hidden_norms.png'
    fieldnames = ['layer_idx', 'input_norm', 'attn_agg_norm', 'post_attn_norm', 'mlp_agg_norm', 'output_norm']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    layer_indices = [row['layer_idx'] for row in rows]
    for key in fieldnames[1:]:
        values = [row[key] for row in rows]
        ax.plot(layer_indices, values, marker='o', label=key)
    ax.set_title('BAR Hidden Norm Statistics')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean L2 norm')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


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
    rng = torch.Generator(device='cpu')
    rng.manual_seed(args.seed)

    device_type, ctx = setup_ctx(args.device, args.dtype)
    checkpoint = load_checkpoint(args.out_dir, args.device)
    dataset = args.dataset or checkpoint.get('config', {}).get('dataset')
    if dataset is None:
        raise ValueError("dataset is not specified and cannot be inferred from checkpoint config")

    model = build_model(checkpoint, args.device)
    if not model.config.use_block_attention_residuals:
        raise ValueError("This checkpoint does not enable use_block_attention_residuals; analyze_bar.py is intended for BAR checkpoints.")

    data = load_split_memmap(dataset, args.split)
    weight_rows = summarize_weights(model)
    save_weight_summary(weight_rows, output_dir)

    all_block_stats = []
    first_batch_stats = None
    losses = []
    batch_pbar = None
    if args.use_tqdm and tqdm is not None:
        batch_pbar = tqdm(total=args.num_batches, desc='analyze', dynamic_ncols=True)
    elif args.use_tqdm and tqdm is None:
        print("WARNING: tqdm is not installed; continuing without progress bar. Install it with `pip install tqdm`.")

    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            x, y = get_batch(data, model.config.block_size, args.batch_size, args.device, device_type, rng)
            with ctx:
                _, loss, aux = model(x, y, return_bar_stats=True)
            losses.append(float(loss.item()))
            all_block_stats.append(aux['block_stats'])
            if batch_idx == 0:
                first_batch_stats = aux['block_stats']
            if batch_pbar is not None:
                batch_pbar.update(1)
                batch_pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    if batch_pbar is not None:
        batch_pbar.close()

    attn_heatmap = scores_to_heatmap(first_batch_stats, 'attn_scores')
    mlp_heatmap = scores_to_heatmap(first_batch_stats, 'mlp_scores')
    save_heatmap(attn_heatmap, mlp_heatmap, output_dir)

    hidden_norm_rows = aggregate_norm_stats(all_block_stats)
    save_hidden_norms(hidden_norm_rows, output_dir)

    meta = {
        'out_dir': args.out_dir,
        'dataset': dataset,
        'split': args.split,
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'mean_loss': float(np.mean(losses)),
        'layers_per_block': model.config.n_layer // model.config.attn_res_num_blocks,
        'attn_res_num_blocks': model.config.attn_res_num_blocks,
    }
    with open(output_dir / 'analysis_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved BAR analysis artifacts to {output_dir}")


if __name__ == '__main__':
    main()
