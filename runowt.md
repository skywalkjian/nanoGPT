# OWT 上运行 GPT-2 128M（仓库对应 124M）对比 baseline / BAR

日期：2026-04-04

说明：这份文档专门给 `OpenWebText` 上的 GPT-2 small 对照实验用。仓库里的对应配置实际是 GPT-2 `124M` 档位，也就是常说的“128M 左右”那一档：

- `n_layer=12`
- `n_head=12`
- `n_embd=768`

这里直接复用现有的 [`config/train_gpt2_124m_pilot.py`](config/train_gpt2_124m_pilot.py) 作为基础，只把输出目录、运行名和训练步数改成这次需要的 baseline / BAR 3500 step 对照。

以下命令本次没有真实执行，是根据仓库当前脚本和配置整理出的推荐跑法。

## 0. 先准备数据和依赖

先安装依赖：

```bash
python -m pip install torch numpy transformers datasets tiktoken tqdm tensorboard matplotlib
```

创建日志目录：

```bash
mkdir -p logs
```

准备 OpenWebText：

```bash
python data/openwebtext/prepare.py
```

生成完成后，检查核心文件：

```bash
ls -lh data/openwebtext/train.bin data/openwebtext/val.bin
```

仓库里已有说明：

- `train.bin` 大约 `17GB`
- `val.bin` 大约 `8.5MB`
- Hugging Face 缓存通常也会占比较多空间

所以建议至少预留 `80GB+` 可用磁盘空间。

## 1. 这次实验的统一约定

基础超参直接沿用 [`config/train_gpt2_124m_pilot.py`](config/train_gpt2_124m_pilot.py)：

- 数据集：`openwebtext`
- `block_size=512`
- `batch_size=4`
- `gradient_accumulation_steps=8`
- 模型规模：`12 x 12 x 768`

这次只改以下几点：

- baseline 和 BAR 分开写到不同 `out_dir`
- TensorBoard run name 分开
- 训练只跑到 `step 3500`
- 学习率衰减也同步改到 `3500`

推荐目录名：

```text
out-gpt2-124m-3500-baseline
out-gpt2-124m-3500-bar
```

## 2. baseline：3500 step

```bash
python train.py config/train_gpt2_124m_pilot.py \
  --out_dir=out-gpt2-124m-3500-baseline \
  --tensorboard_run_name=gpt2-124m-3500-baseline \
  --wandb_run_name=gpt2-124m-3500-baseline \
  --max_iters=3500 \
  --lr_decay_iters=3500 \
  --seed=1337 | tee logs/train_owt_124m_3500_baseline.log
```

## 3. BAR：3500 step

BAR 版本和 baseline 保持同一套基础超参，只额外打开 block attention residuals：

```bash
python train.py config/train_gpt2_124m_pilot.py \
  --out_dir=out-gpt2-124m-3500-bar \
  --tensorboard_run_name=gpt2-124m-3500-bar \
  --wandb_run_name=gpt2-124m-3500-bar \
  --max_iters=3500 \
  --lr_decay_iters=3500 \
  --seed=1337 \
  --use_block_attention_residuals=True \
  --attn_res_num_blocks=3 \
  --attn_res_use_rmsnorm=True | tee logs/train_owt_124m_3500_bar.log
```

如果你还想把训练期 residual diagnostics 一起写进 TensorBoard，可以只在 BAR 这一路额外加上：

```bash
--residual_stats_log=True
```

但这会增加额外诊断开销；如果你当前只想先看 baseline vs BAR 的 loss 曲线，可以先不开。

## 4. 看 baseline vs BAR 的 TensorBoard

```bash
tensorboard \
  --logdir_spec baseline:out-gpt2-124m-3500-baseline/tensorboard/gpt2-124m-3500-baseline,bar:out-gpt2-124m-3500-bar/tensorboard/gpt2-124m-3500-bar \
  --port 6006
```

重点看这些标量：

- `eval/train_loss`
- `eval/val_loss`
- `train/loss`
- `perf/iter_ms`
- `perf/mfu`

## 5. 跑完后导出结果图

当前 3500 step 对照结果的推荐做法，是直接基于 TensorBoard event 导出结果图，而不是依赖额外的 BAR 机制分析脚本。

```bash
python scripts/export_owt_result_figures.py \
  --baseline_dir out-gpt2-124m-3500-baseline \
  --bar_dir out-gpt2-124m-3500-bar \
  --output_dir assets/showcase/owt_3500
```

这会生成：

```text
assets/showcase/owt_3500/train_val_curves.png
assets/showcase/owt_3500/val_gap_curve.png
assets/showcase/owt_3500/efficiency_summary.png
assets/showcase/owt_3500/metrics_summary.json
```

其中：

- `train_val_curves.png` 对比 `eval/train_loss` 和 `eval/val_loss`
- `val_gap_curve.png` 展示 `baseline_val - bar_val`
- `efficiency_summary.png` 比较 steady-state 窗口 `3000..3490` 的 `iter_ms` 与 `mfu`
- `metrics_summary.json` 汇总关键数值，便于 README 直接引用

## 6. 如果训练中断，继续跑

### 继续 baseline

```bash
python train.py config/train_gpt2_124m_pilot.py \
  --init_from=resume \
  --out_dir=out-gpt2-124m-3500-baseline \
  --tensorboard_run_name=gpt2-124m-3500-baseline \
  --wandb_run_name=gpt2-124m-3500-baseline \
  --max_iters=3500 \
  --lr_decay_iters=3500
```

### 继续 BAR

恢复 BAR 时，也要把 BAR 相关开关带上，避免和 checkpoint 结构不一致：

```bash
python train.py config/train_gpt2_124m_pilot.py \
  --init_from=resume \
  --out_dir=out-gpt2-124m-3500-bar \
  --tensorboard_run_name=gpt2-124m-3500-bar \
  --wandb_run_name=gpt2-124m-3500-bar \
  --max_iters=3500 \
  --lr_decay_iters=3500 \
  --use_block_attention_residuals=True \
  --attn_res_num_blocks=3 \
  --attn_res_use_rmsnorm=True
```

## 7. 一个容易混淆的小点

仓库当前训练循环是先按 `iter_num % eval_interval == 0` 做评估，再在循环末尾用 `iter_num > max_iters` 判断结束。

所以这里设置：

```bash
--max_iters=3500
```

你会在日志里看到最后一个评估点和 checkpoint 点是 `step 3500`。这份文档里说的“只跑 3500 步”，就是按这个仓库当前的记法来对齐。

## 8. 最短执行顺序

如果你只想快速开始，最短就是这 5 步：

1. `python data/openwebtext/prepare.py`
2. 跑 baseline 3500 step
3. 跑 BAR 3500 step
4. 用 TensorBoard 对比两条曲线
5. 运行 `python scripts/export_owt_result_figures.py --baseline_dir out-gpt2-124m-3500-baseline --bar_dir out-gpt2-124m-3500-bar --output_dir assets/showcase/owt_3500`
