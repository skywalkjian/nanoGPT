# nanoGPT x Residual Attention 运行指南

日期：2026-03-29

说明：本文件整理 baseline / BAR / FAR 的推荐命令与诊断输出约定。以下命令在这次交付中**没有真实执行**，用于你后续手动跑 Shakespeare 小模型训练与分析。

## 0.data/env 准备
运行 `python data/shakespeare_char/prepare.py`
python -m pip install tensorboard matplotlib tqdm tiktoken datasets

### OpenWebText (OWT) 下载与预处理

`nanoGPT` 在大数据集配置里使用的数据集名是 `openwebtext`，对应目录是：

```text
data/openwebtext
```

先安装依赖：

```bash
python -m pip install datasets tiktoken tqdm numpy
```

然后在仓库根目录执行：

```bash
python data/openwebtext/prepare.py
```

这个脚本会做两件事：

1. 通过 Hugging Face `datasets` 下载 `openwebtext`
2. 在 `data/openwebtext/` 下生成训练用二进制文件

生成完成后，核心产物是：

```text
data/openwebtext/train.bin
data/openwebtext/val.bin
```

脚本中的已知体积大致是：

```text
Hugging Face cache: 约 54GB
train.bin: 约 17GB
val.bin: 约 8.5MB
```

所以建议至少预留 `80GB+` 可用磁盘空间。

如果你想把 Hugging Face 缓存放到别的盘，可以先指定：

```bash
export HF_HOME=/your/bigger/disk/hf
python data/openwebtext/prepare.py
```

下载和预处理结束后，可以简单检查：

```bash
ls -lh data/openwebtext/train.bin data/openwebtext/val.bin
```

确认 `train.bin` / `val.bin` 已生成后，就可以直接跑 OpenWebText 124M pilot：

```bash
python train.py config/train_gpt2_124m_pilot.py
```

如果你想保留日志：

```bash
python train.py config/train_gpt2_124m_pilot.py | tee logs/train_owt_124m_pilot.log
```
## 1. 推荐目录约定

baseline 输出目录：

```text
out-shakespeare-char
```

BAR 输出目录：

```text
out-shakespeare-char-bar
```

FAR 输出目录：

```text
out-shakespeare-char-far
```

分析输出目录：

```text
out-shakespeare-char-bar/bar_analysis
```

如果你想留日志，推荐：

```bash
mkdir -p logs
```

如果你想看 TensorBoard 曲线，额外安装：

```bash
python -m pip install tensorboard
```

## 2. Shakespeare 单 seed 诊断训练

推荐先统一使用 `seed=1337`，并打开训练期残差诊断：

```bash
--seed=1337 --residual_stats_log=True
```

### baseline

标准命令：

```bash
python train.py config/train_shakespeare_char.py --seed=1337 --residual_stats_log=True
```

建议保存日志：

```bash
python train.py config/train_shakespeare_char.py --seed=1337 --residual_stats_log=True | tee logs/train_baseline.log
```

TensorBoard 默认会写到：

```text
out-shakespeare-char/tensorboard/baseline
```

如果你机器较弱，可以先用 CPU/关闭 compile 做最小验证：

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --seed=1337 --residual_stats_log=True
```

### BAR

标准命令：

```bash
python train.py config/train_shakespeare_char_bar.py --seed=1337 --residual_stats_log=True
```

建议保存日志：

```bash
python train.py config/train_shakespeare_char_bar.py --seed=1337 --residual_stats_log=True | tee logs/train_bar.log
```

TensorBoard 默认会写到：

```text
out-shakespeare-char-bar/tensorboard/bar
```

如果你只想先确认 BAR 路径能启动：

```bash
python train.py config/train_shakespeare_char_bar.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --seed=1337 --residual_stats_log=True
```

### FAR

标准命令：

```bash
python train.py config/train_shakespeare_char_far.py --seed=1337 --residual_stats_log=True
```

建议保存日志：

```bash
python train.py config/train_shakespeare_char_far.py --seed=1337 --residual_stats_log=True | tee logs/train_far.log
```

TensorBoard 默认会写到：

```text
out-shakespeare-char-far/tensorboard/far
```

如果你只想先确认 FAR 路径能启动：

```bash
python train.py config/train_shakespeare_char_far.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --seed=1337 --residual_stats_log=True
```

## 3. Shakespeare 采样

### baseline

```bash
python sample.py --out_dir=out-shakespeare-char --start=$'\n' --num_samples=5 --max_new_tokens=300
```

如果要保存输出：

```bash
python sample.py --out_dir=out-shakespeare-char --start=$'\n' --num_samples=5 --max_new_tokens=300 | tee logs/sample_baseline.log
```

### BAR

```bash
python sample.py --out_dir=out-shakespeare-char-bar --start=$'\n' --num_samples=5 --max_new_tokens=300
```

如果要保存输出：

```bash
python sample.py --out_dir=out-shakespeare-char-bar --start=$'\n' --num_samples=5 --max_new_tokens=300 | tee logs/sample_bar.log
```

### FAR

```bash
python sample.py --out_dir=out-shakespeare-char-far --start=$'\n' --num_samples=5 --max_new_tokens=300
```

如果要保存输出：

```bash
python sample.py --out_dir=out-shakespeare-char-far --start=$'\n' --num_samples=5 --max_new_tokens=300 | tee logs/sample_far.log
```

## 4. 统一分析

`analyze_bar.py` 现在兼容 baseline / BAR / FAR checkpoint，并且会额外比较：

- `learned`
- `uniform`
- `current_only`

先确保有画图库和进度条依赖：

```bash
python -m pip install matplotlib tqdm
```

标准分析命令：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8
```

baseline 示例：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8
```

FAR 示例：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char-far --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8
```

如果想指定输出目录：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8 --output_dir=out-shakespeare-char-bar/bar_analysis
```

分析脚本会生成这些核心产物：

```text
<mode>_weight_summary.json
<mode>_weight_summary.csv
<mode>_scores_heatmap.png
<mode>_hidden_norms.csv
<mode>_hidden_norms.png
loss_mode_comparison.json
loss_mode_comparison.csv
diagnosis_summary.json
analysis_meta.json
```

其中 `<mode>` 会按 checkpoint 自动变成：

```text
baseline / bar / far
```

## 5. Shakespeare 三 seed 对照

如果单 seed 显示机制确实活着，但 loss 差异不明显，再跑：

```bash
python train.py config/train_shakespeare_char.py --seed=1337 --residual_stats_log=True
python train.py config/train_shakespeare_char.py --seed=2024 --residual_stats_log=True
python train.py config/train_shakespeare_char.py --seed=3407 --residual_stats_log=True
```

BAR / FAR 同理，只替换配置文件即可。

## 6. resume 对照实验

如果你要继续训练 baseline：

```bash
python train.py config/train_shakespeare_char.py --init_from=resume --out_dir=out-shakespeare-char
```

如果你要继续训练 BAR：

```bash
python train.py config/train_shakespeare_char_bar.py --init_from=resume --out_dir=out-shakespeare-char-bar
```

如果你要继续训练 FAR：

```bash
python train.py config/train_shakespeare_char_far.py --init_from=resume --out_dir=out-shakespeare-char-far
```

## 7. 诊断判读建议

重点先看 4 件事：

1. `*_weight_summary.*` 里的聚合器范数有没有明显离开 0
2. `*_scores_heatmap.png` 是否从接近均匀逐渐学出偏好
3. `loss_mode_comparison.*` 里 `learned` 是否优于 `uniform/current_only`
4. `diagnosis_summary.json` 的分类是否落在 `aggregator_near_init` / `history_not_helping` / `aggregator_affects_loss`

## 8. 面试前最小必跑清单

建议至少手动完成下面 6 条：

1. `python data/shakespeare_char/prepare.py`
2. `python train.py config/train_shakespeare_char.py --seed=1337 --residual_stats_log=True`
3. `python train.py config/train_shakespeare_char_bar.py --seed=1337 --residual_stats_log=True`
4. `python train.py config/train_shakespeare_char_far.py --seed=1337 --residual_stats_log=True`
5. `python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char`
6. `python analyze_bar.py --out_dir=out-shakespeare-char-far --dataset=shakespeare_char`

## 9. TensorBoard 查看

同时看 baseline / BAR / FAR：

```bash
tensorboard --logdir_spec baseline:out-shakespeare-char/tensorboard/baseline,bar:out-shakespeare-char-bar/tensorboard/bar,far:out-shakespeare-char-far/tensorboard/far --port 6006

```

只看 baseline：

```bash
tensorboard --logdir out-shakespeare-char/tensorboard --port 6006
```

只看 BAR：

```bash
tensorboard --logdir out-shakespeare-char-bar/tensorboard --port 6006
```

只看 FAR：

```bash
tensorboard --logdir out-shakespeare-char-far/tensorboard --port 6006
```

## 10. 结果记录建议

建议你边跑边填：

1. [bar_analysis.md](/home/chen/githubku/nanoGPT/bar_analysis.md)
2. [bar_interview.md](/home/chen/githubku/nanoGPT/bar_interview.md)
3. [notes.md](/home/chen/githubku/nanoGPT/notes.md)
