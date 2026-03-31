# nanoGPT x Block Attention Residuals 运行指南

日期：2026-03-29

说明：本文件只整理运行命令与建议输出约定。以下命令在这次交付中**没有真实执行**，用于你后续手动跑 baseline、BAR、采样和分析。

## 0.data/env 准备
运行 `python data/shakespeare_char/prepare.py`
python -m pip install tensorboard matplotlib tqdm tiktoken datasets
## 1. 推荐目录约定

baseline 输出目录：

```text
out-shakespeare-char
```

BAR 输出目录：

```text
out-shakespeare-char-bar
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

## 2. baseline 训练

标准命令：

```bash
python train.py config/train_shakespeare_char.py
```

建议保存日志：

```bash
python train.py config/train_shakespeare_char.py | tee logs/train_baseline.log
```

TensorBoard 默认会写到：

```text
out-shakespeare-char/tensorboard/baseline
```

如果你机器较弱，可以先用 CPU/关闭 compile 做最小验证：

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1
```

## 3. BAR 训练

标准命令：

```bash
python train.py config/train_shakespeare_char_bar.py
```

建议保存日志：

```bash
python train.py config/train_shakespeare_char_bar.py | tee logs/train_bar.log
```

TensorBoard 默认会写到：

```text
out-shakespeare-char-bar/tensorboard/bar
```

如果你只想先确认 BAR 路径能启动：

```bash
python train.py config/train_shakespeare_char_bar.py --device=cpu --compile=False --eval_iters=20 --log_interval=1
```

## 4. baseline 采样

```bash
python sample.py --out_dir=out-shakespeare-char --start=$'\n' --num_samples=5 --max_new_tokens=300
```

如果要保存输出：

```bash
python sample.py --out_dir=out-shakespeare-char --start=$'\n' --num_samples=5 --max_new_tokens=300 | tee logs/sample_baseline.log
```

## 5. BAR 采样

```bash
python sample.py --out_dir=out-shakespeare-char-bar --start=$'\n' --num_samples=5 --max_new_tokens=300
```

如果要保存输出：

```bash
python sample.py --out_dir=out-shakespeare-char-bar --start=$'\n' --num_samples=5 --max_new_tokens=300 | tee logs/sample_bar.log
```

## 6. BAR 分析

先确保有画图库和进度条依赖：

```bash
python -m pip install matplotlib tqdm
```

标准分析命令：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8
```

如果想指定输出目录：

```bash
python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char --split=val --num_batches=4 --batch_size=8 --output_dir=out-shakespeare-char-bar/bar_analysis
```

分析脚本会生成的最小三件套：

```text
bar_weight_summary.json
bar_weight_summary.csv
bar_scores_heatmap.png
hidden_norms.csv
hidden_norms.png
analysis_meta.json
```

## 7. resume 对照实验

如果你要继续训练 baseline：

```bash
python train.py config/train_shakespeare_char.py --init_from=resume --out_dir=out-shakespeare-char
```

如果你要继续训练 BAR：

```bash
python train.py config/train_shakespeare_char_bar.py --init_from=resume --out_dir=out-shakespeare-char-bar
```

## 8. 面试前最小必跑清单

建议至少手动完成下面 4 条：

1. `python data/shakespeare_char/prepare.py`
2. `python train.py config/train_shakespeare_char.py`
3. `python train.py config/train_shakespeare_char_bar.py`
4. `python analyze_bar.py --out_dir=out-shakespeare-char-bar --dataset=shakespeare_char`

## 9. TensorBoard 查看

同时看 baseline 和 BAR：

```bash
tensorboard --logdir_spec baseline:out-shakespeare-char/tensorboard/baseline,bar:out-shakespeare-char-bar/tensorboard/bar --port 6006

```

只看 baseline：

```bash
tensorboard --logdir out-shakespeare-char/tensorboard --port 6006
```

只看 BAR：

```bash
tensorboard --logdir out-shakespeare-char-bar/tensorboard --port 6006
```

## 10. 结果记录建议

建议你边跑边填：

1. [bar_analysis.md](/home/chen/githubku/nanoGPT/bar_analysis.md)
2. [bar_interview.md](/home/chen/githubku/nanoGPT/bar_interview.md)
3. [notes.md](/home/chen/githubku/nanoGPT/notes.md)
