# nanoGPT 学习笔记

日期：2026-03-29

目标：在一天内跑通 nanoGPT 的最小闭环，并看懂数据、配置、模型、训练、采样五条主线。

参考：

1. [plan.md](/home/chen/githubku/nanoGPT/plan.md)
2. [README.md](/home/chen/githubku/nanoGPT/README.md)
3. [interview_prep.md](/home/chen/githubku/nanoGPT/interview_prep.md)

## 今日结论

今天是否完成最小闭环：

- [ ] 已运行 `python data/shakespeare_char/prepare.py`
- [ ] 已运行 `python train.py config/train_shakespeare_char.py`
- [ ] 已运行 `python sample.py --out_dir=out-shakespeare-char`
- [ ] 已看懂 `train.py` 主干
- [ ] 已看懂 `model.py` 主干
- [ ] 已写完这份笔记

一句话总结今天：

```text
在这里写你对 nanoGPT 的一句话理解。
```

## 项目一句话理解

```text
nanoGPT 是一个 ____________________________________________ 的项目。
```

## 项目结构地图

### 我目前对仓库的拆解

```text
数据准备 -> 
配置覆盖 -> 
模型定义 -> 
训练入口 -> 
采样入口 -> 
```

### 关键文件职责

| 文件 | 作用 | 我自己的理解 |
| --- | --- | --- |
| `train.py` | 训练主入口 | |
| `model.py` | GPT 模型定义 | |
| `sample.py` | 采样/推理 | |
| `configurator.py` | 配置覆盖 | |
| `config/train_shakespeare_char.py` | 最小实验配置 | |
| `data/shakespeare_char/prepare.py` | 字符级数据预处理 | |

## 今日环境记录

### 机器与运行环境

```text
Python:
PyTorch:
设备: cpu / cuda / mps
是否支持 torch.compile:
```

### 依赖检查

- [ ] `torch`
- [ ] `numpy`
- [ ] `transformers`
- [ ] `datasets`
- [ ] `tiktoken`

### 我实际使用的命令

```bash
# 在这里记录你实际跑过的命令
```

## 跑通记录

### 1. 数据准备

执行命令：

```bash
python data/shakespeare_char/prepare.py
```

结果：

```text
在这里记录输出、生成了哪些文件、是否成功。
```

我学到的点：

1. 
2. 
3. 

我还不懂的点：

1. 
2. 

### 2. 训练

执行命令：

```bash
python train.py config/train_shakespeare_char.py
```

如果用了缩小版命令，写在这里：

```bash
# 例如 CPU 回退命令
```

训练结果摘要：

```text
loss 变化：
是否保存 checkpoint：
输出目录：
耗时：
```

训练过程中我观察到的现象：

1. 
2. 
3. 

遇到的问题：

1. 
2. 

### 3. 采样

执行命令：

```bash
python sample.py --out_dir=out-shakespeare-char
```

样本输出摘录：

```text
在这里贴一小段你生成的文本
```

我的观察：

1. 
2. 
3. 

## 数据流笔记

请用你自己的话写清楚这条路径：

```text
原始文本
-> 
-> train.bin / val.bin / meta.pkl
-> 
-> get_batch()
-> x, y
-> model(X, Y)
-> loss
```

### `train.bin` / `val.bin` / `meta.pkl` 分别是什么

`train.bin`：

```text
```

`val.bin`：

```text
```

`meta.pkl`：

```text
```

### `x` 和 `y` 是什么关系

```text
在这里解释“下一个 token 预测”。
```

## 配置流笔记

请用你自己的话写清楚这条路径：

```text
train.py 默认参数
-> config/train_shakespeare_char.py
-> 命令行参数
-> 最终训练配置
```

### 我关注的超参数

| 参数 | 当前值 | 作用 | 我的理解 |
| --- | --- | --- | --- |
| `batch_size` | | | |
| `block_size` | | | |
| `n_layer` | | | |
| `n_head` | | | |
| `n_embd` | | | |
| `learning_rate` | | | |
| `max_iters` | | | |
| `dropout` | | | |

### `configurator.py` 在做什么

```text
```

## 训练主线笔记

### 一次训练迭代 10 步

```text
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 
```

### `train.py` 关键函数

#### `get_batch(split)`

它在做什么：

```text
```

输入：

```text
```

输出：

```text
```

为什么重要：

```text
```

#### `estimate_loss()`

它在做什么：

```text
```

#### `get_lr(it)`

它在做什么：

```text
```

#### checkpoint 保存逻辑

checkpoint 里有什么：

```text
model:
optimizer:
model_args:
iter_num:
best_val_loss:
config:
```

## 模型主线笔记

### GPT 结构总览

```text
idx
-> token embedding
-> position embedding
-> 
-> N x Block
-> 
-> logits
-> loss / next token
```

### `Block` 由什么组成

```text
```

### `CausalSelfAttention` 在做什么

```text
```

### causal mask 为什么重要

```text
```

### `GPT.forward()` 在做什么

```text
```

### `generate()` 在做什么

```text
```

## 推理与采样笔记

### `sample.py` 主流程

```text
checkpoint
-> 
-> encode prompt
-> 
-> generate
-> decode
-> 文本输出
```

### 参数理解

| 参数 | 作用 | 我的理解 |
| --- | --- | --- |
| `start` | | |
| `num_samples` | | |
| `max_new_tokens` | | |
| `temperature` | | |
| `top_k` | | |

### `temperature` 和 `top_k` 的影响

```text
```

## 我今天真正看懂的 5 个点

1. 
2. 
3. 
4. 
5. 

## 我还没看懂的 5 个点

1. 
2. 
3. 
4. 
5. 

## 明天如果继续，我会看什么

1. 
2. 
3. 

## 常见卡点记录

### 报错 1

现象：

```text
```

原因：

```text
```

解决：

```text
```

### 报错 2

现象：

```text
```

原因：

```text
```

解决：

```text
```

## 最终复盘

### 我现在能不能脱离代码讲清楚这个项目

- [ ] 能
- [ ] 还不能

### 我是否能回答下面这些问题

- [ ] 数据是怎么变成 token id 的
- [ ] `x` 和 `y` 是怎么构造的
- [ ] 一次训练迭代做了什么
- [ ] `Block` 由什么组成
- [ ] `sample.py` 是怎么把 checkpoint 变成文本的

### 最后一句总结

```text
我现在觉得 nanoGPT 最值得学习的点是：____________________
```

## 面试表达准备

### 30 秒版本

```text
在这里写你自己的 30 秒项目介绍。
```

### 90 秒版本

```text
在这里写你自己的 90 秒项目介绍。
```

### 我最想让面试官记住的 3 个点

1. 
2. 
3. 

### 我提出的 3 个研究问题

#### 问题 1

问题：

```text
```

为什么值得研究：

```text
```

我会怎么做实验：

```text
```

看什么指标：

```text
```

#### 问题 2

问题：

```text
```

为什么值得研究：

```text
```

我会怎么做实验：

```text
```

看什么指标：

```text
```

#### 问题 3

问题：

```text
```

为什么值得研究：

```text
```

我会怎么做实验：

```text
```

看什么指标：

```text
```

### 面试高频追问答题卡

#### Q: 你为什么选这个项目

```text
```

#### Q: 这个项目最核心的技术点是什么

```text
```

#### Q: 你做了哪些实验

```text
```

#### Q: 这个项目的局限是什么

```text
```

#### Q: 如果再给你一周，你会继续做什么

```text
```
