# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

---

**更新（2025 年 11 月）** nanoGPT 现在有一个更新、更完善的“亲戚”项目 [nanochat](https://github.com/karpathy/nanochat)。你很可能其实是想使用/寻找 nanochat。nanoGPT（本仓库）现在已经比较老且已弃用，但我会保留它作为历史资料。

---

这是一个用于训练/微调中等规模 GPT 的最简单、最快速仓库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重写版本，优先考虑“实战能力”而非教学性。项目仍在积极开发中，但目前 `train.py` 可以在 OpenWebText 上复现 GPT-2（124M）：在单个 8xA100 40GB 节点上训练约 4 天即可完成。代码本身直白且可读：`train.py` 是约 300 行的训练循环模板，`model.py` 是约 300 行的 GPT 模型定义（可选加载 OpenAI 的 GPT-2 权重）。就这些。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，你可以很轻松地按需改造、从零训练新模型，或在预训练检查点上做微调（例如当前可用的较大起点是 OpenAI 的 GPT-2 1.3B）。

## 安装

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖项：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers`：huggingface transformers <3（用于加载 GPT-2 检查点）
- `datasets`：huggingface datasets <3（如果你要下载并预处理 OpenWebText）
- `tiktoken`：OpenAI 的高性能 BPE 实现 <3
- `wandb`：可选日志记录 <3
- `tqdm`：进度条 <3

## 快速开始

如果你不是深度学习专业人士，只是想快速体验一下 GPT 的魔力，那么最快方式是先在莎士比亚文本上训练一个字符级 GPT。首先下载并处理数据：

```sh
python data/shakespeare_char/prepare.py
```

这会在该目录下生成 `train.bin` 和 `val.bin`。接下来就可以训练 GPT 了，模型规模取决于你的算力：

**我有 GPU**。太好了，可以直接用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 提供的配置训练一个“小 GPT”：

```sh
python train.py config/train_shakespeare_char.py
```

如果你看配置，会发现模型上下文长度最多 256 字符，384 通道，6 层 Transformer，每层 6 个头。在一张 A100 上，这个训练大约 3 分钟，最佳验证损失约 1.4697。按配置，检查点会写入 `--out_dir` 指定的目录 `out-shakespeare-char`。训练完成后可以这样采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

会生成一些示例文本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol `¯\_(ツ)_/¯`。对一个只训练 3 分钟的字符级模型来说还不错。更好的结果通常可以通过在该数据集上微调预训练 GPT-2 获得（见后面的微调部分）。

**我只有一台 MacBook**（或其他普通电脑）。没关系，也可以训练，只需要把配置调小。建议安装最新的 PyTorch nightly（可在 [这里](https://pytorch.org/get-started/locally/) 选择），通常会更高效。即便不用 nightly，也可以这样跑：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

这里因为在 CPU 上训练，必须设置 `--device=cpu`，并关闭 PyTorch 2.0 编译 `--compile=False`。评估次数改小（`--eval_iters=20`，原为 200）以加快速度；上下文长度降到 64（原 256）；batch size 降到 12（原 64）。模型也缩小为 4 层、4 头、128 维嵌入，并把训练迭代降到 2000（学习率衰减迭代 `--lr_decay_iters` 通常也设到 max_iters 附近）。由于模型小，正则也放松（`--dropout=0.0`）。

这样大概 3 分钟能跑完，损失约 1.88，采样质量会差一些，但依然很有趣：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

可能生成类似：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

对 CPU 上 3 分钟训练来说，这已经能看到正确的“字符风格”了。如果你愿意等更久，可以继续调超参数、增大网络、增大上下文长度（`--block_size`）、延长训练时长等。

最后，如果你使用 Apple Silicon MacBook 且 PyTorch 版本较新，记得加 `--device=mps`（Metal Performance Shaders）。这样会使用芯片上的 GPU，通常可明显提速（2-3 倍），并支持更大网络。见 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## 复现 GPT-2

如果你更偏专业，可能更关心复现 GPT-2 结果。首先需要对数据集分词，这里使用 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)（OpenAI 私有 WebText 的开源复现）：

```sh
python data/openwebtext/prepare.py
```

这会下载并分词 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集，生成 `train.bin` 和 `val.bin`，其中保存 GPT-2 BPE token id 序列（uint16 原始字节）。然后开始训练。要复现 GPT-2（124M），至少建议使用 8x A100 40GB，运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

该训练大约运行 4 天，损失会降到 ~2.85。作为参考，GPT-2 模型直接在 OWT 上评估时验证损失约 3.11，但在 OWT 上再微调后会降到 ~2.85（明显存在领域差异），两者就基本匹配了。

如果你在集群里有多节点，可以这样跑（例如 2 节点）：

```sh
# 在第一台（主）节点运行，示例 IP 123.456.123.456：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# 在工作节点运行：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

建议先测试互联带宽（例如 `iperf3`）。特别是如果没有 Infiniband，请在启动命令前加 `NCCL_IB_DISABLE=1`。多节点训练能跑，但很可能会很慢。默认会周期性把检查点写入 `--out_dir`。训练后可直接 `python sample.py` 采样。

如果只在单卡上训练，直接运行 `python train.py` 即可。建议查看全部参数；脚本尽量保持可读、可改、透明。你大概率需要按自身需求调参。

## 基线结果

OpenAI GPT-2 检查点可用于得到 OpenWebText 基线：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

可得到如下 train/val 损失：

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

但要注意：GPT-2 训练数据是封闭且未公开的 WebText，而 OpenWebText 只是开源复现版本，存在数据域差异。事实上，把 GPT-2（124M）检查点直接在 OWT 上微调一段时间，损失能到 ~2.85，这个值才是更合理的复现实验基线。

## 微调

微调和训练本质一样，只是从预训练模型初始化，并使用更小学习率。示例：进入 `data/shakespeare` 运行 `prepare.py` 下载 tiny shakespeare 并处理成 `train.bin`、`val.bin`，分词使用 GPT-2 的 OpenAI BPE。这个数据处理几秒就能完成。微调也可以很快，例如单卡几分钟。示例命令：

```sh
python train.py config/finetune_shakespeare.py
```

这会加载 `config/finetune_shakespeare.py` 中的参数覆盖（我自己没怎么精调）。核心思路是通过 `init_from` 从 GPT-2 检查点初始化，然后正常训练，只是训练更短、学习率更小。

如果显存不足，尝试减小模型规模（`{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）或减小 `block_size`（上下文长度）。最佳检查点（最低验证损失）会保存在配置的 `out_dir` 中，默认例如 `out-shakespeare`。随后可通过 `sample.py --out_dir=out-shakespeare` 采样：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

好家伙，GPT 有点“黑暗戏剧化”了。我没有细调太多超参数，欢迎你自己继续尝试。

## 采样 / 推理

使用 `sample.py` 可从 OpenAI 发布的预训练 GPT-2 模型采样，也可从你自己训练的模型采样。例如从最大的 `gpt2-xl` 采样：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果从自己训练的模型采样，使用 `--out_dir` 指向对应目录。你也可以从文件读取 prompt，例如：`python sample.py --start=FILE:prompt.txt`。

## 效率说明

如果你想做简单的模型 benchmark 和 profiling，`bench.py` 很有用。它与 `train.py` 主训练循环中的核心部分一致，但省略了其他复杂逻辑。

注意代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在原文写作时（2022-12-29），这使得 `torch.compile()` 可用于 nightly 版本，一行代码就有明显收益，例如迭代时间可从 ~250ms/iter 降到 135ms/iter。PyTorch 团队干得漂亮！

## 待办

- 研究并加入 FSDP 替代 DDP
- 在标准评测上做 zero-shot 困惑度评估（如 LAMBADA、HELM 等）
- 继续调优微调脚本（当前超参可能不理想）
- 支持训练中线性增大 batch size
- 引入其他位置/嵌入方案（rotary、alibi）
- 在检查点中将优化器 buffer 与模型参数分离
- 增加更多网络健康日志（如梯度裁剪触发、梯度幅值）
- 针对更优初始化做更多实验

## 故障排查

默认启用 PyTorch 2.0（即 `torch.compile`）。这仍较新且实验性较强，不是所有平台都可用（例如 Windows）。如果你遇到相关错误，可加 `--compile=False` 关闭。这样会慢一些，但通常能先跑起来。

如果你想了解该仓库、GPT 与语言模型的更多背景，可以看我的 [Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)。其中 [GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) 很受欢迎，前提是你对语言建模已有一点基础。

如需更多讨论，欢迎来 Discord 的 **#nanoGPT**：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

nanoGPT 的所有实验都由 [Lambda Labs](https://lambdalabs.com) 的 GPU 提供支持。感谢 Lambda Labs 对 nanoGPT 的赞助！
