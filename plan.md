# nanoGPT 一日完成计划

适用日期：2026-03-29

## 先说结论

如果你说的“完成这个项目”是指：

1. 搞懂仓库主干结构
2. 跑通一个端到端最小闭环
3. 能清楚解释训练、模型、采样三条主线
4. 留下一份自己的阅读和实验记录

那么一天是可行的。

如果你说的“完成这个项目”是指：

1. 复现 README 里的 OpenWebText + GPT-2 124M 训练
2. 跑出接近论文或 README 中的大规模结果
3. 做完整多机多卡训练

那么正常情况下 2026-03-29 这一天内不现实，除非你已经有现成的 8xA100 级别环境，并且数据也准备好了。

所以这份计划默认采用：

## 明日目标定义

明天结束前，你需要交付以下 6 个结果：

1. 能讲清楚仓库的 5 个模块：数据、配置、模型、训练、采样。
2. 本地成功运行字符级 Shakespeare 最小实验。
3. 至少看懂 `train.py` 和 `model.py` 的主干逻辑。
4. 能解释一次训练迭代里发生了什么。
5. 能从 checkpoint 生成文本并理解采样参数含义。
6. 输出一份你自己的学习记录，至少包含“架构图/流程图 + 核心函数摘要 + 跑通命令 + 遇到的问题”。

## 最小成功标准

只要你明天完成下面这些，就算这次项目冲刺成功：

1. 运行 `python data/shakespeare_char/prepare.py`
2. 运行 `python train.py config/train_shakespeare_char.py`
3. 运行 `python sample.py --out_dir=out-shakespeare-char`
4. 能解释 `get_batch()`、`GPT.forward()`、`generate()` 三个关键路径
5. 写出一页自己的理解笔记

## 时间预算

建议总时长：10 到 12 小时

建议节奏：

1. 4 小时用于理解代码主线
2. 3 小时用于跑通和验证
3. 2 小时用于精读关键代码
4. 1 到 2 小时用于总结、补漏和可选扩展

## 全天详细排程

## 09:00 - 09:30 定目标和检查环境

目标：

1. 确认今天只做“最小闭环 + 主干理解”，不碰大规模复现。
2. 确认 Python、PyTorch、依赖和设备是否可用。

要做的事：

1. 快速读一遍 [README.md](/home/chen/githubku/nanoGPT/README.md)
2. 确认你能运行 Python
3. 确认依赖是否安装
4. 确认设备是 `cuda`、`mps` 还是 `cpu`

建议命令：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, transformers, datasets, tiktoken"
```

完成标准：

1. 你知道今天要走哪条路径
2. 你知道自己跑的是 GPU 还是 CPU

风险：

1. 缺依赖
2. PyTorch 版本太旧
3. 没有 GPU

回退方案：

1. 没 GPU 就走 CPU 最小配置
2. `compile` 出问题就关掉 `--compile=False`

## 09:30 - 10:30 建立项目地图

目标：

1. 用 1 小时把仓库拆成清晰模块
2. 不陷入底层细节

要读的文件：

1. [README.md](/home/chen/githubku/nanoGPT/README.md)
2. [config/train_shakespeare_char.py](/home/chen/githubku/nanoGPT/config/train_shakespeare_char.py)
3. [train.py](/home/chen/githubku/nanoGPT/train.py)
4. [model.py](/home/chen/githubku/nanoGPT/model.py)
5. [sample.py](/home/chen/githubku/nanoGPT/sample.py)
6. [data/shakespeare_char/prepare.py](/home/chen/githubku/nanoGPT/data/shakespeare_char/prepare.py)

你现在只回答这 5 个问题：

1. 数据从哪里来？
2. 配置从哪里覆盖？
3. 模型在哪里定义？
4. 训练循环在哪里？
5. 采样在哪里做？

产出物：

写一个 5 行项目地图草稿，例如：

```text
数据准备 -> prepare.py
配置覆盖 -> configurator.py + config/*.py
模型定义 -> model.py
训练入口 -> train.py
采样入口 -> sample.py
```

## 10:30 - 11:30 跑通数据准备

目标：

1. 明白训练数据最终长什么样
2. 产出 `train.bin`、`val.bin`、`meta.pkl`

执行：

```bash
python data/shakespeare_char/prepare.py
```

需要重点观察：

1. 字符表是怎么构造的
2. `stoi` / `itos` 是什么
3. 为什么保存成 `.bin`
4. `meta.pkl` 为什么后面采样会用到

读代码重点：

1. [data/shakespeare_char/prepare.py](/home/chen/githubku/nanoGPT/data/shakespeare_char/prepare.py)

完成标准：

1. 数据文件生成成功
2. 你能解释 `train.bin` 里装的是整数 token id

## 11:30 - 12:30 跑通训练最小闭环

目标：

1. 先跑起来，再谈理解
2. 生成第一个 checkpoint

优先方案：如果你有 GPU

```bash
python train.py config/train_shakespeare_char.py
```

回退方案：如果只有 CPU 或遇到编译问题

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

此阶段只盯 4 件事：

1. 是否开始打印 loss
2. 是否周期性 eval
3. 是否在输出目录保存 `ckpt.pt`
4. 是否能稳定跑下去

完成标准：

1. 至少成功跑到产生 checkpoint
2. 知道输出目录在哪里

午休前检查点：

1. 数据准备成功
2. 训练成功启动
3. 至少保存过一次 checkpoint

## 12:30 - 13:30 休息

要求：

1. 不继续乱改参数
2. 只记录当前卡点

## 13:30 - 15:00 精读 `train.py`

目标：

1. 看懂训练脚本骨架
2. 搞清楚“一次迭代发生了什么”

阅读顺序：

1. 默认配置区
2. `exec(open('configurator.py').read())`
3. DDP 初始化部分
4. `get_batch(split)`
5. `init_from` 三种模型初始化逻辑
6. `estimate_loss()`
7. `get_lr(it)`
8. 主训练循环
9. checkpoint 保存逻辑

必须回答出来的问题：

1. `x` 和 `y` 怎么构造？
2. 为什么每次 `get_batch` 都重新建 `np.memmap`？
3. `gradient_accumulation_steps` 在解决什么问题？
4. 学习率在哪改？
5. checkpoint 里保存了哪些内容？

产出物：

写一份“训练迭代 10 步流程”，建议像这样：

```text
1. 取 batch
2. 前向计算 logits 和 loss
3. loss 按梯度累积步数缩放
4. 反向传播
5. 梯度裁剪
6. optimizer.step
7. scaler.update
8. 清空梯度
9. 定期 eval
10. 保存 checkpoint
```

## 15:00 - 16:30 精读 `model.py`

目标：

1. 看懂 GPT 的结构
2. 能把 forward 流程说顺

阅读顺序：

1. `LayerNorm`
2. `CausalSelfAttention`
3. `MLP`
4. `Block`
5. `GPTConfig`
6. `GPT.__init__`
7. `GPT.forward`
8. `configure_optimizers`
9. `generate`

必须回答出来的问题：

1. token embedding 和 position embedding 在哪相加？
2. causal mask 在哪保证只看左边？
3. 为什么 `Block` 是残差结构？
4. 为什么训练时返回全序列 logits，而推理时只取最后一个位置？
5. `generate()` 为什么要裁剪上下文到 `block_size`？

产出物：

画一个简化流程图：

```text
idx
-> token embedding
-> position embedding
-> dropout
-> N x Transformer Block
-> final layer norm
-> lm_head
-> logits
-> loss 或下一个 token
```

## 16:30 - 17:00 跑通采样

目标：

1. 让训练产物真正生成文本
2. 理解 checkpoint 到文本输出的路径

执行：

```bash
python sample.py --out_dir=out-shakespeare-char
```

如果是 CPU：

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

额外测试：

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="ROMEO:"
```

需要理解：

1. checkpoint 如何加载
2. `meta.pkl` 如何用于字符级 encode/decode
3. `temperature` 和 `top_k` 如何影响输出

完成标准：

1. 成功打印出生成文本
2. 能看懂采样脚本的主流程

## 17:00 - 18:00 写自己的理解笔记

目标：

1. 把今天获得的理解固化下来
2. 防止“当时看懂了，第二天忘了”

至少写这 4 部分：

1. 项目结构总览
2. 一次训练迭代流程
3. GPT 模型结构摘要
4. 跑通命令和结果

建议文件名：

```text
notes.md
```

如果时间紧，这一步不能省。

## 19:00 - 20:30 可选增强：走第二条路径

只有在前面的最小闭环已经成功后，才做这一段。

可选项 A：看 BPE 版本的数据准备

1. 读 [data/shakespeare/prepare.py](/home/chen/githubku/nanoGPT/data/shakespeare/prepare.py)
2. 读 [data/openwebtext/prepare.py](/home/chen/githubku/nanoGPT/data/openwebtext/prepare.py)
3. 理解字符级和 GPT-2 tokenizer 路线差别

可选项 B：看 finetune 配置

1. 读 [config/finetune_shakespeare.py](/home/chen/githubku/nanoGPT/config/finetune_shakespeare.py)
2. 对比 `train_shakespeare_char.py`
3. 理解 `scratch` 和 `gpt2` 初始化差别

可选项 C：看性能脚本

1. 读 [bench.py](/home/chen/githubku/nanoGPT/bench.py)
2. 理解它和 `train.py` 的关系

## 20:30 - 21:30 复盘和查漏补缺

你要检查自己是否真的完成，而不是“好像看过”：

1. 能不用看代码说出项目主线吗？
2. 能不用看代码说出 `train.py` 的主循环吗？
3. 能解释 `Block` 由什么组成吗？
4. 能说明为什么 `sample.py` 需要 `meta.pkl` 吗？
5. 能说明字符级训练和 GPT-2 BPE 路线差别吗？

如果不能，回去补：

1. `train.py`
2. `model.py`
3. `sample.py`

## 21:30 - 22:00 收尾

明天结束前请确保你留下这些可复用资产：

1. 一份阅读笔记
2. 一组能复现的命令
3. 一个可用 checkpoint
4. 一份你自己画的训练/推理流程图

## 明天的推荐执行顺序

严格按这个顺序，不要乱跳：

1. README
2. `config/train_shakespeare_char.py`
3. `data/shakespeare_char/prepare.py`
4. `train.py`
5. 跑训练
6. `model.py`
7. `sample.py`
8. 写笔记
9. 做可选增强

## 不要做的事

这些事情很容易消耗你一天，但不一定让你完成项目：

1. 一上来就看 OpenWebText 大数据准备
2. 一上来就研究 DDP 细节
3. 一上来就改模型结构
4. 为了“更优雅”重构代码
5. 花很多时间调参追求更好样本
6. 尝试完整复现 GPT-2 124M 结果

## 风险表

## 风险 1：训练太慢

症状：

1. CPU 跑一个 iter 非常慢
2. 迟迟看不到 checkpoint

处理：

1. 减小 `block_size`
2. 减小 `batch_size`
3. 减少 `n_layer` / `n_head` / `n_embd`
4. 减少 `max_iters`
5. 设置 `--compile=False`

## 风险 2：依赖报错

处理：

1. 优先修通 `torch`
2. 其次修通 `numpy`
3. 采样阶段再管 `tiktoken`
4. 不需要马上折腾 `wandb`

## 风险 3：看代码看散了

处理：

1. 永远回到主线：数据 -> 模型 -> 训练 -> 采样
2. 永远优先 `train.py` 和 `model.py`
3. 每看完一段立刻写 3 句总结

## 风险 4：你误把“看过”当成“会了”

处理：

强制自己写出下面 3 段：

1. `get_batch()` 在做什么
2. `GPT.forward()` 在做什么
3. `generate()` 在做什么

## 交付清单

到 2026-03-29 结束时，你应该至少有：

1. `out-shakespeare-char/ckpt.pt`
2. 一次成功的 `sample.py` 输出
3. 一份自己的学习笔记
4. 一份项目流程图
5. 对主文件职责的清晰说明

## 终极判断标准

如果明晚你能做到下面这些，就说明你真的完成了这次项目冲刺：

1. 能从零说出这个仓库的完整执行路径
2. 能本地重新跑出最小实验
3. 能解释 `train.py` 和 `model.py` 的关键函数
4. 能继续基于这个仓库做小改动，而不是只能照抄 README

## 明天如果时间只剩 4 小时

请只做下面这些：

1. 读 README 和 `config/train_shakespeare_char.py`
2. 跑 `data/shakespeare_char/prepare.py`
3. 跑一个缩小版训练命令
4. 跑 `sample.py`
5. 精读 `get_batch()`、`GPT.forward()`、`generate()`

## 明天如果时间还有富余

再考虑下面这些扩展：

1. 跑 `config/finetune_shakespeare.py`
2. 对比字符级和 BPE 两条数据路径
3. 读 `bench.py`
4. 读 `from_pretrained()` 逻辑
5. 尝试改一个小超参数并观察输出变化
