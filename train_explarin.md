# train.py 详细解释

## 1. 这个文件是做什么的

`train.py` 是 nanoGPT 的核心训练入口。它负责把下面这些事情串起来：

- 读取默认配置、配置文件和命令行覆盖参数
- 初始化单卡训练或 DDP 分布式训练环境
- 加载数据批次
- 按指定方式初始化模型
- 创建优化器、混合精度和学习率调度
- 执行训练、评估、保存 checkpoint

可以把它理解成“训练总控脚本”。模型结构定义主要在 `model.py`，配置覆盖逻辑在 `configurator.py`，而 `train.py` 负责把这些部件真正跑起来。

---

## 2. 支持哪些运行方式

文件开头的注释已经说明了三种典型用法：

### 2.1 单卡运行

```bash
python train.py --batch_size=32 --compile=False
```

适合调试、快速实验、或者小数据集训练。

### 2.2 单机多卡 DDP

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

这里会启动 4 个进程，每个进程绑定一张 GPU，使用 PyTorch 的 `DistributedDataParallel` 进行同步训练。

### 2.3 多机多卡 DDP

```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
```

第二台机器把 `node_rank` 改为 `1` 即可。此模式适合大规模预训练。

---

## 3. 文件整体执行流程

`train.py` 可以概括成下面这条主线：

1. 定义一组默认超参数
2. 用 `configurator.py` 接受配置文件和命令行覆盖
3. 判断是否是 DDP 环境，并初始化训练设备
4. 准备随机种子、自动混合精度上下文、输出目录
5. 定义 `get_batch()`，从 `train.bin` / `val.bin` 采样
6. 根据 `init_from` 初始化模型
7. 构建优化器、GradScaler、可选 `torch.compile`
8. 定义验证函数 `estimate_loss()` 和学习率函数 `get_lr()`
9. 进入训练主循环
10. 周期性评估、记录日志、保存 checkpoint
11. 达到最大迭代数后退出，必要时销毁分布式进程组

如果你在读源码，可以按这个顺序往下看，会比较容易。

---

## 4. 默认配置区

这一段从：

```python
out_dir = 'out'
```

一直到：

```python
compile = True
```

它本质上是在文件顶部直接声明了一整套默认训练参数。

### 4.1 I/O 相关参数

- `out_dir`：输出目录，checkpoint 会保存在这里
- `eval_interval`：每多少步做一次验证
- `log_interval`：每多少步打印一次训练日志
- `eval_iters`：验证时抽多少个 batch 来平均 loss
- `eval_only`：如果为 `True`，只做一次评估就退出
- `always_save_checkpoint`：每次验证后是否都保存 checkpoint
- `init_from`：模型初始化方式

`init_from` 有三种常用值：

- `'scratch'`：完全从零开始训练
- `'resume'`：从 `out_dir/ckpt.pt` 恢复训练
- `'gpt2'` / `'gpt2-medium'` / `'gpt2-large'` / `'gpt2-xl'`：从 OpenAI GPT-2 权重开始

### 4.2 wandb 参数

- `wandb_log`：是否启用 Weights & Biases
- `wandb_project`：项目名
- `wandb_run_name`：运行名

默认关闭，避免初学者第一次运行就被外部日志系统干扰。

### 4.3 数据参数

- `dataset`：数据集目录名，对应 `data/<dataset>`
- `gradient_accumulation_steps`：梯度累积步数
- `batch_size`：单次 micro-batch 大小
- `block_size`：上下文长度

这里最容易混淆的是：

- `batch_size` 不是“等效总 batch size”
- 真正参与一次参数更新的 token 数量，和 `gradient_accumulation_steps`、`ddp_world_size`、`block_size` 一起决定

脚本中用下面这句明确打印了每次迭代处理的 token 数：

```python
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
```

### 4.4 模型参数

- `n_layer`：Transformer 层数
- `n_head`：注意力头数
- `n_embd`：隐藏维度
- `dropout`：dropout 比例
- `bias`：是否在线性层和 LayerNorm 中使用 bias

### 4.5 优化器参数

- `learning_rate`
- `max_iters`
- `weight_decay`
- `beta1`
- `beta2`
- `grad_clip`

优化器使用的是 AdamW，但具体分组和创建逻辑在 `model.configure_optimizers(...)` 中。

### 4.6 学习率衰减参数

- `decay_lr`：是否启用学习率衰减
- `warmup_iters`：预热步数
- `lr_decay_iters`：退火到最小学习率所需的步数
- `min_lr`：衰减下限

这里实现的是“线性 warmup + cosine decay”。

### 4.7 系统参数

- `backend`：DDP 通信后端，默认 `nccl`
- `device`：训练设备，通常是 `cuda`
- `dtype`：训练精度，优先 `bfloat16`，否则回退 `float16`
- `compile`：是否使用 `torch.compile`

---

## 5. 配置覆盖机制

这部分代码很关键：

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

含义是：

1. 先把当前文件里定义的这些基础配置项收集起来
2. 执行 `configurator.py`
3. 允许外部配置文件和命令行参数覆盖这些变量
4. 再把最终配置打包成 `config` 字典，供日志和 checkpoint 使用

### 5.1 覆盖顺序

如果你这样运行：

```bash
python train.py config/train_shakespeare_char.py --batch_size=32
```

顺序是：

1. 使用 `train.py` 顶部默认值
2. 执行 `config/train_shakespeare_char.py` 覆盖一批参数
3. 再用 `--batch_size=32` 进行最后覆盖

### 5.2 configurator.py 是怎么工作的

`configurator.py` 的策略非常直接：

- 没有 `=` 的参数，视为一个 Python 配置文件
- 有 `=` 的参数，视为 `--key=value`
- 使用 `literal_eval()` 尝试把字符串转成 Python 类型
- 强制检查覆盖值和原值类型一致

这套设计很“土”，但非常轻量，适合实验脚本。

---

## 6. DDP 初始化逻辑

这一段通过环境变量判断是否在分布式环境中：

```python
ddp = int(os.environ.get('RANK', -1)) != -1
```

如果 `RANK` 存在，就说明当前进程是由 `torchrun` 拉起的。

### 6.1 DDP 模式下做了什么

如果 `ddp == True`：

- 调用 `init_process_group(backend=backend)`
- 读取：
  - `RANK`
  - `LOCAL_RANK`
  - `WORLD_SIZE`
- 当前进程绑定到 `cuda:<LOCAL_RANK>`
- 只让 `rank == 0` 的进程负责日志、建目录、保存 checkpoint
- 用 `seed_offset = ddp_rank` 保证每个进程的随机种子不同
- 把 `gradient_accumulation_steps` 除以 `ddp_world_size`

最后一步非常重要。因为多卡会同时训练，所以每个进程上的梯度累积步数要缩小，否则总 batch 会被意外放大。

### 6.2 非 DDP 模式

如果不是 DDP：

- `master_process = True`
- `seed_offset = 0`
- `ddp_world_size = 1`

也就是普通单进程训练。

---

## 7. 随机种子、精度和 autocast

### 7.1 随机种子

```python
torch.manual_seed(1337 + seed_offset)
```

单卡时就是 `1337`，多卡时每个 rank 稍微偏移，避免所有进程完全一致。

### 7.2 TF32

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

这会让 Ampere 及以上 GPU 在某些计算上使用 TF32，加速矩阵乘法，同时通常保持足够好的训练效果。

### 7.3 混合精度上下文

```python
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

作用是：

- CPU 上不启用 autocast
- GPU 上根据 `dtype` 自动使用 `float16` 或 `bfloat16`

后续前向推理和验证都会写成：

```python
with ctx:
    logits, loss = model(X, Y)
```

---

## 8. 数据加载：get_batch()

这是一个“极简数据加载器”，没有使用 PyTorch 的 `DataLoader`。

### 8.1 数据文件来源

脚本默认从下面的目录找数据：

```python
data_dir = os.path.join('data', dataset)
```

也就是：

- `data/<dataset>/train.bin`
- `data/<dataset>/val.bin`
- 可选的 `data/<dataset>/meta.pkl`

### 8.2 get_batch() 的流程

`get_batch(split)` 做了这些事情：

1. 根据 `split` 选择 `train.bin` 或 `val.bin`
2. 用 `np.memmap` 以只读方式映射数据文件
3. 随机采样一批起始位置 `ix`
4. 从每个位置切出长度为 `block_size` 的 token 序列作为输入 `x`
5. 再向右偏移一位构造标签 `y`

所以：

- `x` 表示当前上下文
- `y` 表示下一个 token 的监督目标

这是标准的自回归语言模型训练方式。

### 8.3 为什么每次都重新创建 memmap

代码里有注释说明：这是为了规避内存泄漏问题。虽然看起来笨，但在这个仓库里优先追求的是稳定和简单。

### 8.4 为什么 CUDA 分支会 `pin_memory`

```python
x, y = x.pin_memory().to(device, non_blocking=True)
```

作用是把 CPU 张量放进 page-locked memory，再异步传到 GPU，减少数据拷贝等待时间。

---

## 9. vocab_size 的推断

脚本会尝试读取：

```python
meta_path = os.path.join(data_dir, 'meta.pkl')
```

如果存在，就从中取出：

```python
meta['vocab_size']
```

这个值常用于从零训练字符级模型或自定义分词器模型时，告诉网络词表大小是多少。

如果没有 `meta.pkl`，并且你选择的是 `scratch`，脚本会把词表大小默认成 `50304`，也就是 GPT-2 的词表规模向上取整后的值。

---

## 10. 模型初始化逻辑

这里是 `train.py` 最核心的分支之一。

```python
if init_from == 'scratch':
elif init_from == 'resume':
elif init_from.startswith('gpt2'):
```

### 10.1 从零开始：`init_from == 'scratch'`

流程如下：

1. 构造 `model_args`
2. 设置 `vocab_size`
3. 用 `GPTConfig(**model_args)` 创建配置
4. 用 `GPT(gptconf)` 构建模型

适合以下情况：

- 训练字符级模型
- 训练自己的 tokenizer
- 完全从头预训练

### 10.2 恢复训练：`init_from == 'resume'`

流程如下：

1. 读取 `out_dir/ckpt.pt`
2. 取出保存时的 `model_args`
3. 强制把若干关键结构参数改成 checkpoint 中的值
4. 创建模型
5. 加载模型参数
6. 恢复 `iter_num`
7. 恢复 `best_val_loss`

这里强制同步的参数包括：

- `n_layer`
- `n_head`
- `n_embd`
- `block_size`
- `bias`
- `vocab_size`

原因很简单：这些参数决定了模型结构，如果不一致，权重根本没法正确加载。

### 10.3 从 OpenAI GPT-2 初始化

当 `init_from` 以 `gpt2` 开头时，会调用：

```python
model = GPT.from_pretrained(init_from, override_args)
```

这条路径通常用于微调。初始化后，脚本还会把模型真实配置再回填到 `model_args` 里，方便将来保存 checkpoint。

### 10.4 `_orig_mod.` 前缀修复

恢复 checkpoint 时有这样一段：

```python
unwanted_prefix = '_orig_mod.'
```

这是为了解决某些情况下保存出来的 state dict key 带额外前缀的问题。脚本会在加载前把这个前缀剥掉，提升兼容性。

---

## 11. block_size 裁剪

这段逻辑：

```python
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
```

表示如果当前要求的 `block_size` 比模型原本支持的上下文更短，就把模型“裁短”。

常见场景：

- 从 GPT-2 权重初始化，但你只想用更短上下文进行微调
- 想减少显存占用

这样做后还会同步更新 `model_args['block_size']`，保证 checkpoint 中记录的是新值。

---

## 12. 优化器和 GradScaler

### 12.1 优化器

```python
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
```

虽然具体细节在 `model.py`，但从调用方式看可以知道：

- 使用 AdamW 风格参数
- 会根据参数类型做 weight decay 分组
- 会考虑设备类型

### 12.2 恢复优化器状态

如果 `init_from == 'resume'`，还会执行：

```python
optimizer.load_state_dict(checkpoint['optimizer'])
```

这能恢复动量、二阶矩估计等内部状态，否则“恢复训练”其实只是恢复了模型权重，而不是完整训练状态。

### 12.3 GradScaler

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

只有 `float16` 训练时才真正启用缩放，避免梯度下溢。

如果是 `bfloat16` 或 `float32`，这个 `scaler` 基本就是空操作包装器。

---

## 13. torch.compile 和 DDP 包装

### 13.1 compile

```python
if compile:
    unoptimized_model = model
    model = torch.compile(model)
```

这会尝试让 PyTorch 2.x 对模型做图编译优化，以提高训练吞吐。

优点：

- 速度可能明显提升

代价：

- 首次编译会很慢
- 某些环境或调试场景下不稳定

所以很多小实验会显式传 `--compile=False`。

### 13.2 DDP 包装

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

这一步把模型变成分布式同步训练版本。后面为了访问原始模型，会写：

```python
raw_model = model.module if ddp else model
```

这是因为：

- `model` 可能已经被 `DDP(...)` 包了一层
- 真正的 GPT 对象藏在 `model.module`

---

## 14. estimate_loss() 做了什么

这是验证函数：

```python
@torch.no_grad()
def estimate_loss():
```

它的行为是：

1. 切到 `model.eval()`
2. 分别对 `train` 和 `val` 做 `eval_iters` 次采样
3. 每次前向得到 `loss`
4. 对 loss 求平均
5. 切回 `model.train()`

返回结果类似：

```python
{
    'train': ...,
    'val': ...
}
```

注意这里没有做完整数据集验证，而是“抽样近似评估”。这样速度更快，非常适合训练中的周期性监控。

---

## 15. 学习率函数 get_lr()

这个函数实现的是三段式逻辑：

### 15.1 预热阶段

当：

```python
it < warmup_iters
```

学习率从 0 附近线性上升到 `learning_rate`。

### 15.2 衰减结束后

当：

```python
it > lr_decay_iters
```

学习率固定为 `min_lr`。

### 15.3 中间阶段

在 `warmup_iters` 和 `lr_decay_iters` 之间，使用 cosine 衰减：

```python
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
```

这是一种非常经典的语言模型训练调度方式。

---

## 16. wandb 日志

如果：

```python
wandb_log and master_process
```

则会初始化 wandb，并在验证时记录：

- `iter`
- `train/loss`
- `val/loss`
- `lr`
- `mfu`

只允许 `master_process` 记录，是为了避免多卡重复上报。

---

## 17. 训练循环逐行理解

主循环从这里开始：

```python
while True:
```

可以拆成 6 个阶段来看。

### 17.1 设置当前学习率

```python
lr = get_lr(iter_num) if decay_lr else learning_rate
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

意思是每轮迭代都动态修改优化器学习率。

### 17.2 周期性评估和保存 checkpoint

当满足：

```python
if iter_num % eval_interval == 0 and master_process:
```

脚本会：

1. 调用 `estimate_loss()`
2. 打印 train / val loss
3. 如启用 wandb，则同步记录
4. 如果验证损失更好，或者 `always_save_checkpoint=True`，就保存 checkpoint

checkpoint 内容包括：

- `model`
- `optimizer`
- `model_args`
- `iter_num`
- `best_val_loss`
- `config`

这说明 checkpoint 不只是参数快照，也是一次训练状态快照。

### 17.3 eval_only 模式

```python
if iter_num == 0 and eval_only:
    break
```

这让 `train.py` 也能临时充当“评估脚本”。

### 17.4 梯度累积训练

真正的训练更新在这里：

```python
for micro_step in range(gradient_accumulation_steps):
```

每个大迭代会做多个 micro step。每一步：

1. 前向计算 `logits, loss`
2. 将 `loss` 除以 `gradient_accumulation_steps`
3. 预取下一个 batch
4. 调用 `scaler.scale(loss).backward()`

这里的“先算当前 batch，马上异步准备下一个 batch”是一个小优化，减少 GPU 等数据的空转。

### 17.5 DDP 下的梯度同步优化

DDP 时有这句：

```python
model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

它的含义是：

- 前几个 micro step 先只做本地反向传播，不立刻同步梯度
- 只有最后一个 micro step 才进行跨卡同步

这样可以减少通信开销。

### 17.6 梯度裁剪、参数更新和清梯度

完成所有 micro step 后：

1. 如果设置了 `grad_clip`，先 `unscale_` 再裁剪梯度
2. `scaler.step(optimizer)` 更新参数
3. `scaler.update()` 更新缩放器状态
4. `optimizer.zero_grad(set_to_none=True)` 清空梯度

这里使用 `set_to_none=True`，通常会比把梯度置零更省内存、也更快。

---

## 18. 日志打印和 MFU

每隔 `log_interval`，主进程会打印：

```python
iter {iter_num}: loss ..., time ...ms, mfu ...%
```

其中：

- `loss`：当前训练损失的近似值
- `time`：本轮迭代耗时
- `mfu`：Model FLOPs Utilization，模型计算利用率

`mfu` 是通过：

```python
raw_model.estimate_mfu(...)
```

估计得到的，用于粗略观察训练效率。

代码里前 5 次迭代不会立刻采用这个值，而是先让训练过程“热起来”，之后再做指数滑动平均：

```python
running_mfu = 0.9 * running_mfu + 0.1 * mfu
```

这样日志更稳定。

---

## 19. 退出条件

训练终止条件很简单：

```python
if iter_num > max_iters:
    break
```

结束后如果是 DDP，还会执行：

```python
destroy_process_group()
```

这一步是分布式训练的标准收尾操作。

---

## 20. 这个脚本设计上的特点

`train.py` 的最大特点不是“功能很多”，而是“把复杂训练流程压缩进了较少代码里”。

它有几个很鲜明的设计取向：

### 20.1 优先可读性

很多框架会把训练流程拆进：

- Trainer
- Engine
- Callback
- DataModule
- Launcher

但这里基本都放在一个文件中，便于初学者从头读到尾。

### 20.2 优先实战而非抽象

例如：

- 直接用全局变量做配置
- 直接 `exec()` 配置文件
- 手写 `get_batch()` 而不是引入更复杂的数据层

这些都不是“最优雅”的工程方案，但非常适合研究和实验。

### 20.3 尽量同时支持小实验和大训练

同一份脚本既能：

- 在 CPU 或单卡上跑小模型
- 也能在多机多卡上训练 GPT-2 级别模型

这是这个文件最厉害的地方之一。

---

## 21. 阅读 train.py 时最值得注意的几个变量

如果你第一次读源码，建议重点盯住下面这些变量，因为它们贯穿全文件：

- `init_from`：决定模型从哪里来
- `gradient_accumulation_steps`：决定等效 batch 大小
- `block_size`：决定上下文长度
- `iter_num`：当前训练走到哪一步
- `best_val_loss`：当前最优验证损失
- `master_process`：当前进程是否负责输出和保存
- `ddp_world_size`：总进程数
- `ctx`：混合精度上下文
- `scaler`：float16 缩放器
- `raw_model`：去掉 DDP 包装后的模型对象

---

## 22. 常见运行场景和它在代码里对应什么

### 22.1 训练莎士比亚字符级模型

```bash
python train.py config/train_shakespeare_char.py
```

对应代码含义：

- `config/train_shakespeare_char.py` 覆盖默认参数
- 一般 `init_from='scratch'`
- 从 `data/shakespeare_char` 读取数据
- 从零开始训练一个小 GPT

### 22.2 恢复中断训练

```bash
python train.py --init_from=resume --out_dir=out-shakespeare-char
```

对应代码含义：

- 读取 `out-shakespeare-char/ckpt.pt`
- 恢复模型权重、优化器状态、迭代计数和最优验证损失

### 22.3 从 GPT-2 微调

```bash
python train.py config/finetune_shakespeare.py
```

通常对应：

- `init_from='gpt2'`
- 加载 GPT-2 预训练权重
- 在更小数据集上继续训练

---

## 23. 这个文件和其他文件的关系

- `train.py`：训练总流程
- `model.py`：GPT 模型结构、优化器分组、预训练权重加载、MFU 估计
- `configurator.py`：配置覆盖
- `config/*.py`：不同任务的参数模板
- `data/*/prepare.py`：生成 `train.bin`、`val.bin`、`meta.pkl`
- `sample.py`：训练后采样生成文本

理解 `train.py` 后，再去读 `model.py`，会顺很多。

---

## 24. 一句话总结 train.py

`train.py` 本质上就是一个“简洁但完整的 GPT 训练循环模板”：它把配置、数据采样、模型初始化、混合精度、梯度累积、分布式训练、验证和 checkpoint 管理都放进了一个相对短小、可直接修改的文件里。

如果你想真正吃透这个仓库，最值得反复读的就是 `train.py` 和 `model.py`。
