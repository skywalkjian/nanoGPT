# model.py 逐函数详细解析

## 概述

`model.py` 是 nanoGPT 的核心文件，在一个文件中完整实现了 GPT（Generative Pre-trained Transformer）语言模型。代码参考了 OpenAI 官方的 GPT-2 TensorFlow 实现和 HuggingFace Transformers 的 PyTorch 实现。

### 整体架构

```
GPT
├── transformer (ModuleDict)
│   ├── wte: 词嵌入 (Embedding)
│   ├── wpe: 位置嵌入 (Embedding)
│   ├── drop: Dropout
│   ├── h: N 个 Block (ModuleList)
│   │   └── Block
│   │       ├── ln_1: LayerNorm
│   │       ├── attn: CausalSelfAttention
│   │       ├── ln_2: LayerNorm
│   │       └── mlp: MLP
│   └── ln_f: 最终 LayerNorm
└── lm_head: 语言模型输出头 (Linear)
```

---

## 导入与依赖

```python
import math            # sqrt 用于注意力缩放
import inspect         # 检查 AdamW 是否支持 fused 参数
from dataclasses import dataclass  # 用于 GPTConfig 配置类

import torch
import torch.nn as nn
from torch.nn import functional as F
```

---

## 一、class LayerNorm（第 18-27 行）

### 设计动机

PyTorch 内置的 `nn.LayerNorm` 不支持单独禁用 bias（只能同时有 weight 和 bias 或都没有）。原始 GPT-2 使用了 bias，但后续研究表明去掉 bias 可以略微提升性能和训练速度。因此这里自定义了一个支持可选 bias 的 LayerNorm。

### `__init__(self, ndim, bias)`

```python
def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))       # 缩放参数 γ，初始化为 1
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 偏移参数 β，初始化为 0，可选
```

**参数**:
- `ndim` (int): 归一化的特征维度，即 `n_embd`
- `bias` (bool): 是否使用偏置项

**行为**: 创建可学习的 `weight`（γ）和可选的 `bias`（β）参数。

### `forward(self, input)`

```python
def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

**计算公式**:

$$\text{output} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

- `self.weight.shape`: 归一化的维度形状
- `1e-5`: epsilon，防止除零的小常数
- 直接调用 PyTorch 的 `F.layer_norm` 实现，效率最高

---

## 二、class CausalSelfAttention（第 29-76 行）

因果自注意力机制——GPT 的核心组件。"因果"意味着每个 token 只能关注它之前（包括自身）的 token，不能"偷看"未来。

### `__init__(self, config)`

```python
def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
```

**断言**: 嵌入维度必须能被头数整除，确保每个头分到相等的维度。

```python
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```

**QKV 投影**: 一个线性层同时计算 Query、Key、Value，输出维度是输入的 3 倍。这比使用三个独立的线性层更高效（一次矩阵乘法代替三次）。

- 命名 `c_attn` 沿用了 OpenAI GPT-2 的命名惯例（`c` 代表 Conv1D，原始实现用的是 1D 卷积）。

```python
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
```

**输出投影**: 将多头注意力拼接后的结果投影回原始维度。

```python
    self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重的 dropout
    self.resid_dropout = nn.Dropout(config.dropout)  # 残差连接前的 dropout
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
```

```python
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
        print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
```

**Flash Attention 检测**:
- PyTorch >= 2.0 提供了 `scaled_dot_product_attention`，自动使用 Flash Attention 内核，显存效率和速度大幅提升。
- 如果不可用，则注册一个**下三角因果掩码**作为 buffer（不参与梯度计算）：

```
bias 矩阵 (block_size x block_size):
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

位置 (i, j) 为 1 表示 token i 可以关注 token j（j ≤ i）。

### `forward(self, x)`

```python
def forward(self, x):
    B, T, C = x.size()  # B=batch, T=序列长度, C=嵌入维度(n_embd)
```

**Step 1: QKV 计算与多头拆分**

```python
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
```

数据流维度变化：
```
x: (B, T, C)
    ↓ c_attn 线性层
(B, T, 3C)
    ↓ split
q, k, v: 各 (B, T, C)
    ↓ view + transpose
q, k, v: 各 (B, n_head, T, head_size)    # head_size = C / n_head
```

**Step 2: 注意力计算**

Flash Attention 路径（PyTorch >= 2.0）:
```python
    if self.flash:
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
```

手动实现路径（回退）:
```python
    else:
        # Q @ K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用因果掩码：未来位置设为 -inf
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Softmax 归一化（-inf 变为 0）
        att = F.softmax(att, dim=-1)
        # 注意力 dropout
        att = self.attn_dropout(att)
        # 加权求和 Value
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```

**注意力公式**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 M 是因果掩码矩阵（未来位置为 -∞）。

**Step 3: 多头合并与输出投影**

```python
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # 拼接所有头
    y = self.resid_dropout(self.c_proj(y))             # 输出投影 + dropout
    return y
```

维度变化：`(B, nh, T, hs)` → transpose → `(B, T, nh, hs)` → view → `(B, T, C)`

---

## 三、class MLP（第 78-92 行）

前馈神经网络（FFN），也叫 Position-wise Feed-Forward Network。

### `__init__(self, config)`

```python
def __init__(self, config):
    super().__init__()
    self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)   # 升维
    self.gelu   = nn.GELU()                                                         # 激活函数
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)    # 降维
    self.dropout = nn.Dropout(config.dropout)
```

**结构**: 经典的"升维 → 激活 → 降维"设计。
- **4 倍扩展**: 隐藏层维度是嵌入维度的 4 倍（GPT-2 原始设计）。例如 `n_embd=768` 时，中间层为 3072。
- **GELU 激活**: Gaussian Error Linear Unit，比 ReLU 更平滑，是 GPT 系列的标准选择。

### `forward(self, x)`

```python
def forward(self, x):
    x = self.c_fc(x)      # (B, T, C) -> (B, T, 4C)  升维
    x = self.gelu(x)       # 非线性激活
    x = self.c_proj(x)     # (B, T, 4C) -> (B, T, C)  降维
    x = self.dropout(x)    # 正则化
    return x
```

---

## 四、class Block（第 94-106 行）

一个完整的 Transformer 解码器块，采用 **Pre-Norm** 架构。

### `__init__(self, config)`

```python
def __init__(self, config):
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # 注意力前的 LayerNorm
    self.attn = CausalSelfAttention(config)                   # 因果自注意力
    self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # MLP 前的 LayerNorm
    self.mlp = MLP(config)                                    # 前馈网络
```

### `forward(self, x)`

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # 残差连接 + 注意力
    x = x + self.mlp(self.ln_2(x))    # 残差连接 + MLP
    return x
```

**Pre-Norm vs Post-Norm**:
- **Pre-Norm**（本实现）: `x + SubLayer(Norm(x))` — 先归一化再送入子层
- **Post-Norm**（原始 Transformer）: `Norm(x + SubLayer(x))` — 先经过子层再归一化

Pre-Norm 的训练更稳定，尤其在深层网络中不容易出现梯度爆炸/消失问题。GPT-2 采用的就是 Pre-Norm。

**残差连接**: `x + f(x)` 结构让梯度可以通过"捷径"直接回传，极大缓解了深层网络的训练困难。

---

## 五、class GPTConfig（第 108-116 行）

使用 `@dataclass` 定义的模型配置类，所有参数都有默认值（对应 GPT-2 124M）。

```python
@dataclass
class GPTConfig:
    block_size: int = 1024     # 最大序列长度（上下文窗口）
    vocab_size: int = 50304    # 词表大小，GPT-2 实际为 50257，填充到 64 的倍数以提升 GPU 效率
    n_layer: int = 12          # Transformer 层数
    n_head: int = 12           # 注意力头数
    n_embd: int = 768          # 嵌入维度
    dropout: float = 0.0       # Dropout 比率（预训练通常为 0，微调可设 0.1+）
    bias: bool = True          # 线性层和 LayerNorm 是否使用 bias
```

**关于 `vocab_size = 50304`**:
GPT-2 的实际词表大小是 50257，但 50304 = 50257 向上取整到 64 的倍数（50304 = 786 × 64）。GPU 在处理 64 对齐的矩阵维度时效率更高（与 CUDA 的内存对齐和 Tensor Core 计算有关）。

---

## 六、class GPT（第 118-330 行）

主模型类，包含完整的 GPT 实现。

### `__init__(self, config)` — 模型构建（第 120-148 行）

```python
def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config
```

**构建 Transformer 主体**:

```python
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),          # 词 Token 嵌入
        wpe = nn.Embedding(config.block_size, config.n_embd),          # 位置嵌入（可学习）
        drop = nn.Dropout(config.dropout),                              # 嵌入后的 dropout
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # N 个 Transformer 块
        ln_f = LayerNorm(config.n_embd, bias=config.bias),             # 最终 LayerNorm
    ))
```

- **`wte`** (word token embedding): 将 token ID 映射为稠密向量。形状 `(vocab_size, n_embd)`
- **`wpe`** (word position embedding): 将位置索引映射为稠密向量。形状 `(block_size, n_embd)`。这是**可学习的**绝对位置编码（不同于 Transformer 原文的正弦位置编码）
- **`h`**: `n_layer` 个 Block 的列表
- **`ln_f`**: 最后一个 Transformer 块之后的 LayerNorm

```python
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

**语言模型头**: 将最终隐藏状态投影回词表维度，产生每个 token 的 logits。

```python
    self.transformer.wte.weight = self.lm_head.weight  # 权重绑定
```

**Weight Tying（权重绑定）**: 输入嵌入层和输出投影层**共享同一个权重矩阵**。
- 理论依据: 输入嵌入将 token 映射到语义空间，输出投影将语义空间映射回 token——这两个映射应该是对称的。
- 实际效果: 减少参数量（`vocab_size × n_embd` 个参数只存一份），同时通常能提升模型性能。
- 参考: [Using the Output Embedding to Improve Language Models (Press & Wolf, 2017)](https://paperswithcode.com/method/weight-tying)

**权重初始化**:

```python
    self.apply(self._init_weights)
    # 残差投影层使用特殊的缩小初始化
    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

- 所有权重先经过 `_init_weights` 标准初始化（正态分布，std=0.02）
- 然后对每个 Block 中的**残差路径输出投影**（`c_proj.weight`，包括注意力和 MLP 中的 `c_proj`）应用缩小初始化：`std = 0.02 / sqrt(2 * n_layer)`
- **原因**: 每个 Block 有 2 条残差路径（注意力 + MLP），总共 `2 * n_layer` 条。缩小初始化确保在模型初始化时，所有残差路径的累积方差不会随深度增长而爆炸。这来自 GPT-2 论文的设计。

---

### `get_num_params(self, non_embedding=True)` — 参数计数（第 150-160 行）

```python
def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params
```

**参数**:
- `non_embedding` (bool): 是否排除位置嵌入的参数量。默认 `True`。

**为什么只减去位置嵌入而不减去词嵌入？**
因为权重绑定：`wte.weight` 和 `lm_head.weight` 是同一个张量。`self.parameters()` 不会重复计数它，而这个权重实际被 `lm_head` 用于最终的预测，所以应该算作"有效参数"。而位置嵌入 `wpe` 只用于编码位置信息，在某些对比场景下不算作模型的核心参数量。

---

### `_init_weights(self, module)` — 权重初始化（第 162-168 行）

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

通过 `self.apply()` 递归地应用于所有子模块：
- **Linear 层**: 权重用 N(0, 0.02) 正态分布初始化，bias 初始化为 0
- **Embedding 层**: 权重用 N(0, 0.02) 正态分布初始化

这是 GPT-2 采用的初始化方案。`std=0.02` 是一个经验值，确保初始输出的方差不会太大。

---

### `forward(self, idx, targets=None)` — 前向传播（第 170-193 行）

这是模型的核心前向计算逻辑。

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)
```

**参数**:
- `idx` (LongTensor): 输入 token 索引，形状 `(batch_size, seq_len)`
- `targets` (LongTensor, optional): 目标 token 索引，形状 `(batch_size, seq_len)`。训练时提供，推理时为 None。

**Step 1: 嵌入**

```python
    tok_emb = self.transformer.wte(idx)    # (b, t) -> (b, t, n_embd)    词嵌入
    pos_emb = self.transformer.wpe(pos)    # (t,)   -> (t, n_embd)       位置嵌入
    x = self.transformer.drop(tok_emb + pos_emb)  # 相加 + dropout
```

词嵌入和位置嵌入相加（broadcasting: `(b, t, n_embd) + (t, n_embd)` → `(b, t, n_embd)`）。

**Step 2: Transformer 块**

```python
    for block in self.transformer.h:
        x = block(x)                        # 依次通过 N 个 Block
    x = self.transformer.ln_f(x)            # 最终 LayerNorm
```

**Step 3: 输出（训练 vs 推理）**

```python
    if targets is not None:
        # 训练：计算所有位置的 logits 和 loss
        logits = self.lm_head(x)            # (b, t, n_embd) -> (b, t, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        # 推理：只计算最后一个位置的 logits（优化）
        logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
        loss = None
```

**训练时**:
- 对所有位置计算 logits，然后用交叉熵损失与目标对比
- `view(-1, ...)` 将 batch 和序列维度展平
- `ignore_index=-1`: 标记为 -1 的位置不参与 loss 计算

**推理时**:
- 只需要最后一个位置的预测（下一个 token），所以只对 `x[:, [-1], :]` 计算 lm_head
- 使用 `[-1]` 而非 `-1` 是为了保留时间维度：`x[:, [-1], :]` 的形状是 `(b, 1, C)` 而非 `(b, C)`

---

### `crop_block_size(self, block_size)` — 裁剪上下文长度（第 195-204 行）

```python
def crop_block_size(self, block_size):
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```

**用途**: 加载预训练模型（如 GPT-2，block_size=1024）后，如果只需要较短的上下文长度，可以裁剪模型。

**操作**:
1. 截断位置嵌入表：只保留前 `block_size` 个位置的嵌入
2. 截断因果掩码（如果存在，即非 Flash Attention 模式下的 `bias` buffer）

这是一种"模型手术"，无需重新训练即可适配较小的上下文。

---

### `from_pretrained(cls, model_type, override_args=None)` — 加载预训练权重（第 206-261 行）

类方法，从 HuggingFace 加载 OpenAI 的 GPT-2 预训练权重。

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
```

**支持的模型**:

| model_type | n_layer | n_head | n_embd | 参数量 |
|-----------|---------|--------|--------|--------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

**核心流程**:

1. **创建 nanoGPT 模型**: 用对应架构配置初始化一个空模型
2. **加载 HuggingFace 模型**: 通过 `GPT2LMHeadModel.from_pretrained()` 下载权重
3. **权重映射与拷贝**:
   - 过滤掉 buffer（`attn.bias`, `attn.masked_bias`）
   - 关键处理：**转置** 4 个权重矩阵

```python
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
```

**为什么需要转置？** OpenAI 原始 GPT-2 使用 `Conv1D` 模块（kernel_size=1 的一维卷积），其权重形状是 `(out_features, in_features)` 的转置。而 nanoGPT 使用标准的 `nn.Linear`，权重形状是 `(out_features, in_features)`，所以需要转置。

---

### `configure_optimizers(self, weight_decay, learning_rate, betas, device_type)` — 配置优化器（第 263-287 行）

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
```

**参数**:
- `weight_decay` (float): 权重衰减系数
- `learning_rate` (float): 学习率
- `betas` (tuple): AdamW 的 (beta1, beta2)
- `device_type` (str): 设备类型，用于决定是否启用 fused AdamW

**核心逻辑: 参数分组**

```python
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]    # 2D+ 参数：weight decay
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]   # 1D 参数：不 decay
```

**分组规则**:
- **应用 weight decay 的参数** (`dim >= 2`): 所有矩阵权重（Linear 的 weight、Embedding 的 weight）。这些参数容易过拟合，weight decay 起正则化作用。
- **不应用 weight decay 的参数** (`dim < 2`): 所有 bias 和 LayerNorm 参数（都是 1D 向量）。对这些参数施加 weight decay 通常没有帮助甚至有害。

**Fused AdamW**:

```python
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
```

PyTorch 较新版本提供了 `fused=True` 选项，将 AdamW 的所有操作融合到一个 CUDA kernel 中执行，减少 kernel launch 开销和内存读写，在 GPU 上显著加速。

---

### `estimate_mfu(self, fwdbwd_per_iter, dt)` — 估算 MFU（第 289-303 行）

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
```

**MFU (Model FLOPs Utilization)**: 模型实际消耗的浮点运算量占 GPU 理论峰值的比例。用于衡量训练效率。

**参数**:
- `fwdbwd_per_iter` (int): 每次迭代的前向+反向传播次数（= `gradient_accumulation_steps * batch_size * ...`）
- `dt` (float): 每次迭代的耗时（秒）

**FLOPS 估算公式**（参考 [PaLM 论文 Appendix B](https://arxiv.org/abs/2204.02311)）:

```python
    N = self.get_num_params()               # 模型参数量
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

    flops_per_token = 6*N + 12*L*H*Q*T
```

- `6N`: 每个 token 的基本 FLOPS（前向 2N + 反向 4N，其中 matmul 占主导）
- `12*L*H*Q*T`: 注意力计算的额外 FLOPS（QK^T 矩阵乘和 attention @ V）

```python
    flops_per_fwdbwd = flops_per_token * T            # 一次前向+反向的总 FLOPS
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # 一次迭代的总 FLOPS

    flops_achieved = flops_per_iter * (1.0/dt)        # 实际 FLOPS/秒
    flops_promised = 312e12                            # A100 bf16 理论峰值: 312 TFLOPS
    mfu = flops_achieved / flops_promised
```

典型的 MFU 值：
- 优秀: > 50%
- 良好: 30-50%
- 一般: < 30%

---

### `generate(self, idx, max_new_tokens, temperature=1.0, top_k=None)` — 文本生成（第 305-330 行）

自回归文本生成方法。

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
```

**参数**:
- `idx` (LongTensor): 初始 token 序列，形状 `(batch_size, seq_len)`
- `max_new_tokens` (int): 要生成的新 token 数量
- `temperature` (float): 温度参数，控制随机性。1.0 为标准，< 1.0 更确定，> 1.0 更随机
- `top_k` (int, optional): 只从概率最高的 k 个 token 中采样

**`@torch.no_grad()`**: 生成过程不需要计算梯度，禁用可节省显存和计算。

**逐步流程**:

```python
    for _ in range(max_new_tokens):
        # 1. 上下文裁剪：超过 block_size 时只取最后 block_size 个 token
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        # 2. 前向传播得到 logits
        logits, _ = self(idx_cond)

        # 3. 取最后一个位置的 logits 并应用温度缩放
        logits = logits[:, -1, :] / temperature
```

**温度缩放原理**: `softmax(logits / T)` —— T < 1 时 logits 被放大，概率分布更尖锐（更确定）；T > 1 时 logits 被缩小，分布更平坦（更随机）。

```python
        # 4. Top-k 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
```

将排名在 top_k 之外的 token 的 logits 设为 -∞，softmax 后概率为 0。

```python
        # 5. 转为概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 6. 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

**生成策略**: 这是**概率采样**（而非贪心解码或 beam search）。通过 `temperature` 和 `top_k` 控制生成质量与多样性的平衡。

---

## 数据流总结

```
输入 token IDs: (B, T)
        ↓
词嵌入 wte: (B, T, C)  +  位置嵌入 wpe: (T, C)  →  (B, T, C)
        ↓
Dropout
        ↓
┌─────────── Block × N ───────────┐
│  LayerNorm → CausalSelfAttention → + (残差)  │
│  LayerNorm → MLP                 → + (残差)  │
└─────────────────────────────────┘
        ↓
最终 LayerNorm: (B, T, C)
        ↓
lm_head 线性投影: (B, T, vocab_size)
        ↓
logits → cross_entropy loss (训练) / 采样 (推理)
```

## 设计亮点总结

| 设计 | 说明 |
|------|------|
| **Pre-Norm** | LayerNorm 在子层之前，训练更稳定 |
| **Weight Tying** | 输入嵌入与输出投影共享权重，减少参数量 |
| **Flash Attention** | 自动检测并使用高效注意力内核 |
| **缩放初始化** | 残差路径的输出投影用 `1/sqrt(2*n_layer)` 缩放 |
| **参数分组** | 矩阵权重用 weight decay，bias/LayerNorm 不用 |
| **Fused AdamW** | GPU 上自动启用融合优化器 |
| **vocab_size 对齐** | 填充到 64 的倍数，提升 GPU 计算效率 |
| **推理优化** | 生成时只计算最后一个位置的 logits |
| **Conv1D→Linear 转置** | 兼容加载 OpenAI 原始 GPT-2 权重 |
