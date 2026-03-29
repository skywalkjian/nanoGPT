# nanoGPT x Kimi Block Attention Residuals 实现计划

适用日期：2026-03-29

目标：基于 `nanoGPT` 做一个**更贴近论文和参考实现**的 `Block Attention Residuals` 面试项目版本，而不是停留在“每层前加一个 depth mixer”的简化原型。

参考实现：

- 论文：<https://arxiv.org/abs/2603.15031>
- 参考代码：<https://github.com/qibin0506/llm_model/blob/master/llm_model/llm_model.py>

---

## 1. 这次计划的核心变化

之前的方案更像：

```text
在 GPT.forward() 里，对每个 block 输入做一次历史层输出的 depth mixer。
```

现在的新方案要升级成：

```text
更接近参考实现的 Block Attention Residuals：
维护 blocks 和 partial_block 两种状态，
在 attention 前和 MLP 前分别做一次 block residual aggregation。
```

一句话总结：

```text
这次不再做“block 入口前的大 mixer”，
而是做“block 内部两处 residual aggregation 都改成 BlockAttnRes”。
```

---

## 2. 为什么这个版本更接近论文

参考实现最关键的结构不是一个大的 `depth_residual_mixer`，而是下面这套状态机：

1. `blocks`
   - 保存已经完成的历史 block 表示
2. `partial_block`
   - 保存当前 block 内正在累积的表示
3. `attn_res_agg`
   - attention 子层前做一次 block residual aggregation
4. `mlp_res_agg`
   - MLP 子层前再做一次 block residual aggregation
5. `layers_per_block`
   - 决定什么时候把当前结果提交进 `blocks`

这意味着：

- BAR 不是只在 block 开头聚合一次
- 而是在 **attention 路径和 MLP 路径前各做一次 residual attention**
- 并且显式区分：
  - 历史完整块 `blocks`
  - 当前块中的中间表示 `partial_block`

这比“直接缓存所有 layer outputs 然后做一次 mixer”更接近论文和参考实现。

---

## 3. 本次最终要实现的版本

## 版本名

### `BAR-v1-faithful`

它的定位是：

```text
面向 nanoGPT 的、尽量贴近参考实现的 Block Attention Residuals 版本。
```

它不是完整论文级复现，但它必须满足下面三点：

1. 机制上明确采用 `blocks + partial_block`
2. attention 前和 MLP 前各有一个 `BlockAttnRes`
3. 用 `layers_per_block` 管理 block 边界

---

## 4. 这次到底改什么，不改什么

## 不改

下面这些保持原样：

1. `CausalSelfAttention` 的 token-time self-attention 公式
2. `q / k / v` 投影方式
3. causal mask
4. `sample.py` 的整体采样流程
5. `train.py` 的训练主循环
6. 数据准备逻辑

## 改

下面这些是主战场：

1. `GPTConfig` 增加 BAR 配置项
2. 在 `model.py` 新增 `BlockAttnRes`
3. 在 `Block` 内部增加 BAR 路径
4. 在 `GPT.forward()` 中维护 `blocks`
5. 让 `sample.py` 能通过 checkpoint 正常恢复新模型

---

## 5. 参考实现里最值得借鉴的代码结构

参考实现最关键的类是：

```python
class BlockAttnRes(nn.Module):
    def __init__(self, hidden_size):
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.norm = RMSNorm(hidden_size)

    def forward(self, blocks, partial_block):
        V = torch.stack(blocks + [partial_block], dim=0)
        K = self.norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.weight, K)
        scores = logits.softmax(dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', scores, V)
        return h
```

这个结构告诉我们 4 件事：

1. Query 可以非常简单，就是一个 learnable vector
2. score 是直接在深度维度上按 token 位置算的，不需要先 sequence mean pooling
3. 权重是在深度维 `dim=0` 做 softmax
4. 聚合值是完整 `(B, T, D)` 表示，而不是 pooled summary

这比之前的“summary 打分 + full value 聚合”更贴近参考实现。

---

## 6. BAR-v1-faithful 的机制定义

## 6.1 block 的定义

为了适配 `nanoGPT`，这次明确用：

```text
layers_per_block = n_layer // num_blocks
```

例如：

- `n_layer = 6`
- `num_blocks = 3`

则：

- `layers_per_block = 2`

也就是每两层 Transformer layer 组成一个 BAR block。

## 6.2 `blocks` 是什么

`blocks` 是一个列表，存的是：

```text
已经完成的历史 block 表示
```

每个元素形状是：

```text
(B, T, C)
```

## 6.3 `partial_block` 是什么

`partial_block` 表示：

```text
当前 block 内部正在逐步构建的 residual state
```

它不是历史块，也不是最终块，而是“当前块进行到一半时的状态”。

## 6.4 两个聚合器

每个 `Block` 内部要有两个聚合器：

1. `attn_res_agg`
   - 在 attention 子层之前聚合
2. `mlp_res_agg`
   - 在 MLP 子层之前聚合

这和参考实现保持一致。

---

## 7. 具体实现设计

## 7.1 `GPTConfig` 新增字段

在 [model.py](/home/chen/githubku/nanoGPT/model.py) 里的 `GPTConfig` 增加：

```text
use_block_attention_residuals: bool = False
attn_res_num_blocks: int = 3
attn_res_use_rmsnorm: bool = True
```

默认行为：

1. `False` 时完全走原版 nanoGPT
2. `True` 时启用 BAR 路径

不新增太多花哨开关，避免面试项目失焦。

## 7.2 新增 `RMSNorm`

因为参考实现的 `BlockAttnRes` 用的是 `RMSNorm`，建议在 [model.py](/home/chen/githubku/nanoGPT/model.py) 里新增一个最小版：

```python
class RMSNorm(nn.Module):
    ...
```

作用：

- 只服务于 `BlockAttnRes`
- 不替换原有 `LayerNorm`

这样可以最大程度保留 nanoGPT 主体结构不变，同时借鉴参考实现。

## 7.3 新增 `BlockAttnRes`

新增：

```python
class BlockAttnRes(nn.Module):
    def __init__(self, hidden_size):
        ...

    def forward(self, blocks, partial_block):
        ...
```

实现必须尽量贴近参考实现：

1. `weight` 是 `(C,)`
2. 把 `blocks + [partial_block]` 沿深度维 stack 成 `(N, B, T, C)`
3. 对 stack 结果做 `RMSNorm`
4. 用 `einsum('d, n b t d -> n b t', ...)` 计算 logits
5. 在 `dim=0` 做 softmax
6. 对完整值张量做加权和，输出 `(B, T, C)`

## 7.4 改 `Block.__init__()`

在 [model.py](/home/chen/githubku/nanoGPT/model.py) 的 `Block` 里新增：

1. `self.layer_idx`
2. `self.layers_per_block`
3. `self.use_block_attention_residuals`
4. `self.attn_res_agg`
5. `self.mlp_res_agg`

注意：

- 原始 `Block(config)` 目前没有 `layer_idx`
- 这次要把 layer index 显式传进去

因此 `GPT.__init__()` 里原本：

```python
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
```

要改成类似：

```python
h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)])
```

## 7.5 改 `Block.forward()`

当前原版 `Block.forward()` 是：

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

这次要改成双路径：

### baseline 路径

保持不变。

### BAR 路径

参考实现风格，伪代码如下：

```python
def forward(self, x, blocks=None):
    if not self.use_block_attention_residuals:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, blocks

    partial_block = x

    h = self.attn_res_agg(blocks, partial_block)
    if self.layer_idx % self.layers_per_block == 0:
        blocks.append(partial_block)
        partial_block = None

    attn_out = self.attn(self.ln_1(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out

    h = self.mlp_res_agg(blocks, partial_block)
    mlp_out = self.mlp(self.ln_2(h))
    partial_block = partial_block + mlp_out

    return partial_block, blocks
```

注意：

1. 这里不是逐点照抄参考实现，而是把它翻译成 nanoGPT 的最小兼容版本
2. baseline 和 BAR 必须共存
3. `Block.forward()` 的返回值会从单个 `x` 变成：
   - baseline：逻辑上返回 `x`
   - BAR：返回 `(x, blocks)`

为了减少分支复杂度，建议统一成：

```python
return x, blocks
```

然后在 `GPT.forward()` 里适配。

## 7.6 改 `GPT.forward()`

当前原版：

```python
x = self.transformer.drop(tok_emb + pos_emb)
for block in self.transformer.h:
    x = block(x)
```

新逻辑：

```python
x = self.transformer.drop(tok_emb + pos_emb)
blocks = []
for block in self.transformer.h:
    x, blocks = block(x, blocks)
```

在 baseline 模式下：

- `blocks` 可以始终原样透传，不参与计算

在 BAR 模式下：

- `blocks` 才真正作为状态列表被使用

这样 `GPT.forward()` 的改动量不大，但机制上更接近参考实现。

---

## 8. 训练与采样兼容要求

## 8.1 `train.py`

只做最小改动：

1. 增加默认配置：
   - `use_block_attention_residuals = False`
   - `attn_res_num_blocks = 3`
   - `attn_res_use_rmsnorm = True`
2. 把新字段纳入 `config_keys`
3. 把新字段纳入 `model_args`
4. `resume` 路径下确保这些结构性字段能从 checkpoint 恢复

## 8.2 新增配置文件

新增：

- [config/train_shakespeare_char_bar.py](/home/chen/githubku/nanoGPT/config/train_shakespeare_char_bar.py)

内容只覆盖最必要的字段：

```text
out_dir = 'out-shakespeare-char-bar'
use_block_attention_residuals = True
attn_res_num_blocks = 3
attn_res_use_rmsnorm = True
```

baseline 和 BAR 一定要分开目录。

## 8.3 `sample.py`

目标：

1. BAR 模型的 checkpoint 能通过 `GPTConfig(**checkpoint['model_args'])` 重建
2. 采样逻辑不需要额外分支

原则：

- `sample.py` 尽量不改，只依赖模型和 checkpoint 兼容性

---

## 9. 分析计划

今天不真实运行训练，但代码和文档要为分析准备好。

重点分析三类信号：

1. `train/val loss`
2. `BlockAttnRes` 的深度权重分布
3. 不同层/不同 block 的 hidden-state norm

最值得做的图：

1. baseline vs BAR 的 loss 曲线
2. `attn_res_agg` 的 depth 权重热力图
3. `mlp_res_agg` 的 depth 权重热力图

这样你就能证明：

```text
你不仅改了代码，而且这个 BAR 机制真的在“选择历史块表示”。
```

---

## 10. 今天的执行路径

今天只走这一条路径：

```text
baseline 跑通
-> 在 model.py 新增 RMSNorm 和 BlockAttnRes
-> 给 Block 增加 BAR 路径
-> 给 GPT.forward() 增加 blocks 状态
-> 训练配置接上
-> checkpoint 和 sample 兼容
-> baseline vs BAR 分析文档准备
-> 面试讲稿准备
```

不允许再退回“简单 depth mixer”方案。

---

## 11. 成功标准

今天完成后，必须能说：

1. 这个版本比之前的 `depth_residual_mixer` 更接近论文
2. 它借鉴了参考实现里的 `BlockAttnRes + blocks + partial_block + attn/mlp 双聚合`
3. 它仍然是面向 nanoGPT 的最小 faithful 版本，而不是完整论文复现

最低成功标准：

1. BAR 代码路径完整设计清楚
2. baseline / BAR 配置分离清楚
3. 训练、采样、分析、面试材料路径全部写清楚

---

## 12. 面试讲述模板

你最好的说法是：

```text
我一开始尝试的是一个更简化的 layer-wise depth mixer，
但后来我对照论文和一个参考实现做了机制修正，
把方案升级成了更接近 Block Attention Residuals 的版本。

这个版本的核心不是在 block 输入前加一个大 mixer，
而是维护历史完整 blocks 和当前 partial_block，
并在 attention 子层前和 MLP 子层前各做一次 block residual aggregation。

这样机制上更接近论文，也更能说明我不是在做一个随意的 attention-inspired 改动，
而是在认真对齐论文里的 residual aggregation 思路。
```
