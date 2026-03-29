# sample.py 逐阶段详细解析

这份文档专门解析 [sample.py](/home/chen/githubku/nanoGPT/sample.py)。

和 `train.py` 不一样，`sample.py` 几乎没有显式定义自己的 `def` 函数，而是一个**顺序执行的推理脚本**。所以这份解析会按下面两个层次展开：

1. `sample.py` 顶层执行阶段逐段解析
2. `sample.py` 真正依赖的核心函数 `model.generate()` 解析

如果你想从“训练结果如何变成文本”这个角度读 nanoGPT，这个文件就是入口。

---

## 1. 这个文件是做什么的

`sample.py` 的职责非常明确：

- 加载一个已经训练好的 checkpoint，或者直接加载 GPT-2 预训练模型
- 准备 prompt
- 调用模型生成若干新 token
- 把 token 解码成人类可读文本

一句话理解：

```text
sample.py = 模型加载 + prompt 编码 + 循环生成 + 文本解码
```

它不负责训练，也不负责数据预处理，只负责**推理阶段**。

---

## 2. 文件整体执行流程

你可以先记住这条主线：

```text
设置默认采样参数
-> 用 configurator.py 覆盖参数
-> 设置随机种子和精度上下文
-> 加载模型
-> 选择编码器/解码器
-> 编码起始 prompt
-> 调用 model.generate()
-> 把生成的 token 解码成文本
-> 打印输出
```

### 全局流程图

```mermaid
flowchart TD
    A[启动 sample.py] --> B[定义默认参数]
    B --> C[exec configurator.py]
    C --> D[设置 seed / device / autocast]
    D --> E{init_from}
    E -->|resume| F[加载 checkpoint]
    E -->|gpt2*| G[加载 GPT-2 预训练权重]
    F --> H[构建模型]
    G --> H
    H --> I[model.eval + to(device)]
    I --> J{是否存在 meta.pkl}
    J -->|是| K[使用数据集自己的编码解码]
    J -->|否| L[回退到 GPT-2 tokenizer]
    K --> M[读取 start prompt]
    L --> M
    M --> N[encode 成 token ids]
    N --> O[调用 model.generate]
    O --> P[decode 成文本]
    P --> Q[打印 num_samples 份结果]
```

---

## 3. 顶部默认参数区

这一段从：

```python
init_from = 'resume'
```

到：

```python
compile = False
```

定义的是**采样脚本的默认行为**。

### 3.1 `init_from`

```python
init_from = 'resume'
```

作用：

- 决定模型从哪里来

可选路径：

1. `'resume'`
   - 从本地训练输出目录中的 `ckpt.pt` 恢复
2. `'gpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`
   - 直接加载 OpenAI GPT-2 权重

这是整个 `sample.py` 中最重要的分支条件之一。

### 3.2 `out_dir`

```python
out_dir = 'out'
```

作用：

- 当 `init_from == 'resume'` 时，表示 checkpoint 所在目录

例如：

```bash
python sample.py --out_dir=out-shakespeare-char
```

表示从 `out-shakespeare-char/ckpt.pt` 读取训练好的模型。

### 3.3 `start`

```python
start = "\n"
```

作用：

- 指定生成的起始 prompt

它支持三种形式：

1. 普通字符串，如 `"ROMEO:"`
2. 特殊 token 字符串，如 `"<|endoftext|>"`
3. 文件输入形式，如 `"FILE:prompt.txt"`

第三种形式表示：

- 不是直接使用字符串本身
- 而是去读 `prompt.txt` 的文件内容，作为 prompt

### 3.4 `num_samples`

```python
num_samples = 10
```

作用：

- 生成多少份独立样本

注意这里不是一次生成 10 个 token，而是生成 10 条完整文本。

### 3.5 `max_new_tokens`

```python
max_new_tokens = 500
```

作用：

- 每条样本最多继续生成多少个新 token

这不是总长度，而是**在已有 prompt 后面再追加**的 token 数。

### 3.6 `temperature`

```python
temperature = 0.8
```

作用：

- 控制采样随机性

直觉上：

- `temperature < 1.0`：更保守，更确定
- `temperature = 1.0`：原始分布
- `temperature > 1.0`：更发散，更随机

### 3.7 `top_k`

```python
top_k = 200
```

作用：

- 只保留概率最高的前 `k` 个候选 token，再做采样

直觉上：

- `top_k` 越小，越保守
- `top_k` 越大，越开放

### 3.8 `seed`

```python
seed = 1337
```

作用：

- 控制随机采样可复现性

如果其他条件不变，固定 seed 有助于复现生成结果。

### 3.9 `device`

```python
device = 'cuda'
```

作用：

- 指定推理运行设备

常见取值：

- `'cpu'`
- `'cuda'`
- `'cuda:0'`
- `'mps'`

### 3.10 `dtype`

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

作用：

- 指定 autocast 使用的数据精度

设计逻辑：

- 如果 GPU 支持 `bfloat16`，优先使用它
- 否则回退到 `float16`

### 3.11 `compile`

```python
compile = False
```

作用：

- 是否对推理模型启用 `torch.compile`

默认关闭，因为：

- 推理脚本往往追求简单稳定
- 短时间采样时，compile 的启动开销可能不划算

---

## 4. 配置覆盖阶段

这一句：

```python
exec(open('configurator.py').read())
```

含义和 `train.py` 一样：

- 允许配置文件和命令行参数覆盖默认值

例如：

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="ROMEO:"
```

运行时就会覆盖：

- `out_dir`
- `device`
- `start`

### 这一阶段的意义

它让 `sample.py` 不需要写复杂的参数解析器，但仍然可以灵活改变行为。

---

## 5. 随机种子与数值环境初始化

接下来这几行：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

主要做两件事：

1. 设置随机种子
2. 允许 CUDA 在合适场景下使用 TF32 加速

### 5.1 `torch.manual_seed(seed)`

作用：

- 设置 PyTorch 的随机数种子

这会影响：

- 采样阶段的随机 token 选择

### 5.2 `torch.cuda.manual_seed(seed)`

作用：

- 设置 CUDA 侧随机数种子

如果使用 GPU 采样，这一步是必要的。

### 5.3 TF32 配置

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

作用：

- 允许在一些矩阵运算中使用 TF32
- 以换取更高吞吐

---

## 6. `device_type`、`ptdtype` 和 `ctx`

这段代码：

```python
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

作用是构造一个统一的推理上下文。

### 6.1 `device_type`

作用：

- 把具体设备名折叠成更粗的类别

例如：

- `cuda`
- `cuda:0`
- `cuda:1`

都会被归为 `'cuda'`

### 6.2 `ptdtype`

作用：

- 把字符串形式的 `dtype` 映射成 PyTorch 真正的数据类型对象

### 6.3 `ctx`

作用：

- 在 CPU 上用空上下文
- 在 GPU 上用自动混合精度 `autocast`

后面推理时会写成：

```python
with torch.no_grad():
    with ctx:
        ...
```

这样就统一了 CPU 和 GPU 的执行逻辑。

---

## 7. 模型加载阶段

这是 `sample.py` 的核心分支。

```python
if init_from == 'resume':
    ...
elif init_from.startswith('gpt2'):
    ...
```

### 7.1 路径一：从 checkpoint 恢复

当：

```python
init_from == 'resume'
```

时，执行以下逻辑。

#### 第一步：定位 checkpoint

```python
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
```

也就是：

- 在 `out_dir` 下寻找 `ckpt.pt`

#### 第二步：读取 checkpoint

```python
checkpoint = torch.load(ckpt_path, map_location=device)
```

作用：

- 把 checkpoint 读到当前设备上

这个 `checkpoint` 一般包含：

- `model`
- `optimizer`
- `model_args`
- `iter_num`
- `best_val_loss`
- `config`

虽然 `sample.py` 只关心其中一部分，但整个训练状态都在里面。

#### 第三步：重建模型配置

```python
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
```

这一步非常关键。

因为要加载权重，必须先构造出一个**结构完全一致**的模型对象。

所以这里不是随便 new 一个 GPT，而是：

- 用训练时保存下来的 `model_args` 重建结构

#### 第四步：读取 `state_dict`

```python
state_dict = checkpoint['model']
```

这里拿到的是参数张量字典。

#### 第五步：修复 `_orig_mod.` 前缀

```python
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
```

作用：

- 兼容某些由 `torch.compile` 或其他包装方式引入的 key 前缀

也就是说：

- 如果参数名形如 `_orig_mod.transformer.h.0...`
- 就把 `_orig_mod.` 去掉

#### 第六步：加载参数

```python
model.load_state_dict(state_dict)
```

至此，本地训练得到的模型就恢复完成了。

---

### 7.2 路径二：直接加载 GPT-2

当：

```python
init_from.startswith('gpt2')
```

时，执行：

```python
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

作用：

- 直接使用 `model.py` 中的 `GPT.from_pretrained(...)`
- 从 Hugging Face / GPT-2 权重构造模型

这里还显式指定：

- `dropout=0.0`

因为推理时不需要 dropout。

---

## 8. `model.eval()`、`model.to(device)` 和可选 compile

模型加载完成后，接着做三件事。

### 8.1 `model.eval()`

```python
model.eval()
```

作用：

- 把模型切换到推理模式

这会影响：

- dropout 关闭
- 某些层的训练态行为关闭

### 8.2 `model.to(device)`

```python
model.to(device)
```

作用：

- 把模型放到指定设备上

### 8.3 可选 compile

```python
if compile:
    model = torch.compile(model)
```

作用：

- 如果显式开启，就对模型做编译优化

默认关闭，是为了减少推理脚本的复杂度和冷启动时间。

---

## 9. 编码器 / 解码器选择阶段

这个阶段很关键，因为模型输出的是 token id，不是文本。

所以必须解决两个问题：

1. 怎么把字符串 prompt 编码成 token id
2. 怎么把生成出来的 token id 解码回文本

### 9.1 `load_meta = False`

```python
load_meta = False
```

这是一个开关变量，用来判断是否能使用训练数据集自己的编码器。

### 9.2 判断是否存在 `meta.pkl`

```python
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
```

这段逻辑的意思是：

只有在以下条件都满足时，才尝试使用 `meta.pkl`：

1. 当前是从本地 checkpoint 恢复
2. checkpoint 中记录了训练配置
3. 配置里有 `dataset`
4. `data/<dataset>/meta.pkl` 文件真实存在

### 为什么这样设计

因为：

- 字符级数据集通常有自己的 `stoi/itos`
- 如果不恢复这个映射，模型输出的整数就没法正确解码成字符

---

### 9.3 路径一：使用数据集自己的 `meta.pkl`

如果 `load_meta == True`，执行：

```python
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

作用：

- 恢复训练时的字符级编码方式

#### `encode`

```python
encode = lambda s: [stoi[c] for c in s]
```

作用：

- 把字符串中的每个字符映射成 id

适用于：

- `shakespeare_char` 这类字符级模型

#### `decode`

```python
decode = lambda l: ''.join([itos[i] for i in l])
```

作用：

- 把 token id 列表还原成字符字符串

---

### 9.4 路径二：回退到 GPT-2 tokenizer

如果没有 `meta.pkl`，执行：

```python
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
```

作用：

- 使用 GPT-2 的 BPE tokenizer

这个分支适用于：

- 直接加载 GPT-2 模型
- 或者训练时没有保存自定义字符映射的情况

### 这里的设计思想

可以理解为：

- 能用“训练时原生编码器”就优先用
- 否则回退到通用的 GPT-2 tokenizer

---

## 10. prompt 读取和编码阶段

这部分代码：

```python
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```

做了三件事：

1. 如果 `start` 是文件路径形式，就读取文件内容
2. 用 `encode` 把字符串变成 token id
3. 构造 batch 维度为 1 的输入张量

### 10.1 `FILE:` 分支

如果：

```python
start = "FILE:prompt.txt"
```

则会读取：

- `prompt.txt` 文件内容

这样可以输入更长、更复杂的 prompt，而不用在命令行里写一大串文本。

### 10.2 `start_ids = encode(start)`

作用：

- 把 prompt 文本转成 token id 列表

### 10.3 构造 `x`

```python
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```

这一步很值得注意。

原本：

- `start_ids` 只是一个一维列表，形状类似 `(t,)`

模型需要的是 batch 形式输入：

- `(b, t)`

所以这里通过 `[None, ...]` 添加了 batch 维，变成：

- `(1, t)`

也就是：

- 一次只生成一条样本的起始上下文

---

## 11. 最终生成阶段

最后这段才是真正把模型跑起来的地方：

```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```

### 11.1 `torch.no_grad()`

作用：

- 关闭梯度计算

因为这里是推理，不需要 backward。

这样可以：

- 节省显存
- 提高速度

### 11.2 `with ctx`

作用：

- 在 GPU 上启用 autocast
- 在 CPU 上保持空上下文

### 11.3 `for k in range(num_samples)`

作用：

- 重复生成多份样本

这里并不是把 `x` 改成多 batch，而是：

- 连续调用多次 `generate()`

### 11.4 `y = model.generate(...)`

这里调用的并不是 `sample.py` 自己的函数，而是 `model.py` 中 `GPT.generate()`。

传入参数包括：

- 初始上下文 `x`
- 最多生成的 token 数
- `temperature`
- `top_k`

### 11.5 `decode(y[0].tolist())`

作用：

1. `y` 是 tensor
2. `y[0]` 取 batch 中第一条样本
3. `tolist()` 转成 Python 列表
4. `decode(...)` 把 token id 序列还原成文本

### 11.6 分隔线

```python
print('---------------')
```

只是为了让多条样本在终端输出中更容易区分。

---

## 12. `sample.py` 自己“没有函数”的理解方式

严格说，这个文件没有显式定义：

- `def foo(...)`

所以如果你问“逐函数解析”，这里更准确的说法应该是：

- **逐执行阶段解析**

因为它本质上是一个命令式脚本。

但它依赖了几个外部函数/方法：

1. `configurator.py` 的配置覆盖逻辑
2. `GPTConfig(...)`
3. `GPT(...)`
4. `GPT.from_pretrained(...)`
5. `model.generate(...)`

其中最核心的是最后一个。

---

## 13. `model.generate()` 是如何配合 `sample.py` 工作的

`sample.py` 负责准备输入和输出，真正的循环生成逻辑在 `model.py` 的 `generate()` 中。

你可以把责任分工理解成：

### `sample.py` 负责

1. 参数准备
2. 模型加载
3. 编码与解码
4. 调用生成接口
5. 打印结果

### `model.generate()` 负责

1. 截断上下文
2. 前向推理
3. 取最后位置 logits
4. 应用 temperature 和 top-k
5. 采样下一个 token
6. 把新 token 拼回序列

---

## 14. `model.generate()` 逐步解析

虽然它不在 `sample.py` 里，但如果不讲它，`sample.py` 的核心就断了。

`generate()` 的逻辑可以概括为：

```text
当前序列
-> 裁剪到 block_size
-> forward
-> 取最后 token 的 logits
-> 调整分布
-> 采样一个新 token
-> 拼接回序列
-> 重复直到达到 max_new_tokens
```

### 对应流程图

```mermaid
flowchart TD
    A[当前 token 序列 idx] --> B[如果太长则裁剪到 block_size]
    B --> C[调用 self(idx_cond)]
    C --> D[取最后一个位置 logits]
    D --> E[除以 temperature]
    E --> F[top_k 截断 可选]
    F --> G[softmax 转概率]
    G --> H[torch.multinomial 采样]
    H --> I[把新 token 拼回 idx]
    I --> J{是否达到 max_new_tokens}
    J -->|否| A
    J -->|是| K[返回完整序列]
```

### 14.1 裁剪上下文

```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
```

作用：

- 如果当前上下文已经超过模型最大上下文长度，就只保留最后 `block_size` 个 token

### 14.2 前向推理

```python
logits, _ = self(idx_cond)
```

作用：

- 调用 `GPT.forward()`

在没有 `targets` 的情况下，`forward()` 会只返回最后一个位置的 logits。

### 14.3 取最后位置 logits

```python
logits = logits[:, -1, :] / temperature
```

作用：

- 只关心“下一个 token”的分布

### 14.4 top-k 截断

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```

作用：

- 把前 `k` 以外的候选 token 直接屏蔽掉

### 14.5 softmax 概率化

```python
probs = F.softmax(logits, dim=-1)
```

作用：

- 把 logits 转成概率分布

### 14.6 采样新 token

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

作用：

- 按概率分布抽样一个新 token

### 14.7 拼回原序列

```python
idx = torch.cat((idx, idx_next), dim=1)
```

作用：

- 把新 token 接到已有序列末尾

这样就进入下一轮生成。

---

## 15. 这个文件最值得记住的 8 个点

1. `sample.py` 是推理脚本，不是训练脚本。
2. 它没有显式函数定义，所以要按“执行阶段”而不是“源码 def”来读。
3. 最关键的分支是 `init_from == 'resume'` 还是 `init_from.startswith('gpt2')`。
4. `meta.pkl` 的作用是恢复训练时的数据集编码方式。
5. 没有 `meta.pkl` 时，会回退到 GPT-2 tokenizer。
6. `start` 支持直接字符串，也支持 `FILE:` 文件输入。
7. 真正的 token-by-token 生成循环在 `model.generate()` 中。
8. `temperature` 和 `top_k` 是控制生成风格的两个核心参数。

---

## 16. 一页总结

如果你只想记住最核心的内容，可以记这四句话：

1. `sample.py` 的本质是：加载模型、编码 prompt、调用 `generate()`、解码文本。
2. 它优先使用 checkpoint 对应数据集的 `meta.pkl` 来做编码解码。
3. 如果没有 `meta.pkl`，就回退到 GPT-2 的 BPE tokenizer。
4. 真正的逐 token 生成逻辑不在 `sample.py`，而在 `model.py` 的 `generate()` 里。
