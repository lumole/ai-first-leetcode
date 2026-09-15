# Transformer 结构

## Embedding

CodeKy：[1430. Embedding 查表](https://codeky.online/problem/1670)
Deep-ML：[Scaled Token Embedding Lookup](https://www.deep-ml.com/problems/1052)
LeetGPU（相关）：[Token Embedding Layer](https://leetgpu.com/challenges/token-embedding-layer)

token 先经 embedding 查表映射成向量，查表结果乘 $\sqrt{d_{model}}$ 让量级和位置编码对齐。很多语言模型让输入 embedding 和输出 logits 投影共享同一套权重，叫 tied embedding，既省参数又让两端的词表示一致。

```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.embed.weight   # 权重共享：输入 embedding 与输出投影同一套

    def encode(self, idx):
        # 查表后乘 sqrt(d_model)，和位置编码量级对齐
        return self.embed(idx) * math.sqrt(self.d_model)

    def decode(self, h):
        # 隐状态投影回词表 logits
        return self.proj(h)

# 输入输出样例
emb = TokenEmbedding(vocab_size=10000, d_model=512)
idx = torch.randint(0, 10000, (2, 10))     # (batch=2, seq=10) 的 token 下标
h = emb.encode(idx)
print(h.shape, emb.decode(h).shape)        # torch.Size([2, 10, 512]) torch.Size([2, 10, 10000])
```

## Positional Encoding

CodeKy：[1377. 正弦位置编码](https://codeky.online/problem/1616)
Deep-ML：[Sinusoidal Positional Encoding](https://www.deep-ml.com/problems/906)
LeetGPU：未找到直接对应题目

self-attention 本身不区分位置，需要额外注入顺序信息。正弦位置编码给每个位置算一组固定的 sin/cos 值，偶数维用 sin、奇数维用 cos，不同维度的波长按几何级数从 $2\pi$ 拉到 $10000\cdot 2\pi$ 。它不带参数，且对没见过的长度也能外推。实现时容易写错的是 $\frac{1}{10000^{2i/d}}$ 这一项，用对数加指数算更稳。

$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$

```python
import torch
import math

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)                    # (T, 1)
    i = torch.arange(0, d_model, 2)                             # 偶数维下标
    div = torch.exp(i * (-math.log(10000.0) / d_model))         # 1 / 10000^(2i/d)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)                          # 偶数维用 sin
    pe[:, 1::2] = torch.cos(pos * div)                          # 奇数维用 cos
    return pe                                                   # (T, d_model)

# 输入输出样例
pe = sinusoidal_positional_encoding(seq_len=4, d_model=8)
print(pe.shape)   # torch.Size([4, 8])
print(pe[0])      # 位置 0 各维为 sin(0)/cos(0)：tensor([0., 1., 0., 1., 0., 1., 0., 1.])
```

## RoPE

CodeKy：[3053. 旋转位置编码 RoPE](https://codeky.online/problem/3721)
Deep-ML：[Rotary Positional Embeddings (RoPE)](https://www.deep-ml.com/problems/381)
LeetGPU：[Rotary Positional Embedding](https://leetgpu.com/challenges/rotary-positional-embedding)

RoPE 不是把位置向量加到输入上，而是对每个位置的 Query 和 Key 做旋转。设每个注意力头的维度为 $d_k$（必须是偶数），把相邻的两维 $(2i, 2i+1)$ 看成一个二维平面，其中 $i=0,1,\ldots,d_k/2-1$。

### 旋转公式

第 $i$ 个二维平面使用一个固定频率：

$$
\theta_i = \mathrm{base}^{-2i/d_k},\qquad \mathrm{base}=10000
$$

位置 $m$ 在这个平面上的旋转角度是：

$$
\phi_{m,i}=m\theta_i
$$

如果当前位置向量在这一对维度上的值是 $(x_{m,2i},x_{m,2i+1})$，旋转后的值就是二维旋转矩阵的结果：

$$
\begin{bmatrix}
x'_{m,2i}\\
x'_{m,2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_{m,i} & -\sin\phi_{m,i}\\
\sin\phi_{m,i} & \phantom{-}\cos\phi_{m,i}
\end{bmatrix}
\begin{bmatrix}
x_{m,2i}\\
x_{m,2i+1}
\end{bmatrix}
$$

展开后，每一对维度分别计算：

$$
x'_{m,2i}=x_{m,2i}\cos\phi_{m,i}-x_{m,2i+1}\sin\phi_{m,i}
$$

$$
x'_{m,2i+1}=x_{m,2i}\sin\phi_{m,i}+x_{m,2i+1}\cos\phi_{m,i}
$$

例如，当某一对维度为 $(1,2)$，旋转角为 $\pi/2$ 时，旋转后为 $(-2,1)$。位置 $m=0$ 时角度为 0，所以所有维度保持不变。

实现时，`build_rope_cache` 先为每个位置和每个二维平面保存 `cos(m * theta_i)` 与 `sin(m * theta_i)`：`cos`、`sin` 的形状都是 `(seq_len, head_dim // 2)`。`apply_rope` 再取出 `x[..., 0::2]` 和 `x[..., 1::2]`，按上面的两条公式计算，最后交错拼回原来的维度。对 Query 和 Key 分别应用同样的操作后，位置 $m$ 和位置 $n$ 的点积等价于使用相对旋转角 $(n-m)\theta_i$，这就是 RoPE 注入相对位置信息的原因。

```python
import torch

def build_rope_cache(seq_len, d_k, base=10000):
    # 每隔两维共用一个频率 theta，形状 (d_k/2,)
    theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)            # (T, d_k/2)，元素是 m * theta
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, heads, T, d_k)，把相邻两维 (x1, x2) 当作复数做旋转
    x1, x2 = x[..., 0::2], x[..., 1::2]        # 偶数维、奇数维
    cos, sin = cos[None, None], sin[None, None]  # 广播到 batch、head 维
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)  # 交错拼回原维度

# 输入输出样例：旋转后形状不变
cos, sin = build_rope_cache(seq_len=16, d_k=64)
x = torch.randn(2, 8, 16, 64)                  # (batch=2, heads=8, seq=16, d_k=64)
print(apply_rope(x, cos, sin).shape)           # torch.Size([2, 8, 16, 64])
```

## Residual Connection

CodeKy：[1403. 残差块](https://codeky.online/problem/1643)
Deep-ML：[Implement a Simple Residual Block with Shortcut Connection](https://www.deep-ml.com/problems/113)
LeetGPU（相关）：[Fused Residual Add and RMS Norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)

残差连接把子层的输出加回它的输入，给梯度留一条直通路径，让深层网络能稳定训练。现代 Transformer 普遍用 Pre-Norm，也就是先做 LayerNorm 再进子层，最后加残差，比原始的 Post-Norm 更容易训深。把这套结构抽成一个通用包装，任意子层都能套用。

```python
import torch
import torch.nn as nn

class ResidualSubLayer(nn.Module):
    def __init__(self, d_model, sublayer, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args):
        # Pre-Norm：先归一化再进子层，子层输出加回原始输入
        return x + self.dropout(self.sublayer(self.norm(x), *args))

# 输入输出样例
sub = ResidualSubLayer(d_model=512, sublayer=nn.Linear(512, 512))
x = torch.randn(2, 10, 512)                    # (batch=2, seq=10, d_model=512)
print(sub(x).shape)                            # torch.Size([2, 10, 512])
```

## SwiGLU

CodeKy：[3071. SwiGLU 前馈网络](https://codeky.online/problem/3739)
Deep-ML：[Implement SwiGLU activation function](https://www.deep-ml.com/problems/156)
LeetGPU：[SwiGLU MLP Block](https://leetgpu.com/challenges/swiglu-mlp-block)

SwiGLU 用门控替代普通前馈网络的单条线性路径。输入分别过两个线性层，一条经 SiLU（也叫 Swish）当门控，逐元素乘到另一条上，再投影回 d_model。它比 ReLU 前馈表达力更强，被 Llama、PaLM 等采用；因为多了一个矩阵，中间维度通常缩到约 8/3 倍 d_model 以保持参数量相当。

$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_1)\odot xW_3\big)W_2,\qquad \mathrm{SiLU}(x)=x\cdot\sigma(x)$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)   # 门控分支
        self.w_up = nn.Linear(d_model, d_ff, bias=False)     # 数值分支
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # 门控分支过 SiLU 后逐元素乘到数值分支，再投影回去
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# 输入输出样例
ffn = SwiGLU(d_model=512, d_ff=1376)         # d_ff 常取约 8/3 倍 d_model
x = torch.randn(2, 10, 512)
print(ffn(x).shape)                          # torch.Size([2, 10, 512])
```

## Transformer Encoder Block

CodeKy：[3299. Post-Norm 编码器层](https://codeky.online/problem/3977)
Deep-ML：[Implement a Transformer Encoder Block](https://www.deep-ml.com/problems/905)
LeetGPU：未找到直接对应题目

把上面的组件拼起来就是一个 encoder 层：multi-head self-attention 子层加残差，再接一个前馈网络子层加残差，两个子层各带一次 Pre-Norm。前馈网络是线性、激活、再线性三步，中间维度通常是 d_model 的四倍，作用是在注意力之后补非线性表达能力。

```python
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)   # 见注意力机制一节
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 两个 Pre-Norm 子层：自注意力、前馈，各带一个残差
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# 输入输出样例
block = TransformerEncoderBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)                    # (batch=2, seq=10, d_model=512)
print(block(x).shape)                          # torch.Size([2, 10, 512])
```

## Transformer Decoder Block

CodeKy：[3300. Post-Norm 解码器层](https://codeky.online/problem/3978)
Deep-ML：未找到直接对应题目
LeetGPU（相关）：[GPT-2 Transformer Block](https://leetgpu.com/challenges/gpt-2-transformer-block)

decoder block 比 encoder block 多一个交叉注意力子层，一共三个子层：带因果掩码的 masked self-attention 保证只看已生成的部分，cross-attention 让它关注 encoder 输出，最后是前馈网络，三个子层各带 Pre-Norm 和残差。

```python
import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)   # 带因果掩码，见注意力机制一节
        self.cross_attn = CrossAttention(d_model, n_heads)      # 见注意力机制一节
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, self_mask=None, cross_mask=None):
        # 三个 Pre-Norm 子层：因果自注意力、交叉注意力、前馈，各带残差
        x = x + self.dropout(self.self_attn(self.norm1(x), self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context, cross_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x

# 输入输出样例
block = TransformerDecoderBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 7, 512)            # decoder 序列
context = torch.randn(2, 10, 512)     # encoder 输出
print(block(x, context).shape)        # torch.Size([2, 7, 512])
```

---

Source: [https://common-doc.pages.dev/ai_basics/ai_coding/transformer/](https://common-doc.pages.dev/ai_basics/ai_coding/transformer/)
