from __future__ import annotations


Contract = tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]


CONTRACTS: dict[str, Contract] = {
    "attention.softmax": (
        "def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:\n    pass",
        ("`class_count`：被归一化维的元素数。",),
        ("`x: np.ndarray`，形状 `(..., class_count)`；`...` 是任意前导维度。", "`axis: int`，归一化维，默认最后一维。"),
        ("`np.ndarray`，形状与 `x` 相同；沿 `axis` 的每组元素和为 1。",),
    ),
    "attention.sdpa": (
        "def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:\n    pass",
        ("`batch`：批大小。", "`heads`：注意力头数。", "`query_len`：Query 序列长度。", "`key_len`：Key/Value 序列长度。", "`head_dim`：每个头的 Query/Key 维度，`q` 与 `k` 必须一致。", "`value_dim`：每个头的 Value 维度，可以与 `head_dim` 不同。"),
        ("`q: torch.Tensor`，形状 `(batch, heads, query_len, head_dim)`。", "`k: torch.Tensor`，形状 `(batch, heads, key_len, head_dim)`。", "`v: torch.Tensor`，形状 `(batch, heads, key_len, value_dim)`；`k` 和 `v` 的 `key_len` 必须一致。", "`mask: torch.Tensor | None`，必须是 `torch.bool`；形状可广播到 `(batch, heads, query_len, key_len)`。`True` 表示允许关注，`False` 表示屏蔽该 Query-Key 位置。"),
        ("返回二元组 `(output, attn)`。`output: torch.Tensor` 的形状为 `(batch, heads, query_len, value_dim)`，是用注意力权重加权求和后的 Value。", "`attn: torch.Tensor` 的形状为 `(batch, heads, query_len, key_len)`，是沿 `key_len` 维 Softmax 后的概率权重；每行和为 1，mask 为 `False` 的位置权重必须为 0。"),
    ),
    "attention.causal_mask": (
        "def causal_mask(seq_len: int) -> torch.Tensor:\n    pass",
        ("`seq_len`：序列长度。",),
        ("`seq_len: int`。",),
        ("`torch.Tensor`，形状 `(seq_len, seq_len)`，类型 `torch.bool`；第 `row` 行只能访问 `0` 到 `row` 列。",),
    ),
    "attention.mha": (
        "class MultiHeadAttention(nn.Module):\n    def __init__(self, model_dim: int, heads: int) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`seq_len`：序列长度。", "`model_dim`：模型总维度。", "`heads`：头数，`model_dim % heads == 0`。", "`head_dim`：每个头维度，等于 `model_dim // heads`。"),
        ("`x: torch.Tensor`，形状 `(batch, seq_len, model_dim)`。", "`mask: torch.Tensor | None`，可广播到 `(batch, heads, seq_len, seq_len)`。"),
        ("`torch.Tensor`，形状 `(batch, seq_len, model_dim)`。",),
    ),
    "attention.gqa": (
        "class GroupedQueryAttention(nn.Module):\n    def __init__(self, model_dim: int, heads: int, kv_heads: int) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`seq_len`：序列长度。", "`model_dim`：模型维度。", "`heads`：Query 头数。", "`kv_heads`：Key/Value 头数，且 `heads % kv_heads == 0`。"),
        ("`x: torch.Tensor`，形状 `(batch, seq_len, model_dim)`。", "`mask: torch.Tensor | None`，可广播到注意力分数形状。"),
        ("`torch.Tensor`，形状 `(batch, seq_len, model_dim)`。",),
    ),
    "attention.cross": (
        "class CrossAttention(nn.Module):\n    def __init__(self, model_dim: int, heads: int) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`query_len`：`x` 的序列长度。", "`context_len`：`context` 的序列长度。", "`model_dim`：模型维度。"),
        ("`x: torch.Tensor`，形状 `(batch, query_len, model_dim)`。", "`context: torch.Tensor`，形状 `(batch, context_len, model_dim)`。", "`mask: torch.Tensor | None`，可广播到 `(batch, heads, query_len, context_len)`。"),
        ("`torch.Tensor`，形状 `(batch, query_len, model_dim)`。",),
    ),
    "attention.kv_cache": (
        "class CachedSelfAttention(nn.Module):\n    def __init__(self, model_dim: int, heads: int) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, cache: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:\n        pass",
        ("`batch`：批大小。", "`new_len`：本次输入 token 数。", "`cached_len`：已有缓存长度。", "`model_dim`：模型维度。", "`head_dim`：每个头的维度。"),
        ("`x: torch.Tensor`，形状 `(batch, new_len, model_dim)`。", "`cache: dict | None`，键 `k`、`v` 的形状均为 `(batch, heads, cached_len, head_dim)`。"),
        ("`output: torch.Tensor`，形状 `(batch, new_len, model_dim)`。", "更新后的 `cache`，长度变为 `cached_len + new_len`。"),
    ),
    "transformer.embedding": (
        "class TokenEmbedding(nn.Module):\n    def __init__(self, vocab_size: int, model_dim: int) -> None:\n        pass\n\n    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:\n        pass\n\n    def decode(self, hidden: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`seq_len`：序列长度。", "`vocab_size`：词表大小。", "`model_dim`：嵌入维度。"),
        ("`token_ids: torch.Tensor`，整数 Tensor，形状 `(batch, seq_len)`。", "`hidden: torch.Tensor`，形状 `(batch, seq_len, model_dim)`。"),
        ("`encode` 返回形状 `(batch, seq_len, model_dim)`。", "`decode` 返回 logits，形状 `(batch, seq_len, vocab_size)`；输入嵌入与输出投影共享权重。"),
    ),
    "transformer.positional_encoding": (
        "def sinusoidal_positional_encoding(seq_len: int, model_dim: int) -> torch.Tensor:\n    pass",
        ("`seq_len`：序列长度。", "`model_dim`：位置向量维度，要求为偶数。"),
        ("`seq_len: int`。", "`model_dim: int`。"),
        ("`torch.Tensor`，形状 `(seq_len, model_dim)`；第 0 维是位置，第 1 维是特征。",),
    ),
    "transformer.rope": (
        "def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:\n    pass\n\n\ndef apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:\n    pass",
        ("`batch`：批大小。", "`heads`：头数。", "`seq_len`：序列长度。", "`head_dim`：每个头维度，要求为偶数。"),
        ("`build_rope_cache(seq_len, head_dim, base)`：`base` 为频率基数，默认 `10000`。", "`x: torch.Tensor`，形状 `(batch, heads, seq_len, head_dim)`。", "`cos`、`sin`：由缓存函数返回的余弦和正弦表，形状均为 `(seq_len, head_dim // 2)`；第 `[position, pair]` 个元素对应 `cos(position * base ** (-2 * pair / head_dim))` 和 `sin(position * base ** (-2 * pair / head_dim))`。"),
        ("`build_rope_cache` 返回二元组 `(cos, sin)`，不是加到 `x` 上的位置向量。", "`apply_rope` 返回一个 Tensor，形状与 `x` 相同；对每个相邻维度对 `(2 * pair, 2 * pair + 1)` 按二维旋转公式计算。"),
    ),
    "transformer.residual": (
        "class ResidualSubLayer(nn.Module):\n    def __init__(self, model_dim: int, sublayer: nn.Module, dropout: float = 0.1) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`model_dim`：最后一维特征数。",),
        ("`x: torch.Tensor`，形状 `(..., model_dim)`。", "`sublayer: nn.Module`，输入输出形状均为 `(..., model_dim)`。"),
        ("`torch.Tensor`，形状与 `x` 相同，使用 Pre-Norm 残差。",),
    ),
    "transformer.swiglu": (
        "class SwiGLU(nn.Module):\n    def __init__(self, model_dim: int, hidden_dim: int) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`seq_len`：序列长度。", "`model_dim`：输入输出维度。", "`hidden_dim`：前馈中间维度。"),
        ("`x: torch.Tensor`，形状 `(batch, seq_len, model_dim)`。",),
        ("`torch.Tensor`，形状 `(batch, seq_len, model_dim)`。",),
    ),
    "transformer.encoder_block": (
        "class TransformerEncoderBlock(nn.Module):\n    def __init__(self, model_dim: int, heads: int, hidden_dim: int, dropout: float = 0.1) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`seq_len`：序列长度。", "`model_dim`：模型维度。", "`heads`：注意力头数。", "`hidden_dim`：前馈中间维度。"),
        ("`x: torch.Tensor`，形状 `(batch, seq_len, model_dim)`。", "`mask: torch.Tensor | None`，可广播到 `(batch, heads, seq_len, seq_len)`。"),
        ("`torch.Tensor`，形状 `(batch, seq_len, model_dim)`。",),
    ),
    "transformer.decoder_block": (
        "class TransformerDecoderBlock(nn.Module):\n    def __init__(self, model_dim: int, heads: int, hidden_dim: int, dropout: float = 0.1) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor, context: torch.Tensor, self_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`target_len`：decoder 序列长度。", "`context_len`：encoder 序列长度。", "`model_dim`：模型维度。"),
        ("`x: torch.Tensor`，形状 `(batch, target_len, model_dim)`。", "`context: torch.Tensor`，形状 `(batch, context_len, model_dim)`。", "`self_mask` 可广播到 `(batch, heads, target_len, target_len)`。", "`cross_mask` 可广播到 `(batch, heads, target_len, context_len)`。"),
        ("`torch.Tensor`，形状 `(batch, target_len, model_dim)`。",),
    ),
    "normalization.batchnorm": (
        "class BatchNorm1d(nn.Module):\n    def __init__(self, feature_dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`batch`：批大小。", "`feature_dim`：特征通道数。"),
        ("`x: torch.Tensor`，形状 `(batch, feature_dim)`。",),
        ("`torch.Tensor`，形状与 `x` 相同；训练沿 batch 维统计，推理使用滑动平均。",),
    ),
    "normalization.layernorm": (
        "class LayerNorm(nn.Module):\n    def __init__(self, model_dim: int, eps: float = 1e-5) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`model_dim`：最后一维的特征数。",),
        ("`x: torch.Tensor`，形状 `(..., model_dim)`。",),
        ("`torch.Tensor`，形状与 `x` 相同；每个前导位置独立沿最后一维归一化。",),
    ),
    "normalization.rmsnorm": (
        "class RMSNorm(nn.Module):\n    def __init__(self, model_dim: int, eps: float = 1e-6) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        pass",
        ("`model_dim`：最后一维的特征数。",),
        ("`x: torch.Tensor`，形状 `(..., model_dim)`。",),
        ("`torch.Tensor`，形状与 `x` 相同；仅用最后一维的均方根缩放。",),
    ),
    "regularization.dropout": (
        "class Dropout(nn.Module):\n    def __init__(self, p: float = 0.5) -> None:\n        pass\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        pass",
        (),
        ("`x: torch.Tensor`，任意形状。", "`p: float`，丢弃概率，范围 `[0, 1)`。"),
        ("`torch.Tensor`，形状与 `x` 相同。",),
    ),
    "activation.relu": ("def relu(x: torch.Tensor) -> torch.Tensor:\n    pass", (), ("`x: torch.Tensor`，任意形状。",), ("`torch.Tensor`，形状与 `x` 相同。",)),
    "activation.gelu": ("def gelu(x: torch.Tensor) -> torch.Tensor:\n    pass", (), ("`x: torch.Tensor`，任意形状。",), ("`torch.Tensor`，形状与 `x` 相同。",)),
    "activation.sigmoid": ("def sigmoid(x: torch.Tensor) -> torch.Tensor:\n    pass", (), ("`x: torch.Tensor`，任意形状。",), ("`torch.Tensor`，形状与 `x` 相同。",)),
    "activation.tanh": ("def tanh(x: torch.Tensor) -> torch.Tensor:\n    pass", (), ("`x: torch.Tensor`，任意形状。",), ("`torch.Tensor`，形状与 `x` 相同。",)),
    "loss.cross_entropy": (
        "def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:\n    pass",
        ("`batch`：样本数。", "`class_count`：类别数。"),
        ("`logits: torch.Tensor`，形状 `(batch, class_count)`。", "`target: torch.Tensor`，整数 Tensor，形状 `(batch,)`。"),
        ("0 维 `torch.Tensor`，batch 平均交叉熵。",),
    ),
    "loss.mse_bce": (
        "def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:\n    pass\n\n\ndef bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:\n    pass",
        (),
        ("`pred`、`target`：形状完全相同的 `torch.Tensor`，形状可以是任意 `(...)`。", "BCE 的 `pred` 表示概率，元素位于 `(0, 1)`。"),
        ("两个函数都返回 0 维 `torch.Tensor`，表示平均损失。",),
    ),
    "vector.l2_normalize": (
        "def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:\n    pass",
        ("`feature_dim`：每个向量的特征数。",),
        ("`x: np.ndarray`，形状 `(..., feature_dim)`。", "`axis: int`，每个向量的特征维。"),
        ("`np.ndarray`，形状与 `x` 相同。",),
    ),
    "vector.minmax": (
        "def min_max_normalize(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:\n    pass",
        ("`sample_count`：样本数。", "`feature_dim`：特征数。"),
        ("`x: np.ndarray`，常见形状 `(sample_count, feature_dim)`。", "`axis: int`，默认 `0`，按特征列缩放。"),
        ("`np.ndarray`，形状与 `x` 相同，元素范围 `[0, 1]`。",),
    ),
    "vector.cosine": (
        "def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:\n    pass\n\n\ndef cosine_similarity_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:\n    pass",
        ("`feature_dim`：向量维度。", "`left_count`、`right_count`：两组向量数量。"),
        ("单向量 `a`、`b`：形状 `(feature_dim,)`。", "批量 `a`：形状 `(left_count, feature_dim)`；批量 `b`：形状 `(right_count, feature_dim)`。"),
        ("单向量函数返回 `float`。", "矩阵函数返回 `np.ndarray`，形状 `(left_count, right_count)`。"),
    ),
    "vector.euclidean": (
        "def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:\n    pass\n\n\ndef euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:\n    pass",
        ("`feature_dim`：向量维度。", "`left_count`、`right_count`：两组向量数量。"),
        ("单向量 `a`、`b`：形状 `(feature_dim,)`。", "批量 `a`：形状 `(left_count, feature_dim)`；批量 `b`：形状 `(right_count, feature_dim)`。"),
        ("单向量函数返回 `float`。", "矩阵函数返回 `np.ndarray`，形状 `(left_count, right_count)`。"),
    ),
}


def contract_markdown(problem_id: str) -> str:
    starter, dimensions, inputs, outputs = CONTRACTS[problem_id]
    parts = ["## 接口说明", "```python\n" + starter + "\n```"]
    if dimensions:
        parts.append("### 维度约定\n\n" + "\n".join(f"- {item}" for item in dimensions))
    parts.append("### 输入\n\n" + "\n".join(f"- {item}" for item in inputs))
    parts.append("### 输出\n\n" + "\n".join(f"- {item}" for item in outputs))
    return "\n\n".join(parts)
