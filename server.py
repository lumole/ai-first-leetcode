from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import io
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from contracts import CONTRACTS, contract_markdown


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "problems"


@dataclasses.dataclass(frozen=True)
class ProblemSpec:
    id: str
    title: str
    category: str
    difficulty: str
    tags: tuple[str, ...]
    source: str
    section: str
    targets: tuple[str, ...]
    subsection: str | None = None


SPECS: tuple[ProblemSpec, ...] = (
    ProblemSpec("attention.softmax", "Softmax", "注意力机制", "easy", ("NumPy", "数值稳定"), "01-注意力机制.md", "Softmax", ("softmax",)),
    ProblemSpec("attention.sdpa", "Scaled Dot-Product Attention", "注意力机制", "medium", ("PyTorch", "Attention"), "01-注意力机制.md", "Scaled Dot-Product Attention", ("scaled_dot_product_attention",)),
    ProblemSpec("attention.causal_mask", "Causal Mask", "注意力机制", "easy", ("PyTorch", "Mask"), "01-注意力机制.md", "Causal Mask", ("causal_mask",)),
    ProblemSpec("attention.mha", "Multi-Head Attention", "注意力机制", "medium", ("PyTorch", "Attention"), "01-注意力机制.md", "Multi-Head Attention", ("MultiHeadAttention",)),
    ProblemSpec("attention.gqa", "Grouped-Query Attention", "注意力机制", "medium", ("PyTorch", "Attention"), "01-注意力机制.md", "Grouped-Query Attention", ("GroupedQueryAttention",)),
    ProblemSpec("attention.cross", "Cross-Attention", "注意力机制", "medium", ("PyTorch", "Attention"), "01-注意力机制.md", "Cross-Attention", ("CrossAttention",)),
    ProblemSpec("attention.kv_cache", "KV Cache", "注意力机制", "medium", ("PyTorch", "Inference"), "01-注意力机制.md", "KV Cache", ("CachedSelfAttention",)),
    ProblemSpec("transformer.embedding", "Embedding", "Transformer 结构", "easy", ("PyTorch", "Embedding"), "02-Transformer 结构.md", "Embedding", ("TokenEmbedding",)),
    ProblemSpec("transformer.positional_encoding", "Positional Encoding", "Transformer 结构", "easy", ("PyTorch", "Position"), "02-Transformer 结构.md", "Positional Encoding", ("sinusoidal_positional_encoding",)),
    ProblemSpec("transformer.rope", "RoPE", "Transformer 结构", "medium", ("PyTorch", "Position"), "02-Transformer 结构.md", "RoPE", ("build_rope_cache", "apply_rope")),
    ProblemSpec("transformer.residual", "Residual Connection", "Transformer 结构", "easy", ("PyTorch", "Block"), "02-Transformer 结构.md", "Residual Connection", ("ResidualSubLayer",)),
    ProblemSpec("transformer.swiglu", "SwiGLU", "Transformer 结构", "medium", ("PyTorch", "FFN"), "02-Transformer 结构.md", "SwiGLU", ("SwiGLU",)),
    ProblemSpec("transformer.encoder_block", "Transformer Encoder Block", "Transformer 结构", "hard", ("PyTorch", "Block"), "02-Transformer 结构.md", "Transformer Encoder Block", ("TransformerEncoderBlock",)),
    ProblemSpec("transformer.decoder_block", "Transformer Decoder Block", "Transformer 结构", "hard", ("PyTorch", "Block"), "02-Transformer 结构.md", "Transformer Decoder Block", ("TransformerDecoderBlock",)),
    ProblemSpec("normalization.batchnorm", "BatchNorm1d", "归一化与激活", "medium", ("PyTorch", "Normalization"), "03-归一化与激活.md", "LayerNorm 与 BatchNorm", ("BatchNorm1d",), "手写 BatchNorm"),
    ProblemSpec("normalization.layernorm", "LayerNorm", "归一化与激活", "medium", ("PyTorch", "Normalization"), "03-归一化与激活.md", "LayerNorm 与 BatchNorm", ("LayerNorm",), "手写 LayerNorm"),
    ProblemSpec("normalization.rmsnorm", "RMSNorm", "归一化与激活", "easy", ("PyTorch", "Normalization"), "03-归一化与激活.md", "RMSNorm", ("RMSNorm",)),
    ProblemSpec("regularization.dropout", "Dropout", "归一化与激活", "easy", ("PyTorch", "Regularization"), "03-归一化与激活.md", "Dropout", ("Dropout",)),
    ProblemSpec("activation.relu", "ReLU", "归一化与激活", "easy", ("PyTorch", "Activation"), "03-归一化与激活.md", "ReLU 与 GELU", ("relu",)),
    ProblemSpec("activation.gelu", "GELU", "归一化与激活", "easy", ("PyTorch", "Activation"), "03-归一化与激活.md", "ReLU 与 GELU", ("gelu",)),
    ProblemSpec("activation.sigmoid", "Sigmoid", "归一化与激活", "easy", ("PyTorch", "Activation"), "03-归一化与激活.md", "Sigmoid 与 Tanh", ("sigmoid",)),
    ProblemSpec("activation.tanh", "Tanh", "归一化与激活", "easy", ("PyTorch", "Activation"), "03-归一化与激活.md", "Sigmoid 与 Tanh", ("tanh",)),
    ProblemSpec("loss.cross_entropy", "CrossEntropy Loss", "归一化与激活", "medium", ("PyTorch", "Loss"), "03-归一化与激活.md", "CrossEntropy Loss", ("cross_entropy",)),
    ProblemSpec("loss.mse_bce", "MSE 与 BCE Loss", "归一化与激活", "easy", ("PyTorch", "Loss"), "03-归一化与激活.md", "MSE 与 BCE Loss", ("mse_loss", "bce_loss")),
    ProblemSpec("vector.l2_normalize", "L2 归一化", "向量相似度与距离", "easy", ("NumPy", "Vector"), "04-向量相似度与距离.md", "L2 归一化 与 Min-Max 归一化", ("l2_normalize",)),
    ProblemSpec("vector.minmax", "Min-Max 归一化", "向量相似度与距离", "easy", ("NumPy", "Vector"), "04-向量相似度与距离.md", "L2 归一化 与 Min-Max 归一化", ("min_max_normalize",)),
    ProblemSpec("vector.cosine", "余弦相似度", "向量相似度与距离", "easy", ("NumPy", "Similarity"), "04-向量相似度与距离.md", "余弦相似度", ("cosine_similarity", "cosine_similarity_matrix")),
    ProblemSpec("vector.euclidean", "欧氏距离", "向量相似度与距离", "easy", ("NumPy", "Distance"), "04-向量相似度与距离.md", "欧氏距离", ("euclidean_distance", "euclidean_distance_matrix")),
)


Example = tuple[str, str, str]


VISIBLE_EXAMPLES: dict[str, tuple[Example, Example, Example]] = {
    "attention.softmax": (
        ("x = [1.0, 2.0, 3.0]", "[0.090031, 0.244728, 0.665241]", "三个概率之和为 1，较大的输入得到较大的概率。"),
        ("x = [[1.0, 1.0], [2.0, 2.0]], axis = -1", "[[0.5, 0.5], [0.5, 0.5]]", "每一行的元素相同，因此行内概率均分。"),
        ("x = [1000.0, 1001.0]", "[0.268941, 0.731059]", "大数输入仍需返回有限概率，不能发生指数溢出。"),
    ),
    "attention.sdpa": (
        ("q, k, v.shape = (2, 4, 5, 8), mask = None", "out.shape = (2, 4, 5, 8)\nattn.shape = (2, 4, 5, 5)", "每个 Query 对 5 个 Key 产生一组权重，再加权聚合 Value。"),
        ("q, k.shape = (1, 2, 3, 4)\nv.shape = (1, 2, 3, 6)", "返回 (out, attn)：\nout.shape = (1, 2, 3, 6)\nattn.shape = (1, 2, 3, 3)", "函数返回两个 Tensor：`out` 聚合 Value，所以最后一维是 `value_dim=6`；`attn` 保存每个 Query 对 3 个 Key 的概率权重。"),
        ("q, k, v.shape = (1, 1, 3, 4)\nmask.shape = (1, 1, 3, 3), dtype = torch.bool\nmask[..., 2] = False", "(out, attn)\nout.shape = (1, 1, 3, 4)\nattn[..., 2] = [[[0., 0., 0.]]]", "输出是二元组 `(out, attn)`，不是单个 Tensor。mask 是布尔 Tensor：`True` 表示允许关注，`False` 表示屏蔽；被屏蔽的位置在 Softmax 前填入负无穷，因此返回的 `attn` 权重为 0。"),
    ),
    "attention.causal_mask": (
        ("seq_len = 1", "[[True]]", "单个 token 只能看到自己。"),
        ("seq_len = 3", "[[True, False, False],\n [True, True, False],\n [True, True, True]]", "第 i 行只允许访问当前位置及之前的位置。"),
        ("seq_len = 5", "shape = (5, 5), dtype = bool", "结果必须是方形布尔矩阵。"),
    ),
    "attention.mha": (
        ("d_model = 16, n_heads = 4\nx.shape = (2, 5, 16)", "shape = (2, 5, 16)", "拆分多头再合并后，输出形状与输入一致。"),
        ("d_model = 32, n_heads = 8\nx.shape = (1, 7, 32)", "shape = (1, 7, 32)", "每个头处理 4 维子空间，序列长度保持不变。"),
        ("d_model = 8, n_heads = 2\nx.shape = (3, 1, 8), causal mask", "shape = (3, 1, 8)", "实现需要兼容可广播的注意力 mask。"),
    ),
    "attention.gqa": (
        ("d_model = 16, n_heads = 4, n_kv_heads = 2\nx.shape = (2, 6, 16)", "shape = (2, 6, 16)", "每组两个 Query 头共享一个 KV 头。"),
        ("d_model = 32, n_heads = 8, n_kv_heads = 1\nx.shape = (1, 4, 32)", "shape = (1, 4, 32)", "只有一个 KV 头时退化为 Multi-Query Attention。"),
        ("d_model = 24, n_heads = 6, n_kv_heads = 3\nx.shape = (3, 2, 24)", "shape = (3, 2, 24)", "KV 头复制后需与 6 个 Query 头对齐。"),
    ),
    "attention.cross": (
        ("d_model = 16, n_heads = 4\nx.shape = (2, 3, 16)\ncontext.shape = (2, 5, 16)", "shape = (2, 3, 16)", "输出序列长度跟随 Query 序列 x。"),
        ("x.shape = (1, 1, 32)\ncontext.shape = (1, 8, 32)", "shape = (1, 1, 32)", "一个 Query 可以关注 context 中的全部 8 个位置。"),
        ("x.shape = (3, 7, 8)\ncontext.shape = (3, 2, 8)", "shape = (3, 7, 8)", "Query 与 context 的序列长度可以不同。"),
    ),
    "attention.kv_cache": (
        ("连续输入 1 个 token，cache = None", "out.shape = (B, 1, D)\ncache['k'].shape[2] = 1", "第一次调用会创建包含一个位置的缓存。"),
        ("在已有 3 个位置的 cache 后再输入 1 个 token", "cache['k'].shape[2] = 4", "新 Key/Value 应追加到序列维。"),
        ("连续解码 5 步", "每步 out.shape = (B, 1, D)\n最终缓存长度 = 5", "增量解码只返回当前 token 的输出，缓存持续增长。"),
    ),
    "transformer.embedding": (
        ("vocab_size = 100, d_model = 16\nidx.shape = (2, 5)", "encode(idx).shape = (2, 5, 16)", "每个 token id 被映射成一个 16 维向量。"),
        ("hidden.shape = (2, 5, 16)", "decode(hidden).shape = (2, 5, 100)", "输出投影为词表中的每个 token 产生一个 logit。"),
        ("检查 embed.weight 与 proj.weight", "两者指向同一组参数", "输入嵌入和输出投影需要共享权重。"),
    ),
    "transformer.positional_encoding": (
        ("seq_len = 4, d_model = 8", "shape = (4, 8)", "每个序列位置得到一个 8 维位置向量。"),
        ("seq_len = 1, d_model = 4", "[[0.0, 1.0, 0.0, 1.0]]", "位置 0 的正弦项为 0，余弦项为 1。"),
        ("seq_len = 6, d_model = 16", "shape = (6, 16), 所有值有限", "实现需要支持不同的偶数模型维度。"),
    ),
    "transformer.rope": (
        ("build_rope_cache(seq_len=5, head_dim=8)", "cos.shape = sin.shape = (5, 4)", "每两个通道共享一个旋转频率。"),
        ("x.shape = (2, 3, 5, 8)", "apply_rope(...).shape = (2, 3, 5, 8)", "旋转只改变数值，不改变张量形状。"),
        ("位置 0 的向量 x", "apply_rope(x)[..., 0, :] = x[..., 0, :]", "位置 0 的旋转角为 0，因此向量保持不变。"),
    ),
    "transformer.residual": (
        ("d_model = 8, dropout = 0\nx.shape = (2, 3, 8)", "shape = (2, 3, 8)", "残差分支不改变输入形状。"),
        ("sublayer(z) = 0", "output = x", "子层输出为零时只剩恒等残差。"),
        ("dropout = 0, eval 模式", "output = x + sublayer(norm(x))", "该题要求实现 Pre-Norm 残差顺序。"),
    ),
    "transformer.swiglu": (
        ("d_model = 8, d_ff = 20\nx.shape = (2, 3, 8)", "shape = (2, 3, 8)", "门控中间层最终投影回模型维度。"),
        ("d_model = 16, d_ff = 32\nx.shape = (1, 1, 16)", "shape = (1, 1, 16)", "实现需要支持单 token 输入。"),
        ("d_model = 24, d_ff = 48\nx.shape = (4, 7, 24)", "shape = (4, 7, 24)", "batch 和序列维都应原样保留。"),
    ),
    "transformer.encoder_block": (
        ("d_model=16, n_heads=4, d_ff=32\nx.shape=(2,5,16)", "shape = (2, 5, 16)", "自注意力与前馈层都保留模型维度。"),
        ("x.shape=(1,8,32), causal mask", "shape = (1, 8, 32)", "编码块需要把 mask 传给注意力层。"),
        ("x.shape=(3,1,8), dropout=0", "shape = (3, 1, 8), 所有值有限", "单 token 与小维度输入也应正常工作。"),
    ),
    "transformer.decoder_block": (
        ("x.shape=(2,4,16)\ncontext.shape=(2,6,16)", "shape = (2, 4, 16)", "输出长度跟随 decoder 输入。"),
        ("x.shape=(1,1,32)\ncontext.shape=(1,9,32)", "shape = (1, 1, 32)", "单步解码仍可关注完整 encoder context。"),
        ("x.shape=(3,7,8)\ncontext.shape=(3,2,8)", "shape = (3, 7, 8)", "自注意力、交叉注意力和前馈层后维度保持一致。"),
    ),
    "normalization.batchnorm": (
        ("x.shape=(16,4), training=True", "y.shape=(16,4)\ny.mean(0) 约为 0", "训练时沿 batch 维计算每个特征的统计量。"),
        ("x.shape=(8,3), training=True", "running_mean 与 running_var 被更新", "训练批次统计量要按 momentum 写入滑动平均。"),
        ("x.shape=(2,3), training=False", "使用 running_mean/running_var，shape=(2,3)", "推理时不能依赖当前 batch 的统计量。"),
    ),
    "normalization.layernorm": (
        ("x.shape=(2,3,4)", "shape=(2,3,4)\ny.mean(-1) 约为 0", "每个 token 独立沿最后一维归一化。"),
        ("x=[[1.0,1.0,1.0]]", "有限值，shape=(1,3)", "零方差输入需要通过 eps 避免除零。"),
        ("x.shape=(4,8), d_model=8", "y.var(-1, unbiased=False) 约为 1", "归一化后的最后一维方差接近 1。"),
    ),
    "normalization.rmsnorm": (
        ("x.shape=(2,3,4)", "shape=(2,3,4)\nsqrt(mean(y^2,-1)) 约为 1", "只按均方根缩放，不减均值。"),
        ("x=[[0.0,0.0,0.0,0.0]]", "有限的全零输出", "eps 保证零向量不会产生 NaN。"),
        ("x.shape=(5,8), d_model=8", "shape=(5,8)", "实现要兼容任意前导维度。"),
    ),
    "regularization.dropout": (
        ("p=0.5, x=全 1, training=True", "元素只能为 0 或 2", "保留元素除以 1-p，以维持输出期望。"),
        ("p=0.3, x=全 1, training=False", "output = x", "推理模式下 Dropout 必须直通。"),
        ("p=0, x 为任意张量, training=True", "output = x", "丢弃概率为 0 时训练模式也不改变输入。"),
    ),
    "activation.relu": (
        ("x=[-2.0,0.0,3.0]", "[0.0,0.0,3.0]", "负数被截断为 0，非负数保持不变。"),
        ("x=[-1.0,-0.5]", "[0.0,0.0]", "全负输入得到全零输出。"),
        ("x=[[1.0,-1.0],[2.0,-2.0]]", "[[1.0,0.0],[2.0,0.0]]", "操作逐元素执行并保持原形状。"),
    ),
    "activation.gelu": (
        ("x=[-1.0,0.0,1.0]", "约 [-0.1588,0.0,0.8412]", "使用题目给出的 tanh 近似公式逐元素计算。"),
        ("x=[0.0]", "[0.0]", "零输入经过 GELU 后仍为零。"),
        ("x.shape=(2,3)", "output.shape=(2,3)", "激活函数不改变张量形状。"),
    ),
    "activation.sigmoid": (
        ("x=[-1.0,0.0,1.0]", "约 [0.2689,0.5,0.7311]", "每个实数被映射到 0 和 1 之间。"),
        ("x=[-20.0,20.0]", "约 [0.0,1.0]", "极端输入应得到有限且接近边界的值。"),
        ("x.shape=(2,4)", "output.shape=(2,4)", "Sigmoid 逐元素运算并保持形状。"),
    ),
    "activation.tanh": (
        ("x=[-1.0,0.0,1.0]", "约 [-0.7616,0.0,0.7616]", "输出位于 -1 和 1 之间。"),
        ("x=[-20.0,20.0]", "约 [-1.0,1.0]", "极端输入趋近双侧饱和值。"),
        ("x.shape=(3,2)", "output.shape=(3,2)", "Tanh 逐元素运算并保持形状。"),
    ),
    "loss.cross_entropy": (
        ("logits=[[2.0,0.5,0.1]], target=[0]", "约 0.3168", "正确类别 0 的 logit 最大，因此损失较低。"),
        ("logits=[[0.0,0.0]], target=[1]", "约 0.6931", "两类概率相同，正确类概率为 1/2。"),
        ("logits.shape=(4,5), target.shape=(4,)", "一个标量", "对 batch 中 4 个样本的损失取平均。"),
    ),
    "loss.mse_bce": (
        ("mse_loss([2.0,0.0], [1.0,0.0])", "0.5", "平方误差为 [1,0]，均值为 0.5。"),
        ("bce_loss([0.9,0.2], [1.0,0.0])", "约 0.1643", "分别计算正类与负类的对数损失后取平均。"),
        ("mse_loss([1.0,1.0], [1.0,1.0])", "0.0", "预测与目标完全一致时均方误差为零。"),
    ),
    "vector.l2_normalize": (
        ("x=[3.0,4.0]", "[0.6,0.8]", "向量范数为 5，逐元素除以 5。"),
        ("x=[[3.0,4.0],[0.0,5.0]], axis=1", "[[0.6,0.8],[0.0,1.0]]", "每一行独立归一化为单位向量。"),
        ("x=[0.0,0.0]", "有限的全零向量", "零向量需要通过 eps 避免除零。"),
    ),
    "vector.minmax": (
        ("x=[1.0,3.0,5.0]", "[0.0,0.5,1.0]", "最小值映射到 0，最大值映射到 1。"),
        ("x=[[1.0,10.0],[3.0,30.0],[5.0,20.0]]", "[[0.0,0.0],[0.5,1.0],[1.0,0.5]]", "默认按列独立缩放。"),
        ("x=[[2.0,1.0],[2.0,3.0]]", "第一列为有限值", "常数列的极差为零，需要避免除零。"),
    ),
    "vector.cosine": (
        ("a=[1.0,0.0], b=[1.0,0.0]", "1.0", "方向完全相同。"),
        ("a=[1.0,0.0], b=[0.0,1.0]", "0.0", "两个向量正交。"),
        ("A.shape=(2,3), B.shape=(4,3)", "cosine_similarity_matrix(A,B).shape=(2,4)", "矩阵结果包含 A 与 B 中每对向量的相似度。"),
    ),
    "vector.euclidean": (
        ("a=[0.0,0.0], b=[3.0,4.0]", "5.0", "坐标差为 [3,4]，距离由勾股定理得到。"),
        ("a=[1.0,2.0], b=[1.0,2.0]", "0.0", "相同向量之间距离为零。"),
        ("A.shape=(2,3), B.shape=(4,3)", "euclidean_distance_matrix(A,B).shape=(2,4)", "矩阵结果包含 A 与 B 中每对向量的距离。"),
    ),
}


def read_source(source: str) -> str:
    return (SOURCE_DIR / source).read_text(encoding="utf-8")


def markdown_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()
    return sections


def markdown_subsection(section: str, title: str) -> str:
    matches = list(re.finditer(r"(?m)^###\s+(.+?)\s*$", section))
    for index, match in enumerate(matches):
        if match.group(1).strip() != title:
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        return section[start:end].strip()
    return section


def first_code_block(markdown: str) -> str:
    match = re.search(r"```python\s*(.*?)```", markdown, flags=re.S)
    if not match:
        return ""
    return strip_demo(match.group(1).strip())


def strip_demo(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if "输入输出样例" in line:
            break
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def imports_from_code(code: str) -> list[str]:
    imports: list[str] = []
    for line in code.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports


def starter_from_solution(problem_id: str, code: str, targets: tuple[str, ...]) -> str:
    contract = CONTRACTS.get(problem_id)
    if contract:
        imports = "\n".join(imports_from_code(code))
        return f"{imports}\n\n{contract[0]}\n" if imports else contract[0] + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        imports = "\n".join(imports_from_code(code))
        stubs = "\n\n".join(f"def {name}(*args, **kwargs):\n    pass" for name in targets)
        return f"{imports}\n\n{stubs}\n" if imports else f"{stubs}\n"

    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
            node.body = [ast.Pass()]
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in targets:
            methods: list[ast.stmt] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child.body = [ast.Pass()]
                    methods.append(child)
            node.body = methods or [ast.Pass()]
            body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module).strip() + "\n"


def hide_solution_code(markdown: str) -> str:
    cleaned = re.sub(r"```python\s*.*?```", "", markdown, flags=re.S | re.I)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def clean_problem_markdown(markdown: str) -> str:
    """Keep the source explanation while removing its duplicate headings and reference links."""
    cleaned = hide_solution_code(markdown)
    lines = cleaned.splitlines()
    visible_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:CodeKy|Deep-ML|LeetGPU)\s*[:：]", stripped, flags=re.I):
            continue
        if re.search(r"https?://", stripped):
            continue
        visible_lines.append(line)

    while visible_lines and not visible_lines[0].strip():
        visible_lines.pop(0)
    if visible_lines and re.match(r"^#{1,3}\s+", visible_lines[0]):
        visible_lines.pop(0)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(visible_lines)).strip()


def examples_markdown(problem_id: str) -> str:
    blocks = []
    for index, (input_text, output_text, explanation) in enumerate(VISIBLE_EXAMPLES.get(problem_id, ()), 1):
        blocks.append(
            f"### 示例 {index}\n\n"
            f"**输入**\n\n```text\n{input_text}\n```\n\n"
            f"**输出**\n\n```text\n{output_text}\n```\n\n"
            f"**解释**\n\n{explanation}"
        )
    return "\n\n".join(blocks)


def spec_markdown(spec: ProblemSpec) -> str:
    source = read_source(spec.source)
    section = markdown_sections(source).get(spec.section, "")
    if not section:
        return f"## {spec.title}\n\n题面暂未读取到。"
    if spec.subsection:
        section = markdown_subsection(section, spec.subsection)
    section = clean_problem_markdown(section)
    target_names = "、".join(f"`{target}`" for target in spec.targets)
    requirement = f"### 作答要求\n\n请在右侧编辑器中从定义开始，实现 {target_names}。"
    return f"{hide_solution_code(section)}\n\n{contract_markdown(spec.id)}\n\n{requirement}\n\n## 测试用例与示例解析\n\n{examples_markdown(spec.id)}"


def spec_solution(spec: ProblemSpec) -> str:
    section = markdown_sections(read_source(spec.source)).get(spec.section, "")
    target_markdown = markdown_subsection(section, spec.subsection) if spec.subsection else section
    return first_code_block(target_markdown)


def load_problem(spec: ProblemSpec, order: int) -> dict[str, Any]:
    solution = spec_solution(spec)
    return {
        "id": spec.id,
        "title": spec.title,
        "category": spec.category,
        "difficulty": spec.difficulty,
        "tags": list(spec.tags),
        "order": order,
        "readme": spec_markdown(spec),
        "examples": [
            {"input": input_text, "output": output_text, "explanation": explanation}
            for input_text, output_text, explanation in VISIBLE_EXAMPLES[spec.id]
        ],
        "starter": starter_from_solution(spec.id, solution, spec.targets),
        "solution": solution,
        "source": str(Path("problems") / spec.source),
    }


def all_problems() -> list[dict[str, Any]]:
    return [load_problem(spec, index + 1) for index, spec in enumerate(SPECS)]


def problem_by_id(problem_id: str) -> dict[str, Any] | None:
    for index, spec in enumerate(SPECS):
        if spec.id == problem_id:
            return load_problem(spec, index + 1)
    return None


class ApiHandler(BaseHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/problems":
            items = [
                {key: problem[key] for key in ("id", "title", "category", "difficulty", "tags", "order")}
                for problem in all_problems()
            ]
            return self.write_json(items)
        match = re.match(r"^/api/problems/([^/]+)$", path)
        if match:
            problem = problem_by_id(unquote(match.group(1)))
            if problem:
                return self.write_json(problem)
            return self.write_error(404, "题目不存在")
        return self.write_error(404, "接口不存在")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        match = re.match(r"^/api/problems/([^/]+)/(run|submit|debug)$", path)
        if not match:
            return self.write_error(404, "接口不存在")
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        code = payload.get("code", "")
        problem_id = unquote(match.group(1))
        action = match.group(2)
        if action == "debug":
            breakpoints = [int(line) for line in payload.get("breakpoints", []) if isinstance(line, int) or str(line).isdigit()]
            result = run_debug(problem_id, code, breakpoints)
        else:
            case_count = payload.get("caseCount") if action == "run" else None
            result = run_submission(problem_id, code, mode=action, case_count=case_count)
        return self.write_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def write_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, status: int, message: str) -> None:
        self.write_json({"error": message}, status)


def run_submission(problem_id: str, code: str, mode: str = "submit", case_count: Any = None) -> dict[str, Any]:
    if not problem_by_id(problem_id):
        return {"score": 0, "all_passed": False, "cases": [], "error": "题目不存在"}
    with tempfile.TemporaryDirectory(prefix="ai-first-judge-") as temp_dir:
        submission = Path(temp_dir) / "submission.py"
        submission.write_text(code, encoding="utf-8")
        try:
            command = [sys.executable, str(Path(__file__).resolve()), "--judge", problem_id, str(submission), mode]
            if mode == "run" and isinstance(case_count, int):
                command.extend(["--judge-case-count", str(max(1, min(case_count, 100)))])
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"score": 0, "all_passed": False, "cases": [], "error": "运行超时"}
    if proc.returncode != 0:
        return {"score": 0, "all_passed": False, "cases": [], "error": proc.stderr or proc.stdout}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"score": 0, "all_passed": False, "cases": [], "error": proc.stdout or proc.stderr}


def run_debug(problem_id: str, code: str, breakpoints: list[int]) -> dict[str, Any]:
    if not problem_by_id(problem_id):
        return {"events": [], "error": "题目不存在"}
    with tempfile.TemporaryDirectory(prefix="ai-first-debug-") as temp_dir:
        submission = Path(temp_dir) / "submission.py"
        submission.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--debug", problem_id, str(submission), json.dumps(breakpoints)],
                text=True,
                capture_output=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"events": [], "error": "调试运行超时"}
    if proc.returncode != 0:
        return {"events": [], "error": proc.stderr or proc.stdout}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"events": [], "error": proc.stdout or proc.stderr}


def execute(code: str, namespace: dict[str, Any]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<submission>", "exec"), namespace)
    return buffer.getvalue()


def build_reference_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "reference"}
    for spec in SPECS:
        code = spec_solution(spec)
        if code:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"<reference:{spec.id}>", "exec"), namespace)
    return namespace


def judge(problem_id: str, submission_path: Path, mode: str = "submit", run_case_count: int | None = None) -> dict[str, Any]:
    code = submission_path.read_text(encoding="utf-8")
    ref = build_reference_namespace()
    user: dict[str, Any] = {"__name__": "submission"}
    try:
        stdout = execute(code, user)
    except Exception:
        return {"score": 0, "all_passed": False, "cases": [], "stdout": "", "error": traceback.format_exc()}

    for name in ("MultiHeadAttention", "CrossAttention"):
        if name not in user and name in ref:
            user[name] = ref[name]

    cases = run_cases(problem_id, user, ref, public=mode == "run", run_case_count=run_case_count)
    passed = sum(1 for case in cases if case["passed"])
    score = 100.0 * passed / max(len(cases), 1)
    return {
        "score": score,
        "all_passed": passed == len(cases),
        "cases": cases,
        "stdout": stdout[-4000:],
        "mode": mode,
    }


def run_cases(
    problem_id: str,
    user: dict[str, Any],
    ref: dict[str, Any],
    public: bool = False,
    run_case_count: int | None = None,
) -> list[dict[str, Any]]:
    if public and run_case_count is not None:
        public_tests = PUBLIC_CASES.get(problem_id, [])
        validation_tests = CASES.get(problem_id, [])
        tests = public_tests[:run_case_count]
        for index in range(max(0, run_case_count - len(tests))):
            if validation_tests:
                tests.append(validation_tests[index % len(validation_tests)])
    else:
        tests = PUBLIC_CASES.get(problem_id, []) if public else CASES.get(problem_id, [])
    results = []
    for index, (name, fn) in enumerate(tests):
        display_name = name if public and index < len(PUBLIC_CASES.get(problem_id, [])) else f"测试用例 {index + 1}"
        start = time.perf_counter()
        try:
            fn(user, ref)
            results.append({
                "name": display_name,
                "passed": True,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            })
        except Exception as exc:
            result = {
                "name": display_name,
                "passed": False,
                "reason": str(exc),
                "error": traceback.format_exc(),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            }
            case_input = extract_case_input(exc.__traceback__)
            if case_input:
                result["input"] = case_input
            if isinstance(exc, OutputMismatch):
                result["actual"] = exc.actual
                result["expected"] = exc.expected
            results.append(result)
    return results


def require(ns: dict[str, Any], name: str) -> Any:
    if name not in ns:
        raise AssertionError(f"没有找到 `{name}`")
    return ns[name]


class OutputMismatch(AssertionError):
    def __init__(self, actual: Any, expected: Any) -> None:
        self.actual = format_judge_value(actual)
        self.expected = format_judge_value(expected)
        super().__init__("输出和参考结果不一致")


def format_judge_value(value: Any) -> str:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size <= 64:
                return np.array2string(value, precision=6, separator=", ")
            return f"array(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, np.generic):
            return repr(value.item())
    except Exception:
        pass
    if isinstance(value, tuple):
        return "(" + ", ".join(format_judge_value(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(format_judge_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {format_judge_value(item)}" for key, item in value.items()) + "}"
    return repr(value)


def assert_sdpa_pair(actual: Any, expected: Any) -> tuple[Any, Any]:
    actual_output, actual_attn = actual
    expected_output, expected_attn = expected
    try:
        assert_close(actual_output, expected_output)
        assert_close(actual_attn, expected_attn)
    except OutputMismatch as exc:
        raise OutputMismatch((actual_output, actual_attn), (expected_output, expected_attn)) from exc
    return actual_output, actual_attn


def extract_case_input(trace: Any) -> str:
    input_names = {
        "x", "q", "k", "v", "mask", "context", "cache", "idx", "hidden",
        "a", "b", "A", "B", "pred", "target", "logits", "prob", "label",
        "seq_len", "axis", "p", "d_model", "n_heads", "n_kv_heads", "d_ff",
        "vocab_size", "head_dim", "features", "batch", "classes",
    }
    frame = None
    while trace:
        candidate = trace.tb_frame
        if candidate.f_code.co_name.startswith("test_"):
            frame = candidate
        trace = trace.tb_next
    if frame is None:
        return ""
    values = []
    for name, value in frame.f_locals.items():
        if name in input_names:
            values.append(f"{name} = {format_judge_value(value)}")
    return "\n".join(values)


def assert_close(actual: Any, expected: Any, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    import numpy as np

    try:
        import torch
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()
    except Exception:
        pass
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise OutputMismatch(actual, expected)


def assert_shape(value: Any, shape: tuple[int, ...]) -> None:
    actual = tuple(value.shape)
    if actual != shape:
        raise AssertionError(f"形状应为 {shape}，实际为 {actual}")


def assert_finite(value: Any) -> None:
    import numpy as np

    try:
        import torch
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise AssertionError("输出中包含非有限数")
            return
    except Exception:
        pass
    if not np.isfinite(value).all():
        raise AssertionError("输出中包含非有限数")


def test_softmax_rows(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    fn = require(user, "softmax")
    rows, cols = random.randint(1, 4), random.randint(2, 9)
    x = np.random.normal(size=(rows, cols)) * random.choice((0.1, 1.0, 10.0))
    x += random.choice((-1000.0, 0.0, 1000.0))
    y = fn(x, axis=-1)
    assert_close(y, ref["softmax"](x, axis=-1))
    assert_close(np.sum(y, axis=-1), np.ones(rows))


def test_softmax_axis(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    shape = (random.randint(2, 7), random.randint(2, 7), random.randint(1, 4))
    axis = random.randrange(len(shape))
    x = np.random.normal(size=shape) * random.choice((1.0, 20.0))
    assert_close(require(user, "softmax")(x, axis=axis), ref["softmax"](x, axis=axis))


def test_sdpa_basic(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    batch, heads = random.randint(1, 3), random.randint(1, 4)
    q_len, kv_len = random.randint(1, 7), random.randint(1, 7)
    d_k, d_v = random.randint(2, 8), random.randint(2, 9)
    q = torch.randn(batch, heads, q_len, d_k)
    k = torch.randn(batch, heads, kv_len, d_k)
    v = torch.randn(batch, heads, kv_len, d_v)
    assert_sdpa_pair(
        require(user, "scaled_dot_product_attention")(q, k, v),
        ref["scaled_dot_product_attention"](q, k, v),
    )


def test_sdpa_mask(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    batch, heads = random.randint(1, 2), random.randint(1, 4)
    q_len, kv_len, d_k = random.randint(2, 7), random.randint(2, 7), random.randint(2, 8)
    q, k, v = torch.randn(batch, heads, q_len, d_k), torch.randn(batch, heads, kv_len, d_k), torch.randn(batch, heads, kv_len, d_k)
    mask = torch.rand(batch, 1, q_len, kv_len) > 0.35
    mask[..., 0] = True
    out, attn = assert_sdpa_pair(
        require(user, "scaled_dot_product_attention")(q, k, v, mask),
        ref["scaled_dot_product_attention"](q, k, v, mask),
    )
    assert_close(attn.masked_select(~mask.expand_as(attn)), torch.zeros_like(attn.masked_select(~mask.expand_as(attn))))


def test_causal_mask(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    seq_len = random.randint(1, 18)
    mask = require(user, "causal_mask")(seq_len)
    assert_close(mask, torch.tril(torch.ones(seq_len, seq_len)).bool())
    if mask.dtype != torch.bool:
        raise AssertionError("mask dtype 应为 bool")


def module_output(cls: Any, args: tuple[Any, ...], input_args: tuple[Any, ...], expected_shape: tuple[int, ...]) -> None:
    import torch

    model = cls(*args)
    model.eval()
    with torch.no_grad():
        output = model(*input_args)
    if isinstance(output, tuple):
        output = output[0]
    assert_shape(output, expected_shape)
    assert_finite(output)


def test_mha_shape(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (24, 6), (32, 8)))
    batch, seq_len = random.randint(1, 3), random.randint(1, 8)
    x = torch.randn(batch, seq_len, d_model)
    module_output(require(user, "MultiHeadAttention"), (d_model, n_heads), (x,), (batch, seq_len, d_model))


def test_mha_mask(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (32, 8)))
    batch, seq_len = random.randint(1, 2), random.randint(2, 8)
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    module_output(require(user, "MultiHeadAttention"), (d_model, n_heads), (x, mask), (batch, seq_len, d_model))


def test_gqa_shape(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads, n_kv_heads = random.choice(((8, 2, 1), (16, 4, 1), (16, 4, 2), (24, 6, 3), (32, 8, 2)))
    batch, seq_len = random.randint(1, 3), random.randint(1, 8)
    x = torch.randn(batch, seq_len, d_model)
    module_output(require(user, "GroupedQueryAttention"), (d_model, n_heads, n_kv_heads), (x,), (batch, seq_len, d_model))


def test_cross_attention(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (24, 6), (32, 8)))
    batch, q_len, kv_len = random.randint(1, 3), random.randint(1, 7), random.randint(1, 9)
    x = torch.randn(batch, q_len, d_model)
    context = torch.randn(batch, kv_len, d_model)
    module_output(require(user, "CrossAttention"), (d_model, n_heads), (x, context), (batch, q_len, d_model))


def test_kv_cache_growth(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (24, 6), (32, 8)))
    batch, steps = random.randint(1, 3), random.randint(1, 7)
    model = require(user, "CachedSelfAttention")(d_model, n_heads)
    cache = None
    for step in range(steps):
        out, cache = model(torch.randn(batch, 1, d_model), cache)
        assert_shape(cache["k"], (batch, n_heads, step + 1, d_model // n_heads))
        assert_shape(cache["v"], (batch, n_heads, step + 1, d_model // n_heads))
    assert_shape(out, (batch, 1, d_model))


def test_embedding(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    cls = require(user, "TokenEmbedding")
    vocab_size = random.randint(16, 80)
    d_model = random.choice((4, 8, 16, 24))
    batch, seq_len = random.randint(1, 4), random.randint(1, 8)
    model = cls(vocab_size, d_model)
    idx = torch.randint(0, vocab_size, (batch, seq_len))
    encoded = model.encode(idx)
    decoded = model.decode(encoded)
    assert_shape(encoded, (batch, seq_len, d_model))
    assert_shape(decoded, (batch, seq_len, vocab_size))
    if model.proj.weight.data_ptr() != model.embed.weight.data_ptr():
        raise AssertionError("输入 embedding 和输出投影需要共享权重")
    assert_close(encoded, model.embed(idx) * math.sqrt(d_model))


def test_positional_encoding(user: dict[str, Any], ref: dict[str, Any]) -> None:
    seq_len, d_model = random.randint(1, 24), random.choice((4, 8, 16, 32))
    pe = require(user, "sinusoidal_positional_encoding")(seq_len, d_model)
    assert_shape(pe, (seq_len, d_model))
    assert_close(pe, ref["sinusoidal_positional_encoding"](seq_len, d_model))


def test_rope_cache(user: dict[str, Any], ref: dict[str, Any]) -> None:
    seq_len, head_dim = random.randint(1, 20), random.choice((4, 8, 16, 32))
    cos, sin = require(user, "build_rope_cache")(seq_len, head_dim)
    exp_cos, exp_sin = ref["build_rope_cache"](seq_len, head_dim)
    assert_close(cos, exp_cos)
    assert_close(sin, exp_sin)


def test_rope_apply(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    batch, heads, seq_len = random.randint(1, 3), random.randint(1, 5), random.randint(1, 12)
    head_dim = random.choice((4, 8, 16))
    x = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = ref["build_rope_cache"](seq_len, head_dim)
    out = require(user, "apply_rope")(x, cos, sin)
    assert_shape(out, (batch, heads, seq_len, head_dim))
    assert_close(out, ref["apply_rope"](x, cos, sin))


def test_residual(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch
    import torch.nn as nn

    d_model = random.choice((4, 8, 16, 32))
    batch, seq_len = random.randint(1, 4), random.randint(1, 8)
    sublayer = nn.Linear(d_model, d_model)
    model = require(user, "ResidualSubLayer")(d_model, sublayer, dropout=0.0)
    x = torch.randn(batch, seq_len, d_model)
    assert_close(model(x), x + sublayer(model.norm(x)))


def test_swiglu(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model = random.choice((4, 8, 16, 24))
    d_ff = random.choice((d_model * 2, d_model * 3, d_model * 4))
    batch, seq_len = random.randint(1, 4), random.randint(1, 8)
    module_output(require(user, "SwiGLU"), (d_model, d_ff), (torch.randn(batch, seq_len, d_model),), (batch, seq_len, d_model))


def test_encoder_block(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (24, 6), (32, 8)))
    batch, seq_len = random.randint(1, 3), random.randint(1, 8)
    d_ff = d_model * random.choice((2, 3, 4))
    x = torch.randn(batch, seq_len, d_model)
    module_output(require(user, "TransformerEncoderBlock"), (d_model, n_heads, d_ff, 0.0), (x,), (batch, seq_len, d_model))


def test_decoder_block(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, n_heads = random.choice(((8, 2), (16, 4), (24, 6), (32, 8)))
    batch, seq_len, context_len = random.randint(1, 3), random.randint(1, 8), random.randint(1, 9)
    d_ff = d_model * random.choice((2, 3, 4))
    x = torch.randn(batch, seq_len, d_model)
    context = torch.randn(batch, context_len, d_model)
    module_output(require(user, "TransformerDecoderBlock"), (d_model, n_heads, d_ff, 0.0), (x, context), (batch, seq_len, d_model))


def test_batchnorm_train(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    features, batch = random.randint(2, 12), random.randint(8, 40)
    model = require(user, "BatchNorm1d")(features)
    x = torch.randn(batch, features) * random.uniform(0.4, 4.0) + random.uniform(-6.0, 6.0)
    y = model(x)
    assert_shape(y, (batch, features))
    assert_close(y.mean(0), torch.zeros(features), atol=1e-4)
    assert_close(y.var(0, unbiased=False), torch.ones(features), atol=2e-4)


def test_batchnorm_eval(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    features = random.randint(2, 10)
    train_batch, eval_batch = random.randint(4, 20), random.randint(1, 6)
    model = require(user, "BatchNorm1d")(features)
    model.train()
    model(torch.randn(train_batch, features) + random.uniform(-3.0, 3.0))
    model.eval()
    y = model(torch.randn(eval_batch, features))
    assert_shape(y, (eval_batch, features))


def test_layernorm(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, batch, seq_len = random.randint(2, 16), random.randint(1, 5), random.randint(1, 8)
    model = require(user, "LayerNorm")(d_model)
    x = torch.randn(batch, seq_len, d_model) * random.uniform(0.5, 5.0) + random.uniform(-4.0, 4.0)
    y = model(x)
    expected = ref["LayerNorm"](d_model)(x)
    assert_shape(y, (batch, seq_len, d_model))
    assert_finite(y)
    assert_close(y, expected)


def test_rmsnorm(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    d_model, batch, seq_len = random.randint(2, 16), random.randint(1, 5), random.randint(1, 8)
    model = require(user, "RMSNorm")(d_model)
    y = model(torch.randn(batch, seq_len, d_model) * random.uniform(0.5, 5.0))
    rms = torch.sqrt(y.pow(2).mean(-1))
    assert_close(rms, torch.ones(batch, seq_len), atol=2e-4)


def test_dropout_eval(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    p = random.choice((0.0, 0.1, 0.3, 0.5, 0.8))
    shape = (random.randint(1, 5), random.randint(2, 20))
    model = require(user, "Dropout")(p)
    x = torch.randn(*shape)
    model.eval()
    assert_close(model(x), x)


def test_dropout_train(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    p = random.choice((0.1, 0.25, 0.5, 0.75))
    model = require(user, "Dropout")(p)
    model.train()
    y = model(torch.ones(4000))
    values = set(float(v) for v in torch.unique(y))
    expected_kept = 1.0 / (1.0 - p)
    if not all(abs(value) < 1e-7 or abs(value - expected_kept) < 1e-5 for value in values):
        raise AssertionError("训练阶段应使用 inverted dropout，保留值应缩放为 1/(1-p)")
    if 0.0 not in values or len(values) < 2:
        raise AssertionError("训练阶段应随机丢弃部分元素")


def test_relu(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    shape = (random.randint(1, 6), random.randint(1, 8))
    x = torch.randn(*shape) * random.uniform(0.1, 10.0)
    assert_close(require(user, "relu")(x), torch.relu(x))


def test_gelu(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch
    import torch.nn.functional as F

    shape = (random.randint(1, 5), random.randint(1, 9))
    x = torch.randn(*shape) * random.uniform(0.2, 4.0)
    assert_close(require(user, "gelu")(x), F.gelu(x, approximate="tanh"))


def test_sigmoid(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    shape = (random.randint(1, 5), random.randint(1, 9))
    x = torch.randn(*shape) * random.uniform(0.2, 12.0)
    assert_close(require(user, "sigmoid")(x), torch.sigmoid(x))


def test_tanh(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    shape = (random.randint(1, 5), random.randint(1, 9))
    x = torch.randn(*shape) * random.uniform(0.2, 12.0)
    assert_close(require(user, "tanh")(x), torch.tanh(x))


def test_cross_entropy(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch
    import torch.nn.functional as F

    batch, classes = random.randint(1, 12), random.randint(2, 10)
    logits = torch.randn(batch, classes) * random.uniform(0.2, 5.0)
    target = torch.randint(0, classes, (batch,))
    assert_close(require(user, "cross_entropy")(logits, target), F.cross_entropy(logits, target))


def test_mse_bce(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import torch

    shape = (random.randint(1, 5), random.randint(1, 8))
    pred = torch.randn(*shape)
    target = torch.randn(*shape)
    assert_close(require(user, "mse_loss")(pred, target), torch.mean((pred - target) ** 2))
    prob = torch.rand(*shape) * 0.98 + 0.01
    label = torch.randint(0, 2, shape).float()
    expected = -(label * torch.log(prob) + (1 - label) * torch.log(1 - prob)).mean()
    assert_close(require(user, "bce_loss")(prob, label), expected)


def test_l2_normalize(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    shape = (random.randint(1, 6), random.randint(2, 9))
    x = np.random.normal(size=shape)
    if random.random() < 0.35:
        x[0] = 0
    assert_close(require(user, "l2_normalize")(x, axis=1), ref["l2_normalize"](x, axis=1))


def test_minmax(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    rows, cols = random.randint(2, 9), random.randint(1, 7)
    x = np.random.normal(size=(rows, cols)) * random.uniform(0.2, 10.0)
    if random.random() < 0.35:
        x[:, 0] = random.uniform(-3.0, 3.0)
    assert_close(require(user, "min_max_normalize")(x), ref["min_max_normalize"](x))


def test_minmax_example_1(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    x = np.array([1.0, 3.0, 5.0])
    expected = np.array([0.0, 0.5, 1.0])
    assert_close(require(user, "min_max_normalize")(x), expected)


def test_minmax_example_2(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    x = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 20.0]])
    expected = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]])
    assert_close(require(user, "min_max_normalize")(x), expected)


def test_minmax_example_3(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    x = np.array([[2.0, 1.0], [2.0, 3.0]])
    output = require(user, "min_max_normalize")(x)
    assert_shape(output, x.shape)
    assert_finite(output)
    assert_close(output, ref["min_max_normalize"](x))


def test_cosine(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    dim = random.randint(2, 12)
    a, b = np.random.normal(size=dim), np.random.normal(size=dim)
    assert_close(require(user, "cosine_similarity")(a, b), ref["cosine_similarity"](a, b))
    A = np.random.normal(size=(random.randint(1, 6), dim))
    B = np.random.normal(size=(random.randint(1, 6), dim))
    assert_close(require(user, "cosine_similarity_matrix")(A, B), ref["cosine_similarity_matrix"](A, B))


def test_euclidean(user: dict[str, Any], ref: dict[str, Any]) -> None:
    import numpy as np

    dim = random.randint(2, 12)
    a, b = np.random.normal(size=dim), np.random.normal(size=dim)
    assert_close(require(user, "euclidean_distance")(a, b), ref["euclidean_distance"](a, b))
    A = np.random.normal(size=(random.randint(1, 6), dim))
    B = np.random.normal(size=(random.randint(1, 6), dim))
    assert_close(require(user, "euclidean_distance_matrix")(A, B), ref["euclidean_distance_matrix"](A, B))


CaseFn = Callable[[dict[str, Any], dict[str, Any]], None]


def fixed_tensor(shape: tuple[int, ...], start: float = 0.0) -> Any:
    import torch

    size = math.prod(shape)
    return (torch.arange(size, dtype=torch.float32).reshape(shape) + start) / max(size, 1)


def sample_softmax(user: dict[str, Any], ref: dict[str, Any], index: int) -> None:
    import numpy as np

    inputs = (
        (np.array([1.0, 2.0, 3.0]), -1),
        (np.array([[1.0, 1.0], [2.0, 2.0]]), -1),
        (np.array([1000.0, 1001.0]), -1),
    )
    x, axis = inputs[index]
    assert_close(require(user, "softmax")(x, axis), ref["softmax"](x, axis))


def sample_sdpa(user: dict[str, Any], ref: dict[str, Any], index: int) -> None:
    import torch

    if index == 0:
        q = k = v = fixed_tensor((2, 4, 5, 8))
        output, attn = require(user, "scaled_dot_product_attention")(q, k, v)
        assert_shape(output, (2, 4, 5, 8)); assert_shape(attn, (2, 4, 5, 5))
    elif index == 1:
        q = k = fixed_tensor((1, 2, 3, 4)); v = fixed_tensor((1, 2, 3, 6))
        output, attn = require(user, "scaled_dot_product_attention")(q, k, v)
        assert_shape(output, (1, 2, 3, 6)); assert_shape(attn, (1, 2, 3, 3))
    else:
        q = k = v = fixed_tensor((1, 1, 3, 4))
        mask = torch.ones((1, 1, 3, 3), dtype=torch.bool); mask[..., 2] = False
        actual = require(user, "scaled_dot_product_attention")(q, k, v, mask)
        expected = ref["scaled_dot_product_attention"](q, k, v, mask)
        _, attn = assert_sdpa_pair(actual, expected)
        assert_close(attn[..., 2], torch.zeros_like(attn[..., 2]))


def sample_causal_mask(user: dict[str, Any], ref: dict[str, Any], seq_len: int) -> None:
    import torch

    output = require(user, "causal_mask")(seq_len)
    assert_close(output, torch.tril(torch.ones(seq_len, seq_len)).bool())
    if seq_len == 5:
        assert output.dtype == torch.bool


def sample_attention_module(user: dict[str, Any], kind: str, index: int) -> None:
    import torch

    if kind == "mha":
        configs = ((16, 4, (2, 5, 16), None), (32, 8, (1, 7, 32), None), (8, 2, (3, 1, 8), torch.ones(1, 1, dtype=torch.bool)))
        model_dim, heads, shape, mask = configs[index]
        model = require(user, "MultiHeadAttention")(model_dim, heads)
        output = model(fixed_tensor(shape), mask)
        assert_shape(output, shape); assert_finite(output)
    elif kind == "gqa":
        configs = ((16, 4, 2, (2, 6, 16)), (32, 8, 1, (1, 4, 32)), (24, 6, 3, (3, 2, 24)))
        model_dim, heads, kv_heads, shape = configs[index]
        output = require(user, "GroupedQueryAttention")(model_dim, heads, kv_heads)(fixed_tensor(shape))
        assert_shape(output, shape); assert_finite(output)
    else:
        configs = (((2, 3, 16), (2, 5, 16), 16, 4), ((1, 1, 32), (1, 8, 32), 32, 8), ((3, 7, 8), (3, 2, 8), 8, 2))
        x_shape, context_shape, model_dim, heads = configs[index]
        output = require(user, "CrossAttention")(model_dim, heads)(fixed_tensor(x_shape), fixed_tensor(context_shape, 1.0))
        assert_shape(output, x_shape); assert_finite(output)


def sample_kv_cache(user: dict[str, Any], index: int) -> None:
    configs = ((1, 1), (1, 4), (2, 5))
    batch, steps = configs[index]
    model = require(user, "CachedSelfAttention")(8, 2)
    cache = None
    for step in range(steps):
        output, cache = model(fixed_tensor((batch, 1, 8), float(step)), cache)
        assert_shape(output, (batch, 1, 8))
    assert cache["k"].shape[2] == steps and cache["v"].shape[2] == steps


def sample_embedding(user: dict[str, Any], index: int) -> None:
    import torch

    configs = ((100, 16, (2, 5)), (32, 8, (1, 1)), (17, 4, (3, 2)))
    vocab, model_dim, shape = configs[index]
    model = require(user, "TokenEmbedding")(vocab, model_dim)
    ids = torch.arange(math.prod(shape)).reshape(shape) % vocab
    assert_shape(model.encode(ids), (*shape, model_dim))
    assert_shape(model.decode(fixed_tensor((*shape, model_dim))), (*shape, vocab))
    assert model.embed.weight.data_ptr() == model.proj.weight.data_ptr()


def sample_positional_encoding(user: dict[str, Any], index: int) -> None:
    import torch

    configs = ((4, 8), (1, 4), (6, 16))
    seq_len, model_dim = configs[index]
    output = require(user, "sinusoidal_positional_encoding")(seq_len, model_dim)
    assert_shape(output, (seq_len, model_dim)); assert_finite(output)
    if index == 1: assert_close(output, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))


def sample_rope(user: dict[str, Any], index: int) -> None:
    import torch

    if index == 0:
        cos, sin = require(user, "build_rope_cache")(5, 8)
        assert_shape(cos, (5, 4)); assert_shape(sin, (5, 4))
    else:
        shape = (2, 3, 5, 8) if index == 1 else (1, 1, 3, 8)
        cos, sin = require(user, "build_rope_cache")(shape[2], shape[3])
        x = fixed_tensor(shape)
        output = require(user, "apply_rope")(x, cos, sin)
        assert_shape(output, shape)
        if index == 2: assert_close(output[..., 0, :], x[..., 0, :])


def sample_residual(user: dict[str, Any], index: int) -> None:
    import torch
    import torch.nn as nn

    shape = ((2, 3, 8), (1, 1, 8), (3, 2, 16))[index]
    model_dim = shape[-1]
    sublayer = nn.Linear(model_dim, model_dim, bias=False)
    nn.init.zeros_(sublayer.weight)
    model = require(user, "ResidualSubLayer")(model_dim, sublayer, 0.0)
    model.eval()
    x = fixed_tensor(shape)
    assert_close(model(x), x)


def sample_swiglu(user: dict[str, Any], index: int) -> None:
    configs = ((8, 20, (2, 3, 8)), (16, 32, (1, 1, 16)), (24, 48, (4, 7, 24)))
    model_dim, hidden_dim, shape = configs[index]
    output = require(user, "SwiGLU")(model_dim, hidden_dim)(fixed_tensor(shape))
    assert_shape(output, shape); assert_finite(output)


def sample_encoder_block(user: dict[str, Any], index: int) -> None:
    import torch

    configs = ((16, 4, 32, (2, 5, 16), None), (32, 8, 64, (1, 8, 32), torch.tril(torch.ones(8, 8)).bool()), (8, 2, 16, (3, 1, 8), None))
    model_dim, heads, hidden_dim, shape, mask = configs[index]
    model = require(user, "TransformerEncoderBlock")(model_dim, heads, hidden_dim, 0.0)
    output = model(fixed_tensor(shape), mask)
    assert_shape(output, shape); assert_finite(output)


def sample_decoder_block(user: dict[str, Any], index: int) -> None:
    configs = (((2, 4, 16), (2, 6, 16), 16, 4, 32), ((1, 1, 32), (1, 9, 32), 32, 8, 64), ((3, 7, 8), (3, 2, 8), 8, 2, 16))
    x_shape, context_shape, model_dim, heads, hidden_dim = configs[index]
    model = require(user, "TransformerDecoderBlock")(model_dim, heads, hidden_dim, 0.0)
    output = model(fixed_tensor(x_shape), fixed_tensor(context_shape, 1.0))
    assert_shape(output, x_shape); assert_finite(output)


def sample_norm(user: dict[str, Any], ref: dict[str, Any], kind: str, index: int) -> None:
    import torch
    if kind == "batchnorm":
        shapes = ((16, 4, True), (8, 3, True), (2, 3, False)); batch, features, training = shapes[index]
        model = require(user, "BatchNorm1d")(features); model.train(); model(fixed_tensor((8, features), 1.0)); model.train(training)
        output = model(fixed_tensor((batch, features), 2.0)); assert_shape(output, (batch, features)); assert_finite(output)
    elif kind == "layernorm":
        shapes = ((2, 3, 4), (1, 1, 3), (4, 1, 8)); shape = shapes[index]
        x = torch.ones(shape) if index == 1 else fixed_tensor(shape)
        assert_close(require(user, "LayerNorm")(shape[-1])(x), ref["LayerNorm"](shape[-1])(x))
    else:
        shapes = ((2, 3, 4), (1, 1, 4), (5, 8)); shape = shapes[index]
        x = torch.zeros(shape) if index == 1 else fixed_tensor(shape)
        output = require(user, "RMSNorm")(shape[-1])(x); assert_shape(output, shape); assert_finite(output)


def sample_dropout(user: dict[str, Any], index: int) -> None:
    import torch
    configs = ((0.5, True), (0.3, False), (0.0, True)); p, training = configs[index]
    model = require(user, "Dropout")(p); model.train(training); x = torch.ones(4000)
    output = model(x)
    if not training or p == 0: assert_close(output, x)
    else:
        values = set(float(value) for value in torch.unique(output)); kept = 1 / (1 - p)
        assert 0.0 in values and any(abs(value - kept) < 1e-5 for value in values)


def sample_activation(user: dict[str, Any], name: str, index: int) -> None:
    import torch
    values = (torch.tensor([-2.0, 0.0, 3.0]), torch.tensor([-20.0, 20.0]), torch.zeros((2, 3)))[index]
    expected = getattr(torch, name)(values) if name != "gelu" else torch.nn.functional.gelu(values, approximate="tanh")
    assert_close(require(user, name)(values), expected)


def sample_losses(user: dict[str, Any], index: int) -> None:
    import torch
    if index == 0: assert_close(require(user, "mse_loss")(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0])), torch.tensor(0.5)); return
    if index == 1: assert_close(require(user, "bce_loss")(torch.tensor([0.9, 0.2]), torch.tensor([1.0, 0.0])), torch.tensor(0.164252)) ; return
    assert_close(require(user, "mse_loss")(torch.ones(2), torch.ones(2)), torch.tensor(0.0))


def sample_cross_entropy(user: dict[str, Any], index: int) -> None:
    import torch
    import torch.nn.functional as F
    logits, target = ((torch.tensor([[2.0, 0.5, 0.1]]), torch.tensor([0])), (torch.zeros((1, 2)), torch.tensor([1])), (fixed_tensor((4, 5)), torch.tensor([0, 1, 2, 3])))[index]
    assert_close(require(user, "cross_entropy")(logits, target), F.cross_entropy(logits, target))


def sample_vector(user: dict[str, Any], ref: dict[str, Any], kind: str, index: int) -> None:
    import numpy as np
    if kind == "l2":
        xs = (np.array([3.0, 4.0]), np.array([[3.0, 4.0], [0.0, 5.0]]), np.array([0.0, 0.0])); x = xs[index]
        assert_close(require(user, "l2_normalize")(x, axis=1 if index == 1 else -1), ref["l2_normalize"](x, axis=1 if index == 1 else -1))
    elif kind == "cosine":
        pairs = ((np.array([1.,0.]),np.array([1.,0.])), (np.array([1.,0.]),np.array([0.,1.])))
        if index < 2: assert_close(require(user, "cosine_similarity")(*pairs[index]), float(index == 0))
        else: assert_shape(require(user, "cosine_similarity_matrix")(np.zeros((2,3)), np.zeros((4,3))), (2,4))
    else:
        pairs = ((np.array([0.,0.]),np.array([3.,4.])), (np.array([1.,2.]),np.array([1.,2.])))
        if index < 2: assert_close(require(user, "euclidean_distance")(*pairs[index]), 5.0 if index == 0 else 0.0)
        else: assert_shape(require(user, "euclidean_distance_matrix")(np.zeros((2,3)), np.zeros((4,3))), (2,4))


BASE_CASES: dict[str, list[tuple[str, CaseFn]]] = {
    "attention.softmax": [("行级数值稳定", test_softmax_rows), ("指定 axis", test_softmax_axis)],
    "attention.sdpa": [("基础输出", test_sdpa_basic), ("mask 屏蔽", test_sdpa_mask)],
    "attention.causal_mask": [("下三角布尔矩阵", test_causal_mask)],
    "attention.mha": [("输出形状", test_mha_shape), ("mask 兼容", test_mha_mask)],
    "attention.gqa": [("GQA 输出形状", test_gqa_shape)],
    "attention.cross": [("交叉注意力输出", test_cross_attention)],
    "attention.kv_cache": [("缓存增长", test_kv_cache_growth)],
    "transformer.embedding": [("查表与权重共享", test_embedding)],
    "transformer.positional_encoding": [("正弦位置编码", test_positional_encoding)],
    "transformer.rope": [("RoPE 缓存", test_rope_cache), ("RoPE 旋转", test_rope_apply)],
    "transformer.residual": [("Pre-Norm 残差", test_residual)],
    "transformer.swiglu": [("SwiGLU 输出", test_swiglu)],
    "transformer.encoder_block": [("Encoder Block 输出", test_encoder_block)],
    "transformer.decoder_block": [("Decoder Block 输出", test_decoder_block)],
    "normalization.batchnorm": [("训练归一化", test_batchnorm_train), ("推理模式", test_batchnorm_eval)],
    "normalization.layernorm": [("样本内归一化", test_layernorm)],
    "normalization.rmsnorm": [("均方根缩放", test_rmsnorm)],
    "regularization.dropout": [("推理直通", test_dropout_eval), ("训练缩放", test_dropout_train)],
    "activation.relu": [("ReLU 截断", test_relu)],
    "activation.gelu": [("GELU tanh 近似", test_gelu)],
    "activation.sigmoid": [("Sigmoid 映射", test_sigmoid)],
    "activation.tanh": [("Tanh 映射", test_tanh)],
    "loss.cross_entropy": [("交叉熵", test_cross_entropy)],
    "loss.mse_bce": [("MSE 与 BCE", test_mse_bce)],
    "vector.l2_normalize": [("L2 单位化", test_l2_normalize)],
    "vector.minmax": [("Min-Max 缩放", test_minmax)],
    "vector.cosine": [("余弦相似度", test_cosine)],
    "vector.euclidean": [("欧氏距离", test_euclidean)],
}


def seeded_case(problem_id: str, case_index: int, fn: CaseFn) -> CaseFn:
    seed = sum(ord(char) for char in problem_id) * 100 + case_index

    def run(user: dict[str, Any], ref: dict[str, Any]) -> None:
        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass
        fn(user, ref)

    return run


def boundary_case(fn: CaseFn) -> CaseFn:
    """Run an existing check at deterministic lower-bound shapes and constant inputs."""
    def run(user: dict[str, Any], ref: dict[str, Any]) -> None:
        import numpy as np

        original_random = {
            "randint": random.randint,
            "choice": random.choice,
            "randrange": random.randrange,
            "random": random.random,
            "uniform": random.uniform,
        }
        original_normal = np.random.normal

        def zero_normal(*args: Any, **kwargs: Any) -> Any:
            size = kwargs.get("size")
            if size is None and len(args) >= 3:
                size = args[2]
            return 0.0 if size is None else np.zeros(size)

        random.randint = lambda low, high: low
        random.choice = lambda values: values[0]
        random.randrange = lambda stop: 0
        random.random = lambda: 0.0
        random.uniform = lambda low, high: high
        np.random.normal = zero_normal
        try:
            fn(user, ref)
        finally:
            random.randint = original_random["randint"]
            random.choice = original_random["choice"]
            random.randrange = original_random["randrange"]
            random.random = original_random["random"]
            random.uniform = original_random["uniform"]
            np.random.normal = original_normal

    return run


def public_case(problem_id: str, case_index: int, fn: CaseFn) -> CaseFn:
    regular = seeded_case(problem_id, case_index, fn)
    edge = boundary_case(fn)

    def run(user: dict[str, Any], ref: dict[str, Any]) -> None:
        regular(user, ref)
        edge(user, ref)

    return run


CASES: dict[str, list[tuple[str, CaseFn]]] = {
    problem_id: [
        (
            f"隐藏用例 {index + 1:02d} · {base_cases[index % len(base_cases)][0]}",
            seeded_case(problem_id, index, base_cases[index % len(base_cases)][1]),
        )
        for index in range(20)
    ]
    for problem_id, base_cases in BASE_CASES.items()
}


PUBLIC_CASES: dict[str, list[tuple[str, CaseFn]]] = {
    problem_id: [
        (f"示例 {index + 1} · 含边界校验", public_case(problem_id, index, base_cases[index % len(base_cases)][1]))
        for index in range(3)
    ]
    for problem_id, base_cases in BASE_CASES.items()
}

# These cases intentionally match the three examples displayed in the problem statement.
PUBLIC_CASES["vector.minmax"] = [
    ("示例 1", test_minmax_example_1),
    ("示例 2", test_minmax_example_2),
    ("示例 3", test_minmax_example_3),
]

PUBLIC_CASES.update({
    "attention.softmax": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_softmax(user, ref, i)) for i in range(3)],
    "attention.sdpa": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_sdpa(user, ref, i)) for i in range(3)],
    "attention.causal_mask": [(f"示例 {i + 1}", lambda user, ref, n=n: sample_causal_mask(user, ref, n)) for i, n in enumerate((1, 3, 5))],
    "attention.mha": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_attention_module(user, "mha", i)) for i in range(3)],
    "attention.gqa": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_attention_module(user, "gqa", i)) for i in range(3)],
    "attention.cross": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_attention_module(user, "cross", i)) for i in range(3)],
    "attention.kv_cache": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_kv_cache(user, i)) for i in range(3)],
    "transformer.embedding": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_embedding(user, i)) for i in range(3)],
    "transformer.positional_encoding": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_positional_encoding(user, i)) for i in range(3)],
    "transformer.rope": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_rope(user, i)) for i in range(3)],
    "transformer.residual": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_residual(user, i)) for i in range(3)],
    "transformer.swiglu": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_swiglu(user, i)) for i in range(3)],
    "transformer.encoder_block": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_encoder_block(user, i)) for i in range(3)],
    "transformer.decoder_block": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_decoder_block(user, i)) for i in range(3)],
    "normalization.batchnorm": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_norm(user, ref, "batchnorm", i)) for i in range(3)],
    "normalization.layernorm": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_norm(user, ref, "layernorm", i)) for i in range(3)],
    "normalization.rmsnorm": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_norm(user, ref, "rmsnorm", i)) for i in range(3)],
    "regularization.dropout": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_dropout(user, i)) for i in range(3)],
    "activation.relu": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_activation(user, "relu", i)) for i in range(3)],
    "activation.gelu": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_activation(user, "gelu", i)) for i in range(3)],
    "activation.sigmoid": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_activation(user, "sigmoid", i)) for i in range(3)],
    "activation.tanh": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_activation(user, "tanh", i)) for i in range(3)],
    "loss.cross_entropy": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_cross_entropy(user, i)) for i in range(3)],
    "loss.mse_bce": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_losses(user, i)) for i in range(3)],
    "vector.l2_normalize": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_vector(user, ref, "l2", i)) for i in range(3)],
    "vector.cosine": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_vector(user, ref, "cosine", i)) for i in range(3)],
    "vector.euclidean": [(f"示例 {i + 1}", lambda user, ref, i=i: sample_vector(user, ref, "euclidean", i)) for i in range(3)],
})


def debug_value(value: Any) -> str:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size <= 12:
                return f"ndarray(shape={value.shape}, dtype={value.dtype}, value={np.array2string(value, precision=4)})"
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    except ImportError:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.numel() <= 12:
                return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, value={detached})"
            return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        if isinstance(value, torch.nn.Module):
            return f"{type(value).__name__}(training={value.training})"
    except ImportError:
        pass
    if isinstance(value, (str, int, float, bool, type(None))):
        text = repr(value)
    elif isinstance(value, (list, tuple, set, dict)):
        text = repr(value)
    else:
        text = f"<{type(value).__name__}>"
    return text if len(text) <= 300 else text[:297] + "..."


def debug_judge(problem_id: str, submission_path: Path, breakpoints: list[int]) -> dict[str, Any]:
    code = submission_path.read_text(encoding="utf-8")
    global_names = debug_global_names(code)
    ref = build_reference_namespace()
    user: dict[str, Any] = {"__name__": "submission"}
    try:
        execute(code, user)
    except Exception:
        return {"events": [], "stdout": "", "error": traceback.format_exc()}

    for name in ("MultiHeadAttention", "CrossAttention"):
        if name not in user and name in ref:
            user[name] = ref[name]

    tests = PUBLIC_CASES.get(problem_id, [])
    if not tests:
        return {"events": [], "stdout": "", "error": "没有可调试的示例"}

    events: list[dict[str, Any]] = []
    breakpoint_set = set(breakpoints)
    active = not breakpoint_set
    hit_breakpoint = False

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal active, hit_breakpoint
        if frame.f_code.co_filename != "<submission>":
            return None
        if event == "line":
            line = frame.f_lineno
            if line in breakpoint_set:
                active = True
                hit_breakpoint = True
            if active and len(events) < 250:
                variables = {
                    name: debug_value(value)
                    for name, value in frame.f_locals.items()
                    if not name.startswith("__")
                }
                globals_snapshot = {
                    name: debug_value(frame.f_globals[name])
                    for name in global_names
                    if name in frame.f_globals
                }
                events.append({
                    "line": line,
                    "function": frame.f_code.co_name,
                    "variables": variables,
                    "globals": globals_snapshot,
                    "breakpoint": line in breakpoint_set,
                    "depth": debug_depth(frame),
                })
        return tracer

    output = io.StringIO()
    case_error = ""
    try:
        sys.settrace(tracer)
        with contextlib.redirect_stdout(output):
            tests[0][1](user, ref)
    except Exception as exc:
        case_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(None)

    if breakpoint_set and not hit_breakpoint and not case_error:
        case_error = "示例 1 没有执行到所设断点，请将断点放在会运行的函数主体行。"
    return {
        "events": events,
        "stdout": output.getvalue()[-4000:],
        "error": case_error,
        "case": "示例 1",
        "breakpoints": sorted(breakpoint_set),
    }


def debug_depth(frame: Any) -> int:
    depth = 0
    current = frame
    while current is not None and current.f_code.co_filename == "<submission>":
        depth += 1
        current = current.f_back
    return depth


def debug_global_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), ApiHandler)
    print(f"AI First LeetCode API: http://127.0.0.1:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--judge", nargs=3, metavar=("PROBLEM_ID", "SUBMISSION", "MODE"))
    parser.add_argument("--judge-case-count", type=int)
    parser.add_argument("--debug", nargs=3, metavar=("PROBLEM_ID", "SUBMISSION", "BREAKPOINTS"))
    args = parser.parse_args()
    if args.judge:
        result = judge(args.judge[0], Path(args.judge[1]), args.judge[2], args.judge_case_count)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.debug:
        result = debug_judge(args.debug[0], Path(args.debug[1]), json.loads(args.debug[2]))
        print(json.dumps(result, ensure_ascii=False))
        return
    serve(args.port)


if __name__ == "__main__":
    main()
