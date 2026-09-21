"""@file contracts.py

@brief 用例与适配器之间的契约：结果对象（用例产出什么）+ 协议（用例需要什么）。

【为什么要有这一层】重构前，用例直接依赖"打印成中文表格"的具体实现，
于是"算一遍 FK"和"输出格式"焊在一起：测试只能断言文本片段，也出不了 JSON。

现在这条边界是这样的：
  * **用例**决定"报什么"——标题、段落、哪些数、哪些表——但不知道终端宽度、不写文件；
  * **适配器**决定"怎么呈现"——走终端、走 JSON、还是走 HTML。

⚠️ 值格式化（`[ 1.0000, -0.5000]` 这种）放在这里，因为它与终端无关；
   而**布局**（表格列宽、对齐、缩进、分节线）在 adapters/render_text.py 里。
   这条分界线的判据：换个渲染器还要不要重写？要重写的就属于布局。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# ── 值格式化：把数字变成人可读的字符串（纯函数，与终端/文件无关）──────────────


def fmt_vector(values, digits: int = 4) -> str:
    """[ 1.0000, -0.5000] 这种紧凑形式。"""
    arr = np.asarray(values, dtype=float).ravel()
    return "[" + ", ".join(f"{v: .{digits}f}" for v in arr) + "]"


def fmt_condition(value: float, limit: float = 1e12) -> str:
    """条件数：大到没有意义时直接写 inf/极大，而不是印一串 18 位数字。"""
    if not np.isfinite(value) or value > limit:
        return "inf/极大"
    return f"{value:.2f}"


# ── 报告块：用例描述"报什么"，渲染器决定"怎么画"───────────────────────────


@dataclass(frozen=True)
class TextBlock:
    """一段文字（可以多行）。已含具体数值，但不含对齐与装饰。"""

    lines: tuple[str, ...]

    @staticmethod
    def of(*lines: str) -> TextBlock:
        """@brief 用一列字符串直接造一个文本块（比写 tuple 顺手）。"""
        return TextBlock(tuple(lines))


@dataclass(frozen=True)
class HeadingBlock:
    """小节标题。怎么画由渲染器决定。"""

    title: str


@dataclass(frozen=True)
class VectorBlock:
    """一行"标签 = 向量"。"""

    label: str
    values: np.ndarray
    digits: int = 4
    suffix: str = ""


@dataclass(frozen=True)
class MatrixBlock:
    """一个"标签 + 矩阵"。矩阵本身不在这里排版。"""

    label: str
    matrix: np.ndarray
    digits: int = 4


@dataclass(frozen=True)
class TableBlock:
    """一张表。rows 里的单元格已经是字符串（数值由用例格式化）。"""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    aligns: tuple[str, ...] | None = None


Block = HeadingBlock | TextBlock | VectorBlock | MatrixBlock | TableBlock


@dataclass(frozen=True)
class Report:
    """用例的产出：一份与呈现方式无关的报告。"""

    title: str
    blocks: tuple[Block, ...] = ()
    output_name: str | None = None
    fields: dict[str, object] = field(default_factory=dict)


# ── 协议：用例需要什么（用例只依赖这些抽象，不认识具体实现）───────────────────


@runtime_checkable
class Renderer(Protocol):
    """把报告渲染成字符串（终端文本 / JSON 都由适配器实现）。"""

    def render(self, report: Report) -> str:
        """@brief 把一份报告渲染成字符串。"""
        ...


@runtime_checkable
class CaseSource(Protocol):
    """算例来源（可选；流水线入口当前不依赖它）。"""

    def coordinate_cases(self) -> list:
        """@brief 取相机→本体变换的算例。"""
        ...

    def panda_cases(self) -> list:
        """@brief 取 Panda 演示姿态。"""
        ...


@runtime_checkable
class Plotter(Protocol):
    """出图。用例只关心"图存到哪了"，不关心 matplotlib。"""

    def plot_panda_skeleton(self, robot, q: np.ndarray) -> Path:
        """@brief 画 Panda 的骨架图（3D 不可用时自动回退三视图）。"""
        ...

    def plot_pick_result(self, task, result, robot) -> Path:
        """@brief 画抓取任务图（可达边界 / 机械臂姿态 / 末端路径 / 奇异点标记）。"""
        ...
