"""@file render_text.py

@brief 把报告渲染成终端文本 —— 布局全在这里：分节线、表格列宽、对齐、缩进。

【为什么要自己算显示宽度】中文字符在等宽终端里占 2 列，Python 的 f"{s:<14}" 只按字符个数
补空格，中文列必然错位。这里用 unicodedata 算显示宽度，保证表格竖线对齐。
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

from ..contracts import (
    HeadingBlock,
    MatrixBlock,
    Report,
    TableBlock,
    TextBlock,
    TwoLinkChartBlock,
    VectorBlock,
    fmt_vector,
)
from ..core.exceptions import KinematicsError


def display_width(text: str) -> int:
    """字符串在等宽终端里占的列数（全角字符按 2 算）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int, align: str = "<") -> str:
    """按显示宽度补空格。align: '<' 左对齐, '>' 右对齐, '^' 居中。"""
    space = width - display_width(text)
    if space <= 0:
        return text
    if align == ">":
        return " " * space + text
    if align == "^":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return text + " " * space


def table(
    headers: list[str] | tuple[str, ...],
    rows: list[list[str]] | tuple[tuple[str, ...], ...],
    aligns: list[str] | tuple[str, ...] | None = None,
    gap: int = 2,
) -> str:
    """渲染一张定宽表格，含表头分隔线。"""
    if not rows:
        return ""
    aligns = list(aligns) if aligns else ["<"] * len(headers)
    if len(aligns) != len(headers):
        raise KinematicsError("表格的列数与对齐方式数量不一致")

    widths = [
        max(display_width(headers[i]), *(display_width(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    sep = " " * gap

    def line(cells) -> str:
        return sep.join(pad(str(cells[i]), widths[i], aligns[i]) for i in range(len(headers)))

    out = [line(headers), "-" * display_width(line(headers))]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def banner(title: str, width: int = 72) -> str:
    """分节标题，与课件里 "=" * 72 的风格一致。"""
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_matrix(matrix: np.ndarray, digits: int = 4) -> str:
    """矩阵按行打印，列宽按最大值对齐 —— 数值大小一眼能比。

    ⚠️ 这是布局，所以它属于渲染器而不是 contracts：换 JSON 输出时不再需要列宽对齐。
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise KinematicsError(f"format_matrix 需要二维矩阵，收到 {arr.shape}")
    cells = [[f"{v: .{digits}f}" for v in row] for row in arr]
    widths = [max(display_width(row[i]) for row in cells) for i in range(arr.shape[1])]
    return "\n".join(
        "[" + "  ".join(pad(cell, widths[i], ">") for i, cell in enumerate(row)) + "]"
        for row in cells
    )


def write_output(name: str, text: str, outputs_dir: str | Path) -> Path:
    """把报告写到 outputs/ 下，返回实际路径。"""
    directory = Path(outputs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


class TextRenderer:
    """终端文本渲染器。outputs_dir 为 None 时只打印、不落盘。"""

    def __init__(self, outputs_dir: str | Path | None = None) -> None:
        """@brief 指定落盘目录。

        @param outputs_dir 目录；None 表示只打印、不写文件
        """
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else None

    def render(self, report: Report) -> str:
        """@brief 把报告渲染成终端文本。

        @param report 用例产出的报告
        @return 文本；若构造时给了 outputs_dir 且报告带 output_name，
            同时把**正文**（不含分节线与"已保存"提示）落盘
        """
        # 正文 = 所有块拼起来；分节线只是终端装饰，不进正文
        body = "\n".join(
            rendered for rendered in (self._render_block(b) for b in report.blocks) if rendered
        ).rstrip()
        text = f"{banner(report.title)}\n{body}" if body else banner(report.title)

        if report.output_name and self.outputs_dir is not None:
            # 落盘的是正文本身（不含分节线、也不含“已保存”这一行），
            # 这样文件才能与课件的交付物逐字符对照
            path = write_output(report.output_name, body, self.outputs_dir)
            text += f"\n\n已保存：{path}"
        return text

    def _render_block(self, block) -> str:
        if isinstance(block, HeadingBlock):
            return banner(block.title)
        if isinstance(block, TextBlock):
            return "\n".join(block.lines)
        if isinstance(block, VectorBlock):
            return f"{block.label} = {fmt_vector(block.values, block.digits)}{block.suffix}"
        if isinstance(block, MatrixBlock):
            return f"{block.label}\n{format_matrix(block.matrix, block.digits)}"
        if isinstance(block, TableBlock):
            return table(block.headers, block.rows, block.aligns)
        if isinstance(block, TwoLinkChartBlock):
            # 网格与坐标范围用渲染器自己的默认值（画法不进契约层）
            chart = ascii_two_link_chart(block.points)
            return f"{block.label}\n{chart}" if block.label else chart
        raise KinematicsError(f"未知的报告块类型: {type(block).__name__}")


def ascii_two_link_chart(points, size: int = 25, limit: float = 2.1) -> str:
    """用字符画一条二连杆 —— 终端里也能看出姿态（不需要图形界面）。

    网格 size×size，范围 [-limit, limit]，B=base、E=elbow、T=tool。
    ⚠️ 这是"画法"，所以它在渲染器里；用例只提供 points（见 contracts.TwoLinkChartBlock）。
    """
    base, elbow, tool = points
    grid = [[" "] * size for _ in range(size)]

    def to_cell(point) -> tuple[int, int]:
        col = round((point[0] + limit) / (2 * limit) * (size - 1))
        row = round((limit - point[1]) / (2 * limit) * (size - 1))
        return min(max(row, 0), size - 1), min(max(col, 0), size - 1)

    for start, end in ((base, elbow), (elbow, tool)):
        for t in np.linspace(0.0, 1.0, 18):
            p = start + t * (end - start)
            row, col = to_cell(p)
            grid[row][col] = "*"
    for marker, point in (("B", base), ("E", elbow), ("T", tool)):
        row, col = to_cell(point)
        grid[row][col] = marker

    border = "+" + "-" * size + "+"
    body = "\n".join("|" + "".join(row) + "|" for row in grid)
    return f"{border}\n{body}\n{border}"
