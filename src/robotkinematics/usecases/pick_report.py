"""@file pick_report.py

@brief 抓取流水线的**报告**：把 `PickResult` 变成 `Report`（报什么，而不是怎么画）。

【为什么报告在用例层】因为"报什么"是领域知识的一部分：七道工序各自的数字、
候选解的对比、失败原因里的下一步建议 —— 这些只有用例知道。至于分节线用几个等号、
表格怎么对齐，那是适配器的事（见 `adapters/render_text.py`）。
"""

from __future__ import annotations

import numpy as np

from ..contracts import (
    HeadingBlock,
    Report,
    TableBlock,
    TextBlock,
    fmt_vector,
)
from .pick_types import PickResult, PickTask


def to_report(task: PickTask, result: PickResult) -> Report:
    """把任务结果渲染成报告（终端 / JSON 都由适配器决定）。"""
    blocks: list = [
        TextBlock.of(
            "机械臂：Panda（末端：夹爪 TCP）",
            f"目标位姿：位置 {fmt_vector(np.asarray(result.target.t, dtype=float))}",
            f"结论：{result.outcome}",
            f"原因：{result.reason}",
        ),
        HeadingBlock("七道工序"),
        TableBlock(
            ("工序", "结论", "关键数字"),
            tuple((s.name, s.detail, numbers_summary(s.numbers)) for s in result.steps),
            ("<", "<", "<"),
        ),
    ]

    if result.q is not None:
        blocks += [
            HeadingBlock("输出"),
            TextBlock.of(
                f"关节解 q = {fmt_vector(result.q)} rad，{fmt_vector(np.rad2deg(result.q), 1)} deg"
            ),
        ]
    if len(result.candidates) > 1:
        blocks += [
            HeadingBlock("候选解对比"),
            TableBlock(
                ("初值", "残差", "σ_min", "离限位", "评分", "结论"),
                tuple(
                    (
                        f"#{c.seed_index + 1}",
                        f"{c.residual:.2e}",
                        f"{c.sigma_min:.4f}" if np.isfinite(c.sigma_min) else "—",
                        f"{np.rad2deg(c.limit_margin):.1f}°"
                        if np.isfinite(c.limit_margin)
                        else "—",
                        f"{c.score:.3f}",
                        c.rejected_by or "存活",
                    )
                    for c in sorted(result.candidates, key=lambda c: -c.score)
                ),
                ("<", ">", ">", ">", ">", "<"),
            ),
        ]

    return Report(
        title="抓取任务流水线",
        blocks=tuple(blocks),
        fields={
            "outcome": result.outcome,
            "reason": result.reason,
            "arm": "panda",
            "frame": task.frame,
            "target_position": np.asarray(result.target.t, dtype=float),
            "q": result.q,
            "q_deg": None if result.q is None else np.rad2deg(result.q),
            "steps": [
                {"name": s.name, "detail": s.detail, "numbers": jsonable(s.numbers)}
                for s in result.steps
            ],
            "candidates": [
                {
                    "seed_index": c.seed_index + 1,
                    "residual": c.residual,
                    "sigma_min": c.sigma_min,
                    "condition": c.condition,
                    "limit_margin_deg": float(np.rad2deg(c.limit_margin))
                    if np.isfinite(c.limit_margin)
                    else None,
                    "score": c.score,
                    "rejected_by": c.rejected_by,
                }
                for c in result.candidates
            ],
        },
    )


def numbers_summary(numbers: dict) -> str:
    """把关键数字压成一行 —— 小量用科学计数法，否则 1e-8 会显示成 0.0000。"""
    parts = []
    for key, value in numbers.items():
        parts.append(f"{key}={format_number(value)}")
        if len(parts) >= 3:
            break
    return "，".join(parts)


def format_number(value) -> str:
    """数字的终端写法：向量按向量、元组按元组，标量按大小选定点/科学计数。"""
    if isinstance(value, np.ndarray):
        return fmt_vector(value, 3)
    if isinstance(value, tuple):
        return "[" + ", ".join(format_number(v) for v in value) + "]"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.2e}" if abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def jsonable(numbers: dict) -> dict:
    """把 numbers 里的 numpy 类型转成 JSON 能编码的 —— JSON 渲染器会用它。"""
    out = {}
    for key, value in numbers.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, tuple):
            out[key] = [float(v) for v in value]
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out
