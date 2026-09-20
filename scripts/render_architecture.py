#!/usr/bin/env python3
"""@file render_architecture.py

@brief 生成架构图（SVG + PNG）—— 三块：分层结构 / 抓取流水线 / 关键配置。

【为什么不用 graphviz 自动布局】自动布局会把**每一行 import 都画成一条边**，
结果是一张 import 图：26 个节点、40 多条边交叉在一起，看不出结构。
架构图要传达的是"意图"，不是"完整的依赖事实"：

  * 层与层之间只画**一根**依赖箭头（层内部的文件列成卡片，不画彼此连线）；
  * 契约层画成一条竖带，说明"用例与适配器只通过它对话，彼此不直接依赖"；
  * 流水线单独一块，标注**步骤之间传的是什么数据**（这才是读者关心的）；
  * 关键数字与配置单独一块。

和 face_recognition_app 的架构图用同一套语言，便于对照。
用脚本画而不是手写 SVG：改一个数字重跑即可，不用手工挪框。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotkinematics.adapters.plot_mpl import configure_chinese_font  # noqa: E402

OUT_DIR = ROOT / "docs" / "architecture"

# 配色与项目其它部分一致（沿用 facegate 那套柔和的层次色）
COLORS = {
    "cli": "#fef3c7",
    "usecases": "#dbeafe",
    "adapters": "#ede9fe",
    "core": "#dcfce7",
    "contracts": "#fee2e2",
    "panel": "#f8fafc",
    "warn": "#b91c1c",
}
EDGE = "#94a3b8"
TEXT = "#0f172a"
MUTED = "#475569"

# ── 各层里有什么（改这里就能改图）──────────────────────────────────────────

LAYERS = [
    (
        "入口 · main.py",
        "唯一入口 = 组合根：命令表一览 + 全部装配 + 渲染 + 唯一的异常出口",
        "cli",
        [
            "① 命令表 —— 命令 → 处理函数（一眼看全部能力）",
            "② 装配 —— 全工程唯一一处“选实现”",
            "③ 入口流程 + 异常出口",
            "④ 处理函数 —— 一条命令一个",
        ],
    ),
    (
        "用例层 · usecases",
        "编排 + 产出 Report：不认识终端、不写文件、不认识 matplotlib",
        "usecases",
        [
            "pick.py ★ —— 抓取流水线（七道工序 + 六类结论）",
            "pose / transforms —— 位姿表示、相机 → 本体变换链",
            "planar / panda —— 二连杆与 Panda 的 FK / IK / 雅可比",
            "batch / env_check —— 批量算例、环境自检",
        ],
    ),
    (
        "适配层 · adapters",
        "唯一认识外部世界的地方 —— 第三方库只出现在这一层的三个文件里",
        "adapters",
        [
            "config_yaml.py —— YAML + 未知项告警 + 范围校验",
            "cases_csv.py —— 读 data/cases/*.csv",
            "render_text / render_json —— 同一份 Report 的两种呈现",
            "plot_mpl.py —— 出图（3D 不可用时回退三视图）",
        ],
    ),
    (
        "核心层 · core",
        "库覆盖不到的那部分（★ = 仍然自己写的）；其余运动学全部交给库",
        "core",
        [
            "robots.py —— 从库建模型：末端帧、限位、link 位置",
            "planar2r.py ★ —— 二连杆解析 FK/IK（库给不出两组解）",
            "workspace.py ★ —— 可达性采样（库没有这个 API）",
            "singularity.py —— SVD 体检 + 库的 manipulability",
            "exceptions.py —— 领域异常（可达 / 收敛 / 奇异）",
        ],
    ),
]

# ── 抓取流水线：七道工序与步骤之间传递的数据 ────────────────────────────────

PIPELINE = [
    ("① 观测 → 本体位姿", "相机坐标 → 本体坐标", "SE3 位姿"),
    ("② 可达性预筛", "沿该方向可达多远的采样估计", "ReachEstimate（余量）"),
    ("③ 多初值 IK", "24 个初值 → 候选解集", "候选解 q"),
    ("④ 雅可比体检", "σ_min / cond / 最难方向", "每组解的体检结果"),
    ("⑤ 限位校验", "逐组过滤越限位的解", "存活的候选"),
    ("⑥ 解择优", "限位余量 / σ_min / 就近 三项评分", "选中的一组 q"),
    ("⑦ 微动可行性校验", "末端压 1mm，关节要动几度", "可执行 / 不可执行"),
]

FAILURES = [
    "不在工作空间（差多少米）",
    "姿态不可达（位置够、方向做不到）",
    "收敛精度不足",
    "所有解越关节限位",
    "所有解接近奇异",
    "微动不可行",
]

# ── 关键配置与数字 ──────────────────────────────────────────────────────────

CONFIG_CARDS = [
    ("多初值个数", "24", "pick.seeds"),
    ("残差容差", "1e-4", "pick.residual_tol"),
    ("接近奇异阈值", "0.02", "pick.min_sigma"),
    ("可达性采样", "8000", "pick.workspace_samples"),
    ("限位余量要求", "0.0°", "pick.limit_margin_deg"),
]

KEY_NUMBERS = (
    "可达边界是采样估计（固定 seed，可复现）：全方向最大 1.19 m，但沿斜下方只有 1.09 m —— 工作空间不是球  ｜  "
    "评分权重 0.5 / 0.3 / 0.2 写在 pick.SCORE_WEIGHTS 里，可审计  ｜  "
    "170 条断言 · 自研的两处（闭式解 / 可达采样）均与库逐位对照"
)


# ── 画图工具 ────────────────────────────────────────────────────────────────


def box(ax, x, y, w, h, text, *, fill, fontsize=9, bold=False, align="left", pad=0.2):
    """画一个圆角卡片：文字默认左对齐、垂直居中。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size=0.6",
            linewidth=1.0,
            edgecolor=EDGE,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center" if align == "center" else "left",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        fontweight="bold" if bold else "normal",
        zorder=3,
        wrap=True,
    )


def label(ax, x, y, text, *, fontsize=9, color=MUTED, ha="left", bold=False):
    """写一段不画框的说明文字（小标题、图例、注释都用它）。"""
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold" if bold else "normal",
        zorder=3,
    )


def arrow(ax, x1, y1, x2, y2, *, color=EDGE, style="-|>", lw=1.4, linestyle="-"):
    """画一根箭头：层间依赖、工序之间的数据流都用它。"""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            mutation_scale=12,
            zorder=4,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_layers(ax) -> None:
    """左panel：分层结构。层间只画一根依赖箭头，层内列文件卡片。"""
    ax.text(2, 91, "分层结构 · Layered Architecture", fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(
        2,
        88.2,
        "依赖只向下；用例与适配器之间没有直接依赖，只通过契约层对话",
        fontsize=8.5,
        color=MUTED,
    )

    top, bottom = 87.5, 27.0
    gap = 2.6
    n = len(LAYERS)
    band_h = (top - bottom - gap * (n - 1)) / n

    y = top
    for index, (name, subtitle, color_key, cards) in enumerate(LAYERS):
        ax.add_patch(
            FancyBboxPatch(
                (2, y - band_h),
                52.5,
                band_h,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                linewidth=1.2,
                edgecolor=EDGE,
                facecolor=COLORS["panel"],
                zorder=1,
            )
        )
        label(ax, 3.6, y - 2.6, name, fontsize=11, color=TEXT, bold=True)
        label(ax, 3.6, y - 5.4, subtitle, fontsize=7.6, color=MUTED)

        # 文件卡片：两列排布，均匀占满
        area_top = y - 6.4
        area_h = band_h - 6.9
        cols = 2 if len(cards) > 2 else len(cards)
        rows = (len(cards) + cols - 1) // cols
        card_h = min(3.4, area_h / rows - 0.4)
        for i, text in enumerate(cards):
            row, col = divmod(i, cols)
            card_w = (51.0 - (cols + 1) * 0.7) / cols
            cx = 3.2 + 0.8 + col * (card_w + 0.8)
            cy = area_top - (row + 1) * card_h - row * 0.4
            box(ax, cx, cy, card_w, card_h, "", fill=COLORS[color_key])
            ax.text(
                cx + card_w / 2,
                cy + card_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=6.9,
                color=TEXT,
                zorder=3,
            )

        # 层间依赖箭头
        if index < n - 1:
            arrow(ax, 28.2, y - band_h - 0.3, 28.2, y - band_h - gap + 0.3, lw=1.6)
            label(ax, 29, y - band_h - gap / 2, "依赖", fontsize=7.5)
        y -= band_h + gap

    # 契约层：竖条，横跨"用例层 + 适配层"（这两层靠它对话，彼此不直接依赖）
    contracts_y = top - band_h - gap
    contracts_h = band_h + gap + band_h
    ax.add_patch(
        FancyBboxPatch(
            (56.0, contracts_y - contracts_h),
            7.6,
            contracts_h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            linewidth=1.2,
            edgecolor="#f87171",
            facecolor=COLORS["contracts"],
            zorder=5,
        )
    )
    ax.text(
        59.8,
        contracts_y - contracts_h / 2 + 3.0,
        "契约层\ncontracts.py",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT,
        fontweight="bold",
        rotation=0,
        zorder=6,
    )
    ax.text(
        59.8,
        contracts_y - contracts_h / 2 - 3.4,
        "Report / 报告块\nRenderer · CaseSource\nPlotter\nKinematicsReference",
        ha="center",
        va="center",
        fontsize=6.4,
        color=MUTED,
        zorder=6,
    )
    arrow(
        ax,
        54.8,
        contracts_y - contracts_h * 0.25,
        55.8,
        contracts_y - contracts_h * 0.25,
        color="#f87171",
        style="<|-|>",
    )
    arrow(
        ax,
        54.8,
        contracts_y - contracts_h * 0.75,
        55.8,
        contracts_y - contracts_h * 0.75,
        color="#f87171",
        style="<|-|>",
    )
    ax.text(
        59.8,
        contracts_y - contracts_h / 2 - 8.4,
        "禁止：用例 → 适配器\n（重构前 8 条，现 0 条）",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLORS["warn"],
        zorder=6,
    )


def draw_pipeline(ax) -> None:
    """右panel：抓取流水线。步骤之间标注传递的数据。"""
    x0, w = 64.0, 24.5
    ax.text(x0, 91, "抓取流水线 · rkin pick", fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(
        x0, 88.2, "一条命令跑完七道工序；每一步都能独立失败、独立验证", fontsize=8.5, color=MUTED
    )

    top, bottom = 86.5, 30.0
    step_h = 5.4
    pitch = (top - bottom) / len(PIPELINE)

    y = top
    for index, (name, duty, data) in enumerate(PIPELINE):
        box(
            ax,
            x0,
            y - step_h,
            w - 1.0,
            step_h,
            f"{name}\n{duty}",
            fill=COLORS["usecases"],
            fontsize=7.6,
        )
        ax.text(
            x0 + w - 1.6,
            y - step_h / 2,
            f"{index + 1}",
            ha="center",
            va="center",
            fontsize=11,
            color=MUTED,
            fontweight="bold",
        )
        if index < len(PIPELINE) - 1:
            arrow(ax, x0 + (w - 1) / 2, y - step_h - 0.3, x0 + (w - 1) / 2, y - pitch + 0.3, lw=1.2)
            label(ax, x0 + (w - 1) / 2 + 1.0, y - step_h - (pitch - step_h) / 2, data, fontsize=6.8)
        y -= pitch

    # 失败分支：右侧单独一列，和流水线并排（不压到左边的分层图）
    fx = 89.6
    ax.plot([fx, fx], [86.0, 32.0], color=COLORS["warn"], lw=1.2, linestyle="--", zorder=3)
    label(ax, fx, 87.6, "任何一步失败", fontsize=7.6, color=COLORS["warn"], ha="center", bold=True)
    for i, text in enumerate(FAILURES):
        yy = 83.0 - i * 8.6
        arrow(ax, fx - 0.5, yy, fx + 0.5, yy, color=COLORS["warn"], lw=1.0)
        label(ax, fx + 0.9, yy, text, fontsize=6.8, color=COLORS["warn"])
    label(
        ax,
        fx,
        30.6,
        "六类结论：有名字、有数字",
        fontsize=7.4,
        color=COLORS["warn"],
        ha="center",
        bold=True,
    )


def draw_config(ax) -> None:
    """下panel：关键配置与数字。"""
    ax.add_patch(
        FancyBboxPatch(
            (2, 2),
            96,
            18.5,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=COLORS["panel"],
            zorder=1,
        )
    )
    ax.text(
        3.4,
        18.4,
        "关键配置 · configs/default.yaml 的 pick 段",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )

    n = len(CONFIG_CARDS)
    card_w = (92 - (n - 1) * 1.6) / n
    for i, (title, value, key) in enumerate(CONFIG_CARDS):
        cx = 4.0 + i * (card_w + 1.6)
        box(ax, cx, 8.8, card_w, 7.6, "", fill="#fef9c3", fontsize=8)
        ax.text(
            cx + card_w / 2,
            14.6,
            title,
            ha="center",
            va="center",
            fontsize=7.4,
            color=MUTED,
            zorder=3,
        )
        ax.text(
            cx + card_w / 2,
            12.1,
            value,
            ha="center",
            va="center",
            fontsize=11.5,
            color=TEXT,
            fontweight="bold",
            zorder=3,
        )
        ax.text(
            cx + card_w / 2,
            10.0,
            key,
            ha="center",
            va="center",
            fontsize=6.4,
            color=MUTED,
            zorder=3,
        )

    ax.text(4.0, 5.2, KEY_NUMBERS, fontsize=7.2, color=MUTED, va="center", zorder=3)


def main() -> int:
    """画出三块内容并存成 SVG + PNG，返回进程退出码。"""
    configure_chinese_font()
    figure, ax = plt.subplots(figsize=(16.2, 11.6))
    figure.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        2, 98.6, "robot_kinematics_app · 系统架构", fontsize=16.5, fontweight="bold", color=TEXT
    )
    ax.text(
        2,
        95.0,
        "运动学交给 spatialmath + roboticstoolbox；本项目写的是「用它们搭一条会告诉你为什么失败的任务流水线」",
        fontsize=9.5,
        color=MUTED,
    )
    ax.text(
        98,
        98.6,
        "Python 3.10 · spatialmath + roboticstoolbox · 170 条断言",
        fontsize=9,
        color=MUTED,
        ha="right",
    )

    draw_layers(ax)
    draw_pipeline(ax)
    draw_config(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(OUT_DIR / f"kinematics-architecture.{suffix}", dpi=140, facecolor="white")
    plt.close(figure)
    print(f"已生成：{(OUT_DIR / 'kinematics-architecture.png').relative_to(ROOT)} / .svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
