"""@file plotting.py

@brief 出图：Panda 骨架、抓取任务图。

【为什么用 Agg 后端】这个项目经常在没有显示器的环境里跑（SSH、容器）。
Agg 是纯文件后端，只写 PNG，不尝试开窗口 —— 这样 `rkin --observe ... --plot` 在任何环境都能出图。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib.pyplot as plt

import numpy as np

from ..core import robots
from ..core.workspace import directional_reach, max_reach

CJK_CANDIDATES = (
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
    "SimHei",
    "Microsoft YaHei",
)

_font_name: str | None = None


def configure_chinese_font() -> str | None:
    """挑一个可用的中文字体并设进 rcParams，返回字体名（找不到返回 None）。"""
    global _font_name
    if _font_name is not None:
        return _font_name

    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for candidate in CJK_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            _font_name = candidate
            return _font_name

    plt.rcParams["axes.unicode_minus"] = False
    _font_name = ""
    return None


def _resolve(outputs_dir: str | Path) -> Path:
    directory = Path(outputs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_panda_skeleton(robot, q: np.ndarray, outputs_dir: str | Path) -> Path:
    """Panda 骨架图：环境支持就用 3D，不支持就退回三视图。"""
    configure_chinese_font()
    available, reason = _axes3d_available()
    if not available:
        print(
            f"提示: 本机 matplotlib 的 3D 模块不可用（{reason}），已改用三视图输出。",
            file=sys.stderr,
        )
        return _plot_panda_three_views(robot, q, outputs_dir)

    points = robots.link_points(robot, q)
    bound = _reach_bound(robot)

    figure = plt.figure(figsize=(7.2, 7.2))
    axes = figure.add_subplot(projection="3d")

    axes.plot(
        points[:, 0], points[:, 1], points[:, 2], "-o", lw=4, ms=7, color="#2563eb", label="关节链"
    )
    axes.plot(*points[0], "s", ms=12, color="#111827", label="base")
    axes.plot(*points[-1], "*", ms=18, color="#dc2626", label="工具端 (TCP)")
    _draw_reference_sphere(axes, bound)

    length = 0.25
    axes.quiver(0, 0, 0, length, 0, 0, color="#dc2626", arrow_length_ratio=0.15)
    axes.quiver(0, 0, 0, 0, length, 0, color="#16a34a", arrow_length_ratio=0.15)
    axes.quiver(0, 0, 0, 0, 0, length, color="#2563eb", arrow_length_ratio=0.15)

    tcp = points[-1]
    axes.set_title(f"Franka Panda 3D 骨架\nTCP = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_zlabel("z [m]")
    axes.set_xlim(-bound, bound)
    axes.set_ylim(-bound, bound)
    axes.set_zlim(0, bound)
    axes.legend(loc="upper left", fontsize=8)

    path = _resolve(outputs_dir) / "panda_fk_pose.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _plot_panda_three_views(robot, q: np.ndarray, outputs_dir: str | Path) -> Path:
    """正交三视图（俯视 XY / 主视 XZ / 侧视 YZ）。"""
    points = robots.link_points(robot, q)
    bound = _reach_bound(robot)
    tcp = points[-1]

    figure, panels = plt.subplots(1, 3, figsize=(14.0, 5.0), layout="constrained")
    for axes, (ix, iy, xlabel, ylabel) in zip(
        panels,
        ((0, 1, "x [m]", "y [m]"), (0, 2, "x [m]", "z [m]"), (1, 2, "y [m]", "z [m]")),
        strict=True,
    ):
        axes.plot(points[:, ix], points[:, iy], "-o", lw=3.5, ms=6, color="#2563eb")
        axes.plot(points[0, ix], points[0, iy], "s", ms=11, color="#111827", label="base")
        axes.plot(tcp[ix], tcp[iy], "*", ms=17, color="#dc2626", label="工具端 (TCP)")
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.25)
        axes.set_aspect("equal")
        axes.axhline(0.0, color="#d1d5db", lw=1)
        axes.axvline(0.0, color="#d1d5db", lw=1)
        span = (
            max(
                float(points[:, ix].max() - points[:, ix].min()),
                float(points[:, iy].max() - points[:, iy].min()),
                0.45 * bound,
            )
            * 1.25
        )
        cx = 0.5 * float(points[:, ix].max() + points[:, ix].min())
        cy = 0.5 * float(points[:, iy].max() + points[:, iy].min())
        axes.set_xlim(cx - span / 2, cx + span / 2)
        axes.set_ylim(cy - span / 2, cy + span / 2)

    panels[0].set_title("俯视图 / XY")
    panels[1].set_title("主视图 / XZ")
    panels[2].set_title("侧视图 / YZ")
    figure.suptitle(
        f"Franka Panda 骨架（三视图，工作空间上界 {bound:.3f} m）"
        f"　TCP = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m",
        fontsize=11,
    )
    panels[0].legend(loc="upper right", fontsize=8)

    path = _resolve(outputs_dir) / "panda_fk_pose.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _reach_bound(robot, samples: int = 4000) -> float:
    return max_reach(robot, samples=samples)


def _axes3d_available() -> tuple[bool, str]:
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _draw_reference_sphere(axes, radius: float, alpha: float = 0.08) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    axes.plot_surface(x, y, z, color="#60a5fa", alpha=alpha, linewidth=0)


class MatplotlibPlotter:
    """实现 contracts.Plotter 协议：把图存到构造时给定的目录。"""

    def __init__(self, outputs_dir: str | Path = "outputs") -> None:
        self.outputs_dir = outputs_dir

    def plot_panda_skeleton(self, robot, q) -> Path:
        return plot_panda_skeleton(robot, q, self.outputs_dir)

    def plot_pick_result(self, task, result, robot) -> Path:
        return plot_pick_result(task, result, robot, self.outputs_dir)


def plot_pick_result(task, result, robot, outputs_dir: str | Path = "outputs") -> Path:
    """抓取任务图：可达边界 / 机械臂姿态 / 末端路径 / 奇异点标记（Panda，x-z 主视）。"""
    configure_chinese_font()

    figure, axes = plt.subplots(figsize=(8.2, 7.4))
    _draw_reachable_slice(axes, robot)

    if result.q is not None:
        points = robots.link_points(robot, result.q)
        px, py = points[:, 0], points[:, 2]
        axes.plot(px, py, "-o", lw=4, ms=6, color="#2563eb", label="解出的姿态", zorder=4)
        axes.plot(px[0], py[0], "s", ms=12, color="#111827", label="base", zorder=5)
        axes.plot(px[-1], py[-1], "*", ms=20, color="#16a34a", label="末端（解）", zorder=6)
        _draw_motion_trace(axes, robot, result.q)

    target = np.asarray(result.target.t, dtype=float)
    mark = "x" if result.q is None else "+"
    axes.plot(
        target[0],
        target[2],
        mark,
        ms=16,
        mew=3,
        color="#dc2626",
        label=f"目标（{result.outcome}）",
        zorder=7,
    )

    axes.set_title(
        f"抓取任务：Panda　结论：{result.outcome}\n"
        f"目标 {np.round(result.target.t, 4)} m"
        + (f"　关节解 {np.round(result.q, 3)} rad" if result.q is not None else "")
    )
    axes.set_xlabel("x [m]")
    axes.set_ylabel("z [m]（主视）")
    axes.set_aspect("equal")
    axes.grid(alpha=0.25)
    axes.axhline(0.0, color="#d1d5db", lw=1)
    axes.axvline(0.0, color="#d1d5db", lw=1)
    axes.legend(loc="upper right", fontsize=8)

    path = _resolve(outputs_dir) / "pick_panda.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _draw_reachable_slice(axes, robot, *, directions: int = 48, samples: int = 3000):
    """x-z 平面内各方向的可达边界（采样估计）。"""
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    radii: list[float] = []
    for angle in angles:
        direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
        radii.append(directional_reach(robot, direction, samples=samples).radius)
    radii_array = np.asarray(radii)
    xs = radii_array * np.cos(angles)
    ys = radii_array * np.sin(angles)
    axes.plot(xs, ys, "--", lw=1.4, color="#9ca3af", label="可达边界（采样估计）", zorder=2)
    axes.fill(xs, ys, color="#e5e7eb", alpha=0.35, zorder=1)


def _draw_motion_trace(axes, robot, q_goal: np.ndarray, *, steps: int = 40):
    """从零位形到解位形的关节空间直线插值：末端轨迹 + 沿途的奇异程度。"""
    from ..core.singularity import analyze

    q_start = np.zeros_like(q_goal)
    weights = np.linspace(0.0, 1.0, steps)[:, None]
    qs = q_start + weights * (q_goal - q_start)
    points = robots.end_positions(robot, qs)
    xs, ys = points[:, 0], points[:, 2]

    sigma_min = np.array([analyze(robot.jacob0(q)).sigma_min for q in qs])
    threshold = max(float(sigma_min.max()) * 0.1, 1e-9)
    risky = sigma_min < threshold
    axes.plot(
        xs,
        ys,
        "-",
        lw=2,
        color="#f59e0b",
        alpha=0.9,
        label="末端路径（关节空间直线插值，仅示意）",
        zorder=3,
    )
    if risky.any():
        axes.plot(
            xs[risky],
            ys[risky],
            "o",
            ms=7,
            color="#dc2626",
            label=f"沿途接近奇异（{int(risky.sum())} 个位形）",
            zorder=4,
        )
