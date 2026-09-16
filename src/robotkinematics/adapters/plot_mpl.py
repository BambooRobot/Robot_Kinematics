"""@file plotting.py
@brief 出图：二连杆 2D 姿态、Panda 3D 骨架、关节扫动 GIF。

【为什么用 Agg 后端】这个项目经常在没有显示器的环境里跑（SSH、容器）。
Agg 是纯文件后端，只写 PNG/GIF，不尝试开窗口 —— 这样 `rkin fk --plot` 在任何环境都能出图。

【中文字体】matplotlib 自带字体没有汉字，不设置的话图上全是方块。
这里按优先级找一个系统里的 CJK 字体（本机有 Noto Sans CJK）；找不到也不报错，
只是图上的中文会缺字，数值和几何图形不受影响。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ⚠️ 必须在 import pyplot 之前设置

with warnings.catch_warnings():
    # 某些环境里 matplotlib 装了两份，pyplot 导入时会喊"3D 模块不可用"。
    # 这条我们自己会在真正需要 3D 时给出更清楚的说明（见 _axes3d_available），
    # 所以在这里只屏蔽这一条固定的警告，别的一概放行。
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib.pyplot as plt

import numpy as np

from ..core import robots
from ..core.planar2r import Planar2R
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
            plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免缺字
            _font_name = candidate
            return _font_name

    plt.rcParams["axes.unicode_minus"] = False
    _font_name = ""
    return None


def _resolve(outputs_dir: str | Path) -> Path:
    directory = Path(outputs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_two_link(
    arm: Planar2R,
    q1_deg: float,
    q2_deg: float,
    outputs_dir: str | Path,
    target: tuple[float, float] | None = None,
) -> Path:
    """二连杆姿态图：两段连杆 + 可达圆环 + 末端位置。"""
    configure_chinese_font()
    q1, q2 = np.deg2rad([q1_deg, q2_deg])
    base, elbow, tool = arm.joint_points(q1, q2)

    figure, axes = plt.subplots(figsize=(6.2, 6.2))
    _draw_workspace_circles(axes, arm)
    axes.plot(
        [base[0], elbow[0]],
        [base[1], elbow[1]],
        "-o",
        lw=5,
        color="#2563eb",
        label=f"link1 L1={arm.l1}",
    )
    axes.plot(
        [elbow[0], tool[0]],
        [elbow[1], tool[1]],
        "-o",
        lw=5,
        color="#16a34a",
        label=f"link2 L2={arm.l2}",
    )
    axes.plot(*tool, "o", ms=11, color="#dc2626", label="末端 tool")
    if target is not None:
        axes.plot(*target, "x", ms=13, mew=3, color="#9333ea", label="目标点")

    axes.set_title(
        f"二连杆  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°\n"
        f"末端 ({tool[0]:.4f}, {tool[1]:.4f})   det(J)={arm.det_jacobian(q2):.4f}"
    )
    _finish_2d_axes(axes, arm)
    path = _resolve(outputs_dir) / f"fk_q1_{q1_deg:g}_q2_{q2_deg:g}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def animate_two_link(
    arm: Planar2R,
    sweep: str = "q2",
    frames: int = 60,
    q1_deg: float = 0.0,
    q2_deg: float = 0.0,
    outputs_dir: str | Path = "outputs",
) -> Path:
    """让一个关节从 0° 扫到 180°，输出 GIF —— 看姿态如何连续变化。"""
    configure_chinese_font()
    figure, axes = plt.subplots(figsize=(5.6, 5.6))

    def draw(index: int) -> tuple:
        angle = 180.0 * index / max(frames - 1, 1)
        q1 = np.deg2rad(angle if sweep == "q1" else q1_deg)
        q2 = np.deg2rad(angle if sweep == "q2" else q2_deg)
        base, elbow, tool = arm.joint_points(q1, q2)

        axes.clear()
        _draw_workspace_circles(axes, arm)
        link1 = axes.plot([base[0], elbow[0]], [base[1], elbow[1]], "-o", lw=5, color="#2563eb")
        link2 = axes.plot([elbow[0], tool[0]], [elbow[1], tool[1]], "-o", lw=5, color="#16a34a")
        end = axes.plot(*tool, "o", ms=11, color="#dc2626")
        axes.set_title(f"扫动 {sweep}：{angle:.0f}°\n末端 ({tool[0]:.3f}, {tool[1]:.3f})")
        _finish_2d_axes(axes, arm)
        figure.tight_layout()
        return (*link1, *link2, *end)

    from matplotlib.animation import FuncAnimation, PillowWriter

    animation = FuncAnimation(figure, draw, frames=frames, interval=60)
    path = _resolve(outputs_dir) / f"two_link_sweep_{sweep}.gif"
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(figure)
    return path


def plot_panda_skeleton(robot, q: np.ndarray, outputs_dir: str | Path) -> Path:
    """Panda 骨架图：环境支持就用 3D，不支持就退回三视图（都是同一份数据）。"""
    configure_chinese_font()
    available, reason = _axes3d_available()
    if not available:
        print(
            f"提示: 本机 matplotlib 的 3D 模块不可用（{reason}），已改用三视图输出。\n"
            "      常见原因是系统里同时装了新旧两个 matplotlib，"
            "老的 nspkg.pth 让 mpl_toolkits 指到了老版本目录。\n"
            "      修法：删掉 /usr/lib/python3/dist-packages/matplotlib-*-nspkg.pth"
            "（或卸载系统的 python3-matplotlib），或改用虚拟环境。",
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

    # 工作空间上界：末端不可能超出所有连杆长度之和（三角不等式）
    _draw_reference_sphere(axes, bound)

    length = 0.25
    axes.quiver(0, 0, 0, length, 0, 0, color="#dc2626", arrow_length_ratio=0.15)
    axes.quiver(0, 0, 0, 0, length, 0, color="#16a34a", arrow_length_ratio=0.15)
    axes.quiver(0, 0, 0, 0, 0, length, color="#2563eb", arrow_length_ratio=0.15)
    axes.text(length, 0, 0, " x", color="#dc2626")
    axes.text(0, length, 0, " y", color="#16a34a")
    axes.text(0, 0, length, " z", color="#2563eb")

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
    """正交三视图（俯视 XY / 主视 XZ / 侧视 YZ）—— 机械图纸的读法，不依赖 3D 模块。

    ⚠️ 每个视图都保持等比例（否则长度会骗人），但视野按各自的数据范围单独取。
    如果所有视图都用统一的大范围，手臂接近平面的姿态会缩成一条线，什么也看不清。
    """
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
        # 以数据范围为准取正方形视野；范围过小时给一个下限，避免“一条线占满整幅图”
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


# ── 内部绘图工具 ────────────────────────────────────────────────────────────


def _draw_workspace_circles(axes, arm: Planar2R) -> None:
    """画出可达范围（外圈 L1+L2）与内圈（|L1-L2|）。"""
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for radius, style, label in (
        (arm.l1 + arm.l2, "--", f"外边界 L1+L2={arm.l1 + arm.l2:g}"),
        (abs(arm.l1 - arm.l2), ":", f"内边界 |L1-L2|={abs(arm.l1 - arm.l2):g}"),
    ):
        if radius <= 1e-12:
            continue
        axes.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            style,
            color="#9ca3af",
            lw=1.2,
            label=label,
        )


def _finish_2d_axes(axes, arm: Planar2R) -> None:
    limit = arm.l1 + arm.l2 + 0.3
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.grid(alpha=0.25)
    axes.axhline(0.0, color="#d1d5db", lw=1)
    axes.axvline(0.0, color="#d1d5db", lw=1)
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.legend(loc="upper right", fontsize=8)


def _reach_bound(robot, samples: int = 4000) -> float:
    """末端离原点的距离上界 —— 采样估计（库的 `robot.reach` 实测返回 0，用不了）。"""
    return max_reach(robot, samples=samples)


def _axes3d_available() -> tuple[bool, str]:
    """探测 mpl_toolkits.mplot3d 能不能用。

    单独探测的原因：某些环境里 matplotlib 装了两份（系统一份、pip 一份），
    老的 nspkg.pth 会把 mpl_toolkits 指向老版本目录，导入时才会炸。
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _draw_reference_sphere(axes, radius: float, alpha: float = 0.08) -> None:
    """半透明的参考球，表示末端不可能超出的距离上界。"""
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    axes.plot_surface(x, y, z, color="#60a5fa", alpha=alpha, linewidth=0)


class MatplotlibPlotter:
    """实现 contracts.Plotter 协议：把图存到构造时给定的目录。

    ⚠️ 这个类与两个绘图函数一样，是唯一碰 matplotlib 的地方；用例和 core 都不认识它。
    """

    def __init__(self, outputs_dir: str | Path = "outputs") -> None:
        self.outputs_dir = outputs_dir

    def plot_two_link(self, arm, q1_deg: float, q2_deg: float, target=None) -> Path:
        return plot_two_link(arm, q1_deg, q2_deg, self.outputs_dir, target=target)

    def plot_panda_skeleton(self, robot, q) -> Path:
        return plot_panda_skeleton(robot, q, self.outputs_dir)

    def plot_pick_result(self, task, result, robot) -> Path:
        return plot_pick_result(task, result, robot, self.outputs_dir)

    def animate_two_link(self, arm, sweep: str, frames: int, q1_deg: float, q2_deg: float) -> Path:
        return animate_two_link(
            arm,
            sweep=sweep,
            frames=frames,
            q1_deg=q1_deg,
            q2_deg=q2_deg,
            outputs_dir=self.outputs_dir,
        )


def plot_pick_result(task, result, robot, outputs_dir: str | Path = "outputs") -> Path:
    """抓取任务图：一张图里放四样东西（愿景里的"四个要素"）。

      a. 工作空间切片 —— 该平面内各方向的可达边界（采样估计）
      b. 机械臂姿态   —— 解出的那组关节角
      c. 末端路径     —— 从初始位形到解位形的关节空间直线插值（**仅示意，不是规划**）
      d. 奇异点标记   —— 沿途每个位形的最小奇异值，低于阈值就标红

    ⚠️ 路径是"关节空间直线插值"，只用来显示这次运动会经过哪些位形，
       它既不是避障轨迹也不是时间最优轨迹 —— 图上会如实标注。
    """
    configure_chinese_font()
    horizontal = task.arm == "planar"  # 二连杆在 x-y 平面里动

    figure, axes = plt.subplots(figsize=(8.2, 7.4))
    _draw_reachable_slice(axes, robot, horizontal=horizontal)

    # b. 机械臂姿态
    if result.q is not None:
        points = robots.link_points(robot, result.q)
        px, py = _project(points, horizontal)
        axes.plot(px, py, "-o", lw=4, ms=6, color="#2563eb", label="解出的姿态", zorder=4)
        axes.plot(px[0], py[0], "s", ms=12, color="#111827", label="base", zorder=5)
        axes.plot(px[-1], py[-1], "*", ms=20, color="#16a34a", label="末端（解）", zorder=6)

        # c. 末端路径 + d. 奇异点标记
        _draw_motion_trace(axes, robot, result.q, horizontal=horizontal)

    # 目标点
    tx, ty = _project(np.asarray(result.target.t).reshape(1, 3), horizontal)
    mark = "x" if result.q is None else "+"
    axes.plot(
        tx, ty, mark, ms=16, mew=3, color="#dc2626", label=f"目标（{result.outcome}）", zorder=7
    )

    axes.set_title(
        f"抓取任务：{task.arm}　结论：{result.outcome}\n"
        f"目标 {np.round(result.target.t, 4)} m"
        + (f"　关节解 {np.round(result.q, 3)} rad" if result.q is not None else "")
    )
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]（俯视）" if horizontal else "z [m]（主视）")
    axes.set_aspect("equal")
    axes.grid(alpha=0.25)
    axes.axhline(0.0, color="#d1d5db", lw=1)
    axes.axvline(0.0, color="#d1d5db", lw=1)
    axes.legend(loc="upper right", fontsize=8)

    path = _resolve(outputs_dir) / f"pick_{task.arm}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _project(points: np.ndarray, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    """投影到绘图平面：二连杆用 x-y，Panda 用 x-z。"""
    arr = np.asarray(points, dtype=float)
    return (arr[:, 0], arr[:, 1]) if horizontal else (arr[:, 0], arr[:, 2])


def _draw_reachable_slice(
    axes, robot, *, horizontal: bool, directions: int = 48, samples: int = 3000
):
    """该平面内各方向的可达边界（采样估计）—— 用来说明"工作空间不是球"。"""
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    radii: list[float] = []
    for angle in angles:
        direction = (
            np.array([np.cos(angle), 0.0, np.sin(angle)])
            if not horizontal
            else np.array([np.cos(angle), np.sin(angle), 0.0])
        )
        radii.append(directional_reach(robot, direction, samples=samples).radius)
    radii_array = np.asarray(radii)
    xs = radii_array * np.cos(angles)
    ys = radii_array * np.sin(angles)
    axes.plot(xs, ys, "--", lw=1.4, color="#9ca3af", label="可达边界（采样估计）", zorder=2)
    axes.fill(xs, ys, color="#e5e7eb", alpha=0.35, zorder=1)


def _draw_motion_trace(axes, robot, q_goal: np.ndarray, *, horizontal: bool, steps: int = 40):
    """从零位形到解位形的关节空间直线插值：末端轨迹 + 沿途的奇异程度。"""
    from ..core.singularity import analyze

    q_start = np.zeros_like(q_goal)
    weights = np.linspace(0.0, 1.0, steps)[:, None]
    qs = q_start + weights * (q_goal - q_start)
    points = robots.end_positions(robot, qs)
    xs, ys = _project(points, horizontal)

    sigma_min = np.array([analyze(robot.jacob0(q)).sigma_min for q in qs])
    # 用颜色区分"沿途有没有接近奇异"：正常点灰、接近奇异的点红
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
