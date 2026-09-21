"""@file main.py

@brief **唯一入口 = 组合根**：Panda 抓取任务流水线；这个程序能干什么、用到哪些实现，都写在这一个文件里。

【三段结构，从上往下读就是它的骨架】

  ① 装配（`Context` / `build_context`）—— **全工程唯一一处"选实现"的地方**：
     配置、渲染器、绘图器都在这一个函数里被选中并注入。
     除了这个文件，任何地方都不该出现"造一个具体实现"。
  ② 入口流程（`main`）—— 解析 → 装配 → 跑流水线 → 渲染，以及唯一的异常出口。
  ③ 参数定义（`build_parser`）—— 扁平 CLI：`rkin --observe X Y Z [...]`，无子命令。

【异常策略】core / usecases / adapters 遇错就抛 `KinematicsError`，不兜底也不打印；
这里作为唯一的顶层统一捕获，写 stderr 并返回非零退出码（走 stderr 而不是 stdout，
这样 `rkin --observe ... > out.txt` 重定向时错误仍显示在终端）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .adapters.config_yaml import Config
from .adapters.render_json import JsonRenderer
from .adapters.render_text import TextRenderer
from .contracts import Renderer, Report
from .core import robots
from .core.exceptions import KinematicsError
from .usecases import pick as pick_uc

# 配置文件里的哪一项对应哪个命令行开关（命令行覆盖优先）
CONFIG_OVERRIDES = {
    "max_iter": "ik_max_iter",
    "tol": "ik_tol",
}


@dataclass(frozen=True)
class CommandOutput:
    """流水线产出：一份报告（可能没有）+ 若干附加行（比如"图已保存到哪"）。"""

    report: Report | None = None
    extra_lines: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Context:
    """装配的产物：这一次运行所依赖的全部具体实现。

    处理逻辑只接收它、不自己造东西 —— 想换实现（比如换成 JSON 渲染器），
    只改 `build_context` 一处。
    """

    config: Config
    renderer: Renderer

    def plotter(self):
        """出图器：**用到时才 import matplotlib** —— 不带 `--plot` 时不该为它付出启动代价。"""
        from .adapters.plot_mpl import MatplotlibPlotter

        return MatplotlibPlotter(outputs_dir=self.config.outputs_dir_path)

    def make_panda(self, frame: str = robots.TCP):
        """Panda 模型的装配点：全工程只有这里造它。抓取末端固定为夹爪 TCP。"""
        return robots.panda(frame)


def build_context(args: argparse.Namespace) -> Context:
    """@brief 读配置、按命令行覆盖、把这次运行要用的实现全部造好。"""
    config = Config.from_yaml(args.config).override(
        **{
            target: getattr(args, source)
            for source, target in CONFIG_OVERRIDES.items()
            if getattr(args, source, None) is not None
        }
    )
    renderer: Renderer = (
        JsonRenderer(outputs_dir=config.outputs_dir_path)
        if args.format == "json"
        else TextRenderer(outputs_dir=config.outputs_dir_path)
    )
    return Context(config=config, renderer=renderer)


def run_pick(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """抓取流水线（七道工序 + 失败分类）→ `usecases.pick.pick / to_report`。"""
    prefer = (
        np.asarray(args.prefer_config, dtype=float) if args.prefer_config is not None else None
    )
    robot = ctx.make_panda(robots.TCP)
    task = pick_uc.PickTask(
        observation=np.asarray(args.observe, dtype=float),
        camera_xyz=(ctx.config.camera.x, ctx.config.camera.y, ctx.config.camera.z),
        camera_yaw_deg=ctx.config.camera.yaw_deg,
        orientation_rpy_deg=tuple(args.rpy),
        mode=args.mode,
        prefer_config=prefer,
    )
    params = pick_uc.PickParams(
        seeds=args.seeds or ctx.config.pick.seeds,
        residual_tol=ctx.config.pick.residual_tol,
        unreachable_residual=ctx.config.pick.unreachable_residual,
        min_sigma=ctx.config.pick.min_sigma,
        workspace_samples=ctx.config.pick.workspace_samples,
        workspace_seed=ctx.config.pick.workspace_seed,
        limit_margin_deg=ctx.config.pick.limit_margin_deg,
        max_iter=ctx.config.ik.max_iter,
        ik_tol=ctx.config.ik.tol,
    )
    result = pick_uc.pick(robot, task, params)
    report = pick_uc.to_report(task, result)

    extra: list[str] = []
    if args.plot:
        path = ctx.plotter().plot_pick_result(task, result, robot)
        extra.append(f"任务图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def main(argv: list[str] | None = None) -> int:
    """@brief 程序入口：解析 → 装配 → 跑流水线 → 渲染 → 打印。

    @param argv 命令行参数；None 表示取 sys.argv
    @return 进程退出码：0 成功、1 领域错误、130 被 Ctrl-C 中断
    """
    args = build_parser().parse_args(argv)
    try:
        context = build_context(args)
        output = run_pick(args, context)

        parts = []
        if output.report is not None:
            parts.append(context.renderer.render(output.report))
        parts.extend(output.extra_lines)
        print("\n".join(parts))
    except KinematicsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误: 文件读写失败: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    return 0


def build_parser() -> argparse.ArgumentParser:
    """@brief 扁平 CLI：无子命令，入口即抓取流水线。"""
    parser = argparse.ArgumentParser(
        prog="rkin",
        description="Panda 抓取任务流水线：七道工序把机器人运动学知识点串成一条链"
        "（运动学由 spatialmath + roboticstoolbox 提供）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认 configs/default.yaml）。命令行参数优先级高于配置文件",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="输出格式（默认 text）"
    )
    parser.add_argument("--version", action="version", version=f"rkin {__version__}")
    parser.add_argument(
        "--max-iter", type=int, dest="max_iter", default=None, help="覆盖 IK 最大迭代次数"
    )
    parser.add_argument("--tol", type=float, default=None, help="覆盖 IK 收敛判据")
    parser.add_argument(
        "--observe",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="观测到的目标点（默认在相机坐标系下，见 --mode）",
    )
    parser.add_argument(
        "--mode",
        choices=["camera", "base"],
        default="camera",
        help="观测所在的坐标系：camera（默认）会先做变换链，base 表示已经是本体坐标",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=[180.0, 0.0, 0.0],
        metavar=("R", "P", "Y"),
        help="抓取姿态（度），默认 180/0/0 = 工具 z 朝下（自上而下抓）",
    )
    parser.add_argument(
        "--prefer-config",
        nargs=7,
        type=float,
        default=None,
        dest="prefer_config",
        metavar="Q",
        help="次要任务：多解时优先靠近这个参考姿态",
    )
    parser.add_argument("--seeds", type=int, default=None, help="覆盖多初值个数")
    parser.add_argument("--plot", action="store_true", help="额外输出一张任务图")
    return parser
