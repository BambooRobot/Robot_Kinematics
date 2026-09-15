"""@file main.py
@brief 程序入口：解析 → 组合 → 渲染 → 打印，以及唯一的异常出口。

【异常策略】core / usecases / adapters 遇到错误就抛 KinematicsError，不兜底也不打印；
这里作为唯一的顶层统一捕获，写 stderr 并返回非零退出码。
（走 stderr 而不是 stdout：这样 `rkin fk ... > out.txt` 重定向时，错误仍然显示在终端。）
"""

from __future__ import annotations

import sys

from ..adapters.config_yaml import Config
from ..adapters.render_json import JsonRenderer
from ..adapters.render_text import TextRenderer
from ..contracts import Renderer
from ..core.exceptions import KinematicsError
from .commands import dispatch
from .parser import build_parser

# 配置文件里的哪一项对应哪个命令行开关
CONFIG_OVERRIDES = {
    "l1": "planar_l1",
    "l2": "planar_l2",
    "dls_lambda": "ik_dls_lambda",
    "max_iter": "ik_max_iter",
    "tol": "ik_tol",
    "step": "ik_step",
    "det_eps": "singularity_det_eps",
    "cond_warn": "singularity_cond_warn",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(args)
        renderer = _make_renderer(args, config)
        output = dispatch(args, config)
    except KinematicsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130

    parts = []
    if output.report is not None:
        parts.append(renderer.render(output.report))
    parts.extend(output.extra_lines)
    print("\n".join(parts))
    return 0


def _make_renderer(args, config) -> Renderer:
    """渲染器按 --format 选。用例产出的 Report 是一样的，换渲染器只是换一种呈现。"""
    if args.format == "json":
        return JsonRenderer(outputs_dir=config.outputs_dir_path)
    return TextRenderer(outputs_dir=config.outputs_dir_path)


def _load_config(args) -> Config:
    config = Config.from_yaml(args.config)
    overrides = {
        target: getattr(args, source)
        for source, target in CONFIG_OVERRIDES.items()
        if getattr(args, source, None) is not None
    }
    return config.override(**overrides)
