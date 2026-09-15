"""@file env_check.py
@brief 环境自检 —— 对应课件 00_check_env.py，但检查的是本项目自己需要的东西。

课件的自检在确认 spatialmath / roboticstoolbox 装没装；本项目把那两个库变成了
**可选项**（只有交叉验证用得到），所以自检的重点变成：numpy 在不在、自己的模块能不能全部导入。
"""

from __future__ import annotations

import importlib
import platform
import sys

import numpy as np

from ..contracts import Report, TextBlock

# 本项目自己的一整套模块：任何一个导入失败，后面所有命令都会失败
SELF_MODULES = [
    "robotkinematics.core.rotations",
    "robotkinematics.core.se3",
    "robotkinematics.core.planar2r",
    "robotkinematics.core.chain",
    "robotkinematics.core.panda",
    "robotkinematics.core.ik_solvers",
    "robotkinematics.core.singularity",
    "robotkinematics.core.numerics",
    "robotkinematics.contracts",
    "robotkinematics.usecases.pose",
    "robotkinematics.usecases.planar",
    "robotkinematics.usecases.panda",
    "robotkinematics.adapters.config_yaml",
    "robotkinematics.adapters.cases_csv",
    "robotkinematics.adapters.render_text",
]

# 只有交叉验证需要的库
OPTIONAL_MODULES = [
    ("spatialmath", "spatialmath-python"),
    ("roboticstoolbox", "roboticstoolbox-python"),
]


def run() -> Report:
    lines = [
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        "说明：本项目的核心数学全部自研，不依赖 spatialmath / roboticstoolbox；",
        "     那两个库只用于 `rkin crosscheck` 的交叉验证，没有也能跑全部功能。",
    ]
    ok = True

    lines.append(f"[OK] numpy: {np.__version__}")
    if int(np.__version__.split(".")[0]) >= 2:
        lines.append(
            "[WARN] 当前 NumPy >= 2；本项目自己不受影响，但若要跑 `rkin crosscheck`，"
            '请先执行：pip install "numpy<2" --force-reinstall'
        )

    modules_ok = {}
    for module_name in ("matplotlib", "yaml"):
        try:
            module = importlib.import_module(module_name)
            lines.append(f"[OK] {module_name}: {getattr(module, '__version__', '未知')}")
            modules_ok[module_name] = True
        except ImportError as exc:
            ok = False
            modules_ok[module_name] = False
            lines.append(f"[MISSING] {module_name}: {exc}")
            lines.append("         请执行：pip install -e .")

    self_broken = []
    for module_name in SELF_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            ok = False
            self_broken.append(module_name)
            lines.append(f"[BROKEN] {module_name}: {type(exc).__name__}: {exc}")
    if not self_broken:
        lines.append(f"[OK] 本项目 {len(SELF_MODULES)} 个模块全部可导入")

    optional_available = []
    for module_name, package in OPTIONAL_MODULES:
        try:
            module = importlib.import_module(module_name)
            lines.append(
                f"[OK] {package}: {getattr(module, '__version__', '未知')}（crosscheck 可用）"
            )
            optional_available.append(package)
        except ImportError:
            lines.append(f"[--] {package}: 未安装（crosscheck 会跳过，不影响其他功能）")

    lines.append("")
    lines.append("环境检查完成，可以开始实战。" if ok else "有依赖缺失，请先执行：pip install -e .")

    return Report(
        title="环境自检",
        blocks=(TextBlock(tuple(lines)),),
        fields={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "ok": ok,
            "broken_modules": self_broken,
            "optional_available": optional_available,
        },
    )
