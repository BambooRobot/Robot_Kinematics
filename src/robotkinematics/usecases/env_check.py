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
    "robotkinematics.core.robots",
    "robotkinematics.core.planar2r",
    "robotkinematics.core.workspace",
    "robotkinematics.core.singularity",
    "robotkinematics.core.exceptions",
    "robotkinematics.contracts",
    "robotkinematics.usecases.pose",
    "robotkinematics.usecases.transforms",
    "robotkinematics.usecases.planar",
    "robotkinematics.usecases.panda",
    "robotkinematics.usecases.pick",
    "robotkinematics.adapters.config_yaml",
    "robotkinematics.adapters.cases_csv",
    "robotkinematics.adapters.render_text",
]

# 运动学依赖：现在是**必需**的（本项目把数学交给了库）
KINEMATICS_MODULES = [
    ("spatialmath", "spatialmath-python"),
    ("roboticstoolbox", "roboticstoolbox-python"),
]


def run() -> Report:
    """@brief 逐项体检当前环境，把结论写成一份可读的自检报告。

    检查顺序是"越靠前越致命"：Python 版本 → numpy → 本项目自己的模块 → 运动学库。
    前两项不合格后面的命令都跑不起来，所以它们排在前面报出来。

    @return 报告；fields["ok"] 汇总是否全部通过，broken_modules 列出导入失败的模块
    """
    lines = [
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        "说明：运动学（FK / IK / 雅可比 / 限位）由 spatialmath + roboticstoolbox 提供；",
        "     本项目自己写的是「用它们搭一条任务流水线」（可达性、择优、失败分类）。",
    ]
    ok = True

    lines.append(f"[OK] numpy: {np.__version__}")
    if int(np.__version__.split(".")[0]) >= 2:
        lines.append(
            "[WARN] 当前 NumPy >= 2；roboticstoolbox 在 numpy 2 下会报 _ARRAY_API not found，"
            '请执行：pip install "numpy<2" --force-reinstall'
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

    kinematics_available = []
    for module_name, package in KINEMATICS_MODULES:
        try:
            module = importlib.import_module(module_name)
            lines.append(f"[OK] {package}: {getattr(module, '__version__', '未知')}")
            kinematics_available.append(package)
        except ImportError as exc:
            ok = False
            lines.append(f"[MISSING] {package}: {exc}")
            lines.append('         请执行：pip install -e .  （或 pip install -e ".[dev]"）')

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
            "kinematics_available": kinematics_available,
        },
    )
