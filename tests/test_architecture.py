"""@file test_architecture.py

@brief 架构守卫：把"依赖只向下""装配只在入口"这两条规矩变成会红的测试。

【为什么要测架构】分层图写在 README 里不会自己生效：过几个月随手加一行
`from ..adapters.render_text import …`，分层就已经名存实亡了，而且**没有任何测试会变红**。
这里用最笨也最可靠的办法 —— 读源码的 import 语句与实际调用 —— 把三条硬规矩钉住：

  ① 装配只在入口：除 `main.py` 外，源码里不出现"造具体适配器"的调用；
  ② 依赖只向下：core 不认识 usecases/adapters；usecases 不认识 adapters；
  ③ 命令表与 argparse 同步：`command_table()` 与子命令集合必须一模一样。

这些测试读的是源码文本（AST），不 import 被检查的模块 —— 所以它们守的是"结构"，与运行无关。
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest

from robotkinematics.main import build_parser, command_table

SRC = Path(__file__).resolve().parents[1] / "src" / "robotkinematics"
LAYERS = {"core", "usecases", "adapters", "contracts", "main"}
ENTRY = SRC / "main.py"


def _source_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)


def _layer(path: Path) -> str:
    """这个文件属于哪一层（顶层模块直接用文件名）。"""
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else path.stem


def _imported_layers(path: Path) -> set[str]:
    """这个文件引用了本项目哪些层（相对导入已解析成绝对层名）。"""
    package = list(path.relative_to(SRC).parent.parts)  # main.py → []
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.split(".")[0] in LAYERS:
                    targets.add(node.module.split(".")[0])
                continue
            base = package[: len(package) - node.level + 1]
            if node.module:
                base = base + node.module.split(".")
            if base and base[0] in LAYERS:
                targets.add(base[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in LAYERS:
                    targets.add(alias.name.split(".")[0])
    return targets


def _called_names(path: Path) -> set[str]:
    """这个文件里调用的函数/构造器名（`X(...)` 取 X，`a.b(...)` 取 b）。"""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


# ── ① 装配只在入口 ──────────────────────────────────────────────────────────

#: "造一个具体适配器"的调用 —— 只允许出现在入口（main.py）与适配器包内部
ASSEMBLY_CALLS = {"CsvCaseSource", "MatplotlibPlotter", "TextRenderer", "JsonRenderer", "from_yaml"}


def test_only_the_entry_constructs_concrete_adapters():
    """除入口外，谁都不许造具体适配器 —— 想换实现只该改 `main.build_context` 一处。"""
    offenders = {
        path.relative_to(SRC).as_posix(): _called_names(path) & ASSEMBLY_CALLS
        for path in _source_files(SRC)
        if path != ENTRY and _layer(path) != "adapters" and _called_names(path) & ASSEMBLY_CALLS
    }
    assert not offenders, f"这些地方绕过了入口自己造适配器：{offenders}"


# ── ② 依赖只向下 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [
        ("core", {"usecases", "adapters", "main", "contracts"}),
        ("usecases", {"adapters", "main"}),
        ("adapters", {"main"}),
    ],
)
def test_dependencies_only_point_downwards(layer, forbidden):
    """core 是纯数学（连契约都不认识）；usecases 不认识适配器；适配器不认识入口。"""
    violations = {}
    for path in _source_files(SRC / layer):
        bad = _imported_layers(path) & forbidden
        if bad:
            violations[path.relative_to(SRC).as_posix()] = sorted(bad)
    assert not violations, f"{layer} 层越界依赖了 {forbidden}：{violations}"


def test_contracts_depends_on_no_layer():
    """契约层是"大家都依赖它、它不依赖任何人" —— 一旦它 import 了某层，分层就成环了。"""
    assert not _imported_layers(SRC / "contracts.py") & LAYERS


# ── ③ 命令表与 argparse 同步 ────────────────────────────────────────────────


def _subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    """枚举子命令名。

    argparse 没有公开 API 做这件事，只能读它的 action 表；读不到就直接报错
    （宁可测试炸掉，也不要静默通过一条什么都验不出来的断言）。
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("解析器里找不到子命令表")


def test_command_table_and_argparse_agree():
    """`rkin --help` 能敲的命令，和命令表里的处理函数，必须一模一样。"""
    table = command_table()
    declared = _subcommand_names(build_parser())
    assert set(table) == declared, {
        "表里有但命令行没有": sorted(set(table) - declared),
        "命令行有但表里没有": sorted(declared - set(table)),
    }
    assert all(callable(handler) for handler in table.values())


def test_pipeline_is_the_single_source_of_truth_for_the_steps():
    """工序序列是数据：执行单元、命名、签名三条都要对得上。

    ⚠️ 一个容易踩的差别：**"七道工序"是报告口径**（7 个 `PickStep`），
       **执行口径只有 6 个单元** —— ④雅可比体检与⑤限位校验共用一次遍历
       （同一份雅可比算"好不好动"和"真机能不能摆出来"两件事），所以它们是同一个
       `step_inspect`。这个测试钉住执行口径；"报告里确实有七道"由
       `tests/usecases/test_pick.py::test_all_seven_steps_are_reported` 钉住。
    """
    from robotkinematics.usecases import pick

    assert len(pick.PIPELINE) == 6
    for step in pick.PIPELINE:
        assert step.__name__.startswith("step_")
        # 签名统一为 (robot, task, params, state) 才能插进这条序列
        assert len(inspect.signature(step).parameters) == 4, step.__name__
