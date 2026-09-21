"""@file test_architecture.py

@brief 架构守卫：把"依赖只向下""装配只在入口"这两条规矩变成会红的测试。
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest

from robotkinematics.main import CONFIG_OVERRIDES, build_parser

SRC = Path(__file__).resolve().parents[1] / "src" / "robotkinematics"
LAYERS = {"core", "usecases", "adapters", "contracts", "main"}
ENTRY = SRC / "main.py"


def _source_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)


def _layer(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else path.stem


def _imported_layers(path: Path) -> set[str]:
    package = list(path.relative_to(SRC).parent.parts)
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
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


ASSEMBLY_CALLS = {"CsvCaseSource", "MatplotlibPlotter", "TextRenderer", "JsonRenderer", "from_yaml"}


def test_only_the_entry_constructs_concrete_adapters():
    """除入口外，谁都不许造具体适配器 —— 想换实现只该改 `main.build_context` 一处。"""
    offenders = {
        path.relative_to(SRC).as_posix(): _called_names(path) & ASSEMBLY_CALLS
        for path in _source_files(SRC)
        if path != ENTRY and _layer(path) != "adapters" and _called_names(path) & ASSEMBLY_CALLS
    }
    assert not offenders, f"这些地方绕过了入口自己造适配器：{offenders}"


def test_config_overrides_keys_are_real_cli_flags():
    """`CONFIG_OVERRIDES` 的每个键必须是解析器真实存在的参数 —— 防"死映射"。"""
    dests = {action.dest for action in build_parser()._actions}  # noqa: SLF001
    dead = {flag for flag in CONFIG_OVERRIDES if flag not in dests}
    assert not dead, f"CONFIG_OVERRIDES 死键（解析器里没有这个开关）：{sorted(dead)}"


def test_only_the_context_builds_robots():
    """机器人构造只出现在 `Context` 的工厂里 —— 处理函数一律走 `ctx.make_*`。"""
    tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
    factory_calls = {"panda"}

    def robot_calls(node: ast.AST) -> set[str]:
        found: set[str] = set()
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "robots"
                and call.func.attr in factory_calls
            ):
                found.add(call.func.attr)
        return found

    offenders: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("make_"):
                continue  # Context 工厂方法允许
            if bad := robot_calls(node):
                # run_pick / build_context 也不该直接造；但 make_panda 是方法
                offenders[node.name] = bad
    # Context 的方法在 ClassDef 内，上面只扫模块级函数 —— 再扫一遍类内方法名
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Context":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # 工厂内调用 robots.panda 是允许的
                    pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in {"run_pick", "build_context", "main", "build_parser"}:
                if bad := robot_calls(node):
                    offenders[node.name] = bad
    assert not offenders, f"绕过 Context 工厂直接造机器人的函数：{offenders}"


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [
        ("core", {"usecases", "adapters", "main", "contracts"}),
        ("usecases", {"adapters", "main"}),
        ("adapters", {"main"}),
    ],
)
def test_dependencies_only_point_downwards(layer, forbidden):
    """core 是纯数学；usecases 不认识适配器；适配器不认识入口。"""
    violations = {}
    for path in _source_files(SRC / layer):
        bad = _imported_layers(path) & forbidden
        if bad:
            violations[path.relative_to(SRC).as_posix()] = sorted(bad)
    assert not violations, f"{layer} 层越界依赖了 {forbidden}：{violations}"


def test_contracts_depends_on_no_layer():
    """契约层是"大家都依赖它、它不依赖任何人"。"""
    assert not _imported_layers(SRC / "contracts.py") & LAYERS


def test_cli_has_no_subcommands():
    """入口是扁平 CLI：解析器里不得再有子命令表。"""
    for action in build_parser()._actions:
        assert not isinstance(action, argparse._SubParsersAction), "不应再有子命令"
    dests = {action.dest for action in build_parser()._actions}
    assert "observe" in dests


def test_pipeline_is_the_single_source_of_truth_for_the_steps():
    """工序序列是数据：执行单元、命名、签名三条都要对得上。"""
    from robotkinematics.usecases import pick

    assert len(pick.PIPELINE) == 6
    for step in pick.PIPELINE:
        assert step.__name__.startswith("step_")
        assert len(inspect.signature(step).parameters) == 4, step.__name__
