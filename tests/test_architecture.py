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

#: 造机器人的工厂（core.robots 里的函数名）
ROBOT_FACTORIES = {"panda", "planar"}


def _parser_actions() -> list[argparse.Action]:
    """解析器的参数表。

    argparse 没有公开 API 能枚举参数，只能读 `_actions` —— 所以这个"越界访问"收在这一个
    函数里、只解释一次，而不是在三处测试里各写一遍（读不到就直接炸，不会静默通过）。
    """
    return list(build_parser()._actions)


def _robot_factory_scopes(path: Path) -> dict[str, set[str]]:
    """返回 {函数/方法的限定名: 它**直接**调用的机器人工厂名}。

    ⚠️ 这里不能用 `ast.walk`：要回答的是"这个调用属于哪个函数"，
      而 walk 丢掉了父链 —— 于是 `Context.make_panda` 里的调用会被算到模块头上，
       分不清是"工厂方法造的"还是"随手造的"。所以自己递归、自己带作用域名。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{scope}.{child.name}" if scope else child.name
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "robots"
                and child.func.attr in ROBOT_FACTORIES
            ):
                found.setdefault(scope, set()).add(child.func.attr)
            visit(child, name)

    visit(tree, "")
    return found


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
    dests = {action.dest for action in _parser_actions()}
    dead = {flag for flag in CONFIG_OVERRIDES if flag not in dests}
    assert not dead, f"CONFIG_OVERRIDES 死键（解析器里没有这个开关）：{sorted(dead)}"


def test_only_the_context_builds_robots():
    """机器人只在 `Context.make_panda` 里被构造 —— 别处一律走 `ctx.make_panda(...)`。

    这是"装配只在入口"的下半句：入口允许装配，但**装配点也只有一个**。
    守它的收益很具体：要换末端帧、换机型（Panda 之外），只改工厂那一处，
    不会漏掉某个角落里偷偷 new 出来的第二台机器人。

    全 src 扫描（不只是入口文件）：`robots.panda(...)` 这种调用出现的位置，
    有且只有 `main.py::Context.make_panda`。
    """
    scopes: dict[str, set[str]] = {}
    for path in _source_files(SRC):
        for scope, names in _robot_factory_scopes(path).items():
            scopes[f"{path.relative_to(SRC).as_posix()}::{scope or '<模块级>'}"] = names

    # 反向守卫：工厂必须真的存在且在造机器人 —— 否则这条测试会"空转也全绿"
    factory = "main.py::Context.make_panda"
    assert factory in scopes, f"没找到唯一的机器人装配点 {factory}；实际调用点：{scopes}"

    offenders = {scope: sorted(names) for scope, names in scopes.items() if scope != factory}
    assert not offenders, f"绕过 Context 工厂直接造机器人的位置：{offenders}"


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
    actions = _parser_actions()
    for action in actions:
        assert not isinstance(action, argparse._SubParsersAction), "不应再有子命令"
    assert "observe" in {action.dest for action in actions}


def test_pipeline_is_the_single_source_of_truth_for_the_steps():
    """工序序列是数据：执行单元、命名、签名三条都要对得上。"""
    from robotkinematics.usecases import pick

    assert len(pick.PIPELINE) == 6
    for step in pick.PIPELINE:
        assert step.__name__.startswith("step_")
        assert len(inspect.signature(step).parameters) == 4, step.__name__
