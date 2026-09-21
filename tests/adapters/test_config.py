"""@file test_config.py

@brief 配置：默认值、范围校验、未知项告警、命令行覆盖优先级。
"""

from __future__ import annotations

import pytest

from robotkinematics.adapters.config_yaml import Config, resolve_path
from robotkinematics.core.exceptions import KinematicsError

VALID_YAML = """
camera:
  x: 0.25
ik:
  max_iter: 123
  tol: 1.0e-8
"""


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_default_config_file_loads():
    """项目自带的 configs/default.yaml 必须能读通。"""
    config = Config.from_yaml()
    assert config.ik.max_iter >= 1
    assert config.pick.seeds >= 1
    assert config.source.endswith("default.yaml")


def test_yaml_values_override_defaults(tmp_path):
    """yaml 里写了的项覆盖默认值，没写的项保持默认。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML))
    assert config.camera.x == 0.25
    assert config.ik.max_iter == 123
    assert config.ik.tol == 1.0e-8
    assert config.pick.seeds == 24


def test_command_line_overrides_win(tmp_path):
    """命令行优先级高于配置文件。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML)).override(ik_max_iter=9)
    assert config.ik.max_iter == 9
    assert config.ik.tol == 1.0e-8


def test_none_values_in_override_are_ignored(tmp_path):
    """命令行没传的参数是 None，不能把配置里的值清成 None。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML)).override(ik_tol=None)
    assert config.ik.tol == 1.0e-8


def test_unknown_key_warns_instead_of_silently_ignoring(tmp_path, capsys):
    """顶层键拼错要告警并忽略。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML + "\nplaenar:\n  l1: 9.9\n"))
    captured = capsys.readouterr()
    assert "plaenar" in captured.err
    assert "无法识别" in captured.err
    assert config.camera.x == 0.25


def test_unknown_field_inside_a_section_warns(tmp_path, capsys):
    """段落里拼错的字段同样要告警。"""
    Config.from_yaml(_write(tmp_path, "ik:\n  max_iter: 10\n  nope: 2.0\n"))
    assert "ik.nope" in capsys.readouterr().err


@pytest.mark.parametrize(
    "yaml_text",
    [
        "ik:\n  max_iter: 0\n",
        "ik:\n  tol: 0\n",
        "pick:\n  seeds: 0\n",
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, yaml_text):
    """越界参数一律拒绝。"""
    with pytest.raises(KinematicsError):
        Config.from_yaml(_write(tmp_path, yaml_text))


def test_non_numeric_value_is_rejected(tmp_path):
    """非数值配置项要报错。"""
    with pytest.raises(KinematicsError):
        Config.from_yaml(_write(tmp_path, "ik:\n  max_iter: abc\n"))


def test_missing_file_is_reported_clearly(tmp_path):
    """配置文件不存在时要提示无法打开。"""
    with pytest.raises(KinematicsError) as excinfo:
        Config.from_yaml(tmp_path / "not-there.yaml")
    assert "无法打开配置文件" in str(excinfo.value)


def test_unknown_override_key_is_rejected():
    """命令行传了不存在的参数要报错。"""
    with pytest.raises(KinematicsError):
        Config().override(nosuch_field=1)


def test_resolve_path_prefers_cwd_then_project_root():
    """相对路径先按当前目录找、再退回工程根目录；绝对路径原样返回。"""
    assert resolve_path("configs/default.yaml").exists()
    absolute = resolve_path("/tmp")
    assert absolute.is_absolute()


def test_project_root_points_at_the_repo_root():
    """`PROJECT_ROOT` 必须真的指向项目根。"""
    from robotkinematics.adapters.config_yaml import PROJECT_ROOT

    assert (PROJECT_ROOT / "pyproject.toml").is_file(), PROJECT_ROOT
    assert (PROJECT_ROOT / "configs" / "default.yaml").is_file(), PROJECT_ROOT
