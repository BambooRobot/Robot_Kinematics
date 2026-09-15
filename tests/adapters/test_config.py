"""@file test_config.py
@brief 配置：默认值、范围校验、未知项告警、命令行覆盖优先级。
"""

from __future__ import annotations

import pytest

from robotkinematics.adapters.config_yaml import Config, resolve_path
from robotkinematics.core.exceptions import KinematicsError

VALID_YAML = """
planar:
  l1: 0.7
  l2: 1.3
ik:
  dls_lambda: 0.02
  max_iter: 123
"""


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_default_config_file_loads():
    """项目自带的 configs/default.yaml 必须能读通。"""
    config = Config.from_yaml()
    assert config.planar.l1 > 0 and config.planar.l2 > 0
    assert config.ik.max_iter >= 1
    assert config.source.endswith("default.yaml")


def test_yaml_values_override_defaults(tmp_path):
    config = Config.from_yaml(_write(tmp_path, VALID_YAML))
    assert config.planar.l1 == 0.7
    assert config.planar.l2 == 1.3
    assert config.ik.dls_lambda == 0.02
    assert config.ik.max_iter == 123
    # 没写的项保持默认值
    assert config.singularity.cond_warn == 100.0


def test_command_line_overrides_win(tmp_path):
    """命令行优先级高于配置文件 —— 配置文件写默认值，命令行做一次性试验。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML)).override(planar_l1=2.5, ik_max_iter=9)
    assert config.planar.l1 == 2.5
    assert config.planar.l2 == 1.3  # 没被覆盖的项不动
    assert config.ik.max_iter == 9


def test_none_values_in_override_are_ignored(tmp_path):
    """命令行没传的参数是 None，不能把配置里的值清成 None。"""
    config = Config.from_yaml(_write(tmp_path, VALID_YAML)).override(planar_l2=None)
    assert config.planar.l2 == 1.3


def test_unknown_key_warns_instead_of_silently_ignoring(tmp_path, capsys):
    config = Config.from_yaml(_write(tmp_path, VALID_YAML + "\nplaenar:\n  l1: 9.9\n"))
    captured = capsys.readouterr()
    assert "plaenar" in captured.err
    assert "无法识别" in captured.err
    assert config.planar.l1 == 0.7  # 拼错的那段没有生效


def test_unknown_field_inside_a_section_warns(tmp_path, capsys):
    Config.from_yaml(_write(tmp_path, "planar:\n  l1: 1.0\n  l3: 2.0\n"))
    assert "planar.l3" in capsys.readouterr().err


@pytest.mark.parametrize(
    "yaml_text",
    [
        "planar:\n  l1: 0\n",  # 连杆长度必须为正
        "planar:\n  l1: -1\n",
        "ik:\n  max_iter: 0\n",  # 迭代次数至少 1
        "ik:\n  step: 2.0\n",  # 步长缩放必须在 (0,1]
        "ik:\n  tol: 0\n",
        "singularity:\n  cond_warn: 0.5\n",
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, yaml_text):
    with pytest.raises(KinematicsError):
        Config.from_yaml(_write(tmp_path, yaml_text))


def test_non_numeric_value_is_rejected(tmp_path):
    with pytest.raises(KinematicsError):
        Config.from_yaml(_write(tmp_path, "planar:\n  l1: abc\n"))


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(KinematicsError) as excinfo:
        Config.from_yaml(tmp_path / "not-there.yaml")
    assert "无法打开配置文件" in str(excinfo.value)


def test_unknown_override_key_is_rejected():
    with pytest.raises(KinematicsError):
        Config().override(nosuch_field=1)


def test_resolve_path_prefers_cwd_then_project_root():
    assert resolve_path("configs/default.yaml").exists()
    # 绝对路径原样返回
    assert resolve_path("/tmp").as_posix() == "/tmp"
