"""@file test_cli.py

@brief 命令行端到端：扁平入口、退出码、输出内容、错误路径、配置覆盖。
"""

from __future__ import annotations

import pytest

from robotkinematics.main import main


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    """跑一次 CLI 并把 exit code 与两路输出一并交回。"""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_reachable_observe_succeeds(capsys):
    """可达观测应跑通流水线并以 0 退出，报告含七道工序。"""
    code, out, _ = run_cli(capsys, "--observe", "0.2", "0", "0")
    assert code == 0
    assert "抓取任务流水线" in out
    assert "七道工序" in out


def test_far_target_reports_not_in_workspace(capsys):
    """远目标应给出「不在工作空间」结论，退出码仍为 0（失败分类是报告，不是崩溃）。"""
    code, out, _ = run_cli(capsys, "--observe", "2", "2", "2")
    assert code == 0
    assert "不在工作空间" in out


def test_json_format(capsys):
    """--format json 应输出可解析的 JSON 字段。"""
    code, out, _ = run_cli(capsys, "--format", "json", "--observe", "0.2", "0", "0")
    assert code == 0
    assert '"outcome"' in out or "可执行" in out


def test_missing_observe_fails():
    """未给 --observe 时 argparse 应 SystemExit，不能静默成功。"""
    with pytest.raises(SystemExit):
        main([])


def test_command_line_overrides_config(capsys):
    """--max-iter 走配置覆盖通道。"""
    code, out, _ = run_cli(capsys, "--observe", "0.2", "0", "0", "--max-iter", "50")
    assert code == 0
    assert "抓取任务流水线" in out


def test_invalid_config_value_exits_nonzero(tmp_path, capsys):
    """配置文件里非法值要在加载阶段就拦下。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("ik:\n  max_iter: 0\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "--config", str(bad), "--observe", "0.2", "0", "0")
    assert code == 1
    assert "超出允许范围" in err


def test_plot_produces_png(capsys):
    """--plot 应生成任务图文件。"""
    from robotkinematics.adapters import plot_mpl as plotting

    code = main(["--observe", "0.2", "0", "0", "--plot"])
    captured = capsys.readouterr()
    assert code == 0
    assert "任务图已保存" in captured.out
    path = plotting._resolve("outputs") / "pick_panda.png"
    assert path.exists() and path.stat().st_size > 1000
