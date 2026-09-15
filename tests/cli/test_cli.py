"""@file test_cli.py
@brief 命令行端到端：退出码、输出内容、错误路径、配置覆盖。
"""

from __future__ import annotations

import pytest

from robotkinematics.cli.main import main


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_check_succeeds(capsys):
    code, out, _ = run_cli(capsys, "check")
    assert code == 0
    assert "numpy" in out
    assert "环境检查完成" in out


def test_fk_prints_the_course_numbers(capsys):
    code, out, _ = run_cli(capsys, "fk", "--q1", "0", "--q2", "90")
    assert code == 0
    assert "[ 1.0000,  1.0000]" in out


def test_ik_reports_both_solutions(capsys):
    code, out, _ = run_cli(capsys, "ik", "--target", "1", "1")
    assert code == 0
    assert "肘上 / 肘下" in out


def test_unreachable_target_exits_nonzero_with_a_clear_message(capsys):
    code, out, err = run_cli(capsys, "ik", "--target", "3", "0")
    assert code == 1
    assert err.startswith("错误:")
    assert "超出二连杆的工作空间" in err
    assert out == ""  # 错误不进 stdout，重定向时才不会污染结果文件


def test_jacobian_singularity_is_explained_not_crashed(capsys):
    code, out, _ = run_cli(capsys, "jac", "--q1", "0", "--q2", "0", "--dx", "0", "0.1")
    assert code == 0
    assert "det(J) = 0.000000" in out


def test_cam_all_lists_every_case(capsys):
    code, out, _ = run_cli(capsys, "cam", "--all")
    assert code == 0
    for case_id in ("ppt_case", "change_x", "change_y", "yaw_zero"):
        assert case_id in out


def test_unknown_case_lists_the_available_ones(capsys):
    code, _, err = run_cli(capsys, "cam", "--case", "nope")
    assert code == 1
    assert "ppt_case" in err


def test_sing_writes_the_report_file(capsys):
    code, out, _ = run_cli(capsys, "sing")
    assert code == 0
    assert "two_link_singularity_summary.txt" in out


def test_panda_fk_with_explicit_q(capsys):
    code, out, _ = run_cli(capsys, "panda-fk", "--q", "0", "0", "0", "0", "0", "0", "0")
    assert code == 0
    assert "0.9260" in out


def test_panda_fk_with_case_id(capsys):
    code, out, _ = run_cli(capsys, "panda-fk", "--case", "ppt_goal")
    assert code == 0
    assert "0.4531" in out


def test_panda_ik_with_explicit_pose(capsys):
    code, out, _ = run_cli(
        capsys,
        "panda-ik",
        "--no-redundancy",
        "--pose",
        "0.4531",
        "0.0",
        "0.5619",
        "0",
        "11.4592",
        "180",
    )
    assert code == 0
    assert "成功" in out


def test_panda_ik_reports_redundancy_by_default(capsys):
    code, out, _ = run_cli(capsys, "panda-ik", "--case", "ppt_goal")
    assert code == 0
    assert "冗余" in out
    assert "零空间" in out


def test_panda_fk_frame_option_switches_the_end_frame(capsys):
    """法兰帧与夹爪 TCP 帧差 103.4mm —— 课件日志用的是后者。"""
    code, out, _ = run_cli(capsys, "panda-fk", "--case", "zero", "--frame", "tcp")
    assert code == 0
    assert "0.8226" in out
    code, out, _ = run_cli(capsys, "panda-fk", "--case", "zero")
    assert code == 0
    assert "0.9260" in out


def test_panda_jacobian_shape(capsys):
    code, out, _ = run_cli(capsys, "panda-jac", "--case", "ik_seed")
    assert code == 0
    assert "6×7" in out


def test_batch_runs_and_writes(capsys):
    code, out, _ = run_cli(capsys, "batch")
    assert code == 0
    assert "ppt_cases_batch_result.txt" in out


def test_command_line_overrides_config(capsys):
    """--l1/--l2 走的是配置覆盖通道，不是各自的私有参数。"""
    code, out, _ = run_cli(capsys, "fk", "--l1", "2.0", "--l2", "3.0", "--q1", "0", "--q2", "0")
    assert code == 0
    assert "[ 5.0000,  0.0000]" in out


def test_invalid_config_value_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("planar:\n  l1: -1\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "--config", str(bad), "fk")
    assert code == 1
    assert "超出允许范围" in err


def test_missing_subcommand_fails(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_anim_produces_a_gif(capsys):
    from robotkinematics.adapters import plot_mpl as plotting

    code = main(["anim", "--sweep", "q2", "--frames", "8"])
    captured = capsys.readouterr()
    assert code == 0
    assert "two_link_sweep_q2.gif" in captured.out
    path = plotting._resolve("outputs") / "two_link_sweep_q2.gif"
    assert path.exists() and path.stat().st_size > 1000
