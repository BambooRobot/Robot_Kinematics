"""@file test_cli.py

@brief 命令行端到端：退出码、输出内容、错误路径、配置覆盖。
"""

from __future__ import annotations

import pytest

from robotkinematics.main import main


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    """跑一次 CLI 并把 exit code 与两路输出一并交回，省去每个用例重复 readouterr。"""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_check_succeeds(capsys):
    """check 子命令是唯一的"环境自证"入口：依赖齐全时以 0 退出并报告检查完成。"""
    code, out, _ = run_cli(capsys, "check")
    assert code == 0
    assert "numpy" in out
    assert "环境检查完成" in out


def test_fk_prints_the_course_numbers(capsys):
    """课件算例：q1=0°, q2=90° 时末端应落在 (1, 1)。"""
    code, out, _ = run_cli(capsys, "fk", "--q1", "0", "--q2", "90")
    assert code == 0
    assert "[ 1.0000,  1.0000]" in out


def test_ik_reports_both_solutions(capsys):
    """目标 (1, 1) 落得进工作空间，肘上与肘下两组解都要列出来，只报一组等于丢解。"""
    code, out, _ = run_cli(capsys, "ik", "--target", "1", "1")
    assert code == 0
    assert "肘上 / 肘下" in out


def test_unreachable_target_exits_nonzero_with_a_clear_message(capsys):
    """够不到的目标要以退出码 1 报错，且错误只走 stderr —— stdout 保持干净，重定向写文件才不被污染。"""
    code, out, err = run_cli(capsys, "ik", "--target", "3", "0")
    assert code == 1
    assert err.startswith("错误:")
    assert "超出二连杆的工作空间" in err
    assert out == ""  # 错误不进 stdout，重定向时才不会污染结果文件


def test_jacobian_singularity_is_explained_not_crashed(capsys):
    """奇异位形是"报告"不是"崩溃"：退出码仍为 0，但要明说奇异，不能返回一脸正常的速度。"""
    code, out, _ = run_cli(capsys, "jac", "--q1", "0", "--q2", "0", "--dx", "0", "0.1")
    assert code == 0  # 奇异是"报告"，不是"崩溃"
    assert "0.000000" in out and "奇异" in out


def test_cam_all_lists_every_case(capsys):
    """--all 必须把内置算例 id 一个不落地列全，漏一个用户就无从调用。"""
    code, out, _ = run_cli(capsys, "cam", "--all")
    assert code == 0
    for case_id in ("ppt_case", "change_x", "change_y", "yaw_zero"):
        assert case_id in out


def test_unknown_case_lists_the_available_ones(capsys):
    """算例名写错时要回显可选算例再退出，让用户照着改，而不是只丢一句 not found。"""
    code, _, err = run_cli(capsys, "cam", "--case", "nope")
    assert code == 1
    assert "ppt_case" in err


def test_sing_writes_the_report_file(capsys):
    """奇异汇总不只在屏幕上闪一下，要落成报告文件并把文件名回报给用户。"""
    code, out, _ = run_cli(capsys, "sing")
    assert code == 0
    assert "two_link_singularity_summary.txt" in out


def test_panda_fk_with_explicit_q(capsys):
    """全零位下默认（法兰帧）末端 z=0.9260 m，这是本工程 Panda 正解的基准数。"""
    code, out, _ = run_cli(capsys, "panda-fk", "--q", "0", "0", "0", "0", "0", "0", "0")
    assert code == 0
    assert "0.9260" in out


def test_panda_fk_with_case_id(capsys):
    """--case 与 --q 是同一套正解的两个入口，ppt_goal 的输出里应出现 0.4531。"""
    code, out, _ = run_cli(capsys, "panda-fk", "--case", "ppt_goal")
    assert code == 0
    assert "0.4531" in out


def test_panda_ik_with_explicit_pose(capsys):
    """显式给位姿并关掉冗余时，可达目标必须报成功，而不是在 7 自由度里搜空。"""
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
    """7 自由度默认就带冗余：冗余度与零空间信息要同时给出，这是 Panda 与二连杆的分水岭。"""
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
    """Panda 雅可比恒为 6×7；形状一错，后面所有零空间与静力学推导都会跟着失真。"""
    code, out, _ = run_cli(capsys, "panda-jac", "--case", "ik_seed")
    assert code == 0
    assert "6×7" in out


def test_batch_runs_and_writes(capsys):
    """batch 要一次跑完全部算例并落盘结果文件，控制台需回报落盘的文件名。"""
    code, out, _ = run_cli(capsys, "batch")
    assert code == 0
    assert "ppt_cases_batch_result.txt" in out


def test_command_line_overrides_config(capsys):
    """--l1/--l2 走的是配置覆盖通道，不是各自的私有参数。"""
    code, out, _ = run_cli(capsys, "fk", "--l1", "2.0", "--l2", "3.0", "--q1", "0", "--q2", "0")
    assert code == 0
    assert "[ 5.0000,  0.0000]" in out


def test_invalid_config_value_exits_nonzero(tmp_path, capsys):
    """配置文件里 l1 为负这类非法值要在加载阶段就拦下，错误信息须点明超出允许范围。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("planar:\n  l1: -1\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "--config", str(bad), "fk")
    assert code == 1
    assert "超出允许范围" in err


def test_missing_subcommand_fails(capsys):
    """敲了命令却没给子命令时，argparse 应直接 SystemExit，绝不能静默返回 0 让脚本误判成功。"""
    with pytest.raises(SystemExit):
        main([])


def test_anim_produces_a_gif(capsys):
    """anim 要真的生成 GIF：光看退出码不够，还要文件存在且非空，才算渲染成功。"""
    from robotkinematics.adapters import plot_mpl as plotting

    code = main(["anim", "--sweep", "q2", "--frames", "8"])
    captured = capsys.readouterr()
    assert code == 0
    assert "two_link_sweep_q2.gif" in captured.out
    path = plotting._resolve("outputs") / "two_link_sweep_q2.gif"
    assert path.exists() and path.stat().st_size > 1000
