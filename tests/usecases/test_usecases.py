"""@file test_usecases.py

@brief 用例层：断言**结构化字段**（不是文本片段），数值以课件算例为准。

换成调库之后，这里验的不再是"公式对不对"（那是库的事），而是：
  * 我们调库的方式对不对（帧、约定、参数）；
  * 报告里给出的结论是否与课件/实测一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.cases_csv import CsvCaseSource
from robotkinematics.adapters.config_yaml import Config
from robotkinematics.core import robots
from robotkinematics.core.exceptions import UnreachableTargetError
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import batch, env_check, planar, pose, transforms
from robotkinematics.usecases import panda as panda_uc


@pytest.fixture
def config():
    """从项目自带的 config.yaml 读配置，作为下面几个夹具的共同依赖。"""
    return Config.from_yaml()


@pytest.fixture
def source(config):
    """用配置里的算例目录建 CSV 算例源 —— 课件算例都从这里读，不写死在测试里。"""
    return CsvCaseSource(config.cases_dir_path)


@pytest.fixture
def arm():
    """等长二连杆（l1=l2=1.0），即课件二连杆算例的标准模型。"""
    return Planar2R(l1=1.0, l2=1.0)


@pytest.fixture
def panda_robot():
    """Panda 配 TCP（夹爪）末端帧 —— 位置与雅可比都取这个帧，别混用法兰帧。"""
    return robots.panda(robots.TCP)


# ── 位姿 ────────────────────────────────────────────────────────────────────


def test_pose_fields_match_the_course_point_transform():
    """课件 01：0.4/0.2/0.3 + Rz(90°)，点 (0.1,0,0) → (0.4,0.3,0.3)。"""
    report = pose.run(0.4, 0.2, 0.3, 0.0, 0.0, 90.0)
    assert np.allclose(report.fields["p_base"], [0.4, 0.3, 0.3])
    assert report.fields["rpy_order"] == "zyx"  # 本项目的 RPY 约定
    assert report.fields["rebuild_gap"] < 1e-12


def test_pose_quaternion_is_wxyz():
    """spatialmath 的四元数是 (w,x,y,z) —— 与 ROS/Eigen 的 (x,y,z,w) 相反，要钉住。"""
    report = pose.run(0.0, 0.0, 0.0, 0.0, 0.0, 90.0)
    q = report.fields["quaternion_wxyz"]
    assert np.isclose(q[0], np.cos(np.pi / 4), atol=1e-9)  # w 在前
    assert np.isclose(q[3], np.sin(np.pi / 4), atol=1e-9)


# ── 坐标变换链 ──────────────────────────────────────────────────────────────


def test_transform_chain_matches_the_course_log(source):
    """相机系到基座系串起整条变换链后，杯子位置与课件日志一致（0.7830, 0.1634, 0.8）。"""
    cases = {c.case_id: c for c in source.coordinate_cases()}
    report = transforms.case_report(cases["ppt_case"])
    assert np.allclose(report.fields["cup_in_base"], [0.7830, 0.1634, 0.8], atol=5e-5)


def test_transform_all_cases_lists_every_case(source):
    """批量报告要把四个坐标变换算例一个不漏地列全，少一个就说明遍历漏了。"""
    report = transforms.all_cases_report(source.coordinate_cases())
    assert set(report.fields["cases"]) == {"ppt_case", "change_x", "change_y", "yaw_zero"}


# ── 二连杆 ──────────────────────────────────────────────────────────────────


def test_planar_fk_fields_and_library_agreement(arm):
    """闭式 FK 与库的结果必须一致（连行列式一起对）；这里 q2=90° 故 detJ=1，不是奇异位形。"""
    report = planar.fk_report(arm, 0.0, 90.0)
    assert np.allclose(report.fields["tool"], [1.0, 1.0])
    assert report.fields["library_gap"] < 1e-12  # 闭式解与库一致
    assert np.isclose(report.fields["det_jacobian"], 1.0)


def test_planar_ik_fields_include_both_solutions(arm):
    """(1,1) 处必须同时给出肘上、肘下两组解 —— 只返回一组就是漏解。"""
    report = planar.ik_report(arm, 1.0, 1.0)
    solutions = np.rad2deg(report.fields["solutions_rad"])
    assert np.allclose(solutions[0], [0.0, 90.0], atol=1e-9)
    assert np.allclose(solutions[1], [90.0, -90.0], atol=1e-9)


def test_planar_ik_raises_for_unreachable_target(arm):
    """够不着的目标要抛 UnreachableTargetError，不能悄悄返回一个近似解糊弄调用方。"""
    with pytest.raises(UnreachableTargetError):
        planar.ik_report(arm, 3.0, 0.0)


def test_planar_jacobian_uses_the_library_and_cross_checks_det(arm):
    """自算雅可比要与库逐元素一致（含行列式），条件数还要复现课件值 2.62。"""
    report = planar.jacobian_report(arm, 0.0, 90.0, (0.0, 0.1))
    assert np.allclose(report.fields["dq_rad"], [0.1, -0.1], atol=1e-9)
    assert np.isclose(report.fields["det_jacobian"], report.fields["det_library"], atol=1e-12)
    assert np.isclose(report.fields["condition"], 2.62, atol=5e-3)


def test_planar_jacobian_at_a_singularity_is_reported_not_raised(arm):
    """奇异位姿下报告说明原因、dq 为空 —— 而不是抛异常（失败是一等公民）。"""
    report = planar.jacobian_report(arm, 0.0, 0.0, (0.0, 0.1))
    assert report.fields["is_singular"] is True
    assert report.fields["dq_rad"] is None
    text = " ".join(line for block in report.blocks for line in getattr(block, "lines", ()))
    assert "奇异" in text


def test_singularity_sweep_matches_the_course_table(arm):
    """扫描复现课件表：条件数依次为 2.62/9.36/28.58，全伸直时 detJ=0 判为奇异。"""
    report = planar.singularity_sweep(arm)
    conditions = [entry["condition"] for entry in report.fields["sweep"]]
    assert np.allclose(conditions[:3], [2.62, 9.36, 28.58], atol=5e-3)
    assert not np.isfinite(conditions[3])


# ── Panda ───────────────────────────────────────────────────────────────────


def test_panda_fk_report_fields(panda_robot):
    """零位下报告要同时给出法兰位置、超限关节（第 4 轴）和 TCP 位置三样。"""
    report = panda_uc.fk_report(panda_robot, panda_uc.Q_ZERO, frame=robots.FLANGE)
    assert np.allclose(report.fields["t"], [0.088, 0.0, 0.926], atol=1e-3)
    assert report.fields["out_of_limits"] == [3]
    assert np.allclose(report.fields["tcp_t"], [0.088, 0.0, 0.8226], atol=1e-3)


def test_panda_ik_report_recovers_the_target(panda_robot):
    """用课件目标反解 IK 必须成功、位置误差小于 1e-5 —— 验的是调库的方式对不对。"""
    target = robots.end_pose(panda_robot, panda_uc.Q_PPT_GOAL, robots.TCP)
    report = panda_uc.ik_report(
        panda_robot, target, panda_uc.Q_IK_SEED, show_redundancy=False, frame=robots.TCP
    )
    assert report.fields["success"] is True
    assert report.fields["position_error"] < 1e-5


def test_panda_jacobian_report_fields(panda_robot):
    """Panda 雅可比是 6x7、初值处不奇异，且自算可操作度与库给出的一致。"""
    report = panda_uc.jacobian_report(panda_robot, panda_uc.Q_IK_SEED)
    assert np.asarray(report.fields["jacobian"]).shape == (6, 7)
    assert report.fields["is_singular"] is False
    # 我们算的可操作度与库的必须一致
    assert np.isclose(report.fields["manipulability"], report.fields["manipulability_library"])


def test_panda_singular_pose_report_marks_zero_as_singular(panda_robot):
    """课件两个姿态里只有零位被判奇异、ppt_goal 不奇异 —— 奇异判定与课件对得上。"""
    report = panda_uc.singular_pose_report(panda_robot)
    by_name = {entry["name"]: entry for entry in report.fields["poses"]}
    assert by_name["zero（课件姿态 1）"]["is_singular"] is True
    assert by_name["ppt_goal（课件姿态 2）"]["is_singular"] is False


# ── 批量算例 ────────────────────────────────────────────────────────────────


def test_batch_lines_reproduce_the_course_output(source):
    """第 1、2 段必须与课件交付物的数值逐条一致（验收标准，期望值内联）。"""
    lines = batch.run(source).fields["lines"]
    for expected in (
        "1) 坐标变换：camera frame -> base frame",
        "- ppt_case: cup_base=[0.783  0.1634 0.8   ]",
        "- change_x: cup_base=[0.9562 0.2634 0.8   ]",
        "- yaw_zero: cup_base=[ 0.8 -0.1  0.8]",
        "2) 二连杆 FK / IK / 奇异点",
        "- fk_0_90: FK tool=[1. 1.], detJ=1.000000",
        "- fk_60_minus30: FK tool=[1.366 1.366], detJ=-0.500000",
        "- ik_1_1: IK sols=(0.00°, 90.00°); (90.00°, -90.00°)",
        "- ik_far: IK no solution",
        "- singular: FK tool=[2. 0.], detJ=0.000000",
    ):
        assert expected in lines, expected


def test_batch_panda_section_uses_the_gripper_frame(source):
    """第 3 段用夹爪 TCP 帧 —— 零位姿 0.8226 与课件日志一致。"""
    lines = batch.run(source).fields["lines"]
    zero = next(line for line in lines if line.startswith("- zero:"))
    assert "0.8226" in zero
    assert "奇异" in zero
    # 位置与奇异值必须同帧：奇异值来自库默认末端（夹爪）的雅可比
    assert any("σmin" in line for line in lines)


# ── 环境自检 ────────────────────────────────────────────────────────────────


def test_env_check_reports_ok_in_this_environment():
    """环境自检在本机（numpy<2 + 已装 spatialmath）应报 ok，且没列出坏掉的模块。"""
    report = env_check.run()
    assert report.fields["ok"] is True
    assert report.fields["broken_modules"] == []
    assert "spatialmath" in " ".join(report.fields["kinematics_available"])
