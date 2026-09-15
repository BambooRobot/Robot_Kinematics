"""@file test_usecases.py
@brief 用例层：每个用例对应课件脚本，断言的是**结构化结果**（不再断言文本片段）。

重构前这里是 `assert "0.7830" in text` 这种写法 —— 改个措辞测试就红，而数字错了却可能漏过。
现在用例返回 Report，数值在 fields 里，文本只在 test_render_text.py 里单独验格式。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.cases_csv import CsvCaseSource
from robotkinematics.adapters.config_yaml import Config
from robotkinematics.core import panda
from robotkinematics.core.exceptions import UnreachableTargetError
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import batch, env_check, planar, pose, transforms
from robotkinematics.usecases import panda as panda_uc


@pytest.fixture
def config():
    return Config.from_yaml()


@pytest.fixture
def source(config):
    return CsvCaseSource(config.cases_dir_path)


@pytest.fixture
def arm():
    return Planar2R(l1=1.0, l2=1.0)


# ── 位姿 ────────────────────────────────────────────────────────────────────


def test_pose_fields_match_the_course_point_transform():
    """课件 01：0.4/0.2/0.3 + Rz(90°)，点 (0.1,0,0) → (0.4,0.3,0.3)。"""
    report = pose.run(0.4, 0.2, 0.3, 0.0, 0.0, 90.0)
    assert np.allclose(report.fields["p_base"], [0.4, 0.3, 0.3])
    assert np.allclose(report.fields["R"], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert report.fields["is_rotation"] is True
    assert report.fields["representation_gap"] < 1e-12


# ── 坐标变换链 ──────────────────────────────────────────────────────────────


def test_transform_chain_matches_the_course_log(source):
    cases = {c.case_id: c for c in source.coordinate_cases()}
    report = transforms.case_report(cases["ppt_case"])
    assert np.allclose(report.fields["cup_in_base"], [0.7830, 0.1634, 0.8], atol=5e-5)


def test_transform_all_cases_lists_every_case(source):
    report = transforms.all_cases_report(source.coordinate_cases())
    assert set(report.fields["cases"]) == {"ppt_case", "change_x", "change_y", "yaw_zero"}
    assert (
        report.fields["cases"]["change_x"]["cup_in_base"][0]
        > report.fields["cases"]["ppt_case"]["cup_in_base"][0]
    )


# ── 二连杆 ──────────────────────────────────────────────────────────────────


def test_planar_fk_fields(arm):
    report = planar.fk_report(arm, 0.0, 90.0)
    assert np.allclose(report.fields["tool"], [1.0, 1.0])
    assert np.allclose(report.fields["elbow"], [1.0, 0.0])
    assert np.isclose(report.fields["det_jacobian"], 1.0)


def test_planar_ik_fields_include_both_solutions_and_their_verification(arm):
    report = planar.ik_report(arm, 1.0, 1.0)
    solutions = np.rad2deg(report.fields["solutions_rad"])
    assert np.allclose(solutions[0], [0.0, 90.0], atol=1e-9)
    assert np.allclose(solutions[1], [90.0, -90.0], atol=1e-9)
    # FK 回验是报告的一部分：表格里两行都必须是 ✔
    table = next(b for b in report.blocks if b.__class__.__name__ == "TableBlock")
    assert [row[-1] for row in table.rows] == ["✔", "✔"]


def test_planar_ik_raises_for_unreachable_target(arm):
    with pytest.raises(UnreachableTargetError):
        planar.ik_report(arm, 3.0, 0.0)


def test_planar_jacobian_fields(arm):
    report = planar.jacobian_report(arm, 0.0, 90.0, (0.0, 0.1))
    assert np.allclose(report.fields["jacobian"], [[-1, -1], [1, 0]], atol=1e-12)
    assert np.allclose(report.fields["dq_rad"], [0.1, -0.1], atol=1e-9)
    assert np.isclose(report.fields["det_jacobian"], 1.0)
    assert np.isclose(report.fields["condition"], 2.62, atol=5e-3)
    assert report.fields["is_singular"] is False


def test_planar_jacobian_fields_at_a_singularity(arm):
    """奇异位姿下 dq 为 None（而不是抛异常）—— 报告里会说明为什么解不出来。"""
    report = planar.jacobian_report(arm, 0.0, 0.0, (0.0, 0.1))
    assert report.fields["is_singular"] is True
    assert report.fields["dq_rad"] is None
    assert np.isclose(report.fields["det_jacobian"], 0.0)


def test_singularity_sweep_matches_the_course_table(arm):
    report = planar.singularity_sweep(arm)
    conditions = [entry["condition"] for entry in report.fields["sweep"]]
    assert np.allclose(conditions[:3], [2.62, 9.36, 28.58], atol=5e-3)
    assert not np.isfinite(conditions[3])
    assert report.output_name == "two_link_singularity_summary.txt"


# ── Panda ───────────────────────────────────────────────────────────────────


def test_panda_fk_fields_and_limit_warning():
    report = panda_uc.fk_report(panda.panda_urdf_chain(), panda.PANDA_Q_ZERO)
    assert np.allclose(report.fields["t"], [0.088, 0.0, 0.926], atol=1e-3)
    assert report.fields["out_of_limits"] == ["panda_joint4"]
    assert report.fields["ets_mdh_deviation"] < 1e-12


def test_panda_fk_tcp_frame_reproduces_the_course_log():
    report = panda_uc.fk_report(panda.panda_hand_tcp_chain(), panda.PANDA_Q_ZERO)
    assert np.allclose(report.fields["t"], panda.COURSE_LOG_ZERO_POSE_TOOL_T, atol=5e-5)


def test_panda_jacobian_fields():
    report = panda_uc.jacobian_report(panda.panda_urdf_chain(), panda.PANDA_Q_IK_SEED)
    assert np.asarray(report.fields["jacobian"]).shape == (6, 7)
    assert np.isclose(report.fields["condition"], 8.7606, atol=1e-3)
    assert report.fields["is_singular"] is False


def test_panda_ik_fields():
    chain = panda.panda_urdf_chain()
    target = chain.fk(panda.PANDA_Q_PPT_GOAL)
    report = panda_uc.ik_report(chain, target, panda.PANDA_Q_IK_SEED, show_redundancy=False)
    assert report.fields["success"] is True
    assert report.fields["position_error"] < 1e-9
    assert np.allclose(chain.fk(report.fields["q"]).t, target.t, atol=1e-8)


def test_panda_ik_redundancy_entries():
    chain = panda.panda_urdf_chain()
    target = chain.fk(panda.PANDA_Q_PPT_GOAL)
    report = panda_uc.ik_report(chain, target, panda.PANDA_Q_IK_SEED)
    entries = report.fields["redundancy"]
    assert len(entries) == 4
    assert sum(1 for e in entries if e["success"]) >= 2
    # 至少有两个不同的解 —— 这才是"冗余"
    solved = [e["q_first3"] for e in entries if e["success"]]
    assert np.linalg.norm(np.subtract(solved[0], solved[1])) > 0.1


def test_panda_singular_pose_report_marks_zero_as_singular():
    report = panda_uc.singular_pose_report(panda.panda_urdf_chain())
    by_name = {entry["name"]: entry for entry in report.fields["poses"]}
    assert by_name["zero（课件姿态 1）"]["is_singular"] is True
    assert by_name["ppt_goal（课件姿态 2）"]["is_singular"] is False
    assert report.output_name == "panda_singularity_summary.txt"


# ── 批量算例 ────────────────────────────────────────────────────────────────


def test_batch_lines_reproduce_the_course_output(source):
    """第 1、2 段必须与课件 ppt_cases_batch_result.txt 逐条一致（验收标准）。"""
    report = batch.run(source)
    lines = report.fields["lines"]
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


def test_batch_adds_the_panda_section_the_course_left_out(source):
    lines = batch.run(source).fields["lines"]
    assert any("Franka Panda" in line for line in lines)
    for case_id in ("zero", "ppt_goal", "ik_seed"):
        assert any(line.startswith(f"- {case_id}:") for line in lines)


def test_batch_reports_singular_panda_pose_with_readable_condition(source):
    lines = batch.run(source).fields["lines"]
    zero_line = next(line for line in lines if line.startswith("- zero:"))
    assert "inf/极大" in zero_line
    assert "奇异" in zero_line


def test_every_two_link_case_runs(source):
    """算例表里的每一条都要能跑通；只有 ik_far 允许无解。"""
    for case in source.two_link_cases():
        if case.kind == "fk":
            assert planar.fk_report(Planar2R(case.l1, case.l2), *case.q_deg)
            continue
        try:
            assert planar.ik_report(Planar2R(case.l1, case.l2), *case.target)
        except UnreachableTargetError:
            assert case.case_id == "ik_far"


# ── 环境自检 ────────────────────────────────────────────────────────────────


def test_env_check_reports_ok_in_this_environment():
    report = env_check.run()
    assert report.fields["ok"] is True
    assert report.fields["broken_modules"] == []
