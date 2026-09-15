"""@file test_pick.py
@brief 抓取流水线：七道工序串起来之后，六类结论都要能复现、且各有数字。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import panda
from robotkinematics.core.ik_solvers import planar_chain
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import pick as pick_uc

CAMERA = (0.30, 0.00, 0.60)
CAMERA_YAW = 30.0

# 测试用的快速参数：初值少一些，采样少一些，结论不变
FAST = pick_uc.PickParams(seeds=12, workspace_samples=2000)


def make_task(observation, *, arm="panda", rpy=(180.0, 0.0, 0.0), prefer=None, mode="camera"):
    if arm == "planar":
        chain = planar_chain()
        limits = None
        planar = Planar2R()
    else:
        chain = panda.panda_urdf_chain()
        limits = np.array(panda.PANDA_JOINT_LIMITS)
        planar = None
    task = pick_uc.PickTask(
        observation=np.asarray(observation, dtype=float),
        camera_xyz=CAMERA,
        camera_yaw_deg=CAMERA_YAW,
        orientation_rpy_deg=rpy,
        arm=arm,
        limits=limits,
        planar=planar,
        mode=mode,
        prefer_config=prefer,
    )
    return chain, task


def run(observation, *, arm="panda", rpy=(180.0, 0.0, 0.0), prefer=None, params=FAST):
    chain, task = make_task(observation, arm=arm, rpy=rpy, prefer=prefer)
    return chain, task, pick_uc.pick(chain, task, params)


# ── 成功路径 ────────────────────────────────────────────────────────────────


def test_reachable_pick_returns_a_usable_solution():
    chain, _task, result = run([0.2, 0.0, 0.0], params=pick_uc.PickParams(seeds=40))
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert result.q is not None
    # 解必须真的把末端送到目标（位置误差在容差内）
    reached = chain.fk(result.q).t
    assert np.linalg.norm(reached - result.target.t) < 1e-6
    assert "离限位余量" in result.reason and "离初始位形" in result.reason


def test_all_seven_steps_are_reported():
    """七道工序一道都不能少 —— 这是"项目"与"七个 demo"的分界线。"""
    _, _, result = run([0.2, 0.0, 0.0], params=pick_uc.PickParams(seeds=40))
    names = [step.name for step in result.steps]
    assert len(names) == 7
    for keyword in (
        "观测 → 本体位姿",
        "可达性预筛",
        "多初值 IK",
        "雅可比体检",
        "限位校验",
        "解择优",
        "微动可行性校验",
    ):
        assert any(keyword in name for name in names), keyword


def test_solution_prefers_configurations_near_home():
    """就近择优：数值 IK 能找到转两圈多的解，但评分必须压住它。

    真实踩过的坑：二连杆曾解出 q=(-13.6, 15.1) rad —— 运动学正确、真机不可能执行。
    """
    _, _, result = run([0.35, -0.05, 0.15], arm="planar")
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert np.abs(result.q).max() < np.pi  # 不允许出现多圈解


def test_two_link_solution_matches_the_closed_form():
    """对照组的意义：数值流水线的解必须与闭式解一致。"""
    _chain, task, result = run([0.35, -0.05, 0.15], arm="planar")
    analytic = task.planar.ik(float(result.target.t[0]), float(result.target.t[1]))
    matches = [np.linalg.norm(result.q - np.array(sol)) for sol in analytic]
    assert min(matches) < 1e-6


def test_planar_task_ignores_orientation_and_says_so():
    """二连杆没有姿态自由度：报告里要如实说明，而不是假装姿态也满足了。"""
    _, _, result = run([0.35, -0.05, 0.15], arm="planar")
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert "无姿态自由度" in result.reason
    # 任务残差只看位置，不能拿 1.45 rad 的姿态误差去判失败（这是修过的一个 bug）
    assert result.candidates[0].residual < 1e-6


# ── 六类失败 ────────────────────────────────────────────────────────────────


def test_not_in_workspace_when_target_is_far_away():
    """观测一个远处的点：位置本身就够不着，换初值没用。"""
    _, _, result = run([2.0, 2.0, 2.0])
    assert result.outcome == pick_uc.OUTCOME_NOT_IN_WORKSPACE
    assert "超出" in result.reason and "换初值没用" in result.reason
    assert result.steps[1].numbers["margin"] < 0


@pytest.mark.slow
def test_pose_not_reachable_when_only_orientation_fails():
    """位置够得着、姿态做不到 —— 位置可达 ≠ 位姿可达。"""
    _chain, _, result = run([0.35, -0.05, 0.15], params=pick_uc.PickParams(seeds=40))
    assert result.outcome == pick_uc.OUTCOME_POSE_NOT_REACHABLE
    assert result.steps[1].numbers["inside"] is True  # 预筛说位置没问题
    assert "换个抓取姿态" in result.reason


@pytest.mark.slow
def test_same_position_with_another_orientation_succeeds():
    """上一条失败的原因确实是姿态：换个姿态就通了（验证失败分类没有冤枉目标）。"""
    _, _, result = run(
        [0.35, -0.05, 0.15], rpy=(0.0, 90.0, 0.0), params=pick_uc.PickParams(seeds=40)
    )
    assert result.outcome == pick_uc.OUTCOME_REACHABLE


@pytest.mark.slow
def test_all_limit_violated_has_its_own_outcome():
    _, _, result = run(
        [0.35, -0.05, 0.15], rpy=(0.0, 0.0, 0.0), params=pick_uc.PickParams(seeds=40)
    )
    assert result.outcome == pick_uc.OUTCOME_ALL_LIMIT_VIOLATED
    assert "全部被淘汰" in result.reason
    assert all(not c.survived for c in result.candidates)


@pytest.mark.slow
def test_residual_too_large_when_best_residual_is_between_the_two_thresholds():
    """ "精度不足"这一类的判定门槛：最佳残差高于残差容差、但还没到"姿态不可达"。

    这里把"不可达"的门槛抬到 10，让那个差 4.3cm 的目标落进"精度不足"分支
    （正常情况下它该被判为"姿态不可达"—— 那是另一条测试）。
    """
    params = pick_uc.PickParams(seeds=20, unreachable_residual=10.0)
    _, _, result = run([0.35, -0.05, 0.15], params=params)
    assert result.outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE
    assert "换更多初值或放宽精度" in result.reason


def test_classifier_separates_unreachable_from_imprecise():
    """直接测判定逻辑本身：同一个残差，配合不同的门槛会落到不同结论。"""
    from robotkinematics.core.se3 import SE3
    from robotkinematics.core.workspace import ReachEstimate

    chain, task = make_task([0.2, 0.0, 0.0])
    inside = ReachEstimate(
        chain_name=chain.name,
        direction=np.array([0.0, 0.0, 1.0]),
        radius=10.0,
        samples=100,
        seed=0,
        within_limits=True,
    )
    outside = ReachEstimate(
        chain_name=chain.name,
        direction=np.array([0.0, 0.0, 1.0]),
        radius=0.01,
        samples=100,
        seed=0,
        within_limits=True,
    )
    target = SE3.Trans(0.4, 0.0, 0.5)

    # 位置在界外 → 无论残差多大都是"不在工作空间"
    outcome, _ = pick_uc._classify_no_candidate(task, outside, target, 0.5, pick_uc.PickParams())
    assert outcome == pick_uc.OUTCOME_NOT_IN_WORKSPACE

    # 位置在界内、残差很大 → 姿态不可达
    outcome, reason = pick_uc._classify_no_candidate(
        task, inside, target, 0.5, pick_uc.PickParams()
    )
    assert outcome == pick_uc.OUTCOME_POSE_NOT_REACHABLE
    assert "换个抓取姿态" in reason

    # 位置在界内、残差很小但没到容差 → 精度不足
    outcome, _ = pick_uc._classify_no_candidate(task, inside, target, 1e-4, pick_uc.PickParams())
    assert outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE

    # 平面机械臂没有姿态自由度：同样情形不能报"姿态不可达"
    _, planar_task = make_task([0.2, 0.0, 0.0], arm="planar")
    outcome, reason = pick_uc._classify_no_candidate(
        planar_task, inside, target, 0.5, pick_uc.PickParams()
    )
    assert outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE
    assert "姿态" not in reason


def test_all_near_singular_when_sigma_threshold_is_high():
    """把"接近奇异"的门槛提高，所有解都会被淘汰 —— 分支要能走到。"""
    _, _, result = run([0.2, 0.0, 0.0], params=pick_uc.PickParams(seeds=8, min_sigma=100.0))
    assert result.outcome == pick_uc.OUTCOME_ALL_NEAR_SINGULAR


def test_micro_motion_infeasible_when_joint_budget_is_tiny():
    """微动校验：把关节步长上限压到极小，同一组解就变成"不可执行"。

    （真实场景里这条对应"末端要动 1mm，关节却要转好几度"的接近奇异位形。）
    """
    _, _, result = run(
        [0.2, 0.0, 0.0], params=pick_uc.PickParams(seeds=40, max_joint_step_deg=1e-6)
    )
    assert result.outcome == pick_uc.OUTCOME_MICRO_MOTION_INFEASIBLE
    assert result.q is None
    assert "微动" in result.reason


def test_outcome_labels_are_unique_and_complete():
    """六类失败 + 一类成功，名字互不相同（改了名字就会在这里暴露）。"""
    assert len(set(pick_uc.OUTCOMES)) == len(pick_uc.OUTCOMES) == 7
    assert pick_uc.OUTCOME_REACHABLE in pick_uc.OUTCOMES


# ── 报告 ────────────────────────────────────────────────────────────────────


def test_report_carries_outcome_steps_and_candidates():
    _, task, result = run([0.2, 0.0, 0.0], params=pick_uc.PickParams(seeds=20))
    report = pick_uc.to_report(task, result)
    assert report.fields["outcome"] == result.outcome
    assert len(report.fields["steps"]) == 7
    assert report.fields["q_deg"] is not None
    assert len(report.fields["candidates"]) == len(result.candidates)
    # 报告里要有结论、原因、七道工序
    text = " ".join(line for block in report.blocks for line in getattr(block, "lines", ()))
    assert result.outcome in text


def test_report_of_a_failure_has_no_solution():
    _, task, result = run([2.0, 2.0, 2.0])
    report = pick_uc.to_report(task, result)
    assert report.fields["q"] is None
    assert report.fields["outcome"] == pick_uc.OUTCOME_NOT_IN_WORKSPACE


def test_planar_report_includes_the_ascii_chart():
    _, task, result = run([0.35, -0.05, 0.15], arm="planar")
    report = pick_uc.to_report(task, result)
    kinds = {type(block).__name__ for block in report.blocks}
    assert "TwoLinkChartBlock" in kinds
