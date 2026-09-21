"""@file test_pick.py

@brief 抓取流水线：七道工序 + 七类结论（一类成功、六类失败），每类都要能复现。机器人固定 Panda。
"""

from __future__ import annotations

import numpy as np

from robotkinematics.core import robots
from robotkinematics.usecases import pick as pick_uc

CAMERA = (0.30, 0.00, 0.60)
CAMERA_YAW = 30.0
FAST = pick_uc.PickParams(seeds=10, workspace_samples=2000)

TARGET_REACHABLE = [0.2, 0.0, 0.0]
TARGET_POSE_UNREACHABLE = [0.35, -0.05, 0.15]
TARGET_FAR = [2.0, 2.0, 2.0]


def make_task(observation, *, rpy=(180.0, 0.0, 0.0), prefer=None):
    """组装 Panda 与抓取任务。"""
    robot = robots.panda(robots.TCP)
    task = pick_uc.PickTask(
        observation=np.asarray(observation, dtype=float),
        camera_xyz=CAMERA,
        camera_yaw_deg=CAMERA_YAW,
        orientation_rpy_deg=rpy,
        prefer_config=prefer,
    )
    return robot, task


def run(observation, *, rpy=(180.0, 0.0, 0.0), params=FAST):
    """一次跑完整条流水线。"""
    robot, task = make_task(observation, rpy=rpy)
    return robot, task, pick_uc.pick(robot, task, params)


def test_reachable_pick_returns_a_usable_solution():
    """可达目标的解要真能用：末端到位误差小于 1e-3，且理由里带上离限位余量。"""
    robot, task, result = run(TARGET_REACHABLE)
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert result.q is not None
    reached = np.asarray(robots.end_pose(robot, result.q, task.frame).t)
    assert np.linalg.norm(reached - np.asarray(result.target.t)) < 1e-3
    assert "离限位余量" in result.reason


def test_all_seven_steps_are_reported():
    """七道工序一道都不能少。"""
    _, _, result = run(TARGET_REACHABLE)
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
    """就近择优：不允许出现"关节转两圈"的解。"""
    _, _, result = run(TARGET_REACHABLE)
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert np.abs(result.q).max() < 2 * np.pi


def test_not_in_workspace_when_target_is_far_away():
    """远目标归为「不在工作空间」。"""
    _, _, result = run(TARGET_FAR)
    assert result.outcome == pick_uc.OUTCOME_NOT_IN_WORKSPACE
    assert "换初值没用" in result.reason
    assert result.steps[1].numbers["margin"] < 0


def test_pose_not_reachable_when_only_orientation_fails():
    """位置在界内、只有姿态做不到时，要归为姿态不可达。"""
    _, _, result = run(TARGET_POSE_UNREACHABLE)
    assert result.outcome == pick_uc.OUTCOME_POSE_NOT_REACHABLE
    assert result.steps[1].numbers["inside"] is True
    assert "换个抓取姿态" in result.reason


def test_same_position_with_another_orientation_succeeds():
    """换个姿态就通了（验证分类没冤枉目标）。"""
    _, _, result = run(TARGET_POSE_UNREACHABLE, rpy=(0.0, 90.0, 0.0))
    assert result.outcome == pick_uc.OUTCOME_REACHABLE


def test_residual_too_large_when_tolerance_is_impossible():
    """容差压到 1e-14 时归为残差过大。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, residual_tol=1e-14)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE


def test_all_near_singular_when_sigma_threshold_is_high():
    """σmin 阈值抬到 100 后所有解都算近奇异。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, min_sigma=100.0)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_ALL_NEAR_SINGULAR


def test_all_limit_violated_when_margin_is_demanding():
    """要求离限位至少 170° → 所有解都被限位淘汰。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, limit_margin_deg=170.0)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_ALL_LIMIT_VIOLATED
    assert all(not c.survived for c in result.candidates)


def test_micro_motion_infeasible_when_joint_budget_is_tiny():
    """关节单步上限压到 1e-6 后微动校验过不了。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, max_joint_step_deg=1e-6)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_MICRO_MOTION_INFEASIBLE
    assert result.q is None


def test_outcome_labels_are_unique_and_complete():
    """七类结论的标签必须互不重复且恰好七类。"""
    assert len(set(pick_uc.OUTCOMES)) == len(pick_uc.OUTCOMES) == 7
    assert pick_uc.OUTCOME_REACHABLE in pick_uc.OUTCOMES


def test_classifier_separates_unreachable_from_imprecise():
    """同一个「无候选」入口要按预筛余量和残差给出不同结论。"""
    from robotkinematics.core.workspace import ReachEstimate

    robot, task = make_task(TARGET_REACHABLE)
    inside = ReachEstimate(
        robot_name=robot.name,
        direction=np.array([0.0, 0.0, 1.0]),
        radius=10.0,
        samples=100,
        seed=0,
        within_limits=True,
    )
    outside = ReachEstimate(
        robot_name=robot.name,
        direction=np.array([0.0, 0.0, 1.0]),
        radius=0.01,
        samples=100,
        seed=0,
        within_limits=True,
    )
    target = robots.end_pose(robot, pick_uc.PickTask(observation=np.zeros(3)).home(7), robots.TCP)

    outcome, _ = pick_uc.classify_no_candidate(task, outside, target, 0.5, pick_uc.PickParams())
    assert outcome == pick_uc.OUTCOME_NOT_IN_WORKSPACE

    outcome, reason = pick_uc.classify_no_candidate(task, inside, target, 0.5, pick_uc.PickParams())
    assert outcome == pick_uc.OUTCOME_POSE_NOT_REACHABLE
    assert "换个抓取姿态" in reason

    outcome, _ = pick_uc.classify_no_candidate(task, inside, target, 1e-4, pick_uc.PickParams())
    assert outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE


def test_report_carries_outcome_steps_and_candidates():
    """成功的报告要一次带齐结论、七道工序、关节角和全部候选。"""
    _, task, result = run(TARGET_REACHABLE)
    report = pick_uc.to_report(task, result)
    assert report.fields["outcome"] == result.outcome
    assert report.fields["arm"] == "panda"
    assert len(report.fields["steps"]) == 7
    assert report.fields["q_deg"] is not None
    assert len(report.fields["candidates"]) == len(result.candidates)


def test_report_of_a_failure_has_no_solution():
    """失败的报告不能夹带解（q 为空）。"""
    _, task, result = run(TARGET_FAR)
    report = pick_uc.to_report(task, result)
    assert report.fields["q"] is None
    assert report.fields["outcome"] == pick_uc.OUTCOME_NOT_IN_WORKSPACE
