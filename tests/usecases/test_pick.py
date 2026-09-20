"""@file test_pick.py

@brief 抓取流水线：七道工序 + 七类结论（一类成功、六类失败），每类都要能复现。

运动学交给库之后，这里验的是**流水线本身**：编排、判定、择优、失败分类。
"""

from __future__ import annotations

import numpy as np

from robotkinematics.core import robots
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import pick as pick_uc

CAMERA = (0.30, 0.00, 0.60)
CAMERA_YAW = 30.0

# 快速参数：初值少、采样少，结论不变
FAST = pick_uc.PickParams(seeds=10, workspace_samples=2000)

# 实测过的三个"基准场景"
TARGET_REACHABLE = [0.2, 0.0, 0.0]  # 可达（自上而下抓）
TARGET_POSE_UNREACHABLE = [0.35, -0.05, 0.15]  # 位置够、姿态做不到
TARGET_FAR = [2.0, 2.0, 2.0]  # 够不着


def make_task(observation, *, arm="panda", rpy=(180.0, 0.0, 0.0), prefer=None):
    """按臂型组装机器人与抓取任务，让下面的用例只关心目标点和参数。"""
    if arm == "planar":
        robot, planar = robots.planar(), Planar2R()
    else:
        robot, planar = robots.panda(robots.TCP), None
    task = pick_uc.PickTask(
        observation=np.asarray(observation, dtype=float),
        camera_xyz=CAMERA,
        camera_yaw_deg=CAMERA_YAW,
        orientation_rpy_deg=rpy,
        arm=arm,
        planar=planar,
        prefer_config=prefer,
    )
    return robot, task


def run(observation, *, arm="panda", rpy=(180.0, 0.0, 0.0), params=FAST):
    """一次跑完整条流水线，并把机器人一并返回，好让断言能自己正运动学复核。"""
    robot, task = make_task(observation, arm=arm, rpy=rpy)
    return robot, task, pick_uc.pick(robot, task, params)


# ── 成功路径 ────────────────────────────────────────────────────────────────


def test_reachable_pick_returns_a_usable_solution():
    """可达目标的解要真能用：末端到位误差小于 1e-3，且理由里带上离限位余量。"""
    robot, task, result = run(TARGET_REACHABLE)
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert result.q is not None
    reached = np.asarray(robots.end_pose(robot, result.q, task.frame).t)
    assert np.linalg.norm(reached - np.asarray(result.target.t)) < 1e-3
    assert "离限位余量" in result.reason


def test_all_seven_steps_are_reported():
    """七道工序一道都不能少 —— 这是"项目"与"七个 demo"的分界线。"""
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
    """就近择优：不允许出现"关节转两圈"的解（真机没法执行）。"""
    _, _, result = run([0.35, -0.05, 0.15], arm="planar")
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    assert np.abs(result.q).max() < np.pi


def test_planar_solution_comes_from_the_closed_form():
    """二连杆的候选解来自闭式解 —— 库的数值 IK 给不出"肘上/肘下"两组。"""
    _, task, result = run([0.35, -0.05, 0.15], arm="planar")
    analytic = task.planar.ik(float(result.target.t[0]), float(result.target.t[1]))
    assert min(np.linalg.norm(result.q - np.asarray(sol)) for sol in analytic) < 1e-9


def test_planar_step_reports_the_closed_form_as_its_source():
    """二连杆的 IK 工序要如实标注解来自闭式解，而不是库的数值 IK。"""
    _, _, result = run([0.35, -0.05, 0.15], arm="planar")
    ik_step = next(s for s in result.steps if "多初值 IK" in s.name)
    assert "闭式解" in ik_step.numbers["source"]


def test_planar_task_ignores_orientation_and_says_so():
    """平面臂没有姿态自由度：第一道工序要写明姿态要求被忽略，不能假装约束都满足了。"""
    _, _, result = run([0.35, -0.05, 0.15], arm="planar")
    assert result.outcome == pick_uc.OUTCOME_REACHABLE
    # 第 ① 道工序要如实说明"z 与姿态要求被忽略"，而不是假装约束都满足了
    locate = result.steps[0].detail
    assert "忽略" in locate and "姿态要求" in locate


# ── 六类失败 ────────────────────────────────────────────────────────────────


def test_not_in_workspace_when_target_is_far_away():
    """(2,2,2) 这种远目标归为「不在工作空间」，并直说换初值没用、预筛余量为负。"""
    _, _, result = run(TARGET_FAR)
    assert result.outcome == pick_uc.OUTCOME_NOT_IN_WORKSPACE
    assert "换初值没用" in result.reason
    assert result.steps[1].numbers["margin"] < 0


def test_pose_not_reachable_when_only_orientation_fails():
    """位置在界内、只有姿态做不到时，要归为姿态不可达并建议换个抓取姿态。"""
    _, _, result = run(TARGET_POSE_UNREACHABLE)
    assert result.outcome == pick_uc.OUTCOME_POSE_NOT_REACHABLE
    assert result.steps[1].numbers["inside"] is True  # 位置在界内
    assert "换个抓取姿态" in result.reason


def test_same_position_with_another_orientation_succeeds():
    """上一条失败的原因确实是姿态：换个姿态就通了（验证分类没冤枉目标）。"""
    _, _, result = run(TARGET_POSE_UNREACHABLE, rpy=(0.0, 90.0, 0.0))
    assert result.outcome == pick_uc.OUTCOME_REACHABLE


def test_residual_too_large_when_tolerance_is_impossible():
    """容差压到 1e-14 时谁也过不了，应归为残差过大，而不是赖到「不可达」头上。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, residual_tol=1e-14)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE


def test_all_near_singular_when_sigma_threshold_is_high():
    """σmin 阈值抬到 100 后所有解都算近奇异 —— 失败分类要跟着阈值走。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, min_sigma=100.0)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_ALL_NEAR_SINGULAR


def test_all_limit_violated_when_margin_is_demanding():
    """要求离限位至少 170°（几乎不可能）→ 所有解都被限位淘汰。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, limit_margin_deg=170.0)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_ALL_LIMIT_VIOLATED
    assert all(not c.survived for c in result.candidates)


def test_micro_motion_infeasible_when_joint_budget_is_tiny():
    """关节单步上限压到 1e-6 后微动校验过不了，这一档也不该给出解（q 为空）。"""
    params = pick_uc.PickParams(seeds=6, workspace_samples=2000, max_joint_step_deg=1e-6)
    _, _, result = run(TARGET_REACHABLE, params=params)
    assert result.outcome == pick_uc.OUTCOME_MICRO_MOTION_INFEASIBLE
    assert result.q is None


def test_outcome_labels_are_unique_and_complete():
    """七类结论的标签必须互不重复且恰好七类（一类成功 + 六类失败），否则报告没法机器判定。"""
    assert len(set(pick_uc.OUTCOMES)) == len(pick_uc.OUTCOMES) == 7
    assert pick_uc.OUTCOME_REACHABLE in pick_uc.OUTCOMES


# ── 判定逻辑（不依赖具体目标，直接测分类器）────────────────────────────────


def test_classifier_separates_unreachable_from_imprecise():
    """同一个「无候选」入口要按预筛余量和残差给出不同结论；平面臂不能报姿态不可达。"""
    # 分类器是流水线对外的一部分（失败词汇的实现），所以是公开名，不带下划线
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

    # 平面机械臂没有姿态自由度：同样情形不能报"姿态不可达"
    _, planar_task = make_task(TARGET_REACHABLE, arm="planar")
    outcome, reason = pick_uc.classify_no_candidate(
        planar_task, inside, target, 0.5, pick_uc.PickParams()
    )
    assert outcome == pick_uc.OUTCOME_RESIDUAL_TOO_LARGE
    assert "姿态" not in reason


# ── 报告 ────────────────────────────────────────────────────────────────────


def test_report_carries_outcome_steps_and_candidates():
    """成功的报告要一次带齐结论、七道工序、关节角和全部候选，供 CLI 直接打印。"""
    _, task, result = run(TARGET_REACHABLE)
    report = pick_uc.to_report(task, result)
    assert report.fields["outcome"] == result.outcome
    assert len(report.fields["steps"]) == 7
    assert report.fields["q_deg"] is not None
    assert len(report.fields["candidates"]) == len(result.candidates)


def test_report_of_a_failure_has_no_solution():
    """失败的报告不能夹带解（q 为空），结论照实写「不在工作空间」。"""
    _, task, result = run(TARGET_FAR)
    report = pick_uc.to_report(task, result)
    assert report.fields["q"] is None
    assert report.fields["outcome"] == pick_uc.OUTCOME_NOT_IN_WORKSPACE


def test_planar_report_includes_the_ascii_chart():
    """二连杆的报告要多挂一个 ASCII 构型图块，好让人在终端里直接看姿态。"""
    _, task, result = run([0.35, -0.05, 0.15], arm="planar")
    report = pick_uc.to_report(task, result)
    assert "TwoLinkChartBlock" in {type(b).__name__ for b in report.blocks}
