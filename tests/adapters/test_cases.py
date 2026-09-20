"""@file test_cases.py

@brief 算例文件读取：列名与课件一致、FK/IK 分流、坏数据要报错。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.cases_csv import (
    load_coordinate_cases,
    load_panda_cases,
    load_two_link_cases,
)
from robotkinematics.adapters.config_yaml import Config
from robotkinematics.core.exceptions import KinematicsError


@pytest.fixture
def cases_dir():
    """算例目录从配置里取，测试不写死路径。"""
    return Config.from_yaml().cases_dir_path


def test_coordinate_cases_match_the_course_dataset(cases_dir):
    """coordinate_cases.csv 的列名与顺序要和课件一致，含 ppt 那条基准算例。"""
    cases = load_coordinate_cases(cases_dir)
    assert [c.case_id for c in cases] == ["ppt_case", "change_x", "change_y", "yaw_zero"]
    ppt = cases[0]
    assert ppt.cup_xyz == (0.50, -0.10, 0.20)
    assert ppt.camera_xyz == (0.30, 0.00, 0.60)
    assert ppt.camera_yaw_deg == 30.0


def test_two_link_cases_split_into_fk_and_ik(cases_dir):
    """kind 列决定走正解还是逆解：fk 只给角度，ik 只给目标点。"""
    cases = {c.case_id: c for c in load_two_link_cases(cases_dir)}
    assert cases["fk_0_90"].kind == "fk"
    assert cases["fk_0_90"].q_deg == (0.0, 90.0)
    assert cases["fk_0_90"].target is None
    assert cases["ik_1_1"].kind == "ik"
    assert cases["ik_1_1"].target == (1.0, 1.0)
    assert cases["singular"].q_deg == (0.0, 0.0)


def test_panda_cases_are_seven_dimensional(cases_dir):
    """panda 算例必须是 7 维关节角，课件里没人读的 csv 本项目也接进来。"""
    cases = {c.case_id: c for c in load_panda_cases(cases_dir)}
    assert set(cases) == {"zero", "ppt_goal", "ik_seed"}
    assert cases["zero"].q.shape == (7,)
    assert np.allclose(cases["zero"].q, 0.0)
    assert np.allclose(cases["ppt_goal"].q[1], -0.4)
    # panda_cases.csv 在课件里没有任何脚本读它，本项目把它接进来了
    assert cases["ppt_goal"].description


def test_row_without_joint_angles_or_target_is_rejected(tmp_path):
    """既没关节角也没目标点的行必须报错，不能被当成合法算例放过。"""
    path = tmp_path / "two_link_cases.csv"
    path.write_text(
        "case_id,q1_deg,q2_deg,target_x,target_y,L1,L2,description\n"
        "broken,,,,,1.0,1.0,既没角度也没目标\n",
        encoding="utf-8",
    )
    with pytest.raises(KinematicsError) as excinfo:
        load_two_link_cases(tmp_path)
    assert "FK" in str(excinfo.value) or "IK" in str(excinfo.value)


def test_non_numeric_value_is_reported_with_context(tmp_path):
    """坏数据的报错要带上列名，用户才知道该改哪一格。"""
    path = tmp_path / "coordinate_cases.csv"
    path.write_text(
        "case_id,cup_x_camera,cup_y_camera,cup_z_camera,camera_x_base,camera_y_base,"
        "camera_z_base,camera_yaw_deg,description\n"
        "bad,0.5,abc,0.2,0.3,0.0,0.6,30,坏数据\n",
        encoding="utf-8",
    )
    with pytest.raises(KinematicsError) as excinfo:
        load_coordinate_cases(tmp_path)
    assert "cup_y_camera" in str(excinfo.value)


def test_utf8_bom_and_blank_lines_are_handled(tmp_path):
    """课件 CSV 带 BOM，还有空行 —— 都得能读。"""
    path = tmp_path / "coordinate_cases.csv"
    path.write_text(
        "﻿case_id,cup_x_camera,cup_y_camera,cup_z_camera,camera_x_base,"
        "camera_y_base,camera_z_base,camera_yaw_deg,description\n"
        "only,0.5,-0.1,0.2,0.3,0.0,0.6,30,唯一一条\n"
        "\n",
        encoding="utf-8",
    )
    cases = load_coordinate_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "only"


def test_missing_file_is_reported(tmp_path):
    """算例文件不存在时要给中文提示，而不是抛底层 FileNotFoundError。"""
    with pytest.raises(KinematicsError) as excinfo:
        load_panda_cases(tmp_path)
    assert "找不到算例文件" in str(excinfo.value)
