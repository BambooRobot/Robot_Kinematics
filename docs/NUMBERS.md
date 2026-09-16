# 数值出处

项目里每个被断言的数值，都能追到出处。三类来源：

| 标记 | 含义 | 可信度 |
|---|---|---|
| **PPT** | 课件里印出来的算例，学生能手算复核 | 最高 |
| **课程日志** | 课件 `run_logs/*.log` 里记录的库输出 | 高 |
| **库** | 由 spatialmath / roboticstoolbox 算出 | 高（就是运动学本身） |
| **自洽** | 本项目自己推导的性质（二连杆闭式解、可达采样） | 由测试与库对照保证 |

> 换成调库之后，**验收方式变了**：不再"与库逐项对照"（那成了拿库跟自己比），
> 而是 **自研的那两处必须与库一致** + **课件算例必须复现**。

---

## 1. 位姿与坐标变换

| 数值 | 来源 | 断言在 |
|---|---|---|
| `R = Rz(yaw)·Ry(pitch)·Rx(roll)` ⇔ `SE3.RPY(…, order='zyx')` | **PPT**（课件 01 的 `rpy_to_R`） | `tests/core/test_robots.py`、`test_usecases.py` |
| 六元组 (0.4,0.2,0.3,0,0,90°) 下点 (0.1,0,0) → **(0.4, 0.3, 0.3)** | **PPT** | `test_usecases.py::test_pose_fields_match_the_course_point_transform` |
| 四元数顺序是 **(w,x,y,z)**（库的约定，与 ROS/Eigen 相反） | **库** | `test_usecases.py::test_pose_quaternion_is_wxyz` |
| `^base T_cup.t = [0.7830, 0.1634, 0.8]` | **课程日志** | `test_usecases.py::test_transform_chain_matches_the_course_log` |
| `change_x = [0.9562, 0.2634, 0.8]`、`change_y = [0.6330, 0.4232, 0.8]`、`yaw_zero = [0.8, -0.1, 0.8]` | **课程日志** | `test_usecases.py::test_batch_lines_reproduce_the_course_output` |

## 2. 二连杆（唯一保留的手推公式）

| 数值 | 来源 | 断言在 |
|---|---|---|
| `fk(0°,90°) = [1,1]`、`fk(30°,60°) = [0.866,1.5]`、`fk(60°,-30°) = [1.366,1.366]` | **PPT** | `tests/core/test_planar.py` |
| 闭式 FK 与**库的 `fkine` 逐位一致**（1e-12） | **自洽 + 库** | `test_planar.py::test_fk_agrees_with_the_library` |
| `ik(1,1)` → **(0°,90°)** 与 **(90°,-90°)** | **课程日志** | `test_planar.py::test_ik_returns_both_elbow_configurations` |
| `ik(3,0)` → 无解 | **课程日志** | `test_planar.py::test_ik_reports_no_solution_when_target_is_too_far` |
| `det(J) = L1·L2·sin(q2)` 等于**库的 `jacob0` 行列式** | **自洽 + 库** | `test_planar.py::test_det_jacobian_closed_form_matches_the_library` |
| `cond(J)` @ q2 = 90°/30°/10° → **2.62 / 9.36 / 28.58** | **课程日志**（课件 07 的表） | `tests/core/test_singularity.py` |
| 1/σ_min @ q2 = 90°/30°/10°/1° → **1.62 / 4.33 / 12.83 / 128.12** | **库**（SVD） | `docs/GUIDE.md` 第 8 章 |

## 3. Franka Panda（全部由库给出）

| 数值 | 来源 | 断言在 |
|---|---|---|
| 7 关节；雅可比形状 **(6, 7)** | **库** | `tests/core/test_panda.py` |
| 关节限位（q4 区间不含 0，所以"全零姿态"真机摆不出） | **库**（`robot.qlim`） | `test_panda.py::test_fk_matches_course_numbers` |
| 零位姿：法兰 **0.926** / 夹爪 TCP **0.8226**（差 103.4 mm） | **库 + 课程日志** | `test_robots.py::test_flange_and_tcp_differ_by_1034_millimetres` |
| `fk(ppt_goal)`：法兰 [0.4531, 0, 0.5619] / 夹爪 [0.4737, 0, 0.4606] | **库** | `test_panda.py::test_fk_matches_course_numbers` |
| 全零姿态是**奇异位形**（σ_min ≈ 1e-16） | **库 + 自洽** | `test_panda.py::test_zero_pose_is_singular` |
| 雅可比第 7 列线速度为 0（工具挂在第 7 关节轴上） | **库** | `test_panda.py::test_jacobian_last_column_is_pure_rotation` |
| 可操作度：本项目算的（numpy SVD）与库的 `manipulability()` **逐位一致** | **自洽 + 库** | `test_singularity.py::test_manipulability_matches_the_library` |
| FK→IK 往返位置残差 **< 1e-5**（库的 tol 是关节步长判据，见 GUIDE 7.4） | **库** | `test_panda.py::test_ik_round_trip_recovers_the_target` |

## 4. 抓取流水线（自研部分）

| 数值 | 来源 | 断言在 |
|---|---|---|
| 沿"杯子方向"可达 **1.0893 m**，目标 1.1313 m → 余量 **-0.0420 m**（够不着） | **自洽**（采样估计，固定 seed 可复现） | `tests/core/test_workspace.py` |
| 全方向最大可达 **1.19 m** ＞ 沿某方向 **1.09 m** —— 工作空间不是球 | **自洽** | `test_workspace.py::test_workspace_is_not_a_sphere` |
| 评分权重：限位余量 0.5 + σ_min 0.3 + 就近 0.2 | **自洽**（写在 `pick.SCORE_WEIGHTS`，可审计） | `tests/usecases/test_pick.py` |
| 七类结论（一类成功 + 六类失败）每类至少一条端到端测试 | **自洽** | `test_pick.py` |
| 二连杆流水线的解 = 闭式解（与解析解距离 < 1e-9） | **自洽** | `test_pick.py::test_planar_solution_comes_from_the_closed_form` |

## 5. 与课件批量输出的逐字符对照

`rkin batch` 的前两段与课件 `outputs/ppt_cases_batch_result.txt` **逐字符一致**
（仅文件末尾换行符不同；期望值内联在测试与 CI 里，不随仓库分发课件原件）：

```
1) 坐标变换：camera frame -> base frame
- ppt_case: cup_base=[0.783  0.1634 0.8   ]
- change_x: cup_base=[0.9562 0.2634 0.8   ]
- change_y: cup_base=[0.633  0.4232 0.8   ]
- yaw_zero: cup_base=[ 0.8 -0.1  0.8]

2) 二连杆 FK / IK / 奇异点
- fk_0_90: FK tool=[1. 1.], detJ=1.000000
- fk_30_60: FK tool=[0.866 1.5  ], detJ=0.866025
- fk_60_minus30: FK tool=[1.366 1.366], detJ=-0.500000
- ik_1_1: IK sols=(0.00°, 90.00°); (90.00°, -90.00°)
- ik_far: IK no solution
- singular: FK tool=[2. 0.], detJ=0.000000
```

第 3 段（Panda）是本项目补的；用**夹爪 TCP 帧**，所以零位姿是 `0.8226` —— 与课件日志同帧。

## 6. 换库过程中实测到的库行为（留档）

这些都写进了 `docs/GUIDE.md` 第 7 章，因为它们**只有跑起来才会暴露**：

| 现象 | 实测 |
|---|---|
| `ikine_LM` 的 `tol` 是**关节步长**判据 | 默认值 → 位姿残差 7.85e-4；`tol=1e-9` → 5.22e-7 |
| 平面机器人传 `mask` | 库内部维数不匹配，抛 `ValueError` |
| `robot.reach` | 对本项目模型返回 `0`（int），不可用 |
| `Planar2.a` | 只读属性，改杆长必须自建 `DHRobot` |
| `manipulability(method='asada')` | 某些位形返回 `nan` |
| `fkine(batch)` 的返回值 | `isinstance(poses, SE3)` 为 True、`.shape` 报 (4,4) —— 只能用 `.t` 的维数判 |
| `SE3.RPY(…, order='xyz')` vs `'zyx'` | 两套相反的约定；课件里两种都出现过 |
