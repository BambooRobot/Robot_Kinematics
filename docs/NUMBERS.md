# 数值清单：每个数从哪来、凭什么算对

本项目的每一个验收数值都能追到出处。三类来源，可信度递减：

| 标记 | 含义 |
|---|---|
| **PPT** | 课件里印出来的算例，学生可以手算复核 |
| **课程日志** | 课件 `run_logs/*.log` 里记录的库输出，来自讲师本机 |
| **自洽** | 本项目自己推导的性质（如三角不等式上界、ETS/MDH 等价），用内部一致性验证 |

这些数值全部写进了 `tests/`，`pytest -q` 会逐条核对。

---

## 1. 位姿与坐标变换

| 数值 | 来源 | 出现在 |
|---|---|---|
| `R = Rz(yaw)·Ry(pitch)·Rx(roll)` | **PPT**（课件 01 的 `rpy_to_R`） | `test_rotations.py::test_rpy_convention_is_rz_ry_rx` |
| 六元组 (0.4, 0.2, 0.3, 0, 0, 90°) 下点 (0.1,0,0) → **(0.4, 0.3, 0.3)** | **PPT**（课件 01） | `test_se3.py::test_point_transform_matches_ppt_case` |
| `^base T_cup.t = [0.7830, 0.1634, 0.8]` | **课程日志**（`ppt_cases_batch_result.txt`） | `test_applications.py::test_transform_chain_case_matches_the_course_log` |
| `change_x = [0.9562, 0.2634, 0.8]`、`change_y = [0.6330, 0.4232, 0.8]`、`yaw_zero = [0.8, -0.1, 0.8]` | **课程日志** | `test_applications.py::test_batch_report_reproduces_the_course_output` |
| 万向锁时约定 roll = 0、角度全部记在 yaw 上 | **自洽**（重建出的 R 必须一致） | `test_rotations.py::test_rpy_gimbal_lock_puts_all_angle_on_yaw` |

## 2. 二连杆

| 数值 | 来源 | 出现在 |
|---|---|---|
| `fk(0°,90°) = [1, 1]` | **PPT** | `test_planar_fk.py` |
| `fk(30°,60°) = [0.866, 1.5]`、`fk(60°,-30°) = [1.366, 1.366]` | **课程日志** | `test_planar_fk.py` |
| `joint_points(45°,45°)`：elbow `[0.7071, 0.7071]`、tool `[0.7071, 1.7071]` | **课程日志**（课件 04） | `test_planar_fk.py::test_joint_points_match_course_ascii_demo` |
| `ik(1,1)` → **(0°, 90°)** 与 **(90°, -90°)** | **课程日志**（课件 05） | `test_planar_ik.py::test_ik_returns_both_elbow_configurations` |
| `ik(3,0)` → 无解 | **课程日志**（课件 05） | `test_planar_ik.py::test_ik_reports_no_solution_when_target_is_too_far` |
| `J(0°,90°) = [[-1,-1],[1,0]]` | **PPT** | `test_planar_jacobian.py::test_jacobian_matches_ppt_value` |
| `solve(J, [0,0.1]) = [0.1, -0.1] rad = [5.73°, -5.73°]` | **PPT** | `test_planar_jacobian.py::test_solve_step_reproduces_ppt_example` |
| `det(J) = L1·L2·sin(q2)` → 1.0 / 0.866025 / -0.5 / 0 | **PPT** | `test_planar_singularity.py` |
| `cond(J)` @ q2 = 90°/30°/10°/0° → **2.62 / 9.36 / 28.58 / inf** | **课程日志**（课件 07 的表） | `test_planar_singularity.py::test_condition_number_matches_course_table` |
| 雅可比第 i 列 = 该关节绕自身轴的线速度贡献 | **自洽**（与数值微分逐项对照） | `test_planar_jacobian.py::test_jacobian_matches_numeric_differentiation` |

## 3. Franka Panda

| 数值 | 来源 | 出现在 |
|---|---|---|
| 关节数 n = 7，雅可比 6×7 | **PPT** | `test_panda_jacobian.py::test_jacobian_shape_is_6_by_7` |
| 连杆几何 0.333 / 0.316 / 0.384 / 0.0825 / 0.088 / 0.107 m | **自洽**（Franka 官方 URDF 的关节 origin） | `test_panda_fk.py::test_zero_pose_flange_height_is_the_sum_of_link_offsets` |
| 零位姿法兰高度 = 0.333+0.316+0.384 = **1.033 m**，法兰 z = **0.926 m** | **自洽** + **真库**（见下） | 同上 + `test_zero_pose_tool_is_107mm_below_the_flange` |
| 夹爪 TCP = 法兰 + **103.4mm** 并绕 z 转 **-45°** | **真库实测**（roboticstoolbox 1.4.3） | `test_panda_fk.py::test_hand_tcp_offset_is_a_fixed_transform` |
| 课件日志的零位姿 `t = [0.088, 0, 0.8226]` | **课程日志**（夹爪 TCP 帧） | `test_panda_fk.py::test_hand_tcp_frame_reproduces_the_course_log` |
| `fk(ppt_goal).t = [0.4531, 0, 0.5619]`（法兰帧） | **自洽** + **真库** | `test_panda_fk.py::test_ppt_goal_pose_position` |
| **ETS 与 MDH 两套参数化 FK 逐位相等** | **自洽**（两套独立实现互为交叉验证） | `test_panda_fk.py::test_ets_and_mdh_give_identical_fk` |
| 几何雅可比 = 数值微分（< 1e-6） | **自洽** | `test_panda_jacobian.py::test_jacobian_matches_numeric_differentiation` |
| 雅可比第 7 列：线速度 0、角速度 = 第 7 关节 z 轴 | **自洽** | `test_panda_jacobian.py::test_last_column_is_a_pure_rotation_about_the_last_joint_axis` |
| 全零姿态是**奇异位姿**（σ_min ≈ 1e-17） | **自洽**（SVD） | `test_panda_jacobian.py::test_zero_pose_is_singular` |
| 全零姿态越过了 q4 的关节限位 `[-3.0718, -0.0698]` | **自洽**（Franka 数据表限位） | `test_panda_fk.py::test_joint_limits_flag_the_mathematically_impossible_zero_pose` |
| 7 自由度存在 1 维零空间（不动末端也能动关节） | **自洽** | `test_panda_jacobian.py::test_redundancy_shows_up_as_a_nonzero_null_space` |
| FK→IK 往返残差 < 1e-9 | **自洽** | `test_panda_ik.py` |

### ✅ 已结案：课件日志那 0.103 m 的差异（原本是待核项）

**现象**：课件 `run_logs/08_franka_panda_fk.log` 的零位姿是 `t = [0.088, 0, 0.8226]`，
本项目按 Franka 官方 URDF 建模却得 `[0.088, 0, 0.926]` —— 位置差 0.103 m，
旋转还差一个绕 z 的 45° 耦合。

**结论（roboticstoolbox 1.4.3 实测判定，`rkin crosscheck` 可复现）**：

```
课件日志记录          t = [0.0880, 0.0000, 0.8226]
本项目 · 法兰帧       t = [0.0880, 0.0000, 0.9260]
本项目 · 夹爪 TCP 帧  t = [0.0880, 0.0000, 0.8226]   ← 与课件日志逐位一致
真库 · panda_link8    t = [0.0880, 0.0000, 0.9260]
真库 · 默认（夹爪）    t = [0.0880, 0.0000, 0.8226]
```

**差异来源是末端帧的定义**：roboticstoolbox 的 Panda 模型自带一个夹爪（`panda_hand`），
`robot.fkine(q)` 默认返回**夹爪 TCP** = 法兰 + 103.4mm + 绕 z 转 45°（Franka 官方 TCP 约定），
不是法兰 `panda_link8`。课件的日志用的是默认末端，所以是 0.8226。

**两边的模型其实完全一致**：本项目与真库在
法兰帧、夹爪 TCP 帧、以及两个帧下的雅可比，逐项偏差都是 `0.00e+00` 或 `1e-16` 量级。

最初我判断"日志与它自己打印的 ETS 表不自洽"是**错的** —— ETS 表描述的是机器人本体
（到法兰），`fkine` 静默地又接了夹爪偏移。这条教训已写进 GUIDE 第 7 章。
想直接看夹爪帧：`rkin panda-fk --case zero --frame tcp`。

## 4. 与课件批量输出的逐字符对照

`rkin batch` 的前两段与课件 `outputs/ppt_cases_batch_result.txt` **逐字符一致**
（仅文件末尾换行符不同）：

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

第 3 段是本项目补上的（课件的 `panda_cases.csv` 声明了却没有任何脚本读它）。

## 5. 本项目自己抓出来的两个问题（留档）

1. **`matrix_to_axang` 的数值精度**：最初用 `angle = arccos((tr-1)/2)`，在**小转角**时条件数爆炸
   （arccos 在自变量趋近 1 时导数无穷大），导致 IK 里的角速度误差项偏差约 1%。
   是"解析雅可比 vs 数值微分"这条测试抓出来的（偏差 3e-2），改用
   `atan2(‖反对称部‖/2, (tr-1)/2)` 后降到 9e-10。
2. **阻尼伪逆不能用来做零空间投影**：`J·(I − J_damped⁺J) ≠ 0`，会让次要任务偷偷扰动主任务，
   IK 在目标附近卡住不收敛（残差停在 2e-2）。换成未阻尼的 `pinv` 后正常收敛。

这两个坑都写进了代码注释 —— 它们是这类项目最容易踩、又最难靠肉眼发现的那一类错误。
