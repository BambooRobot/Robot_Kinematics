# robot_kinematics_app (rkin)

**具身智能机器人学基础（第一章）全内容实现**：位姿描述 → 坐标变换链 → 二连杆 FK/IK/雅可比/奇异点 →
7 自由度 Franka Panda。核心数学全部自研（只需要 numpy），不依赖 `spatialmath` / `roboticstoolbox`。

> 📖 **第一次接触本项目 / 没学过机器人学？**
> 先读 [`docs/GUIDE.md`](docs/GUIDE.md) —— 面向初学者的从零指南，
> 包含推荐读码顺序、数学直觉、踩坑清单和动手实验。
> 本 README 只讲"怎么用"，GUIDE 讲"为什么这么写"。
>
> 📐 所有数值的出处见 [`docs/NUMBERS.md`](docs/NUMBERS.md)。
> 🗺️ 七个知识点分别落在哪 → [`docs/KNOWLEDGE_MAP.md`](docs/KNOWLEDGE_MAP.md)

## 这个项目解决什么问题

课件里那套实验（`0625robotics_python_lab_cloud_terminal`）把 FK/IK/雅可比的公式在 4 个脚本里
各抄了一份，Panda 部分直接调库，`panda_cases.csv` 声明了却没有任何脚本读它。
本项目把同样的内容重写成一套分层、可测试、可出图的工程，并且**每个数都是自己算的**：

```text
cli/                      组合根：造适配器 → 调用例 → 交给渲染器
  parser                  参数定义（只有开关，没有业务）
  commands                装配 + 派发（想知道某条命令用了哪些实现，只看这里）
  main                    渲染 + 唯一的异常出口

usecases/                 用例：编排 + 产出 Report（不认识终端/文件/matplotlib）
  pick                    ★ 抓取任务流水线：七道工序 + 六类失败结论（把七个知识点串起来）
  pose                    位姿的几种表示互转（对应课件 01）
  transforms              相机目标 → 机器人本体（对应课件 02）
  planar                  二连杆 FK/IK/雅可比/奇异点（对应课件 03~07）
  panda                   Panda FK/IK/雅可比/冗余（对应课件 08~10）
  batch                   批量算例报告（对应课件 11，并补上 Panda 段）
  compare                 与第三方库交叉验证的判定逻辑
  env_check               环境自检（对应课件 00）

contracts.py              契约：Report / 报告块 + 4 个协议
                          Renderer · CaseSource · Plotter · KinematicsReference

adapters/                 唯一认识外部世界的地方（实现上面的协议）
  config_yaml             YAML 配置：未知项告警 + 范围校验 + 命令行覆盖
  cases_csv               读 data/cases/*.csv
  render_text             终端文本（中文宽度对齐、表格、分节线）
  render_json             同一份 Report 输出 JSON
  plot_mpl                matplotlib 出图 + 3D 不可用时的三视图回退
  reference_rtb          第三方库取数（唯一 import 那两个库的文件）

core/                     纯数学，零 IO、只依赖 numpy
  rotations  se3  planar2r  chain  panda  ik_solvers  singularity  numerics
  workspace               工作空间采样估计（可达性预筛：工作空间不是球）
  exceptions
```

**依赖规则**（只允许这几条）：

| 从 | 到 | 说明 |
|---|---|---|
| `cli` | 全部 | 组合根，负责装配 |
| `usecases` | `core` / `contracts` | **不允许**直接依赖 `adapters` |
| `adapters` | `contracts` / `core` / 第三方库 | 实现协议 |
| `core` | 只有 numpy | 数学 |

用例产出的是 `Report`（结构化），所以同一份结果既能走终端，也能 `rkin --format json` 出 JSON。


## 明确不做

| 不做 | 原因 |
|---|---|
| 真机控制、实时回路 | 那是控制器的事，不是运动学的事 |
| 动力学、力控 | 本章不涉及 |
| 真视觉 | 观测是**输入**（与脸门项目划清界限） |
| 避障与运动规划 | 只回答"这个位姿解不解得出来"，不回答"怎么绕过去" |
| 相机标定 | 外参是给定常数 |

## 依赖

- Python ≥ 3.10
- numpy < 2（与课件一致；仅 `crosscheck` 需要这个约束）
- matplotlib ≥ 3.5、PyYAML ≥ 6（运行时）
- 开发工具（ruff / mypy / pytest / coverage）：`pip install -e ".[dev]"`
- **可选**：spatialmath-python、roboticstoolbox-python（只有 `rkin crosscheck` 用得到）：
  `pip install -e ".[crosscheck]"`

> 依赖只声明在 **`pyproject.toml` 一处**（`dependencies` + `optional-dependencies`），
> 不另外维护 requirements 文件 —— 两份清单必然漂移。

## 安装与运行

```bash
cd robot_kinematics_app
pip install -e .            # 依赖 + 命令行入口 rkin

# 方式一：不安装，直接跑（推荐先这样试）
PYTHONPATH=src python3 -m robotkinematics check

# 方式二：安装成命令 `rkin`
pip install -e .
rkin check
```

## 用法

**一条主线**（把七个知识点串成一个任务）：

```bash
rkin pick --observe 0.2 0 0                   # 抓取：观测 → 关节解（含失败分类）
rkin pick --observe 0.2 0 0 --plot            # 同一件事，再给一张任务图
rkin pick --observe 0.2 0 0 --arm planar      # 二连杆跑同一条流水线（教学对照）
rkin --format json pick --observe 0.2 0 0     # 同一份结果输出 JSON
```

**按知识点逐个验证**：

```bash
rkin check                                    # 环境自检
rkin pose --xyz 0.4 0.2 0.3 --rpy 0 0 90      # 六元组 → R/T，RPY/轴角/四元数互转
rkin cam --case ppt_case                      # 相机目标 → 本体（--all 跑整张算例表）
rkin fk  --q1 0 --q2 90                       # 二连杆 FK（字符画 + 数值）
rkin ik  --target 1 1                         # 两组解 + FK 回验
rkin jac --q1 0 --q2 90 --dx 0 0.1            # J、dq、det、cond
rkin sing                                     # 奇异点扫描表 → outputs/
rkin panda-fk  --case ppt_goal [--plot]       # Panda FK（--q 也可直接给 7 个关节角）
rkin panda-fk  --case zero --frame tcp        # 换成夹爪 TCP 帧（与课件日志一致）
rkin panda-ik  --case ppt_goal                # 数值 IK + FK 回验 + 冗余演示
rkin panda-jac --case ik_seed                 # 6×7 几何雅可比 + 奇异分析
rkin panda-sing                               # 三个典型姿态的运动能力对比
rkin anim --sweep q2 --frames 60              # 关节扫动 GIF
rkin batch                                    # 批量算例报告 → outputs/
rkin crosscheck                               # 需可选依赖，未安装会说明“已跳过”
rkin --format json panda-jac                  # 同一份报告输出 JSON（供别的程序调用）
```

配置在 `configs/default.yaml`。**命令行参数优先级高于配置文件**，且 `--l1` 这类开关
写在子命令前后都可以：

```bash
rkin --l1 2 --l2 3 fk --q1 0 --q2 0
rkin fk --l1 2 --l2 3 --q1 0 --q2 0     # 等价
```

程序按相对路径查找 `configs/` 与 `data/`，从项目根目录运行最省事。

## 验证

```bash
make test-fast            # 6 秒跑完大部分断言（跳过 slow 标记的长搜索用例）
make check-all            # 提交前跑这条：ruff + mypy + pytest
bash scripts/run_all.sh   # 跑完全部命令，产物落到 outputs/
rkin crosscheck           # 可选：装了真库才生效
```

判据（详见 `docs/NUMBERS.md`）：
- `rkin batch` 的前两段与课件 `outputs/ppt_cases_batch_result.txt` **逐字符一致**；
- 自研雅可比与数值微分的偏差 < 1e-9；ETS 与 MDH 两套参数化的 FK 逐位一致；
- Panda 的 FK↔IK 往返残差 < 1e-9；
- `rkin crosscheck`（装可选依赖后）与 roboticstoolbox 逐项对照，偏差 **0.00e+00 ~ 1e-16**；
- `rkin pick` 的六类失败结论各有端到端测试；二连杆解与闭式解逐位一致。

## 已知环境问题

若 `--plot` 输出的是三视图而不是 3D 图，说明本机 matplotlib 的 3D 模块不可用
（常见于系统与 pip 各装了一份 matplotlib，老的 `nspkg.pth` 把 `mpl_toolkits` 指到了老版本目录）。
命令会在 stderr 说明原因；修法是卸掉系统的 `python3-matplotlib` 或删除
`/usr/lib/python3/dist-packages/matplotlib-*-nspkg.pth`，或改用虚拟环境。
两种输出用的是同一份数据，三视图按正交投影画，数值不受影响。

⚠️ 同一个原因还会让 **`spatialmath` 装不上/导入失败**（它依赖 matplotlib 的 3D 模块，
报 `cannot import name 'plotvol3'`）。也就是说 `rkin crosscheck` 在本机需要先修好上面这一条。
本项目不依赖这两个库，所以不影响其他任何功能。
