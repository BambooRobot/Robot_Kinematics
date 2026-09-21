# robot_kinematics_app (rkin)

**一条 Franka Panda 抓取任务流水线**：把机器人运动学的七个知识点串成七道工序 ——
观测 → 可达性预筛 → 多初值 IK → 雅可比体检 → 限位校验 → 解择优 → 微动校验，
**并且会告诉你为什么失败**（六类失败结论，有名字、有数字）。

入口是扁平 CLI：`rkin --observe X Y Z`（无子命令）。运动学（FK / IK / 雅可比 / 关节限位）由
`spatialmath` + `roboticstoolbox` 提供；本项目写的是"用它们搭一条任务流水线"——可达性预筛、
失败分类、解择优、微动校验，库都不提供。

> 📖 **第一次接触本项目 / 没学过机器人学？**
> 先读 [`docs/GUIDE.md`](docs/GUIDE.md) —— 面向初学者的从零指南，
> 包含推荐读码顺序、数学直觉、踩坑清单和动手实验。
> 本 README 只讲"怎么用"，GUIDE 讲"为什么这么写"。
>
> 📐 所有数值的出处见 [`docs/NUMBERS.md`](docs/NUMBERS.md)。
> 🗺️ 七个知识点分别落在哪 → [`docs/KNOWLEDGE_MAP.md`](docs/KNOWLEDGE_MAP.md)

## 这个项目解决什么问题

课件里那套实验是一批扁平脚本：每一步彼此孤立 —— 算了 FK 不会用去判可达，解了 IK
不知道这组解能不能执行。本项目把它们重写成一套分层、可测试的工程，核心是**一条流水线**：

```text
main.py                   ★ 唯一入口 = 组合根：
                          ① 装配（唯一一处"选实现"）
                          ② 入口流程（解析→装配→跑流水线→渲染）
                          ③ 扁平参数定义（--observe 等）
                          想知道"这个程序用到哪些实现"，只看这一个文件

usecases/                 用例：只做抓取流水线（依赖靠参数传入，不认识终端/文件）
  pick                    ★ 门面：工序序列 + 执行器
  pick_types                └ 词汇与数据契约：结论常量、Task/Params/Candidate/Result
  pick_steps                └ 七道工序：每道一个函数，失败时自己归类
  pick_report               └ 报什么（Report 的结构；怎么画是适配器的事）

contracts.py              契约：Report / 报告块 + 协议（Renderer · Plotter · …）

adapters/                 唯一认识外部世界的地方
  config_yaml             YAML 配置：未知项告警 + 范围校验 + 命令行覆盖
  render_text             终端文本
  render_json             同一份 Report 输出 JSON
  plot_mpl                matplotlib 出图 + 3D 不可用时的三视图回退

core/                     库覆盖不到的那部分
  robots                  从库建 Franka Panda：末端帧、关节限位、link 位置
  workspace ★             可达性预筛（库没有这个 API）
  singularity             SVD 体检 + 库的 manipulability
  exceptions              领域异常
```

**依赖规则**（只允许这几条，**由 `tests/test_architecture.py` 守着 —— 违反了会红**）：

| 从 | 到 | 说明 |
|---|---|---|
| `main.py` | 全部 | 组合根：**唯一**允许"造具体实现"的地方 |
| `usecases` | `core` / `contracts` | **不允许**直接依赖 `adapters` |
| `adapters` | `contracts` / `core` / 第三方库 | 实现协议 |
| `core` | numpy + 运动学库 | 只做数学与模型构造，不认识契约、不做 IO |

还有一条不是靠 import 而是靠"读代码顺序"的规矩：**除了 `main.py`，任何文件里都不该出现
"造一个具体实现"** —— 于是"换成 JSON 渲染器""换个出图器"都只改 `build_context` 一处。

用例产出的是 `Report`（结构化），所以同一份结果既能走终端，也能 `rkin --format json` 出 JSON。

## 明确不做

| 不做 | 原因 |
|---|---|
| 真机控制、实时回路 | 那是控制器的事，不是运动学的事 |
| 动力学、力控 | 本章不涉及 |
| 真视觉 | 观测是**输入** |
| 避障与运动规划 | 只回答"这个位姿解不解得出来"，不回答"怎么绕过去" |
| 相机标定 | 外参是给定常数 |
| 其它机型 / 平面臂 | 本仓库只服务 Franka Panda |

## 依赖

- Python ≥ 3.10
- **运动学库**：`spatialmath-python` + `roboticstoolbox-python`（FK / IK / 雅可比 / 限位）
- numpy < 2（roboticstoolbox 在 numpy 2 下会报 `_ARRAY_API not found`）
- matplotlib ≥ 3.5、PyYAML ≥ 6
- 开发工具（ruff / mypy / pytest / coverage）：`pip install -e ".[dev]"`

> 依赖只声明在 **`pyproject.toml` 一处**（`dependencies` + `optional-dependencies`），
> 不另外维护 requirements 文件 —— 两份清单必然漂移。

## 安装与运行

```bash
cd robot_kinematics_app
pip install -e .            # 依赖 + 命令行入口 rkin

# 方式一：不安装，直接跑（推荐先这样试）
PYTHONPATH=src python -m robotkinematics --observe 0.2 0 0

# 方式二：安装成命令 `rkin`
pip install -e .
rkin --observe 0.2 0 0
```

## 用法

扁平入口，**没有子命令**：

```bash
rkin --observe 0.2 0 0                              # 抓取：观测 → 关节解（含失败分类）
rkin --observe 0.2 0 0 --plot                       # 同一件事，再给一张任务图
rkin --observe 0.2 0 0 --format json                # 同一份结果输出 JSON
rkin --observe 0.2 0 0 --mode base                  # 观测已是本体坐标，跳过相机变换
rkin --observe 0.35 -0.05 0.15 --rpy 0 90 0          # 换抓取姿态（默认 180 0 0 = 自上而下）
rkin --observe 0.2 0 0 --prefer-config 0 -0.3 0 -2.2 0 2.0 0.8
rkin --observe 0.2 0 0 --seeds 32                   # 覆盖多初值个数
```

配置在 `configs/default.yaml`。**命令行参数优先级高于配置文件**。
程序按相对路径查找 `configs/` 与 `data/`，从项目根目录运行最省事。

## 验证

```bash
make test-fast            # 快速跑完大部分断言（跳过 slow 标记的长搜索用例）
make check-all            # 提交前跑这条：ruff + mypy + pytest
bash scripts/run_all.sh   # 跑演示流水线，产物落到 outputs/
```

判据（详见 `docs/NUMBERS.md`）：
- Panda 的 FK↔IK 往返残差 < 1e-5（库的 tol 是关节步长判据，换算到位姿就是这个量级）；
- 可达性采样给出方向相关边界（工作空间不是球）；
- `rkin --observe …` 的六类失败结论各有端到端测试。

## 已知环境问题

若 `--plot` 输出的是三视图而不是 3D 图，说明本机 matplotlib 的 3D 模块不可用
（常见于系统与 pip 各装了一份 matplotlib，老的 `nspkg.pth` 把 `mpl_toolkits` 指到了老版本目录）。
命令会在 stderr 说明原因；修法是卸掉系统的 `python3-matplotlib` 或删除
`/usr/lib/python3/dist-packages/matplotlib-*-nspkg.pth`，或改用虚拟环境。
两种输出用的是同一份数据，三视图按正交投影画，数值不受影响。

⚠️ **同一个原因会让本项目跑不起来**：`spatialmath` 导入时会连带 matplotlib 的 3D 模块，
在这个环境下直接报 `cannot import name 'plotvol3'`。而运动学现在由它提供 ——
所以这条不是"影响出图"，而是**必须先修好才能运行**。
