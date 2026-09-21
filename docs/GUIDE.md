# robot_kinematics_app 从零理解指南

> 面向**会一点 Python、但没系统学过机器人学**的读者。
> 目标：读完之后，你能独立看懂这个项目的每一行代码，能自己改参数做实验，
> 并且知道**用运动学库时哪些坑会让人算错**（第 7 章是这份文档最值钱的部分）。

---

## 目录

- [第 0 章 · 这个项目是什么](#第-0-章--这个项目是什么)
- [第 1 章 · 先补三个概念](#第-1-章--先补三个概念)
- [第 2 章 · 先跑起来](#第-2-章--先跑起来)
- [第 3 章 · 代码地图](#第-3-章--代码地图)
- [第 4 章 · 一次抓取的一生](#第-4-章--一次抓取的一生)
- [第 5 章 · 逐层读代码（推荐顺序）](#第-5-章--逐层读代码推荐顺序)
- [第 6 章 · 库的用法速查](#第-6-章--库的用法速查)
- [第 7 章 · 库的坑（★ 实测踩出来的）](#第-7-章--库的坑-实测踩出来的)
- [第 8 章 · 动手实验](#第-8-章--动手实验)
- [第 9 章 · 卡住时怎么办](#第-9-章--卡住时怎么办)
- [附录 · 一分钟速查](#附录--一分钟速查)

---

## 第 0 章 · 这个项目是什么

对应课程**第一章《具身智能机器人学基础》**。课件那套实验是一批扁平脚本：每一步彼此孤立 ——
算了 FK 不会用去判可达，解了 IK 不知道这组解能不能执行。

本项目重写成一条 **Franka Panda 抓取任务流水线**：七道工序，每道对应一个知识点，
**并且会告诉你为什么失败**。入口是扁平 CLI：`rkin --observe X Y Z`（无子命令）。

| 谁做什么 | 内容 |
|---|---|
| **库负责**（spatialmath + roboticstoolbox） | FK、IK（数值）、雅可比、关节限位、可操作度、SE3 表示 |
| **本项目负责** | 可达性预筛、多初值搜索、解择优、微动校验、失败分类、一条命令的编排 |
| **仍然自己写** | **可达性采样**（库没有这个 API；`robot.reach` 实测返回 0） |

⚠️ 这个定位很重要：本项目**不是**"自己实现运动学"，而是"**用运动学库搭一条任务流水线**"。
所以代码里没有推导公式，取而代之的是"库怎么用、它的坑在哪"——后者才是这份文档的重点。

---

## 第 1 章 · 先补三个概念

### 1.1 位姿 = 位置 + 姿态；坐标系回答"相对谁"

说"杯子在 (0.5, -0.1, 0.2)"是没意义的 —— **相对谁**？

- `camera_link`：相机看到目标时用的坐标系
- `base_link`：机器人本体的基座坐标系（机械臂规划认这个）
- `panda_link8` / 夹爪 TCP：末端的两种定义（差一个固定变换，见第 7 章）

**位姿**比位置多一层：不只"在哪里"，还有"朝哪边"。数学上打包成 4×4 矩阵（`SE3`），
核心用途只有一个：把"B 坐标系下的量"换成"A 坐标系下的量"，写作 `^A T_B`。

### 1.2 FK 与 IK：一个正向、一个反向

| | 输入 | 输出 | 何时用 |
|---|---|---|---|
| **FK** | 每个关节的角度 q | 末端位姿 | 想知道"手现在在哪" |
| **IK** | 目标位姿 | 关节角 q | 想抓东西 |

必须建立的直觉：**FK 永远有唯一答案，IK 不一定**。

- IK 可能**多解**：同一末端位姿对应多组关节角
- IK 可能**无解**：目标太远或太近，物理上够不着
- IK 可能**无穷多解**：Panda 有 7 个关节、末端只有 6 个自由度 → **冗余**

### 1.3 雅可比：末端微调 ↔ 关节微调

手臂伸得笔直时，让末端再往外挪一点 —— 关节几乎动不了。这就是**奇异点**。
雅可比 J 描述这个局部关系：`Δx ≈ J(q) · Δq`。

- **只在小步时成立**（当前姿态下的线性近似）
- **依赖当前姿态**：姿态变了 J 就变
- Panda 的 J 是 **6×7**（末端 6 个自由度、7 个关节）
- `σ_min = 0`（或接近 0）就是奇异位姿：某个方向的瞬时运动能力**丧失**了

---

## 第 2 章 · 先跑起来

```bash
cd robot_kinematics_app
pip install -e .            # 会把运动学库一起装上

make pick                   # 抓取流水线：一条命令跑完七道工序 + 出任务图
# 或：PYTHONPATH=src python -m robotkinematics --observe 0.2 0 0 --plot
```

`make pick` 会打印一份**任务结论**：结论（可执行 / 六类失败之一）、原因、
七道工序各自的数字、关节解、候选解对比。图上四样东西：可达边界（采样估计）、
机械臂姿态、末端路径、沿途的奇异点标记。

**几条开关看完全貌**：

```bash
rkin --observe 0.2 0 0 --plot
rkin --observe 0.2 0 0 --format json
rkin --observe 0.35 -0.05 0.15 --rpy 0 90 0
rkin --observe 0.2 0 0 --mode base
rkin --observe 0.2 0 0 --prefer-config 0 -0.3 0 -2.2 0 2.0 0.8
rkin --observe 0.2 0 0 --seeds 32
```

---

## 第 3 章 · 代码地图

```text
robot_kinematics_app/
├── configs/default.yaml      相机外参、IK 参数、流水线阈值
├── data/cases/*.csv          课件算例（原样保留，供对照与扩展）
├── outputs/                  运行产物
├── docs/                     本文、KNOWLEDGE_MAP、NUMBERS、架构图
├── scripts/                  run_all.sh、架构图脚本
├── src/robotkinematics/
│   ├── main.py               ★ 唯一入口 = 组合根：装配 + 扁平 CLI + 异常出口
│   │                           读它一个文件就知道"这个程序用到哪些实现"
│   ├── core/                 ★ 库覆盖不到的那部分
│   │   ├── robots.py           从库建 Panda：末端帧、关节限位、link 位置
│   │   ├── workspace.py        可达性采样（库没有这个 API）
│   │   ├── singularity.py      SVD 体检 + 库的 manipulability
│   │   └── exceptions.py       领域异常
│   ├── contracts.py          契约：Report / 报告块 + 协议
│   ├── usecases/             用例层（只 pick*：pick + pick_types/steps/report）
│   ├── adapters/             配置、两种渲染、绘图
│   └── __main__.py           三行：`python -m robotkinematics` 的桥
└── tests/                    架构守卫 + 流水线端到端断言
```

**依赖方向**：`main.py → usecases → contracts / core`，`adapters` 实现 contracts 里的协议；
**用例不直接依赖适配器**（这条边界让"算一遍"和"打印成表格"解耦，同一份结果能出文本也能出 JSON）。
这三条不是靠自觉：`tests/test_architecture.py` 读源码的 import 与构造调用，违反了就红。

---

## 第 4 章 · 一次抓取的一生

```bash
rkin --observe 0.2 0 0 --plot
```

| 工序 | 做什么 | 关键点 |
|---|---|---|
| ① 观测 → 本体位姿 | `SE3.Trans(camera) * SE3.Rz(yaw) * SE3(观测)` | 变换链顺序不能反；`--mode base` 可跳过 |
| ② 可达性预筛 | 采样几千个姿态，求沿目标方向的最大投影 | **工作空间不是球**：全方向最大 1.19m，但沿斜下方只有 1.09m |
| ③ 多初值 IK | 对每个初值调 `ikine_LM` | 默认 24 个初值（可用 `--seeds` 覆盖） |
| ④ 雅可比体检 | `robot.jacob0(q)` 做 SVD | σ_min 太小 → 这组解"不好动" |
| ⑤ 限位校验 | `robot.qlim` | 越限位的解直接淘汰 |
| ⑥ 解择优 | 评分：限位余量 0.5 + σ_min 0.3 + 就近 0.2 | 避免挑到"转两圈"的解 |
| ⑦ 微动校验 | 末端压 1mm，关节要动多少度 | 太夸张说明接近奇异 |

**失败时的结论**（六类，各有名字和数字）：不在工作空间 / 姿态不可达 / 收敛精度不足 /
所有解越关节限位 / 所有解接近奇异 / 微动不可行。

★ 最值得体会的一条：**位置可达 ≠ 位姿可达**。同一个观测点，自上而下抓可能"姿态不可达"，
换成侧向接近就"可执行" —— 报错信息里会直接给这个建议。

---

## 第 5 章 · 逐层读代码（推荐顺序）

### 第 1 轮 · 骨架（约 30 分钟）

1. `core/robots.py` —— 看末端帧、限位、link 位置是怎么从库里取的
2. `contracts.py` —— `Report` 与几种报告块；这是"用例产出什么"的定义
3. `main.py` —— 装配、扁平参数、`run_pick`；看它怎么把模型和用例接起来

**读完能回答**：`rkin --observe …` 到底用了库的哪几个 API？

### 第 2 轮 · 五脏（约 60 分钟）

4. `usecases/pick.py` —— 七道工序的主线，重点看每道工序**失败时怎么归类**
5. `core/workspace.py` —— 可达性预筛：为什么库给不了、采样怎么做的
6. `core/singularity.py` —— 体检报告的那几个数从哪来
7. `usecases/pick_steps.py` —— 每道工序的细节

### 第 3 轮 · 外围（约 45 分钟）

8. `adapters/render_text.py` —— 中文表格为什么不能直接用 `f"{s:<14}"`
9. `adapters/config_yaml.py` —— 未知项告警、范围校验、命令行覆盖
10. `tests/usecases/test_pick.py` —— 七类结论各一条测试，等于行为说明书

---

## 第 6 章 · 库的用法速查

| 想干的事 | 怎么写 |
|---|---|
| 位姿 | `SE3(x, y, z) * SE3.RPY(roll, pitch, yaw, order='zyx')` |
| 纯平移 / 纯旋转 | `SE3.Trans(x, y, z)` / `SE3.Rz(theta)` |
| 变换链 | `T1 * T2`（顺序不能反） |
| 逆 | `T.inv()` |
| 取位置 / 矩阵 | `T.t`（(3,) 或 (N,3)）/ `T.A`（4×4） |
| 取 RPY | `T.rpy(order='zyx')` —— ⚠️ 必须带 order |
| 取四元数 | `spatialmath.base.r2q(T.R)` —— ⚠️ 返回 **(w,x,y,z)** |
| 取轴角 | `spatialmath.base.tr2angvec(R)` —— ⚠️ 返回 **(角度, 轴)** |
| 建机器人 | `rtb.models.Panda()` |
| FK | `robot.fkine(q)`（批量：`robot.fkine(qs)`，qs 是 (N,n)） |
| IK | `robot.ikine_LM(T, q0=…, tol=1e-9)` |
| 雅可比 | `robot.jacob0(q)`（6×n） |
| 可操作度 | `robot.manipulability(q, method='yoshikawa')` |
| 关节限位 | `robot.qlim`（形状是 (2, n)，不是 (n, 2)） |
| 每个 link 的位置 | `robot.fkine_all(q)` |

---

## 第 7 章 · 库的坑（★ 实测踩出来的）

这一章是这个项目最有价值的部分 —— 下面每一条都是**跑起来才暴露**的，光看文档看不出来。

### 7.1 RPY 有两套相反的约定 ⚠️

```python
SE3.RPY(r, p, y, order='xyz')   # 默认：等于 Rx(y)·Ry(p)·Rz(r) —— 注意首尾是反的
SE3.RPY(r, p, y, order='zyx')   # 本项目采用：等于 Rz(y)·Ry(p)·Rx(r)
```

`order='zyx'` 才等于课件里手写的 `Rz(yaw)·Ry(pitch)·Rx(roll)`。
**课件自己两种都出现过**（手写的公式是一种、`T.rpy(order="xyz")` 打的数是另一种），
这就是"零位姿 RPY 对不上"的根源。取角时也必须 `T.rpy(order='zyx')`。

### 7.2 四元数是 (w, x, y, z)

`spatialmath.base.r2q()` 返回实部在前；ROS / Eigen 惯例是 (x, y, z, w)。
**差一个位置，接错了不会报错，只会静默算错**。

### 7.3 "末端"是哪个坐标系 ⚠️（最大的一个坑）

```python
robot.fkine(q)                      # 默认：夹爪 TCP   → 零位姿 [0.088, 0, 0.8226]
robot.fkine(q, end="panda_link8")   # 法兰           → 零位姿 [0.088, 0, 0.926]
```

差 **103.4 mm**（法兰再往前 103.4mm、绕 z 转 45° 才是夹爪 TCP）。同一组关节角、两个都正确的数。
课件的 `run_logs` 记的是默认末端（夹爪）；本项目抓取流水线固定在夹爪帧。

而且 `robot.ikine_LM` **只对模型默认末端求解**（传 `end=` 会报 `unexpected keyword argument`），
所以抓取流水线固定在夹爪帧 —— 物理上抓东西的本来就是夹爪。

### 7.4 `ikine_LM` 的 `tol` 是**关节步长**判据，不是位姿残差

实测：默认值下位姿残差能到 **0.8 mm**；传 `tol=1e-9` 才压到 **5e-7**。
想要"位姿多准"，要么把 `tol` 调小、要么迭代完自己用 FK 回验再决定收不收
（本项目两步都做了）。

### 7.5 `robot.reach` 返回 0

文档里它像"最大可达距离"，实测对本项目的模型返回 `0`（int）。
想判"够不够得着"只能自己采样 —— 而采样恰好还更准（能给出**沿某方向**的边界，而不是一个球）。

### 7.6 `manipulability(method='asada')` 可能返回 nan

库给了 4 种可操作度定义，其中 `asada` 在某些位形下算出 nan。
本项目用 `yoshikawa`，并拿自己算的（numpy SVD）与它对照 —— 两者逐位一致。

### 7.7 `fkine` 批量返回值的类型会骗人

```python
poses = robot.fkine(np.zeros((5, 7)))   # 批量
isinstance(poses, SE3)                  # True！—— 不是单个位姿
poses.shape                             # (4, 4) —— 也是骗人的
poses.t.shape                           # (5, 3) —— 这个才是真的
```

**只能用 `.t` 的维数判**：`(3,)` 是单个、`(N,3)` 是批量。

---

## 第 8 章 · 动手实验

⚠️ 下面的结果全部是**实测值**，不是推测。

### 实验 A · 位置可达 ≠ 位姿可达

```bash
rkin --observe 0.35 -0.05 0.15                # 姿态不可达（自上而下做不到）
rkin --observe 0.35 -0.05 0.15 --rpy 0 90 0   # 换个姿态就通了
```

### 实验 B · 工作空间不是球

```bash
rkin --observe 2 2 2    # 不在工作空间：会告诉你沿该方向差多少米
```

### 实验 C · 库的 `tol` 怎么影响精度

把 `configs/default.yaml` 的 `ik.tol` 从 `1e-9` 改成 `1e-3`，再跑
`rkin --observe 0.2 0 0` —— 报告里的位置残差会明显变大。**这就是 7.4 那条坑**。

### 实验 D · 偏好姿态与多初值

```bash
rkin --observe 0.2 0 0 --prefer-config 0 -0.3 0 -2.2 0 2.0 0.8
rkin --observe 0.2 0 0 --seeds 32
```

### 实验 E · 同一份结果出 JSON / 出图

```bash
rkin --observe 0.2 0 0 --format json
rkin --observe 0.2 0 0 --plot
```

### 实验 F · 观测已在本体坐标

```bash
rkin --observe 0.5 0 0.4 --mode base
```

---

## 第 9 章 · 卡住时怎么办

### 9.1 三个最容易卡住的地方

1. **`import spatialmath` 失败**（`cannot import name 'plotvol3'`）→ 见 README「已知环境问题」，
   这是 matplotlib 版本冲突，**必须先修好**（运动学全靠它）。
2. **末端帧搞错** → 数值整体差 0.1 m。第一件事是问"末端是哪个坐标系"。
3. **RPY 约定搞错** → 姿态角差 90°/180°。记住本项目一律 `order='zyx'`。

### 9.2 常见运行错误

| 报错 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: robotkinematics` | 没设 PYTHONPATH 也没安装 | `pip install -e .` 或 `PYTHONPATH=src python -m robotkinematics --observe …` |
| `cannot import name 'plotvol3'` | 本机 matplotlib 3D 被老 `.pth` 劫持 | 见 README「已知环境问题」 |
| 结论是 `姿态不可达` | 位置够但方向做不到 | 换抓取姿态（`--rpy`） |
| 结论是 `不在工作空间` | 物理上够不着 | 换观测点或挪机器人 |
| 图里中文是方块 | 系统没装 CJK 字体 | `apt install fonts-noto-cjk` |

---

## 附录 · 一分钟速查

**关键公式**

```
R = Rz(yaw)·Ry(pitch)·Rx(roll)      ⇔  SE3.RPY(roll, pitch, yaw, order='zyx')
^A T_C = ^A T_B · ^B T_C            顺序不能反
Δx ≈ J(q)·Δq                        雅可比：局部线性
```

**关键数字**

- 坐标变换：`^base T_cup.t = [0.7830, 0.1634, 0.8]`（课件基准算例）
- Panda：7 关节、雅可比 6×7；零位姿法兰 0.926 / 夹爪 TCP 0.8226（差 103.4 mm）
- 抓取流水线评分权重：限位余量 0.5 + σ_min 0.3 + 就近 0.2
- 可达边界（采样）：全方向最大约 1.19 m，沿斜下方约 1.09 m —— 工作空间不是球

**关键命令**

```bash
make test-fast / pick / run          # 测试 / 抓取 / 全流程演示
rkin --observe X Y Z [--plot] [--format json] [--mode …] [--rpy …] [--prefer-config …] [--seeds …]
PYTHONPATH=src python -m robotkinematics --observe 0.2 0 0
```

**分层依赖**：`main.py → usecases → contracts/core`；**用例不许直接依赖 adapters** —— 这三条由 `tests/test_architecture.py` 守着。
