# 算例数据集

本数据集直接取自课程实验，**列名与内容都保持原样**，便于与课件数字对照（见 `docs/NUMBERS.md`）。

当前 CLI 只跑 Franka Panda 抓取流水线（`rkin --observe X Y Z`）；下列 CSV 可作为对照材料保留。

- `coordinate_cases.csv`：camera frame 到 base frame 的坐标变换练习
- `panda_cases.csv`：Franka Panda 的演示姿态

> 流水线入口不依赖这些 CSV；它们保留作数值出处与扩展对照。
