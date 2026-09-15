# 算例数据集

本数据集直接取自课程实验（`0625robotics_python_lab_cloud_terminal/data/`），
**列名与内容都保持原样** —— 只有这样，`rkin batch` 的输出才能与课件的
`outputs/ppt_cases_batch_result.txt` 逐字符对照（见 `docs/NUMBERS.md`）。

本数据集只服务于课件算例的复现与验证，不做训练用途。

- `coordinate_cases.csv`：对应 PPT 中 camera frame 到 base frame 的坐标变换练习
- `two_link_cases.csv`：对应二连杆 FK、IK、奇异点练习（填了 q 就是 FK 算例，填了 target 就是 IK 算例）
- `panda_cases.csv`：对应 Franka Panda 的三个演示姿态

> 课件里 `panda_cases.csv` 声明了却**没有任何脚本读它**（08/09/10 把姿态硬编码在代码里）。
> 本项目把它接进了 `rkin batch` 的第 3 段。
