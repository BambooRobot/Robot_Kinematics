按层组织的测试：目录结构与 src/robotkinematics/ 一一对应。

- core/       纯数学（旋转、SE3、二连杆、串联链、Panda、IK、奇异点、工作空间）
- contracts/  契约层（报告块、协议）
- usecases/   用例层（含抓取流水线的六类结论）
- adapters/   适配层（配置、算例、渲染、绘图）
- cli/        端到端（命令行、退出码、错误路径）
