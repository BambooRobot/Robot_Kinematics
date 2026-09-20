按层组织的测试：目录结构与 `src/robotkinematics/` 一一对应。

- core/       库覆盖不到的那部分（robots 建模/末端帧、planar2r 闭式解、singularity、workspace、exceptions）
- contracts/  契约层（报告块、协议）
- usecases/   用例层（含抓取流水线的七类结论，以及逐道工序）
- adapters/   适配层（配置、算例、渲染、绘图）
- cli/        端到端跑入口 `main.py`（命令行、退出码、错误路径）
- test_architecture.py  **架构守卫**（跨层）：读源码的 import 与构造调用，钉住三条规矩 ——
              ① 装配只在入口：除 `main.py` 与 `adapters/` 外，任何文件都不许"造具体实现"
              ② 依赖只向下：core 不认识 usecases/adapters/contracts；usecases 不认识 adapters
              ③ 命令表与 argparse 枚举出的子命令一模一样

图省事的话，这几条最容易在重构中被无声破坏（分层图写在 README 里是不会自己生效的），
所以它们由测试守着 —— 违反了会红。
