# Laya 机械臂 Agent：代码和权重交接包

本包是当前冻结 T4 方法的完整推理与机械臂仿真交接版，供组员运行、阅读和后续开发。
流程：DINO + RGBD 当前视觉证据 → 动态语义问题 → Laya 概率 → 结构化聚合/任务图/恢复 → 技能 → 新观察。
运行时不调用生成式大语言模型；Laya 本身仍是语言判别模型。

## 包含什么

- method/src：我方当前方法的完整运行代码（14 模块）。
- weights/T4：选定的完整 T4 权重、选择/校准参数及训练来源记录。
- runtime/models/laya_en：基础 Laya（目录名 laya_en） 权重、配置、tokenizer。现有流程先构建基础模型再严格载入 T4，故两套均保留。
- runtime/models/grounding_dino_tiny：DINO 权重及预处理/tokenizer 配置。
- runtime/vendor/laya：固定上游 Python 实现与许可。
- vendor/capx：固定 CaP-X 仿真、任务与运动依赖子集；不含其 LLM Agent/对照运行器。
- vendor/robosuite：实际 robosuite 源码与仿真资产。
- vendor/source_archives：固定 JAXLS / PyRoKi 源码包。
- robot_assets：Panda URDF、mesh 和许可。
- requirements、setup.sh、run.py：依赖版本、安装及可迁移启动入口。
- evidence、RESULTS_AND_LIMITS.md：既有实测事实及限制。

不含 Codex/Phi/Smol/Qwen 权重、生成式回退、API 密钥、账号、SSH 配置或旧试错目录。
这是推理复现包，不是历史全部实验归档或从头训练复现包；T4 完整权重已含，无需重训。
训练数据与历史失败 raw 仍在服务器保留；包内摘要不取代原始证据。

## 环境

Linux x86_64、Python 3.12、NVIDIA GPU、可用 EGL/MuJoCo。
实际用 PyTorch 2.8.0+cu128，驱动需兼容。建议 12GB 以上显存、16GB 以上内存（不是已测最低要求）。
Windows 可存储/转交 ZIP，但这不是 Windows 原生一键运行包，组员应在 Linux GPU 服务器运行。
安装 Python 依赖需要访问 PyPI / PyTorch 索引；不是自带操作系统、驱动及全部 wheel 的离线容器。
所有模型、tokenizer、机器人资产随包提供；运行阶段 Hugging Face 离线，无需账号。

Ubuntu 常见系统依赖：python3.12-venv libegl1 libgl1 libglib2.0-0 libgomp1 libstdc++6。
按机器实际情况安装，勿盲改系统。
三套环境隔离视觉 NumPy 2.x、仿真 NumPy 1.26、JAX CPU 运动库，不建议随意合并。
requirements 的 environment.json 保留实际版本和未启用的可选依赖记录；仅承诺四类已接通路径。

## 安装与运行

在解压后顶层目录：

```bash
bash setup.sh
python3.12 run.py --seeds 1061201
```

默认 CubeLift / CubeStack / CubeRestack / SpillWipe 四类各一回合。
结果存入新的 runs/时间戳/，只有 COMPLETE.json 实际终局才说明完成，PID 不算结果。

```bash
python3.12 run.py --tasks CubeLift CubeStack --seeds 1061201 1061202 --out /绝对路径/new_run
```

输出目录必须不存在，不覆盖历史。上述种子是已见开发例，不是独立 TEST。
启动器启动自己的本机 CPU IK 服务，默认端口18116，结束只关闭自己启动的服务。
端口已占用时改 --ik-port 18117，不关闭他人进程。

已有兼容环境可以指定：

```bash
python3.12 run.py --sim-python /path/sim/bin/python --vision-python /path/vision/bin/python --motion-python /path/motion/bin/python
```

## 代码阅读顺序

1. method/METHOD.md：模块与方法关系。
2. method/src/run_native.py：观察—决策—执行—反馈。
3. question_direct.py、question_scope.py、semantic_core.py：动态问题与语义接口。
4. semantic_agent.py、aggregation.py：概率、任务图、恢复/澄清。
5. worker.py、native_cube.py、native_spill.py、vision_spill.py：视觉与技能绑定。
6. motion.py：复用 CaP-X / PyRoKi 运动方法，不是原创控制器。

动态出题仍受技能目录/规则边界限制，不是无边界规划器；语义概率不是物理成功率。
不用仿真真值补感知；环境成功标志只作独立结果评价。
打包没有改变出题/聚合算法、权重、物理步长或等待，只迁移路径和开放本机 IK 端口配置。

## 验证、许可、安全

原代码整理已四类各一例实际运行；本完整包另做路径/权重/独立 IK 集成检查，见 evidence/portable_runtime_check.json。
新机器从零依赖安装尚未替组员执行，不宣称任何机器解压即保证成功。
只供仿真，不能直接接真机；真机安全与硬件适配尚未验证。
来源版本见 COMPONENTS.json，各 vendor/模型 README、method/THIRD_PARTY.md 和 licenses 保留第三方归属。
对外发布须继续遵守上游条款，不对上游权重/代码主张独占版权。
