# evla-lab

**端到端的视觉-语言-动作（VLA）流水线：微调 450M VLA → 推理优化 → 带安全层的真机运行时 → 可辩护的成功率评测。**

[![ci](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

![chunking](assets/chunking_ab.gif)

*同一个策略、同一个场景、两种执行方式。面对**移动**目标时，每步重规划赢（成功率 77% vs 40%），
因为 8 步的计划会过期；而在**静态**的 Push-T 任务上排序会反过来（见结论 3）：没有东西要追，
重规划只是在重新采样噪声。执行策略是一个需要测量的选择，而不是默认值——这正是本仓库要说明的事。*

---

## 为什么做这个项目

多数 VLA 仓库停在"这里有一个 checkpoint 和一个训练脚本"。而具身智能比赛的结果由另外四个问题决定：

| 环节 | 问题 | 本项目的位置 |
|---|---|---|
| 1. 微调 | 把公开 VLA 适配到我的任务上，到底有没有用？ | `evla finetune`：SmolVLA-450M + Push-T |
| 2. 推理 | 一次决策多快？提速要拿多少精度去换？ | `evla optimize`：Pareto 研究 |
| 3. 部署 | 推理迟到了怎么办？策略输出离谱怎么办？ | `evla deploy`：运行时 + 安全层 + HAL |
| 4. 评测 | "高了 3 个点"是真差异还是噪声？ | `evla eval`：Wilson 区间 + 配对 McNemar |

每个环节都落盘成 JSON，于是"结论"是一个可以打开的文件，而不是一段回忆。

## 结果

SmolVLA-450M 在 Push-T 上微调 20k 步（单张 3090，4.3 小时），在 40 个配对种子上评测
（receding 配置每回合需 300 次推理，故用 20 个）。`best coverage` 是回合内达到的最佳位姿重叠度
（成功阈值 0.95）。

| 配置 | 执行/块长 | 成功率（95% CI） | 控制频率 | p50 推理 | best coverage |
|---|---|---|---:|---:|---:|
| 零样本 checkpoint | 50 / 50 | 0%（0–9%） | 137 | 237 ms | 0.33 |
| 微调后，fp32 | 50 / 50 | 5%（1–17%） | 130 | 250 ms | 0.89 |
| 微调后，bf16 | 50 / 50 | 2%（0–13%） | 127 | 261 ms | 0.95 |
| + `torch.compile` | 50 / 50 | 8%（3–20%） | **229** | **86 ms** | 0.91 |
| + compile + 4 步 flow | 50 / 50 | 10%（4–23%） | **270** | **54 ms** | 0.95 |
| + compile + 4 步，exec 20 | 20 / 50 | 10%（4–23%） | 185 | 55 ms | 0.95 |
| receding（每步重规划） | 1 / 50 | 0%（0–16%） | 18 | 52 ms | 0.79 |
| 时序集成（exec 10） | 10 / 50 | 5%（1–17%） | 18 | 52 ms | 0.94 |

![pusht](assets/pusht_rollout.gif)

*一次微调后的真实 rollout（固定种子列表的第 0 个回合，没有挑选）。方块被推到目标附近，但姿态偏了约 45°，
覆盖率约 0.9（成功阈值 0.95）——这就是"near miss"长什么样，也是下表同时报告覆盖率的原因。*

完整产物：[`results/pusht_sweep/REPORT.md`](results/pusht_sweep/REPORT.md)、
[`LEADERBOARD.md`](results/pusht_sweep/LEADERBOARD.md)（配对 McNemar）、
[`results/pusht_learning_curve/`](results/pusht_learning_curve)、
[`results/pusht_lora/`](results/pusht_lora)。

**1. 微调改变的是"失败方式"，不只是分数。**
位姿覆盖率 0.33 → 0.95；失败分类彻底翻转：预训练 checkpoint 在 **40 个回合里有 29 次根本没碰到方块**；
微调后 40 次全部能到达，其中 **36 次只差姿态的几度**。成功率 0/40 → 4/40——这个 10 个点的提升
在当前样本量下还无法给出显著性结论（配对 McNemar p = 0.125；要分辨 10% vs 2.5% 每组需要 160 回合）。
支撑结论的是覆盖率和失败分类，不是成功率本身。

**2. `torch.compile` 是最大单项收益——4.8 倍，且没有可测量的精度损失。**
237 ms（fp32）→ 261 ms（bf16，batch 1 下 bf16 并没有更快）→ 86 ms（compile）→ 54 ms（compile + 4 步 flow）。
action expert 是一长串形状固定的小矩阵乘，eager 模式的 kernel 发射开销占主导。没有它 10 Hz 根本跑不动；
有了它，控制回路只用了 3% 的时间预算。

**3. 每步重规划反而更差。**
同一组权重：开环执行动作块覆盖率 0.95，receding 只有 0.79。flow-matching 策略每次调用都重新采样动作噪声，
每秒重规划 20 次注入的是抖动而不是反馈。"延迟越低越安全"是错的，真正的问题是任务能容忍多久的计划过期。

**4. 微调不是单调变好——热重启甚至会把效果练回去。** 覆盖率随 checkpoint：0.49（4k）→ 0.76（8k）→
0.92（12k）→ **0.73（16k）** → 0.95（20k）。把同一条 run 再续训 5k 步（学习率减半、新的 cosine 调度）
结果更差：22.5k 时 0.91，25k 时 0.83（见 [`results/pusht_stage2/`](results/pusht_stage2)）。
20k 是那条调度曲线的最优点，而不是"步数越多越好"。选 checkpoint 一定要按验证集覆盖率，不能按步数。

**5. 参数高效微调在这里受容量限制。** 相同步数下 LoRA r=16（2.4M 可训练参数）覆盖率停在 0.48，
而默认的 expert-only 微调（约 100M 可训练）从 0.49 升到 0.76。对这种量级的域偏移，低秩适配需要更高 rank 或更多数据，
而不只是更多步数。

## 部署实测：仿真环境、10 Hz、安全层开启

```bash
evla deploy -c configs/deploy.yaml      # 定频回路 + 策略服务 + HAL + 安全层
```

```
steps                 900        （3 回合 × 300 步，真实 10 Hz 节奏）
inference calls        45        （每执行 20 个动作才推理一次）
deadline misses         0        （平均只用了 3.1% 的时间预算，最慢一步 76 ms / 100 ms）
safety                 900 个动作里 12 次被限速，0 次急停
wall time            89.8 s      （正好等于 900 步 / 10 Hz）
```

这就是"模型能跑"和"回路能守住节拍"的区别：预算花在**摊薄后的推理**上，
而策略输出跳变超过 80 像素时，是安全层限速而不是让机械臂自己跳。
原始报告：[`results/deploy_sim.json`](results/deploy_sim.json)。

## 快速开始（不需要 GPU，也不需要机器人，约 1 分钟）

```bash
git clone https://github.com/mystic-qaq/evla-lab && cd evla-lab
pip install -e ".[dev]" torch --extra-index-url https://download.pytorch.org/whl/cpu

evla doctor          # 环境自检：装了什么、CUDA 是否可见
evla quickstart      # 采集 -> 训练 -> 评测 -> 报告，全在 CPU 上跑通
```

`evla quickstart` 在一个零依赖的玩具任务上跑完整条流水线，输出
`outputs/quickstart/eval/REPORT.md`。CI 跑的就是它，所以这条链路不会悄悄腐烂。

```bash
evla toy-eval --policy outputs/quickstart/toy_flow.npz --episodes 30 --drift 0.03
```

```
receding-1       76.7% [59.1%, 88.2%] (23/30)   582.4 Hz
open_loop-8      40.0% [24.6%, 57.7%] (12/30)   802.7 Hz
temporal_ens-8   60.0% [42.3%, 75.4%] (18/30)   575.0 Hz
```

同样的权重、同样的 30 个场景、三种执行方式：**开环执行快 38%，但成功率低 37 个点**；时序集成在几乎相同的控制频率下挽回了大部分精度。这就是"不能只优化单一指标"的完整论证。

## 复现 VLA 实验

需要一张 24GB 显卡。RTX 3090 上的实测耗时：微调 4.3 小时、学习曲线约 10 分钟、
优化矩阵约 35 分钟、部署演示约 2 分钟。

```bash
bash scripts/setup_env.sh                    # 建 .venv，装 torch + lerobot + evla
bash scripts/download_assets.sh              # SmolVLA-base（873MB）+ Push-T 数据集
evla finetune -c configs/finetune_smolvla_pusht.yaml
evla sweep    -c configs/sweep_learning_curve.yaml   # 6 个 checkpoint，各 20 个种子
evla sweep    -c configs/sweep_pusht.yaml            # 8 组配置，各 40 个种子
evla deploy   -c configs/deploy.yaml                 # 10 Hz 定频回路
```

`evla sweep` 产出 `results/pusht_sweep/{REPORT.md, LEADERBOARD.md, *.json, *.png}`。

## 一个策略，四种接口

因为所有组件都只讲 `Observation` / `Policy` 两个类型，450M 的 SmolVLA、2M 的
flow-matching 小模型、手写的脚本控制器可以互相替换——包括在单元测试里替换。

```
观测 Observation(state + images + task)
        │
        ▼  predict_chunk(obs) -> (H, action_dim)
   ┌────┴─────────────┬──────────────────┬─────────────────┐
ChunkExecutor   InstrumentedPolicy   RemotePolicy      (训练/回放)
receding /      （同步计时）         （TCP 策略服务）
open-loop /
temporal-ens
        │
   SafetyLayer   限幅 · 限速 · 看门狗 · 急停
        │
      HAL        MuJoCo 仿真 · SO-101 真机
```

## 目录结构

```
src/evla/
  types.py            Observation / StepResult / Policy（全部接口就这些）
  envs/               toy_reach（纯 numpy，零依赖）· pusht（MuJoCo）
  models/             smolvla 适配层（450M）· flow_chunk（从零实现，2M）
  data/               演示采集、动作块切分、归一化统计
  control/            ChunkExecutor：receding / open_loop / temporal_ens
  optimize/           延迟剖析器 · 能力探测式优化管线
  deploy/             运行时 · 策略服务端/客户端 · HAL（仿真、SO-101）· 安全层
  eval/               回合执行器 · 统计检验 · 报告生成 · 配对排行榜
  cli.py              evla doctor/quickstart/finetune/bench/eval/sweep/serve/deploy
configs/              每个实验都是一个 YAML，而不是一段 shell 历史
docs/                 架构 · 优化研究 · 部署 · 评测协议 · 比赛路线图
results/              提交进仓库的实验产物（JSON + 图 + 报告）
tests/                39 个测试，纯 CPU，约 40 秒
```

## 几个关键设计决定

- **包装 LeRobot 的训练器，自己做其余部分。** 重写 SmolVLA 微调只会得到一份带新 bug 的副本；价值在推理成本、部署和可辩护的度量上。`evla finetune` 会把实际执行的上游命令打印出来，没有黑箱。
- **策略返回"动作块"而不是单个动作。** 执行策略从此是一个显式、可测量的选择，而不是藏在模型包装器里的队列。
- **延迟必须在控制回路内、带 `cuda.synchronize()` 测量。** 异步计时会把推理时间少报 3–10 倍，而决定机器人会不会卡顿的是 p95 而不是均值。
- **限幅只计数，不触发急停。** 策略输出超范围是标定问题，不是危险；急停留给非有限值。
- **不出现裸百分比。** 每个成功率都带 Wilson 区间与回合数；比较用同一批种子 + McNemar 精确检验。

更多：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、[`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md)、
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)、[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)、
[`docs/COMPETITION_PLAYBOOK.md`](docs/COMPETITION_PLAYBOOK.md)。

## 本项目**不**声称的东西

- **没有用真机。** 所有数字都来自仿真。HAL、运行时、安全层、SO-101 后端都已写好并在仿真中验证；真机验证是路线图的第一项。
- **Push-T 只是小基准。** 它适合在 12 小时内研究推理成本与执行策略，不是操作能力的证据。
- **绝对成功率不是重点。** 流水线、失败插桩和评测协议才是。

## 路线图

1. SO-101 真机打通：标定、动作尺度、2Hz 干跑、失败分类。
2. 数据规模研究：成功率 vs 演示条数（模仿学习里最有说服力的一张图）。
3. VLM 主干 int8 量化（需要标定数据 + 精度校验）。
4. 多任务套件：按任务报区间，而不是一个总分。

## 参考

SmolVLA（Shukor et al., 2025）· ACT / temporal ensembling（Zhao et al., 2023）·
Flow Matching（Lipman et al., 2023）· π0（Black et al., 2024）·
Diffusion Policy（Chi et al., 2023）· LeRobot

MIT License。
