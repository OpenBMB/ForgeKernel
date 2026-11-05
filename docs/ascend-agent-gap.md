# 当前 PyPTO Ascend Agent 与论文 Ascend Agent 的 Gap

> 对比日期：2026-11-03  
> 论文：[`../59558_ForgeKernel_Self_Evolvin.pdf`](../59558_ForgeKernel_Self_Evolvin.pdf)，重点为 §3.1–3.5、§4.2 的 Cross-architecture test、Appendix G 和 Figure 6。  
> 当前实现证据：`tasks/swiglu-pypto/`、`src/forgekernel/card_template/eval/cann_bench.py`、`src/forgekernel/card_template/pypto_gpu_entry.sh`，以及 run `.artifacts/swiglu-pypto_261103-210743-f995/`。

## 标记约定

- **[MATCH]**：与论文明确描述一致，并已被当前真实 run 验证。
- **[PARTIAL]**：主体或代码骨架存在，但实验协议、数据保真度或覆盖范围不同。
- **[GAP]**：论文明确需要、当前缺失或不等价的能力。
- **[UNVERIFIED]**：共享代码可能存在，但当前 Ascend run 没有实际覆盖，不能声称可用。
- **[INFO]**：论文公开信息不足，不能据此判定双方等价或不等价。

## 0. 结论

> **2026-11-04 更新**：本文件保留为初始 gap 分析。后续实测已补齐官方字段/
> 原始报告、真实 msprof、Ascend 专用 prompt/KB routing、密封参照晋升、
> 确认协议和新的 probe-certified recipes。全 DeepSeek 的 SwiGlu campaign
> 已完成 `1.20165×` 最终确认（3 student + 3 fresh ref jobs，cross-job）。
> Level-4 GQA 参照和 M0 正确性已完成，但候选仍约 40.24 ms vs 强库
> 0.06116 ms，未达到论文同级性能。严格论文复现仍不可能：本机为
> Ascend950PR / CANN-Bench 2.0.0 / 本地公开评分，论文为 Ascend910C /
> CANN-Bench 1.1.2 / hosted scoring / 8 个 Level-4 算子。
> 当前准确状态见 `docs/ascend-agent-validation.md`。

当前实现已经验证了：

```text
公开 PyPTO 参照
  → teacher 蒸馏 milestone
  → student 提交 M0 架构修改
  → CANN-Bench 20-case 正确性 + trace_view 时间
  → KEEP
  → best confirmation + report
```

因此，它是 **ForgeKernel 控制骨架在 PyPTO/Ascend 上的 source-mode、单算子、timing-only 端到端适配**。

但它尚不是论文所述的完整 Ascend Agent：

1. **没有接入论文明确点名的 msprof**；
2. **没有执行 M1、plateau 诊断、外层架构重开或跨代参照晋升**；
3. **CANN 原始报告中已有的 baseline、hardware bound 和评分数据被 adapter 丢弃**；
4. **没有 B200 式的 per-iteration profiler 轨迹、定向硬件探针和 probe-certified KB 积累**。

论文的 Ascend 结果是 **Ascend 910C + CANN-Bench 1.1.2 + 8 个 Level-4 算子**。当前结果是 **Ascend 950PR + CANN-Bench 2.0.0 + 一个 Level-1 SwiGlu**，两者的性能数字不能直接比较。

---

# 第一部分：迭代流程

## 1. 实验输入和停止条件

| 项目 | 论文 Ascend Agent | 当前实现 | 判定与 Gap |
|---|---|---|---|
| 硬件 | Ascend 910C | Ascend 950PR | **[INFO]** 不同硬件，不是同一结果的复现。 |
| Benchmark | CANN-Bench 1.1.2 | CANN-Bench 2.0.0，固定 commit `059cb61d…` | **[INFO]** 任务包和评分元数据不同。 |
| 覆盖范围 | 8 个 Level-4 fused attention / RNN 算子 | 1 个 Level-1 SwiGlu，20 个公开 case | **[GAP]** 尚未证明对论文同级复杂算子的适用性。 |
| 优化参照 | 华为当前算子库中最快实现 | CANN-Bench 公开 PyPTO 实现 | **[GAP]** 参照强度和来源不同，不能直接比较最终收益。 |
| 参照披露 | 论文主实验为密封参照，可运行/profile、不可读源码 | `oracle_mode=source`，teacher 可读；初始 student kernel 已是公开参照实现 | **[GAP]** 不能证明“仅从硬件行为重建架构”。当前只是自动修改已有实现。 |
| 模型 | 论文实验设置为 arch=Fable 5.1、coding=Opus 5 | teacher/student 均为 `deepseek-flash` | **[INFO]** 不是同模型对照，性能差异不能全部归因于 harness。 |
| 停止门槛 | 按 campaign target 和噪声规则停止 | `ratio=0.5`，允许 device time 约为参照的 2 倍 | **[GAP]** 当前是宽松闭环验收门槛，不是论文强度的超越参照实验。 |
| 最终确认 | 通用协议为最多 3 次确认；Ascend 分数由 hosted CANN-Bench 计算 | `confirm_jobs=1`，最终 1 个 student job + 1 个 ref job | **[PARTIAL]** 有确认和同期 re-anchor，但统计强度较低。 |

当前 run 的参照为 `0.0666214970 ms`，停止门槛约为 `0.133243 ms`。student 首轮 M0 达到 `0.0664000291 ms` 后立即停止，因此没有进入后续调优阶段。

## 2. 两层循环与知识反馈

| 流程 | 论文要求 | 当前实现与运行证据 | 判定与 Gap |
|---|---|---|---|
| M0 架构课程 | teacher 生成架构优先 DAG；student 实现；M0 作为架构单元/验收 | teacher 生成 `milestone v1`，`violations=none`；student 提交 D0.1–D0.9，20/20、KEEP | **[MATCH]** 这是本次已完整验证的部分。 |
| 后续 M1→Mk | 在冻结拓扑内调整 tile、stage、unroll、buffer 等杠杆 | milestone 写了 D1.1–D1.3，但 run 在 M0 后因达到宽松目标停止 | **[UNVERIFIED]** 文档存在不代表 Ascend 上已执行有效。 |
| 内层测量反馈 | 正确性、计时、profiling 驱动 KEEP/REVERT | 20-case 正确性和 trace_view 时间驱动 KEEP；`profiler={}` | **[PARTIAL]** 时间闭环存在，硬件机制反馈缺失。 |
| 外层 arch loop | plateau、DAG exhausted 或 gate impasse 后，teacher 定价、probe、修改 milestone | 共享 driver 保留该逻辑；本次 `meta_steps=0`，未触发 | **[UNVERIFIED]** 不能声称 Ascend 外层诊断已可用。 |
| M0 架构重开 | 测量证明拓扑是瓶颈时，重开最早依赖决策 | 未发生；当前没有 msprof/stall/timeline 等证据支持同类诊断 | **[UNVERIFIED]** 控制代码存在，但 Ascend 证据链尚未验证。 |
| 跨代参照晋升 | best kernel 成为下一代参照，重新分析并设置更高目标 | `scripts/self_evolve.py` 对非 `cutedsl` 直接报错 | **[GAP]** PyPTO 当前不能执行 self-evolution。 |
| KB 沉淀与复用 | 硬件 probe 认证后入库，跨算子复用 | `hw_kb/pypto/ascend950/` 只有初始 `index.md`；本次无新增 recipe | **[GAP]** 没有形成论文意义上的 Ascend 知识积累闭环。 |
| 跨算子通用性 | harness/prompt 不含算子专属信息 | `meta/pypto/student_prompt.md` 硬编码 `swi_glu` 和八个 class | **[GAP]** 当前仍是 SwiGlu 专用适配，不是通用 Ascend Agent。 |

### Self-evolution 的直接代码限制

`scripts/self_evolve.py` 当前包含：

```python
if task.language != "cutedsl":
    raise ValueError(...)
```

仅删除此判断还不够。PyPTO leaderboard 任务还需要实现：

- 下一代 CANN 参照树和任务快照；
- best kernel 作为下一代参照及初始候选的打包；
- 下一代参照评测与确认比值；
- 参照密封和 student 访问边界。

### 外层循环未被执行不是“代码不存在”

当前共享 driver 中仍有外层循环实现。本次没有执行，是因为第一轮已经满足 `ratio=0.5` 的停止条件。正确结论是 **未验证**，而不是已证明外层循环失效。下一次验收必须让 run 真正进入 M1 和至少一次外层诊断。

---

# 第二部分：Agent 能力与性能数据

## 3. 论文明确公开的 Ascend 数据

论文 Appendix G 和 Figure 6 明确给出以下数据通道：

- msprof `kernel_details` 的 kernel-only device time；
- CANN-Bench 为每个 case 定义的 benchmark baseline；
- CANN-Bench hardware bound；
- compilation、accuracy、hardware-attainable performance 组成的分数；
- hosted CANN-Bench 计算的算子和整体结果。

论文**没有明确列出** Ascend 是否具有以下 B200 数据：

- register/spill、bank conflict、DRAM bytes；
- per-instruction stall；
- SASS 等价物；
- PM timeline、role exposure。

因此，不能仅凭“only profiler swapped”就推断论文 Ascend 与 B200 的信息量完全一致。可确认的是控制流复用；具体 profiler 证据种类仍需看实现，而论文没有公开完整明细。

## 4. 当前保留的数据与 Gap

| 数据/能力 | 论文 Ascend Agent | 当前实现 | 判定与 Gap |
|---|---|---|---|
| 正确性 | CANN-Bench accuracy，计入评分 | 官方 20-case 结果驱动 KEEP/REVERT | **[MATCH]** 官方正确性闭环已验证；当前未覆盖隐藏集。 |
| kernel device time | msprof `kernel_details` | CANN `trace_view` 的 `elapsed_us` / AI-Core E2E | **[PARTIAL]** 时间货币适合当前 PyPTO，但不是论文点名的 msprof 通道。 |
| benchmark baseline | 用于 speedup 和评分 | 原始报告有，adapter 不导出 | **[GAP]** 已有数据未进入 agent 轨迹。 |
| hardware bound | 用于 hardware-attainable performance | 原始报告有 `t_hw_us`，adapter 不导出 | **[GAP]** 无法让 agent 判断距离硬件锚还有多远。 |
| 综合评分 | compilation + accuracy + performance | 原始报告有，adapter 只保留 pass 和 time | **[GAP]** agent 看不到论文 Figure 6 使用的评分体系。 |
| 架构参照比较 | 由 campaign 参照和 Ascend hosted 评测共同提供证据 | 参照在独立 tree/job 测量，`ref_in_job=false` | **[PARTIAL]** 有 cross-job re-anchor，但不是 B200 同作业 paired timing；论文 Ascend 本身也使用 hosted harness，协议不同需明确标注。 |
| 每轮 profiler digest | 论文称替换 profiler 后沿用 loop | `profiler={}`、`kernels=[]` | **[GAP]** teacher 缺少独立的每轮硬件事实。 |
| launch / memory / stall / instruction | 论文未给出完整 Ascend 清单 | 当前均无 | **[INFO] + [GAP]** 不能要求与 B200 字段一一对应，但当前也没有任何 Ascend 替代证据通道。 |
| 定向硬件探针 | 用于架构假设、杠杆定价和 KB 认证 | 本次没有真实 NPU probe | **[GAP]** 无法形成论文式机制归因和 KB 认证。 |
| 噪声与稳定性 | 同作业噪声地板；Ascend hosted score | 有同期复测，但最终确认各 1 次；原始 jitter 未导出 | **[PARTIAL]** 最终 1.16% 比值不应视为稳定显著提速。 |

## 5. 最容易补齐的数据：CANN 原始报告已经提供

对本次 950PR 原始 CANN-Bench JSON 的检查表明，下列字段已经存在：

| 原始字段 | 可提供的信息 | 当前是否进入 ForgeKernel 轨迹 |
|---|---|---|
| `baseline_perf_us` | CANN benchmark baseline 时间 | 否 |
| `speedup` | 相对 benchmark baseline 的加速比 | 否 |
| `t_hw_us` | benchmark hardware bound | 否 |
| `perf_score` | 单 case 性能分 | 否 |
| `compilation_score` / `compile_runtime_score` | 编译分 | 否 |
| `function_score` | 功能/正确性分 | 否 |
| `performance_score` | 算子性能分 | 否 |
| `overall_score` | 综合分 | 否 |
| `op_times.trace_view.aicore_e2e` | AI-Core E2E 时间 | 仅通过 `elapsed_us` 保留 |
| `aicore_e2e_jitter` | AI-Core 时间抖动 | 否 |
| `aicpukernel_gap` | AICPU/kernel gap | 否 |
| `memory_peak_mb` | 峰值内存，当前可为 null | 否 |

`src/forgekernel/card_template/eval/cann_bench.py` 的 `report_to_eval()` 当前主要读取：

```text
status
accuracy.passed
elapsed_us
```

所以这里的首要问题不是硬件没有数据，而是 **adapter 没有将已有数据交给 agent**。

实现时必须区分：

- `baseline_perf_us`：CANN **benchmark baseline**；
- 当前公开 PyPTO 实现：campaign 的 **architecture/library reference**；
- `t_hw_us`：benchmark **hardware bound**。

三者语义不同，不能都写入 `ref_*`，否则会破坏仓库的 single-ref 合约。

## 6. PyPTO 目前仍存在 NCU 假通道

`src/forgekernel/card_template/pypto_gpu_entry.sh` 的 `ncu` 分支只打印“不是 Ascend/PyPTO channel”，但共享 driver 仍按空 NCU 结果重试。本次日志出现：

```text
ncu empty on a passing kernel -> retry 1/2
ncu empty on a passing kernel -> retry 2/2
```

同时，当前 milestone 中仍有 `[VERIFY: ncu-derived]` 标记，而实际证据来自 trace_view 时间或解析推导。

**[GAP]** 这不只是命名问题：它会让轨迹、gate 和 teacher 对证据来源产生错误表达。profiler backend 和 verification channel 都需要显式建模。

---

# 第三部分：Gap 修复优先级与验收标准

## P0 — 先补已有数据和真实控制流覆盖

### G1. 完整保留 CANN 官方数据

**改动方向**：

- 扩展 `report_to_eval()`、eval JSON、iteration record 和 baseline archive；
- 保存 benchmark baseline、hardware bound、speedup、各项 score、jitter、AICPU gap 和 memory 字段；
- 将原始官方 JSON 归档到本机 run artifacts，而不只留下远端路径；
- 将结构化数据提供给 milestone、diagnose digest 和最终报告。

**验收**：一次 run 的 `status.jsonl` 能同时看到每个 case 的 device time、CANN baseline、hardware bound 和 score，且不与 campaign reference 混淆。

### G2. 建立真实 profiler backend，清理 NCU 假设

**改动方向**：

- 将 `ncu` / `msprof` / `cann_trace_view` 建模为不同 backend；
- 无硬件 profiler 时跳过 NCU 采集及 retry；
- 为 CANN/msprof 增加真实 verification channel，停止把 trace 推导标成 `ncu-derived`；
- 对不可用数据显式写 `unavailable`，不伪造 NCU 等价字段。

**验收**：PyPTO run 不再出现无意义 NCU retry，milestone validator 和 trajectory 中的证据来源均与实际采集通道一致。

### G3. 做一次覆盖 M1 和外层诊断的 campaign

**改动方向**：

- 将目标提高到至少 `1.05`，而不是 `0.5`；
- 增加 `confirm_jobs` 和合理预算；
- 覆盖 M1 lever sweep，并至少触发一次 plateau 或 gate impasse；
- 检查 teacher 是否能在只有 Ascend 数据的情况下作出诊断和 milestone patch。

**验收**：轨迹中同时存在 M1、外层 meta step，以及基于实测数据的 patch KEEP/REVERT 记录。

## P1 — 补齐论文级 Ascend 能力

### G4. 接入真实 msprof

**改动方向**：实现采集入口、报告归档、解析器、per-iteration digest、环境 preflight 和失败分类。trace_view 可继续作为 PyPTO 的计时货币，msprof 补充论文所称的 profiler 通道，不要求机械复制 NCU 字段。

**验收**：reference 和 student 都能产生可审计的 msprof 数据，teacher 的至少一项机制判断由该数据独立支持。

### G5. 支持 PyPTO self-evolution

**改动方向**：补齐 leaderboard 任务的参照晋升、任务生成、参照评测、disclosure boundary 和确认协议，而不是只删除语言判断。

**验收**：完成 `gen0 best → gen1 reference → gen1 confirmed result`；若收益低于噪声地板，应按规则停止而不是强行晋升。

### G6. 建立 PyPTO 知识闭环并去掉算子硬编码

**改动方向**：

- 将 `swi_glu` 和 class 要求从通用 prompt 移入任务卡；
- 增加 PyPTO 的 recon/distill/diagnose/harvest overlay 和工具；
- 用真实 NPU probe 认证 operator-agnostic recipe；
- 记录 no-go 和 retry precondition。

**验收**：至少一个 recipe 经 NPU probe 入库，并在另一个算子的 run 中被读取和使用；通用 prompt 不出现 `swi_glu`。

### G7. 做论文同级实验，而非只做 Level-1 迁移

**改动方向**：选择 Level-4 算子、华为强库参照和明确的 source/blackbox 协议；若要比较论文机制收益，固定 teacher/student 模型和预算。

**验收**：公开任务版本、参照版本、模型、硬件、隐藏/公开 case 范围和评分口径，不能只比较一个加速比数字。

---

## 当前最准确的定位

> 当前实现完成了 ForgeKernel 骨架在 PyPTO/Ascend 上的首个端到端 M0 优化闭环，但尚未复现论文 Ascend Agent 的 profiler 证据链、完整两层优化、跨代自进化和知识积累。

最小下一步不是先发明一套“Ascend 版 NCU 指标”，而是：

1. **把 CANN 已经提供但被丢弃的数据交给 agent；**
2. **让 M1 和外层诊断真正运行；**
3. **接入 msprof；**
4. **完成一次 PyPTO 参照晋升和 probe-certified KB 入库。**
