# ForgeKernel PyPTO / Ascend 后端技术报告

> 版本：`feat/pypto-ascend` 当前实现  
> 目标平台：Ascend 950PR、CANN 9.2、PyPTO 0.3.0、torch_npu 2.13  
> 评测任务：CANN-Bench 2.0.0，固定 commit `059cb61d…`  
> 说明：本文描述的是 ForgeKernel 的 PyPTO 后端实现，不是华为 CANN/PyPTO 的官方架构文档。

## 1. 设计目标与边界

ForgeKernel 的核心不是“让模型直接写出一个 kernel 文件”，而是把一次 kernel 优化组织成可测量、可回退、可复现的 campaign：

```text
参照实现
  → 可运行/可计时/可 profile 的行为测量
  → 架构优先的 milestone 课程
  → coding agent 逐项实现
  → 官方正确性与固定计时口径
  → KEEP / REVERT / 外层重新诊断
  → 下一轮或下一代参照
```

PyPTO 后端要解决的不是一个单纯的“语言适配”问题，而是把上述闭环完整接到 Ascend 上。它必须同时满足四个约束：

1. **执行面真实**：候选必须经过官方 CANN-Bench 构建、正确性和计时，不接受 agent 自测数字。
2. **参照边界真实**：在 blackbox 模式下，student 只能看到 milestone，不接触参照源码或可解码载荷。
3. **证据口径真实**：AI-Core E2E、msprof kernel task row、官方 benchmark baseline 和 campaign reference 是不同概念，不能混用。
4. **失败可恢复**：编译失败、精度失败、超时、profile 不可用和 runner 中断必须被区分，不能都记成“精度失败”。

因此，后端的工程重点落在“冻结任务、隔离参照、统一证据格式、驱动双层循环”上，而非引入一套 Ascend 版 NCU 指标。

## 2. 总体架构

整体可以分成控制面与执行面。

```text
┌──────────────────────────── 控制面（controller） ────────────────────────────┐
│  forge CLI / run snapshot / task card                                        │
│  teacher: recon, distill, diagnose, patch, KB certification                 │
│  student: edit kernel.py, local probes, commit                               │
│  deterministic driver: gates, KEEP/REVERT, confirmation, evolution           │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ SSH / local delivery
┌───────────────────────────────▼──────────────────────────────────────────────┐
│ 执行面（Ascend 950PR）                                                        │
│  CANN-Bench frozen worktree + isolated venv                                  │
│  gpu_entry.sh → eval/cann_bench.py → official kernel_eval                    │
│  kernel build / correctness / trace_view or kernel_details                   │
│  msprof raw PROF export → kernel_details.csv / op_summary.csv                │
└──────────────────────────────────────────────────────────────────────────────┘
```

控制面负责“决策和记录”，执行面负责“构建和测量”。两者之间只传任务树、提交分支和带校验的证据文件，不传“agent 认为的性能数字”。

## 3. 任务模型与冻结任务卡

### 3.1 任务声明

`src/forgekernel/task.py` 在通用任务模型中加入了：

- `language = "pypto"`；
- `arch = "ascend950"`；
- CANN-Bench leaderboard suite；
- PyPTO 对应的 `hw_kb/pypto/<arch>` 知识库。

任务本身只描述问题，不描述某次实验的环境。运行时由 `forge new` 创建快照，再由 `taskcard.py` 安装任务卡和参照树。

### 3.2 两类树

一次 campaign 至少有两棵树：

```text
<run>/operators/<op>/           student 的 card
<run>/operators/_<op>ref/       teacher-only reference tree
```

- card 包含 `kernel.py`、`build.sh`、`setup.py`、`eval/`、`gpu_entry.sh`、`window.toml`。
- reference tree 包含参照实现或密封参照，以及独立的评测配置。
- student workspace 是 card 的 clone；参照树不在 student 的可见路径中。

任务卡在 run 开始时被复制和 git 化，因此后续修改 `tasks/<op>/` 不会追溯性改变已存在 run。这个“快照优先”规则是复现实验边界的一部分。

## 4. 参照、密封与 source/blackbox

### 4.1 三类“参照”必须分开

后端明确区分三种时间/锚点：

| 名称 | 意义 | 是否决定停止门槛 |
|---|---|---|
| campaign architecture reference | 当前 campaign 要复现/超过的实现 | 是 |
| CANN benchmark baseline | 官方 benchmark 的 baseline | 否，只做官方评分背景 |
| CANN `t_hw_us` | 官方定义的理论/硬件下界 | 否 |

它们不会写入同一个 `ref_*` 字段。CANN 官方的 `baseline_perf_us`、`t_hw_us`、`perf_score` 等保存在 `cann_bench` 结构中；campaign 参照则通过单独测量和 `best_confirmation.json` 管理。

### 4.2 CANN sealed reference

`src/forgekernel/cannref.py` 将 PyPTO 参照源码打包为 keyed payload：

- 控制器只保留 `ref_bin/ref_payload.bin` 和 gitignored 的 key；
- 执行机上的 `eval/cann_bench.py` 在临时目录内解密并物化源码；
- 构建完成后删除临时源码；
- student job 不包含参照树，也没有 key；
- blackbox teacher 可以运行、计时和 profile，但不能读取源码。

这不是在宣称一个针对恶意代码的强沙箱；它解决的是“参照源码不跨过 agent/学生边界”的工程问题。真正的隔离仍由 agent policy、路径 deny 和作业树分离共同保证。

### 4.3 source 模式

source 模式下 teacher 可以读取明确声明的 oracle，但：

- student 仍然不能读；
- `hw_kb` 不能包含 oracle 的实现细节；
- provenance 与 agent-readable recipe 分离；
- teacher 的 recon 仍是独立测量，不把 “source 说了” 当作证据。

## 5. 官方评测链路

### 5.1 入口

`src/forgekernel/card_template/pypto_gpu_entry.sh` 提供最小任务面：

```text
preflight   环境/msprof/NPU smoke
correctness/perf/all/cardref/ref/cann
profiler   （PyPTO 专用）与官方评测一起采集
ncu        明确不可用，不伪装成 Ascend profiler
cmd         teacher/student probe
```

`preflight` 会检查 CANN-Bench、PyPTO/torch_npu 和一次 NPU tensor 运算；它只说明环境可用，不代表候选正确。

### 5.2 Adapter

`src/forgekernel/card_template/eval/cann_bench.py` 是官方评测与 ForgeKernel 之间的唯一转换层。它做四件事：

1. 清理旧 wheel、旧 `kernel_impl.py`，保证每个 job 构建当前提交；
2. 调用固定版本的官方 `kernel_eval.cli`；
3. 将官方 JSON 转成 ForgeKernel 的 correctness/performance 合同；
4. 归档官方 JSON、evaluator log、msprof CSV，并输出带 SHA-256 的 artifact 行。

关键语义：

- **官方 20 个 case 全量通过**才允许产生 performance reading；
- `baseline_perf_us`、`t_hw_us`、官方 score 原样保存，不改写成 campaign reference；
- 官方 evaluator 非零退出时，即使存在部分报告，也不得作为 gate 成功；
- 失败时保留官方 per-case 错误，不把“构建失败”“超时”“精度失败”折叠成同一个字符串。

### 5.3 两种计时口径

后端不强行统一所有 PyPTO 任务的计时口径，而是让 card 声明：

| `timing_basis` | 数据来源 | 含义 |
|---|---|---|
| `cann_trace_view` | `op_times.trace_view.aicore_e2e` | AI-Core E2E，排除 host dispatch；SwiGlu 使用 |
| `cann_kernel_details` | 官方 `kernel_details` 的 per-case 时间 | kernel-only；GQA 参照和候选使用 |

两者的边界必须一致，且不能跨 ladder 混用。一个常见问题是 card 采集 `kernel_details`，却要求 adapter 读取 trace_view；这时 adapter 会把该次评测标为“测量不可用”，而不是回退到假数据。新的任务卡生成器会检查这两项的一致性。

### 5.4 多 case 聚合

CANN-Bench 每个 case 的官方时间先被保留，然后取几何平均：

\[
T=\exp\left(\frac{1}{N}\sum_{i=1}^{N}\ln t_i\right)
\]

这使一个极慢 case 会显著拖累整体。SwiGlu 的优化之所以有意义，正是因为它针对了两个主导几何平均的尾部 case。

## 6. msprof 证据链

### 6.1 为什么需要独立 profiler

正确性和计时只回答“是否通过、多久”，不回答“时间花在哪个 task 上”。`eval/ascend_profiler.py` 在官方评测完成后：

- 找到原始 `PROF_*` session；
- 调用真实 `msprof --export=on`；
- 解析 `kernel_details.csv` 和 `op_summary_*.csv`；
- 记录 task type、duration、block dim、launch count；
- 导出原始 CSV 和 msprof 日志的校验和。

### 6.2 证据解释规则

- `kernel_details` 和 `op_summary` 是同一批 session 的两个视图，不能把时间相加；
- `launches` 是所有记录到的 launch，不是“每次调用一次”的证明；
- PyPTO 的一次 JIT 调用可能对应多个 task row，必须阅读 task type/block 信息，而不是数行数；
- 缺失的字段写 `unavailable`，不构造 NCU 等价字段；
- 每次迭代记录 `profiler`，KEEP 和 REVERT 都有硬件事实可查。

### 6.3 profiler backend 路由

`src/forgekernel/profiling.py` 将 backend 显式化：

```text
ncu              仅 NVIDIA
msprof           Ascend 原始采集/导出
cann_trace_view  Ascend 官方计时口径
none             不采集
```

PyPTO 不会提交 NCU job，也不会重试“empty NCU”。milestone 只能使用对应 backend 的验证通道：`msprof-direct`、`msprof-derived`、`cann-trace-view`、`probe`，不能把 parse 出来的时间标成 `ncu-derived`。

## 7. 轨迹与证据模型

### 7.1 每条迭代记录

`src/forgekernel/inner/trajectory.py` 的 `IterationRecord` 在通用字段之上增加了：

- `cann_bench`：官方 score/baseline/hardware bound/per-case 信息；
- `profiler`：backend、CSV task table、不可用字段和 raw artifacts 索引；
- 逐 workload 的 device median、KEEP/REVERT、base SHA、确认信息。

这样一次诊断可以回答：

1. 这次是否真的通过官方正确性？
2. 哪个 case 最慢？
3. 慢在什么 task 上？
4. 这个判断是 KEEP 还是 REVERT，基于哪次 job？
5. 这次变化有没有对应的 commit？

### 7.2 artifact 传输

`src/forgekernel/evidence.py` 接收 job 输出中的 `FORGE-ARTIFACT:` 行：

- 只接受相对路径；
- 限制单文件/总大小；
- 校验 SHA-256；
- 写入 `.jobs/<job>/artifacts/`；
- 不依赖远端路径在本机可访问。

因此报告中的数字可以追溯到本机保存的官方 JSON、CSV 和日志。

## 8. 双层 agent 循环

### 8.1 teacher / arch agent

`meta/pypto/*.md` 为 PyPTO 提供独立 overlay：

- `distill_prompt.md` / `distill_blackbox_prompt.md`：从测量和 recon 生成 M0→Mk；
- `recon_prompt.md`：使用官方测量、msprof 和定向 NPU probe；
- `diagnose_prompt.md`：基于 KEEP/REVERT、case 时间、score、msprof 行做外层诊断；
- `patch_prompt.md`：只修改 milestone 的失败点；
- `kb_recipe_prompt.md` / `kb_harvest_prompt.md`：probe 认证与 finish-time 收割。

teacher 的核心职责不是“讲 CUDA 经验”，而是把 Ascend 可观测事实转成可执行决策：buffer 预算、tile 形状、数据布局、DMA/计算顺序、同步边界、shape 特化。

### 8.2 student / coding agent

student 只能：

- 读 milestone、objective trajectory、`hw_kb/pypto/ascend950` 和本地允许的 PyPTO/PTO API；
- 编辑 `kernel.py`；
- 写 `notes/*.py` 做单合同 probe；
- 请求驱动提交 probe 或 commit。

student 不能读参照树、其他 run、密钥或未认证的参照源码。

### 8.3 M0 与 M1

- **M0** 固定真实架构：算子拆分、数据流、dtype、tile、buffer、core 角色、同步和 shape 分派。
- **M1+** 只调 milestone 已声明的 lever，不在拓扑之外自由发挥。
- M0 的 gate 依赖官方正确性和可测的架构合同；M1 进入性能优化。
- 失败的 probe、编译错误和 case 错误都写回 student，避免“agent 自己认为对”。

## 9. KB 与跨算子复用

PyPTO KB 位于 `hw_kb/pypto/ascend950/`。当前收录的都是 operator-agnostic、真实 NPU probe 认证的机制，而不是“某次成功配置”的复刻：

- `valid_shape_assemble`：valid-only 写回；明确 no-go：不能假设无效 view 区域自动为零。
- `pypto-operand-tile-construction`：rank-3 的 per-axis `view` + `reshape` 构造 batched cube operand。
- `pypto-dynamic-execution`：一次 PyPTO 调用对应多个 msprof task row，不能把行数当调用数、把行时间相加当 kernel 时间。
- `device_floor_sandwich`：device-event sandwich 测得的 per-launch floor，以及不可用的 kineto/msprof export 通道。

KB 的进入条件是 probe 认证，不是“agent 解释合理”。失败假设和 retry precondition 留在 provenance / gap queue，不写成可复用 recipe。

## 10. 跨代自进化

`scripts/self_evolve.py` 支持 PyPTO 的下一代任务生成：

1. 只有当前代存在完整 confirmed best 才允许晋升；
2. student 和 fresh reference 各至少 3 个 job；
3. 按两侧测量离散度计算噪声地板；
4. 收益超过噪声才晋升；
5. 生成 `tasks/<op>-g<k>/`：
   - 上一代 best 进入新的 sealed reference tree；
   - candidate 只得到 signature-only stub；
   - `oracle_mode=blackbox`；
   - 重新测量 reference、设置更高的 target。

晋升不是把代码复制给 student。参照源码只在 reference 树的密文 payload 中，student 看到的是接口桩和 milestone。

## 11. 运行配置与可复现性

当前 PyPTO 任务显式声明：

```toml
[agent]
provider = "deepseek"
model = "deepseek-flash[1m]"
effort = "max"
```

这保证 teacher/student 都走 DeepSeek 官方 Anthropic 接口，而不是被历史 `TEACHER_*` 环境变量带到其他 provider。密钥只从 `.env` 读取，不进入命令行、任务快照或 git。

一次 run 的可复现信息包括：

- task snapshot；
- `run.json`（机器、task source、创建时间）；
- operator card 的 git HEAD；
- 每次 job 的官方 JSON、日志和 artifacts；
- `status.jsonl` 和 `meta_status.jsonl`；
- `best_confirmation.json` 和 `report.md`。

## 12. 设计不变量

以下规则是后端可审计性的基础：

1. **不把官方 benchmark baseline 当 campaign reference。**
2. **不把 trace_view、kernel_details 和 msprof 内部 task 时间相加或互换。**
3. **不把 agent 自报的 probe PASS 当官方正确性。**
4. **不把部分报告当 gate 成功。**
5. **不把参照源码或密钥交给 student。**
6. **不把不同测量口径的数字放在同一 ladder 上。**
7. **不把一次乐观停止当最终结论；必须经过完整 confirmation。**
8. **不把没有 probe 的机制写成 KB recipe。**

## 13. 已知限制

- 论文 Ascend 结果是 **910C + CANN-Bench 1.1.2 + hosted scoring + 8 个 Level-4 算子**；当前环境是 **950PR + CANN-Bench 2.0.0 + 本地公开 case**，不能严格复现论文分数。
- PyPTO 的编译/运行时行为强依赖 CANN、torch_npu 和 JIT 版本；升级后需要重新 probe。
- msprof 的 task row 模型并不等价于 NCU 的 warp/stall 视角；当前没有 register、bank conflict、SASS 等证据。
- 大规模 shape 特化会放大 JIT 编译时间，因此 campaign 必须区分“架构探索”和“全量 gate 评测”的成本。
- 本后端能保证“每个数字有来源、每次回退有原因”，但不能保证模型在有限预算内找到接近硬件上限的架构。

## 相关文档

- 两次真实优化尝试：[pypto-optimization-case-studies-zh.md](pypto-optimization-case-studies-zh.md)
- 当前验证状态与证据索引：[ascend-agent-validation.md](ascend-agent-validation.md)
- 迁移与运行交接：[../MIGRATION_PYPTO.md](../MIGRATION_PYPTO.md)
