# ForgeKernel 的 PyPTO / Ascend 后端

> 对应分支 `feat/pypto-ascend` 的当前实现
>
> 目标后端 PyPTO on Ascend，支持 910B/C、950PR/DT 等变体
>
> 文中的实测示例来自 Ascend 950PR、CANN 9.2、PyPTO 0.3.0、torch_npu 2.13
>
> 这份文档只讲 ForgeKernel 怎样接 Ascend，不代替华为的 CANN/PyPTO 文档。

把 PyPTO 接进 ForgeKernel，第一件要解决的事是让一次 kernel 优化在 Ascend 上完整跑完。代码能编译，能跑出一个时间，离这个目标还很远。候选要经过官方 CANN-Bench 的构建和正确性检查，参照要留在 student 看不到的地方，测出来的时间要知道属于哪一层，失败以后还要能判断是编译、精度、超时还是 runner 出了问题。后端的大部分工作都花在这些边界上。

## 1. 后端要守住的四件事

执行面必须真实。候选每次都要重新构建，再跑官方 20 个 case 的正确性和计时。agent 自己写的 probe 只能用来找方向，不能替代官方结果。一个 job 如果只跑出部分报告，或者 evaluator 非零退出，这次评测就不能算通过。

参照边界必须真实。blackbox 模式下，student 能看到 milestone、接口桩和公开知识库，看不到参照源码，也拿不到能解开参照载荷的密钥。teacher 可以运行、计时和 profile 参照，但不能把源码交给 student。这个边界由任务树、agent policy 和路径 deny 一起维持。

证据口径必须真实。AI-Core E2E、官方 `kernel_details`、msprof 里的 task row、CANN benchmark baseline 和 campaign 自己测的参照，是五种不同的东西。它们可以互相解释，不能直接相加，也不能换一个名字以后混进同一个比值。

失败必须能恢复。编译失败、精度失败、官方超时、profile 缺失和 runner 中断，在轨迹里要留下不同的原因。全都写成“精度失败”以后，下一轮 agent 会沿着错误方向修，外层诊断也失去了依据。

这四件事决定了后端的工程重点。任务要冻结，参照要隔离，证据要统一格式，双层循环要能根据失败类型重新决策。Ascend 版的 NCU 指标并不是这里的重点。

## 2. 一次 campaign 怎样跑

整套系统分成控制面和执行面。

```text
┌──────────────────────────── 控制面 ────────────────────────────┐
│  forge CLI / run snapshot / task card                          │
│  teacher  recon、distill、diagnose、patch、KB certification     │
│  student  编辑 kernel.py、写 probe、提交 commit                 │
│  driver   gate、KEEP/REVERT、confirmation、evolution            │
└───────────────────────────────┬─────────────────────────────────┘
                                │ SSH 或本机投递
┌───────────────────────────────▼─────────────────────────────────┐
│ 执行面  Ascend 950PR                                            │
│  CANN-Bench 冻结 worktree + 隔离 venv                           │
│  gpu_entry.sh → eval/cann_bench.py → 官方 kernel_eval           │
│  构建、正确性、trace_view 或 kernel_details                     │
│  msprof 原始 PROF 导出 → kernel_details.csv / op_summary.csv    │
└─────────────────────────────────────────────────────────────────┘
```

控制面保存任务快照，组织 teacher 和 student，记录每一轮 gate 的结果。执行面只负责构建和测量。两边之间传的是任务树、提交分支和带校验的证据文件。agent 在对话里说“这次应该快了”不会进入任何 gate。

## 3. 任务卡和两棵树

`src/forgekernel/task.py` 在通用任务模型里加了 PyPTO 需要的几项声明。任务语言是 `pypto`，架构是具体 Ascend 变体（如 `ascend950pr`），榜单套件是 CANN-Bench，知识库指向 `hw_kb/pypto/<arch>`。任务本身只描述问题和评测方式，不描述某一次实验的机器状态。

硬件不是写死在代码里的。`arch` 记录具体变体，支持 `ascend910b`、`ascend910c`、`ascend950pr`、`ascend950dt` 以及后续的 `ascend*` 名称。知识库查找先找 `hw_kb/pypto/<arch>/`，没有精确树时回退到 `hw_kb/pypto/ascend/`；后一个目录只放变体无关的接口说明，不放别的变体的认证 recipe。需要的话，`loop.toml` 的 `kb_arch` 还可以单独指定知识库版本，而不改任务的目标硬件。

`forge new` 创建 run 时会把任务卡复制进 run，交给 git 管理。之后即使有人修改 `tasks/<op>/`，已经存在的 run 也不会被追溯性改写。这个快照规则看起来笨，却让复现实验有了一个固定起点。

一次 campaign 至少有两棵树。

| 目录 | 内容 | 可见性 |
|---|---|---|
| `<run>/operators/<op>/` | `kernel.py`、`build.sh`、`setup.py`、`eval/`、`gpu_entry.sh`、`window.toml` | student 的 card |
| `<run>/operators/_<op>ref/` | 参照实现或密封参照，以及独立评测配置 | teacher 使用，student 不可见 |

student workspace 是 card 的 clone。参照树不在它的可见路径里，密钥也不会跟着 student job 走。任务卡和执行树分开以后，teacher 可以在参照树上跑 recon，student 仍然只看到自己的 card。

## 4. 三种参照，三种用途

后端把三类锚点分开保存。

| 名称 | 用途 | 是否决定停止门槛 |
|---|---|---|
| campaign architecture reference | 当前 campaign 要复现或超过的实现 | 是 |
| CANN benchmark baseline | 官方榜单的 baseline | 否，只作评分背景 |
| CANN `t_hw_us` | 官方定义的理论或硬件下界 | 否 |

CANN 官方的 `baseline_perf_us`、`t_hw_us` 和 `perf_score` 都留在 `cann_bench` 结构里。campaign 参照通过单独的测量和 `best_confirmation.json` 管理。两类数字可以放在同一份报告里解释，不能写进同一个 `ref_*` 字段。

CANN 参照的密封由 `src/forgekernel/cannref.py` 完成。控制器保留 `ref_bin/ref_payload.bin` 和 gitignored 的 key，参照源码被打包成 keyed payload。执行机上的 `eval/cann_bench.py` 在临时目录里解密、物化源码，构建结束就删掉临时目录。student job 没有参照树，也没有 key。blackbox teacher 可以运行和 profile 这份参照，但读不到源码。

这套机制解决的是参照源码跨过 agent 边界的问题。恶意代码隔离仍然要靠 agent policy、路径 deny 和作业树分离，不能把密封 payload 当成一个完整的沙箱。

source 模式下，teacher 可以读明确声明的 oracle。student 仍然不能读。`hw_kb` 里也不能出现 oracle 的实现细节。provenance 记录和 agent 可读的 recipe 要分开，teacher 的 recon 仍然以独立测量为准，不能因为“源码里这样写”就跳过实测。

## 5. 官方评测怎样接

`src/forgekernel/card_template/pypto_gpu_entry.sh` 提供了一组很小的任务面。

```text
preflight   环境、msprof、NPU smoke
correctness 正确性
perf        性能
all         正确性加性能
cardref     当前 card 的参照
ref         参照树
cann        官方 CANN-Bench
profiler    PyPTO 专用，和官方评测一起采集
ncu         明确不可用
cmd         teacher/student probe
```

`preflight` 会检查 CANN-Bench、PyPTO、torch_npu 和一次 NPU tensor 运算。它只能说明环境可用，不能说明候选正确。`ncu` 在 PyPTO 任务上会直接报不可用，不会伪装成一个空的 Ascend profiler。

`src/forgekernel/card_template/eval/cann_bench.py` 是官方评测和 ForgeKernel 之间的转换层。每次 job 开始时，它先清掉旧 wheel、旧 `kernel_impl.py` 和旧的构建目录，保证这次构建的是当前提交。然后它调用固定版本的 `kernel_eval.cli`，把官方 JSON 转成 ForgeKernel 的 correctness 和 performance 合同，再把官方报告、evaluator log、msprof CSV 归档成带 SHA-256 的 artifact 行。

这里有几条不能松的语义。

| 规则 | 结果 |
|---|---|
| 官方 20 个 case 全部通过 | 才允许产生 performance reading |
| `baseline_perf_us`、`t_hw_us`、官方 score | 原样保存，不改写成 campaign reference |
| evaluator 非零退出 | 即使有部分报告，也不能作为 gate 成功 |
| 构建失败、超时、精度失败 | 保留各自的原始错误，不折叠成同一个字符串 |

计时口径由 card 声明，adapter 按声明读取。

| `timing_basis` | 数据来源 | 含义 |
|---|---|---|
| `cann_trace_view` | `op_times.trace_view.aicore_e2e` | AI-Core E2E，排除 host dispatch，SwiGlu 使用 |
| `cann_kernel_details` | 官方 `kernel_details` 的 per-case 时间 | kernel-only，GQA 的参照和候选使用 |

两边的边界必须一致，也不能跨 ladder 混用。card 采集 `kernel_details`，adapter 却去读 trace_view，这种评测会被标成“测量不可用”，不会回退到一个看起来能用的数字。新的任务卡生成器会检查这两项是否一致。

ForgeKernel 的 campaign 参照先保留每个 case 的官方时间，再取几何平均。

\[
T=\exp\left(\frac{1}{N}\sum_{i=1}^{N}\ln t_i\right)
\]

一个极慢的 case 会明显拖累整体。SwiGlu 的优化后来盯住了两个尾部 case，原因就在这里。

## 6. msprof 能回答什么

正确性和计时回答“有没有过、花了多久”，不回答“时间花在哪个 task 上”。`eval/ascend_profiler.py` 在官方评测结束后找到原始 `PROF_*` session，调用真实的 `msprof --export=on`，解析 `kernel_details.csv` 和 `op_summary_*.csv`，记录 task type、duration、block dim 和 launch count，再把原始 CSV 和 msprof 日志一起导出。

读这些表时，有几条边界要先记住。

`kernel_details` 和 `op_summary` 是同一批 session 的两个视图。把两边的时间相加，会得到一份重复计算的账。

`launches` 是所有记录到的 launch，不是“每次调用只有一次”的证明。PyPTO 的一次 JIT 调用可能对应多个 task row。必须看 task type 和 block 信息，不能数行数。

缺失的字段要写 `unavailable`。msprof 没有 NCU 的 warp stall、register 或 bank conflict 字段，不要为了表格好看构造一组等价指标。

每次迭代都要记录 profiler，KEEP 和 REVERT 都一样。这样回退的时候还能查到当时的硬件事实，而不是只留下一句“更慢了”。

`src/forgekernel/profiling.py` 把 backend 显式化。

| backend | 用途 |
|---|---|
| `ncu` | 仅 NVIDIA |
| `msprof` | Ascend 原始采集和导出 |
| `cann_trace_view` | Ascend 官方计时口径 |
| `none` | 不采集 |

PyPTO 不会提交 NCU job，也不会重试“empty NCU”。milestone 只能引用对应 backend 允许的通道，`msprof-direct`、`msprof-derived`、`cann-trace-view`、`probe` 和 `correctness`。把 parse 出来的时间标成 `ncu-derived` 会被 `channel_violations` 拦下。

## 7. 一次迭代留下什么

`src/forgekernel/inner/trajectory.py` 的 `IterationRecord` 在通用字段之外加了 `cann_bench` 和 `profiler`。`cann_bench` 保存官方 score、baseline、hardware bound 和 per-case 信息。`profiler` 保存 backend、CSV task table、不可用字段和 raw artifact 索引。每条记录还带有逐 workload 的 device median、KEEP/REVERT、base SHA 和确认信息。

一条记录要能回答五件事。这次有没有通过官方正确性。哪个 case 最慢。慢在什么 task 上。这个判断是 KEEP 还是 REVERT，基于哪次 job。这次变化有没有对应的 commit。五个问题答不上来，这条记录就只是一段聊天。

`src/forgekernel/evidence.py` 接收 job 输出里的 `FORGE-ARTIFACT:` 行，只接受相对路径，限制单文件和总大小，校验 SHA-256，然后写进 `.jobs/<job>/artifacts/`。本机不需要挂载远端路径，报告里的数字也能追到当时的官方 JSON、CSV 和日志。

## 8. teacher 和 student 怎样分工

`meta/pypto/*.md` 给 PyPTO 准备了一套独立的 prompt overlay。`distill_prompt.md` 和 `distill_blackbox_prompt.md` 从测量和 recon 生成 M0 到 Mk。`recon_prompt.md` 使用官方测量、msprof 和定向 NPU probe。`diagnose_prompt.md` 看 KEEP/REVERT、case 时间、score 和 msprof 行，做外层诊断。`patch_prompt.md` 只改 milestone 指出的失败点。`kb_recipe_prompt.md` 和 `kb_harvest_prompt.md` 负责 probe 认证和收尾收割。

teacher 要把 Ascend 上测到的事实翻成可执行决策。buffer 预算、tile 形状、数据布局、DMA 和计算顺序、同步边界、shape 特化，都要落到 milestone 里。泛泛讲 CUDA 经验没有用，PyPTO 的 lowering 和 UB 预算不会因为一段熟悉的 GPU 说法就改变。

student 能读 milestone、objective trajectory、`hw_kb/pypto/<arch>` 和本地允许的 PyPTO/PTO API。它能编辑 `kernel.py`，能写 `notes/*.py` 做单合同 probe，也能请求驱动提交 probe 或 commit。它不能读参照树、其他 run、密钥或未认证的参照源码。

M0 先把真实架构定下来，包括算子拆分、数据流、dtype、tile、buffer、core 角色、同步和 shape 分派。M1 以后只调 milestone 已经声明的 lever，不在拓扑之外自由发挥。M0 的 gate 看官方正确性和可测的架构合同，M1 才进入性能优化。失败的 probe、编译错误和 case 错误都会写回 student，避免 agent 自己给自己判对。

## 9. 知识库只收 probe 认过的机制

PyPTO 的知识库在 `hw_kb/pypto/<arch>/`。现在收录的条目都和具体算子解耦，并且经过真实 NPU probe 认证。

| 条目 | 已验证的结论 |
|---|---|
| `valid_shape_assemble` | valid-only 写回可行，不能假设无效 view 区域自动为零 |
| `pypto-operand-tile-construction` | rank-3 的 per-axis `view` 加 `reshape` 可以构造 batched cube operand |
| `pypto-dynamic-execution` | 一次 PyPTO 调用对应多个 msprof task row，行数不等于调用数 |
| `device_floor_sandwich` | device-event sandwich 能测出每 launch 的 floor，也记录了不可用的 kineto/msprof 导出通道 |

KB 的进入条件是 probe 认证。agent 解释得再顺，没有实测就只是一条假设。失败的假设和重试条件留在 provenance 和 gap queue 里，不写成可复用 recipe。这样做会让知识库长得慢一些，但下一位 agent 读到的是能够重新验证的东西。

## 10. 下一代怎样长出来

`scripts/self_evolve.py` 支持 PyPTO 的下一代任务生成。当前代先要有完整的 confirmed best，student 和新鲜 reference 各至少 3 个 job，然后按两侧测量的离散度算噪声地板。收益超过噪声，才允许晋升。

晋升时，脚本生成 `tasks/<op>-g<k>/`。上一代 best 进入新的 sealed reference tree，candidate 只拿到 signature-only stub，`oracle_mode` 设成 `blackbox`。新的 reference 会重新测量，target 也会调高。student 拿不到上一代的代码，只拿到接口和 milestone。

这里没有“把 best 复制一份继续改”的捷径。参照源码留在 reference 树的密文 payload 里，student 看到的是接口桩。跨代积累靠的是重新测量和重新实现，不靠把旧代码换个目录。

## 11. 运行配置和复现

teacher 和 student 的模型由任务配置决定，后端代码不写死具体模型。`[agent]` 给两者共同默认值，`[agent.teacher]` 和 `[agent.student]` 可以分别覆盖 provider、model、effort 和 timeout。

```toml
[agent]
provider = "native"
model = "your-model"

[agent.teacher]
provider = "teacher-gateway"
model = "your-teacher-model"

[agent.student]
provider = "student-gateway"
model = "your-student-model"
```

provider 名来自 `global_config/backends.toml`，key 由该表的 `key_env` 指向的变量提供。后端没有隐藏的 teacher 环境变量覆盖；换模型必须同时显式配置 provider。密钥只从环境变量读取，不进入命令行、任务快照或 git。

一次 run 要留下这些复现信息。task snapshot，`run.json` 里的机器、task source 和创建时间，operator card 的 git HEAD，每次 job 的官方 JSON、日志和 artifacts，`status.jsonl` 和 `meta_status.jsonl`，以及 `best_confirmation.json` 和 `report.md`。少一项，后面就只能靠记忆补。

## 12. 八条不变量

下面这些规则是后端能自证的基础。

| 规则 | 原因 |
|---|---|
| 不把官方 benchmark baseline 当 campaign reference | 两者的测量对象和停止门槛不同 |
| 不把 trace_view、kernel_details 和 msprof task 时间相加或互换 | 它们统计的范围不同，相加会重复计算 |
| 不把 agent 自报的 probe PASS 当官方正确性 | probe 只覆盖单合同，官方评测覆盖 20 个 case |
| 不把部分报告当 gate 成功 | 构建或 evaluator 中断时，剩余 case 没有测到 |
| 不把参照源码或密钥交给 student | 否则 blackbox 失去意义 |
| 不把不同测量口径的数字放在同一 ladder 上 | 比值会失去可比性 |
| 不把一次乐观停止当最终结论 | 必须经过完整的 student 和 reference confirmation |
| 不把没有 probe 的机制写成 KB recipe | 知识库要能重新验证 |

## 13. 已知限制

论文的 Ascend 结果是 910C、CANN-Bench 1.1.2、hosted scoring 和 8 个 Level-4 算子。当前实现支持多种 Ascend 变体，但具体 run 的硬件、榜单版本和评分方式由任务卡决定；本报告中的示例不能当作论文分数的复现。

PyPTO 的编译和运行时行为强依赖 CANN、torch_npu 和 JIT 版本。升级以后，原来的 tile、view 和动态执行结论都要重新 probe。msprof 的 task row 也不等于 NCU 的 warp/stall 视角，这里没有 register、bank conflict 和 SASS 证据。

大规模 shape 特化会放大 JIT 编译时间，所以 campaign 要把架构探索和全量 gate 评测的成本分开算。一次 20-case 评测可能要几分钟到几十分钟，几次 probe 编译就能吃掉一个外层循环的预算。

这套后端能保证每个数字有来源，每次回退有原因。它不能保证模型在有限预算内找到接近硬件上限的架构，GQA 的尝试就是一个明确的例子。

## 相关文档

- 两次真实优化尝试 [pypto-optimization-case-studies-zh.md](pypto-optimization-case-studies-zh.md)
- 边界与安全 [blackbox-mode.md](blackbox-mode.md)
- 知识库使用与收割 [kb-harvest.md](kb-harvest.md)
