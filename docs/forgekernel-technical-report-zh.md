# ForgeKernel 技术思路汇报

> 基于 ForgeKernel 仓库、18 页主论文，以及公开的 CANNBench / JustGoal 资料整理。
>
> 整理日期：2026-11-03。950DT 榜单会动态更新，本文明确区分论文结果、新闻快照和当前 API 数据。

## 1. 一句话概括

**ForgeKernel 不是提出了一种新的 GPU kernel 算法，而是提出了一套“以参照 kernel 为老师、以硬件测量为真值、能重建架构并跨代积累知识”的自动优化框架。**

它的“自进化”**不是训练或更新模型权重**，而是测试时的迭代搜索：

\[
R_{n+1}=K_n
\]

即把这一代确认过的最佳 kernel \(K_n\)，变成下一代的参照 \(R_{n+1}\)，再继续优化。

---

## 2. 它要解决的不是“不会写 kernel”，而是“过早封顶”

论文先用通用 Claude Code `/goal` agent 做 baseline。结果是：

- 10 个任务中 7 个超过专家参照，几何平均约 **+17%**；
- 但无论最终超过还是落后，都会收敛到一个自身无法继续改进的 fixed point。

作者把这个 fixed point 归因于两个问题：

1. **架构过早冻结**  
   Agent 在最初几轮决定 tile、warp 分工、pipeline、数据布局，之后只做局部调参。局部参数全部无收益，并不意味着不存在更好的架构。

2. **知识随 session 消失**  
   一次运行留下了 kernel，却没有留下经过验证的硬件经验；下次运行还要重新支付相同的探索和 GPU 测量成本。

所以 ForgeKernel 的目标不是单纯提高一次搜索的强度，而是让搜索能够：

- 回到架构层重新决策；
- 越过当前最好实现继续运行；
- 把测量所得保留下来。

---

## 3. 总体工作流

一次 campaign 的输入可以抽象为：

\[
(\text{operator},\text{shape},R,\tau)\rightarrow K
\]

其中 \(R\) 是参照 kernel，\(\tau\) 是目标加速比。收益定义为：

\[
g(K,R)=t_R/t_K
\]

候选 kernel 必须同时满足：

- 独立正确性检查通过；
- 与参照在**同一个 GPU 作业中**测得 \(g(K,R)\geq\tau\)。

整体流程是：

```text
参照 kernel
   │
   ├─ 运行 / 计时 / NCU / SASS / 定向硬件探针
   ▼
arch agent（仓库中的 teacher）
   │
   ├─ 生成架构优先的 milestone：M0 → M1 → … → Mk
   ▼
coding agent（仓库中的 student）
   │
   ├─ 实现一项或一组决策
   ▼
GPU：正确性检查 + 配对计时 + profiling
   │
   ├─ 内层：继续实现、重试或回退
   ├─ 外层：诊断瓶颈、给杠杆定价、修改 milestone
   ▼
最佳 kernel
   ├─ 晋升为下一代参照
   └─ 探针认证的硬件经验进入共享知识库
```

对应实现主要位于：

- `src/forgekernel/loop/driver.py`
- `meta/recon_prompt.md`
- `meta/distill_prompt.md`
- `meta/student_prompt.md`
- `meta/diagnose_prompt.md`

---

## 4. 核心技术思路

### 4.1 从硬件行为恢复“架构”，而不是恢复源码

论文主实验把参照作为密封二进制：agent 可以运行、计时、profile、反汇编，但不能读取源码。

它试图恢复的是决定性能上限的结构，例如：

- kernel 如何拆分；
- grid/block/cluster 和 tile 形状；
- warp 角色及寄存器、shared memory 预算；
- 数据驻留与搬运方式；
- pipeline stage、同步与等待关系；
- 哪些计算与哪些 MMA/内存操作重叠。

仓库中的 `meta/recon_prompt.md` 比论文正文更具体，使用：

- NCU launch/occupancy/traffic/stall 指标；
- SASS 指令与内存访问 footprint；
- 主循环 schedule 和 wait-target 关系；
- 寄存器 live-range 分析；
- per-instruction stall ledger；
- 能区分多个架构假设的定向探针。

因此它不是简单的黑盒随机搜索，而是**测量引导的架构逆向与重建**。

> 论文主实验是黑盒模式；仓库也支持 `source` 模式，此时 teacher 可以读参照源码，但 student 始终不能读。两种模式不能混为一谈。

### 4.2 架构优先的 milestone 课程

参照分析结果被蒸馏为自然语言决策图。每项决策包含五元组：

- **Action**：做什么；
- **Purpose**：消除什么瓶颈；
- **Trigger condition**：什么测量症状触发；
- **Cost**：寄存器、SMEM、同步等代价；
- **Depends on**：依赖与原子组合关系。

关键在于 **M0 先确定全局架构**：

- 算法契约：kernel 拆分、数据流、中间 tensor ABI、精度；
- 硬件架构：tile、warp 分工、寄存器/SMEM 布局、MMA atom；
- 资源拓扑：buffer 生命周期、访问原子、驻留策略、发射顺序和延迟隐藏关系。

M0 验收后，student 在后续 `M1 → Mk` 中只调：

- stage 数；
- swizzle 模式；
- wait 深度；
- 其他既定拓扑内的参数。

这样可以避免“先写一个容易写的版本，最后再通过大重构补架构”的死路。

但 **M0 并非不可更改**：如果测量证明当前拓扑本身无法达到目标，outer loop 会重新打开 M0。也就是说：

> 对 student 而言架构被冻结，对 arch agent 而言架构仍可被证据推翻。

相关文件：

- `meta_milestone.md`
- `docs/milestones-template.md`
- `docs/design/milestone-distillation.md`

### 4.3 双层循环：内层优化代码，外层优化“优化路线”

**内层 student loop**

- 按 DAG 顺序实现一项或一组决策；
- 每轮由 GPU 作业判正确性、配对计时并 profiling；
- 正确性失败硬回退；
- 不能凭 agent 自己的描述认定优化生效。

**外层 arch loop**

在连续无新最佳、决策图耗尽或 gate 卡住时介入：

- 读取目标、性能曲线、NCU 和结构化轨迹；
- 给候选优化杠杆做 A/B 定价；
- 运行 verdict probe 区分独立效果和耦合效果；
- 判断是指导不足，还是架构本身达到上限；
- 修改 milestone。

patch 分为两种成本：

- **additive patch**：保留已有前缀，继续执行；
- **structural patch**：使相关依赖失效，从依赖闭包重新执行。

`src/forgekernel/resume/invalidation.py` 实现了依赖失效与局部重开；`src/forgekernel/inner/trajectory.py` 则只记录客观事实——代码改动、正确性、计时、NCU、KEEP/REVERT，而不记录 student 的主观解释。

另一个重要细节是 `⊗` 原子组合：有些优化单独落地必然变慢，只有组合完成才获益，不能因中间版本回退就误判整个方向无效。

### 4.4 把“测量纪律”做成系统机制，而不是依赖 prompt 自觉

这是仓库工程中很重要的一部分：

- 参照密封，只在 GPU job 内解封；
- student 的访问权限由确定性 driver 控制；
- milestone 不包含参照 kernel 实现体；
- 数值必须标为 `[MEASURED]` 或 `[ESTIMATE]`；
- milestone gate 同时检查正确性、性能和**机制指标**；
- 每个知识条目必须由 GPU 探针认证。

计时也不是简单比较两个绝对延迟。`src/forgekernel/card_template/eval/timing.py` 使用：

1. burn-in，让 GPU 进入相对稳定的热状态；
2. `ref → kernel → kernel → ref` 的 ABBA 镜像测量；
3. 同窗口计算 `ref_time / kernel_time`；
4. 最终结果再做多次确认。

其目的是减少节点差异、温度和测量顺序造成的几个百分点偏差——这些偏差足以淹没大部分单轮收益。

### 4.5 跨算子、跨代的硬件知识库

`hw_kb/` 按 GPU 架构保存与具体算子无关的硬件 recipe：

- 机制及触发症状；
- 最小可运行示例；
- 已验证的用法和坑；
- 对应的失败方向/no-go 记录。

知识入库要求 GPU 探针认证，避免把“某次成功实现中恰好出现的设计”误当成因果经验。后续 campaign 的 arch agent 可以按名字把 recipe 写进 milestone，student 再查具体硬件接口代码。

论文记录了：

- 79 次运行；
- 71 个条目、9 类知识；
- 553 次“既有知识在交付 kernel 中落地”的 entry-run 记录。

两条反馈边因此承担不同职责：

- **kernel 晋升**：抬高下一代的性能起点；
- **知识积累**：降低达到这个新起点的成本。

### 4.6 参照晋升：把一次收敛变成迭代局部搜索的重启

每代结束后，`scripts/self_evolve.py` 将确认过的最佳 kernel 复制为下一代参照，并设置新的目标，论文中使用每代 **+5%**。

新一代会重新：

- 测量参照；
- 分析架构；
- 锚定 gate 和机制指标；
- 搜索超过它的实现。

当提升低于同作业噪声地板时停止；当前实现中噪声地板至少为 **1.5%**，也会考虑确认测量的离散度。

本质上，这是一种带状态的 **iterated local search**：

> 上一代结束只代表上一次 campaign 的停止条件被触发，不代表硬件极限；晋升相当于从当前最优点、以更高目标重新开始搜索。

---

## 5. 论文结果应该怎样解读

论文覆盖 27 个算子、29 个“算子 × shape × GPU”对照 cell：

| 对照 | 结果 |
|---|---|
| 专家库 kernel | 22 个中有 18 个追平或超过，另有 2 个差距不到 1%；几何平均 **+1.6%**，最高 **+7.6%** |
| 三个其他 kernel 生成系统 | 公开可比的 7 个全部追平或超过；几何平均 **+39.4%**，最高 **+170.6%** |
| 自进化：GDN prefill | 在 `/goal` 已领先 FlashInfer 的 kernel 上再快 9.5%，累计约 **+100%** |
| 自进化：GQA attention forward | 在 `/goal` kernel 上再快 5%，累计比 FlashAttention-4 快约 **29%**；另一条自身晋升链连续两代后累计约 **+15%** |
| Ascend 910C | 仅将 NCU 换成 msprof，在 CANN-Bench 8 个 Level-4 算子上均超过最佳公开提交，几何平均 **+33.1%** |

需要注意：**+170.6% 是相对其他 agent 系统的 kernel，不是相对 FlashAttention/cuDNN 等最强专家库。** 对高度优化的专家库，论文中的典型收益只有几个百分点，这更符合任务难度。

---

## 6. 消融揭示了它真正的边界

### 6.1 参照不是可有可无，而是核心信息源

论文对三个算子做参照消融：

- 只给 PyTorch 实现和一个延迟数字：**三个全部停滞**；
- 给可 profile 的密封二进制：三者恢复；
- 再给源码：几何平均收益从约 +1.5% 提升到 +2.5%，并非数量级差异。

说明 ForgeKernel 更擅长：

> **分析并超过一个已经存在的好实现，而不是凭空写出第一个好实现。**

### 6.2 参照强度会传递到最终结果

DSA FP8 indexer 实验中：

- 以 DeepGEMM 强参照学习，最终为 **1.808 ms**；
- 以慢约 29% 的 cuDNN-frontend 为参照，最终为 **2.138 ms**；
- 两者都超过了各自参照，但最终结果仍相差约 **18%**。

所以“超过参照”不等于“达到同一水平”。这也是为什么自进化必须不断晋升当前最强 kernel。

---

## 7. 有公开的 Ascend 950DT 优化结果吗？

**有。公开程度可分为三层：论文、新闻快照、CANNBench 当前榜单。**

### 7.1 论文没有 950DT 结果

主论文公开的是 **Ascend 910C、CANN-Bench v1.1.2、8 个 Level-4 算子**的结果，不是 950DT。仓库 README 只额外声明 ForgeKernel 曾在 950DT 榜单取得第一名。

因此，**不能把论文中 910C 的 +33.1% 直接写成 950DT 的结果。**

### 7.2 2026-10-27 的新闻快照

[京报网转载的面壁智能消息](https://news.bjd.com.cn/2026/10/28/12009317.shtml)公布了当时的 950DT 截图数据：

| 指标 | 当时数值 |
|---|---:|
| 总分 | 3903.96 |
| 基准加速比 | 1.523× |
| 平均分 | 70.98 |
| 提交算子 | 55/60 |
| 第二名总分 | 2846.61 |

后续报道还出现过 4010.75 分、1.533× 等快照。榜单会持续更新，这些数字应理解为不同时间/提交状态的历史记录，不宜混成同一组指标。

### 7.3 CANNBench 当前公开 API：总分和 60 个算子逐项数据都有

CANNBench 的公开接口可以按 `hardware=950dt` 直接查询：

- [950DT / official-tasks / latest / solution  leaderboard JSON](https://cannbench.com/api/public/leaderboard?benchmark_slug=official-tasks&mode=solution&detail=operators&version=latest&hardware=950dt)
- [CANNBench 网页榜单](https://cannbench.com/leaderboard)

本文查询时，ForgeKernel 的最新记录更新时间为 **2026-11-02 23:02:37**，任务包为 **official-tasks 2.0.0**：

| 指标 | ForgeKernel 当前公开值 |
|---|---:|
| 总分 | **4232.63** |
| 第二名总分（CANNBot PyPTO S2 Frontier） | 4045.39 |
| 相对第二名的总分领先 | **约 4.63%** |
| 平均分（60 个算子） | **70.54** |
| 基准几何平均加速比 | **1.614×** |
| 理论/硬件锚几何平均比值 | 0.163× |
| 提交算子 | **60/60** |
| 通过用例 | **1178/1200（98.17%）** |
| 隐藏集 | 当前记录 `has_hidden_eval=false`，`hidden_metric=null` |

在这份 API 返回的 950DT 榜单中，ForgeKernel 排第一。第二名的基准几何平均加速比为 1.431×；ForgeKernel 相对其约高 12.76%。

**注意：CANNBench 总分同时包含编译、正确性和性能评分，不能只凭总分推断纯 kernel 加速比；上述记录也不应被描述为隐藏集成绩。**

### 7.4 950DT 的部分逐算子结果

公开 API 的 `metric.operators` 给出了全部 60 个算子的分数、通过用例和平均加速比。以下选取若干代表值：

| 算子 | Level | 平均加速比 | 通过用例 | 说明 |
|---|---:|---:|---:|---|
| LinearCrossEntropy | 4 | **7.868×** | 20/20 | 当前最高的算子级加速比 |
| KvCacheAppend | 3 | **7.276×** | 20/20 | 高收益算子 |
| ScatterBlockUpdate | 3 | **5.812×** | 20/20 | 高收益算子 |
| AttentionResidual | 4 | **3.949×** | 20/20 | 高收益 attention 相关算子 |
| MLA | 4 | **1.441×** | 20/20 | attention 类结果 |
| GQA | 4 | **1.069×** | 20/20 | 小幅超过基准 |
| FlashAttentionScoreGradEnhance | 4 | **0.849×** | 20/20 | 正确性全过，但慢于基准 |
| SparseFlashAttention | 4 | **0.589×** | 20/20 | 正确性全过，但慢于基准 |
| LSTM | 4 | **0.646×** | 20/20 | 正确性全过，但慢于基准 |
| TensorProgramVm | 5 | **0.424×** | 20/20 | 当前加速比最低的算子 |
| Unique | 3 | 2.157× | **9/20** | 不能只看加速比，覆盖不足显著拉低分数 |
| DynamicMxQuant | 4 | 1.247× | **10/20** | 同样存在覆盖不足 |

所以，950DT 的公开结果并不是“所有算子都快于基准”，而是**总榜第一、整体几何平均加速 1.614×，同时仍有若干慢于基准或用例未全过的算子**。

### 7.5 公开结果不等于本仓库已公开 950DT 复现材料

当前仓库的 `kernels/` 只列出 H100/B200 kernel，没有 950DT kernel 源码、对应 milestone 或优化轨迹。因而：

- **公开可核实**：CANNBench 总分、逐算子指标、通过率和关联 job；
- **本仓库未提供**：950DT kernel 源码及完整优化过程；
- 所以可以从榜单确认性能结果，但不能仅凭这个仓库逐项复盘 950DT 的具体优化手法。

---

## 8. 我的评价与保留意见

### 8.1 最有价值的创新

我认为不是某个 milestone 模板，而是下面三者的组合：

1. **把强参照变成架构先验**，避免从零搜索；
2. **允许测量证据推翻早期架构**，避免局部最优被封死；
3. **把 kernel 和硬件知识都变成后续运行的输入**，让优化过程有长期状态。

这使它区别于一次性代码生成、无状态 evolutionary search，以及只在一个 session 内运行的 agent loop。

### 8.2 证据较强与较弱的部分

- **系统级结果证据较强**：正确性门槛、同作业配对计时、多次确认和参照版本固定都比较严格。
- **组件级因果归因较弱**：
  - `/goal` baseline 使用 Fable 5.1，而 ForgeKernel 使用 Fable 5.1 + Opus 5，并非完全同模型对照；
  - 自进化实验同时改变参照、知识积累和外层架构循环，没有分别隔离贡献；
  - KB 的 553 次复用证明了“确实被使用”，但不是严格的 KB-on/KB-off 性能消融。

因此，论文较好证明了“**整个系统有效**”，但尚不能精确回答“**每个组件分别贡献多少**”。

### 8.3 工程边界

- 输出 kernel 深度特化于给定 shape 和硬件；通用的是 ForgeKernel 流程，不是同一个 kernel。
- 一次 campaign 需要数小时乃至更长的 GPU/前沿模型时间，不是廉价编译过程。
- 噪声地板上的 plateau 只是“本循环没找到更多”，不是硬件最优性证明。
- 闭源前沿模型使轨迹难以逐字复现；论文主要保证的是测量协议可复现。
- 950DT 的公开榜单结果与论文 campaign 是不同证据来源，应分别说明任务包、时间、覆盖范围和隐藏集状态。

---

## 9. 与 JustGoal 的关系

[JustGoal](https://arxiv.org/abs/2610.36434) 是另一篇论文，提出了一个很重要的评估原则：

> 专用 kernel agent 的每个机制，都应在**同模型、同任务范围**下，与“只给目标和约束”的 baseline 比较净增益。

其实验中：

- 单 shape 下，KDA-0.5 专用 harness 约为 JustGoal 的 **0.95×**，没有总体收益；
- 扩展到整个算子的全部 workload 后，KDA harness 相对 JustGoal 有约 **1.37×** 优势。

这不是对 ForgeKernel 的直接消融，不能据此否定 ForgeKernel；但它提示：随着基础模型增强，专用流程的价值可能从“提高单点极限”转向“组织更大范围任务、保留状态和保证测量可靠性”。

**ForgeKernel 最可能长期保留下来的部分，也未必是复杂的 prompt 和 milestone 形式，而是论文最后强调的两项资产：可验证的硬件知识，以及能把当前最佳实现继续作为输入的闭环。**

---

## 10. 主要材料

- [ForgeKernel 主论文（OpenReview）](https://openreview.net/pdf?id=EtV3f5BPD4)
- [ForgeKernel 仓库 README（中文）](../README_zh.md)
- [milestone 蒸馏设计](./design/milestone-distillation.md)
- [milestone 模板](./milestones-template.md)
- [黑盒参照与配对测量](./blackbox-mode.md)
- [知识库收割](./kb-harvest.md)
- [JustGoal 论文](https://arxiv.org/abs/2610.36434)
- [CANNBench 950DT 公开 JSON 数据](https://cannbench.com/api/public/leaderboard?benchmark_slug=official-tasks&mode=solution&detail=operators&version=latest&hardware=950dt)
- [950DT 新闻报道快照](https://news.bjd.com.cn/2026/10/28/12009317.shtml)
