# PyPTO 算子优化案例报告：SwiGlu 与 GQA

> 平台：Ascend 950PR、CANN 9.2、PyPTO 0.3.0、CANN-Bench 2.0.0  
> 模型：teacher/student 均为 `deepseek-flash[1m]`，全流程 DeepSeek  
> 说明：本文比较的是当前环境下、同一套 ForgeKernel 驱动的两次真实 campaign，不是论文 910C hosted 分数的复现。

## 0. 先看结论

两次尝试的结果差异很大：

| 算子 | 参照 | 结果 | 结论 |
|---|---|---|---|
| SwiGlu（20 个公开 case） | CANN-Bench 公开 PyPTO 实现 | 最终确认 **0.054738 ms**，参照 **0.065776 ms**，**1.20165×** | 达到目标，完成 3+3 确认 |
| GQA（20 个 Level-4 case） | 华为 `FusedInferAttentionScore` | M0 **20/20 正确**，但 **40.24 ms**，参照 **0.06116 ms** | 正确但远未达到性能目标；campaign 停止 |

SwiGlu 的收益不是“换个更大的 tile 就结束了”，而是找到了两个会主导几何平均的 shape 家族，并分别改变了它们的访问模式和循环结构。GQA 的问题也不是“某一个 tile 参数没调好”，而是通用 attention 实现与强库 fused kernel 之间存在架构级差距。

下面的读者可以重点看两件事：

1. **SwiGlu 如何用 case 级证据定位真正影响几何平均的尾部。**
2. **GQA 如何证明“正确”与“可优化”不是同一个问题。**

---

## 1. 实验协议与证据口径

两次 campaign 都使用同一套控制面：

- 官方 CANN-Bench 负责构建、20 个 case 的正确性和计时；
- 每次最终结论要求 **3 个 student job + 3 个新鲜 reference job**；
- 最终比值为跨作业的 median-vs-median，不是 CUDA 式同作业 paired timing；
- 原始 CANN JSON、evaluator log、msprof kernel/op CSV 都归档到 job artifacts；
- milestone 只能引用 `cann-trace-view`、`msprof-*`、`probe`、`correctness` 等真实通道。

SwiGlu 任务使用 `cann_trace_view`（AI-Core E2E）作为候选计时；GQA 的参照和候选使用 `cann_kernel_details`，比较对象分别是两个真实作业的 kernel 时间。两种口径不能互相换算。

---

## 2. SwiGlu：从公开实现到 1.20×

### 2.1 起点不是空白

SwiGlu 的初始候选是 CANN-Bench 公开 PyPTO 实现的单文件合并版。因此这次工作的准确定位是：

> **在已有公开实现的基础上，识别并优化两个 shape 家族的瓶颈，而不是从零发明一个 SwiGlu 算法。**

这一点很重要：agent 的贡献是可审计的增量，而不是“重写了所有代码”的叙事。

### 2.2 先找到几何平均的尾部

参照的 20 个 case 中，多数在 30–150 µs 附近，但有两个明显离群点：

| case | 参照时间 | 形状特征 |
|---|---:|---|
| `swi_glu_12` | 约 **484 µs** | `[1000003, 2]`，最后一维只有 2 |
| `swi_glu_20` | 约 **269 µs** | 5-D 输入，最后维 100 |

20 个 case 取几何平均后，这两个 case 对总时间有很强的拉动作用。优化它们，比在所有已经很快的 case 上再挤 2% 更有效。

最终确认时，这两个 case 的中位数分别为：

| case | 参照 | 优化后 | 近似变化 |
|---|---:|---:|---:|
| `swi_glu_12` | 0.484 ms | **0.0586 ms** | 约 8.3× 更快 |
| `swi_glu_20` | 0.269 ms | **0.0515 ms** | 约 5.2× 更快 |

这两个数字不是同作业 paired 比值，而是同一官方口径下参照测量与最终确认中位数的对照；总体 1.20165× 则来自最终 3+3 的正式确认。

### 2.3 优化一：为“薄尾维”建立专用路径

`swi_glu_12` 的输入是 `[1000003, 2]`，按最后一维 split 后，输出半维只有 1。原始实现把它当成普通的二维 tile 处理，等价于反复读取“偶数列/奇数列”的 2 字节元素，形成 strided access，并产生大量行循环。

agent 增加了一个 host 侧 O(1) 的路由条件：

```python
if split_dim == 1 and N_out <= THIN_MINOR_MAX:
    # deinterleave path
```

`THIN_MINOR_MAX = 8`，公开 case 中只有 case 12 命中。关键实现不是把 tile 调小，而是把两个交错的列看成一条连续数据：

```python
blk = pypto.view(input, shape=[2 * THIN_T0], offsets=[2 * r_off],
                 valid_shape=[2 * v])
parts = pypto.deinterleave(blk)   # even = x0, odd = x1
```

这带来三个变化：

1. **连续读取代 strided 读取**：一次读入 `2*T0` 个连续元素，而不是逐行取两个 2 字节列。
2. **循环次数下降**：该路径把约 3907 次 trip 降到约 123 次。
3. **计算仍保持 FP32**：deinterleave 之后分别 cast、sigmoid、乘 gate，再做一次 dtype 回归。

`THIN_T0` 不是凭感觉选的。probe 比较了 1024、4096、8192、16384 四个候选，8192 的 event time 最快；这组数据只用于确定这个实现参数，最终成绩仍由官方 20-case 确认决定。

### 2.4 优化二：重新配对 c8 的 tile 与 unroll

5-D case 使用 `c8` 类。原始实现的配置是：

```text
tile  = (32, 50)
unroll = [1]
trips ≈ 1632
```

agent 把 tile 改为 `(128, 50)`，并把 row loop unroll 设为 `[4]`：

```text
tile  = (128, 50)
unroll = [4]
trips ≈ 408
```

为什么这两个参数必须一起改？

- 只放大 tile，循环体变重，但每次 trip 的启动/调度比例未必下降；
- 只加 unroll，单次 trip 内并行度提高，但 tile 太小仍然产生过多循环；
- `(128, 50)` 的 tile 加上 `[4]` unroll，才把 trip 数从 1632 压到 408，同时没有超过 UB 预算。

这个选择来自一次联合 probe：在 `Tp ∈ {32,64,128,256}` 与 `unroll ∈ {1,2,4}` 的组合中，`128/[4]` 的 probe event time 最优；而原始 `32/[1]` 明显更慢。probe 只负责找方向，最终是否 KEEP 由官方全量评测决定。

同时，c8 的 host wrapper 去掉了无意义的 FP16/BF16→FP32 再转回的路径。kernel 直接在内部完成 cast，减少一次 host 侧的 dtype 往返。

### 2.5 为什么这次优化有效

可以把有效原因压缩成四点：

1. **优化对象是几何平均的主要贡献者，不是平均值最小的 case。**
2. **针对 shape 不变量做专用分派**：只有当 `split_dim=1` 且输出半维极小时才走薄路径；其他 case 不受影响。
3. **改变访问模式，而不只是改参数**：`deinterleave` 把“逐行取两列”变成“连续块拆分”，这是数量级上的变化。
4. **把 tile、unroll、loop 次数和 UB 预算一起考虑**，而不是孤立地扫一个参数。

最终结果：

```text
reference: 0.0676196 ms
student:   0.0547384 ms
ratio:     1.20165x
confirmation: 3 student jobs + 3 fresh reference jobs
status: done / target met
```

### 2.6 对其他算子的启发

这个案例可迁移的不是“SwiGlu 的 128×50 tile”，而是方法：

- 先看每个 case 对几何平均的贡献，再决定优化谁；
- 找出 shape 的稳定不变量，建立一个明确的、可回退的专用路径；
- 对“宽高比极端”的 shape，优先检查访问模式，而不是继续加循环并行；
- 用 probe 缩小参数空间，用官方全量评测做最终判决；
- 对不适用的 shape 保持原路径，避免“一个优化拖慢所有 case”。

---

## 3. GQA：正确但为什么会慢近三个数量级

### 3.1 任务比 SwiGlu 复杂得多

GQA 是 Level-4 fused attention 算子，20 个公开 case 覆盖：

- prefill、decode、MTP 等不同序列长度；
- causal / non-causal；
- MHA、GQA、MQA；
- `D=128` 与 `D=256`；
- fp16 / bf16；
- `B` 从 2 到 128，`S_kv` 到 2048。

参照是华为 CANN 的 `FusedInferAttentionScore`，一个高度工程化的库实现，20-case 几何平均 **0.06116 ms**。

### 3.2 实际做出来的候选

student 最终做出的是一个可运行的通用实现，核心结构是：

```text
按 (batch, kv head, q block) 拆分 unit
  → Q 与 K 做 matmul 得到 score
  → 加 scale，causal mask
  → 在线 max / sum / rescale
  → P 与 V 做 matmul
  → 写回输出
```

代码里有两种布局：

- `_build_batched`：rank-3 batched cube 形式；
- `_build_flat`：rank-2 flat cube 形式。

它使用 `pypto.loop(..., parallel=True)` 并行 unit，对 Q/K/V 做 per-axis view，使用 FP32 score/softmax，online rescaling，再把 P cast 回输入 dtype 做 PV matmul。

这不是一个“玩具 kernel”：它有正确的 GQA 分组、causal mask、decode/prefill 共用路径、shape-specialized JIT，并且最终 **20/20 通过官方正确性**。M0 gate 的 device geomean 为 **40.2399959 ms**。

### 3.3 差距在哪里

与参照的 case 级对照非常清楚：

| case | 参照 | 候选 M0 | 现象 |
|---|---:|---:|---|
| `gqa_8` | 166.85 µs | **2245.98 ms** | 约 1.35 万倍慢 |
| `gqa_10` | 184.09 µs | **617.09 ms** | 约 3350 倍慢 |
| `gqa_9` | 130.49 µs | **372.45 ms** | 约 2855 倍慢 |
| `gqa_18` | 11.24 µs | **1.37 ms** | 约 122 倍慢 |

差距不只出现在一个 case 上。小 decode case 也有约 50–120 倍差距，大 prefill case 更是差了三个数量级以上。因此：

> GQA 的问题不是“最后一个尾块没处理好”，而是整个执行架构和库实现不在一个层级。

### 3.4 主要阻碍一：通用 attention 架构 vs 专用 fused kernel

候选是“通用 online softmax + 两个 matmul”的结构，参照是专门的 fused attention：

- 参照对 prefill 和 decode 有不同调度路径；
- 参考实现会把 Q/K/V 的加载、score、softmax、PV、写回组织成适合硬件流水和 buffer 复用的结构；
- 候选的 KV tile 循环中，每个 tile 都执行 matmul、mask、softmax、rescale、PV，缺少与之匹配的预取、双缓冲和跨阶段重叠；
- 候选是一个“能覆盖所有 case 的同一家族”，而库实现会针对不同 regime 选择不同策略。

这里有一个重要的口径说明：候选的 40.24 ms 是 card 声明的 `cann_kernel_details` 窗口，包含 PyPTO 动态执行的控制行；按已认证的 KB，控制税约为 compute row 的 2.35–2.6 倍。即便只取 compute row，候选仍比参照慢两个数量级以上，因此口径差异不能解释主要差距。

在 SwiGlu 中，一个 shape 特化就能改变数量级；在 GQA 中，需要的是 **prefill/decode 分流、tile/pipeline、buffer 生命周期和 persistent scheduling 的整体重构**，不是单个参数。

### 3.5 主要阻碍二：PyPTO 的布局与 lowering 约束

GQA 的关键路径需要频繁构造 cube operand。实际遇到的约束包括：

- `pypto.view` 是 **逐轴 slice**，不是 reshape；中间轴的 run 长度不能直接当作 N 轴；
- rank-3 的 grouped K/V 必须构造成 `view(x, [N, 1, D])` 再 `reshape([1, N, D])`，否则会读到不同的窗口；
- tile rank、vec tile shape 和 view 的 rank 必须一致，否则编译期报 `REGISTER_COPY`/`CAST` 尺寸不匹配；
- UB 有明确容量上限（该平台约 253,952 B），不能把生产环境里的“先物化再切”随意搬进来；
- Q/K/V 的 origin 可能是 4-D，而 cube 需要 2-D/3-D 的合法 operand，转换路径必须逐项验证。

这些不是“文档里查一下”的问题。它们直接决定某个 attention 分解能不能 lower，进而决定 agent 是否有机会做性能优化。为此 campaign 认证了两个关键 KB：

- `pypto-operand-tile-construction`：rank-3 的 per-axis view + reshape；
- `pypto-dynamic-execution`：一次 JIT 调用对应多个 msprof task row，不能把控制行时间当 kernel 时间。

但这两个 KB 解决的是“能不能构造”，没有解决“构造后是否具备库级性能”。

### 3.6 主要阻碍三：探索成本与搜索空间

GQA 的每次尝试都要经过：

```text
写 probe → JIT 编译 → 单合同验证 → 修正布局
→ 合并成完整 kernel → 20-case 官方评测
```

其中：

- 20 个 case 各自按 shape 特化，JIT 编译成本高；
- 某些 probe 需要几十分钟编译，接近 7200 秒 job 上限；
- 一次全量评测本身也要数分钟到数十分钟；
- 多个 probe/编译失败会消耗整个外层循环的迭代预算。

SwiGlu 可以把搜索集中在两个 case 上；GQA 必须在 20 个 case 上同时保持正确性，且每个 case 的最优数据流不同。这使 agent 很难在有限预算内从“通用实现”走到“库级实现”。

### 3.7 尝试过程总结

GQA campaign 的路径可以概括为：

1. **接口与数值 contract**：从 4/8、16/17 逐步做到 18/18，最终 21/21 smoke；
2. **M0 架构**：形成 batched/flat 两种在线 softmax 路径，20/20 正确；
3. **第一次 M0 gate**：因计时配置不一致被标记无效（`kernel_details` 与 `trace_view` 混用）；
4. **修正计时**：统一为 `cann_kernel_details`，得到真实 40.24 ms；
5. **外层诊断**：不是“精度失败”，而是 G3 硬件接口/执行模型问题；
6. **KB 认证**：认证 operand construction、dynamic execution，并应用到候选；
7. **M1 尝试**：继续探索 tile/布局，但多次 probe/编译失败或超时，未能形成性能突破；
8. **停止**：没有 best confirmation，记录为未完成优化。

这个结果本身是有价值的：它证明后端能正确识别“正确但慢”的算子，并阻止把 20/20 correctness 误报为优化成功。

---

## 4. 两次尝试的对照

| 维度 | SwiGlu | GQA |
|---|---|---|
| 初始点 | CANN-Bench 公开 PyPTO 实现 | 从接口桩自行编写 |
| 参照强度 | 公开 PyPTO 实现 | 华为强库 fused kernel |
| 关键收益来源 | 两个尾部 shape 的专用路径 | 未形成 |
| 优化粒度 | 访问模式 + tile/unroll + 分派 | 需要完整 fused attention 架构 |
| 正确性 | 20/20 | 20/20 |
| 最终状态 | 1.20165×，3+3 确认 | 40.24 ms，未确认 |
| 可复用产出 | 两个 case 的优化方法 | 两个 NPU 接口 KB + 差距分析 |

### 对读者的直接启发

1. **优化几何平均时，先找“尾部”**：优化一个 0.5 ms 的 case，比优化十个 0.05 ms 的 case 更容易产生可见收益。
2. **优先级是“访问模式 > 循环结构 > 参数”**：SwiGlu 的 deinterleave 改变的是数据如何进入计算，而不是只改 tile。
3. **专用路径要有明确不变量和回退边界**：只对 `split_dim=1 && half_extent<=8` 生效，不让特化污染其他 case。
4. **正确性通过只是进入优化阶段的入场券**：GQA 20/20 但慢约 658 倍，说明 correctness 不能代表 architecture。
5. **强库参照需要架构级追赶**：当参照是 fused attention 时，单纯在现有循环里扫 tile，通常只能得到局部最优。
6. **把“不可用”当一等证据**：GQA 的 trace/kernel_details 口径问题和动态 task row 问题都被显式记录，避免了把测量错误当成 kernel 错误。

## 5. 后续建议

如果继续 GQA，优先顺序应是：

1. 按 prefill/decode、causal/non-causal、D=128/256、GQA/MQA 分成清晰的子家族；
2. 每个子家族先写一个只覆盖代表 case 的专用 kernel，再用官方全量 case 验证；
3. 在进入大规模 20-case 评测前，用官方 kernel_details 对单 case 做代价拆解，确认瓶颈在 QK、softmax、PV 还是搬运；
4. 把 persistent scheduling、双缓冲、KV tile 预取作为架构级决策，而不是 M1 的局部 lever；
5. 只有在单家族已接近强库时，再考虑统一实现。

对 SwiGlu，后续可以继续扩展：

- 对更多“薄维/极窄维” shape 建立受控专用路径；
- 保持所有专用路径的 no-go 条件和回退测试；
- 用同样的 case 贡献分析寻找下一个几何平均尾部，而不是平均地扫所有参数。

## 相关文档

- PyPTO 后端架构：[pypto-backend-technical-report-zh.md](pypto-backend-technical-report-zh.md)
- 验证状态与完整证据索引：[ascend-agent-validation.md](ascend-agent-validation.md)
- NPU 接口 KB：[../hw_kb/pypto/ascend950/index.md](../hw_kb/pypto/ascend950/index.md)
