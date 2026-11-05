# SwiGlu 与 GQA 的两次 PyPTO 优化

> 平台 Ascend 950PR、CANN 9.2、PyPTO 0.3.0、CANN-Bench 2.0.0
>
> 模型 teacher 和 student 都是 `deepseek-flash[1m]`，全流程走 DeepSeek
>
> 两次 campaign 都在当前环境里跑完，不能当作论文 910C hosted 分数的复现。

## 结果

先把两次结果放在一起。

| 算子 | 参照 | 结果 | 状态 |
|---|---|---|---|
| SwiGlu，20 个公开 case | CANN-Bench 公开 PyPTO 实现 | 最终确认 **0.054738 ms**，参照 **0.065776 ms**，**1.20165×** | 达到目标，完成 3+3 确认 |
| GQA，20 个 Level-4 case | 华为 `FusedInferAttentionScore` | M0 **20/20 正确**，但 **40.24 ms**，参照 **0.06116 ms** | 正确但远未达到性能目标，campaign 停止 |

一个完成，一个中止。

SwiGlu 做成了，GQA 只过了正确性。两个结果都值得留下来。SwiGlu 的过程能看出一个通用 agent 怎样在已有实现上找到影响最大的尾部。GQA 的过程能看出“能写对”和“能写得像库一样快”之间隔着什么。

## 证据口径

两次 campaign 都走官方 CANN-Bench，构建、20 个 case 的正确性和计时都在官方 evaluator 里完成。最终结论要求 3 个 student job 加 3 个新鲜 reference job，比值取跨作业的 median-vs-median。这里没有 CUDA 那种同作业 paired timing，报告里也不能把它写成 paired。原始 CANN JSON、evaluator log、msprof kernel 和 op CSV 都会归档到 job artifacts。

SwiGlu 使用 `cann_trace_view`，读的是 AI-Core E2E 时间，排除了 host dispatch。GQA 的参照和候选使用 `cann_kernel_details`，比较的是两个真实作业的 kernel 窗口。两种口径不能互相换算。GQA 的 40.24 ms 里还包含 PyPTO 动态执行的控制行，这一点后面会单独说明。

## SwiGlu 的起点

起点并不空白。

SwiGlu 的初始候选是 CANN-Bench 公开 PyPTO 实现合并成的单文件版本。agent 做的是在这份实现上找瓶颈，再加两条受控的专用路径。这个定位要写清楚，免得把一次增量优化说成从零重写。

参照的 20 个 case 里，大多数在 30 到 150 微秒之间，有两个明显离群。

| case | 参照时间 | 形状特征 |
|---|---|---|
| `swi_glu_12` | 约 **484 微秒** | `[1000003, 2]`，最后一维只有 2 |
| `swi_glu_20` | 约 **269 微秒** | 5-D 输入，最后维 100 |

20 个 case 取几何平均。这两个 case 会把总时间往上拽。优化它们，比在所有已经很快的 case 上再挤 2% 更划算。

最终确认时，这两个 case 的中位数变成了下面这样。

| case | 参照 | 优化后 | 近似变化 |
|---|---|---|---|---|
| `swi_glu_12` | 0.484 ms | **0.0586 ms** | 约 8.3× 更快 |
| `swi_glu_20` | 0.269 ms | **0.0515 ms** | 约 5.2× 更快 |

这两个数字来自不同作业，不能当成 paired 比值。它们仍然是同一官方口径下参照测量与最终确认中位数的对照。总体 1.20165× 来自最后那组 3+3 确认。

## 薄尾维的专用路径

`swi_glu_12` 的输入是 `[1000003, 2]`，按最后一维 split 以后，输出半维只有 1。原始实现把它当成普通二维 tile 处理，读偶数列和奇数列时是 2 字节的 strided access，行循环也因此很多。

agent 在 host 侧加了一个 O(1) 的路由条件。

```python
if split_dim == 1 and N_out <= THIN_MINOR_MAX:
    # deinterleave path
```

`THIN_MINOR_MAX = 8`。公开 case 里只有 case 12 命中。关键改动在数据怎样进入计算。它把两个交错的列看成一条连续数据，一次读入 `2 * T0` 个元素，再拆成偶数列和奇数列。

```python
blk = pypto.view(input, shape=[2 * THIN_T0], offsets=[2 * r_off],
                 valid_shape=[2 * v])
parts = pypto.deinterleave(blk)   # even = x0, odd = x1
```

这样改完，读入从逐行取两个 2 字节列变成了连续块读取。循环次数从约 3907 次 trip 降到约 123 次。计算仍然在 FP32 里做，deinterleave 之后分别 cast、sigmoid、乘 gate，最后再转回输入 dtype。

`THIN_T0` 不是拍脑袋定的。probe 比较了 1024、4096、8192 和 16384 四个候选，8192 的 event time 最快。这个结果只用来选实现参数。最终成绩仍然由官方 20-case 确认决定。

## c8 的 tile 和 unroll

5-D case 走 `c8` 类。原始配置是 tile `(32, 50)`、unroll `[1]`，大约 1632 次 trip。agent 把 tile 改成 `(128, 50)`，把 row loop 的 unroll 改成 `[4]`，trip 数降到约 408。

| 配置 | tile | unroll | trips |
|---|---|---|---|
| 原始 | 32 × 50 | 1 | 约 1632 |
| 优化后 | 128 × 50 | 4 | 约 408 |

两个参数要一起改。只放大 tile，循环体变重，单次 trip 的启动和调度比例未必下降。只加 unroll，单次 trip 内并行度上去了，tile 太小仍然会产生过多循环。`(128, 50)` 加 `[4]` 把 trip 数压到 408，同时没有超过 UB 预算。

这个组合来自一次联合 probe。`Tp` 取 32、64、128、256，`unroll` 取 1、2、4。`128/[4]` 的 probe event time 最好，原始的 `32/[1]` 明显更慢。probe 只负责找方向，最后是否 KEEP 由官方全量评测决定。

c8 的 host wrapper 还去掉了 FP16 或 BF16 先转 FP32 再转回来的路径。kernel 内部直接完成 cast，少了一次 host 侧的 dtype 往返。

## 这次为什么成了

优化对象选对了。两个尾部 case 对几何平均的贡献最大，改它们比平均地扫所有 case 更容易看到整体变化。

专用路径的边界很清楚。只有 `split_dim = 1` 且输出半维不超过 8 时才走薄路径，其他 case 仍然走原实现。这个条件简单到可以在 host 侧用常数时间判断，也不会把特化扩散到不该去的地方。

数据进入计算的方式变了。`deinterleave` 把逐行取两列变成连续块拆分，循环次数和访存形状一起变化。这类改动的收益通常比单独调一个 tile 参数大。

tile、unroll、trip 数和 UB 预算是一起考虑的。放大 tile 以后，循环体是否还装得下，unroll 以后单次 trip 是否还划算，都要在同一轮 probe 里看。孤立地扫一个参数，很容易在错误的方向上得到一个小数点后的提升。

最终确认的数字是这组。

```text
reference  0.0676196 ms
student    0.0547384 ms
ratio      1.20165x
jobs       3 student + 3 fresh reference
status     done / target met
```

## 对其他算子的启发

先看每个 case 对几何平均的贡献，再决定优化谁。一个 0.5 ms 的 case 掉到 0.06 ms，和一个 0.05 ms 的 case 掉到 0.045 ms，对总成绩的影响完全不同。

专用路径要有稳定的不变量和明确的回退边界。`split_dim = 1`、输出半维不超过 8，这两个条件在 dispatch 时就能判断。条件越清楚，越不容易在别的 shape 上引入回归。

访问模式比循环参数更值得先看。宽高比极端的 shape，往往先坏在读取方式上。把 strided access 改成连续块，再考虑 tile 和 unroll，顺序通常更顺。

probe 用来缩小范围，官方全量评测用来判决。probe 快，能试很多组合，但它只覆盖一个合同。官方 20-case 会告诉你在真实榜单上有没有收益。

不命中的 shape 保持原路径。一个优化如果只在两个 case 上有效，就让它在两个 case 上生效。为了“统一”把所有 shape 都赶到新路径，常常会把原来的快 case 拖慢。

## GQA 的任务

GQA 是 Level-4 fused attention 算子，20 个公开 case 覆盖 prefill、decode、MTP 等不同序列长度，也覆盖 causal 和 non-causal、MHA、GQA、MQA，`D` 取 128 或 256，输入有 fp16 和 bf16，`B` 从 2 到 128，`S_kv` 到 2048。

参照是华为 CANN 的 `FusedInferAttentionScore`，20-case 几何平均 **0.06116 ms**。这是一个高度工程化的库实现，prefill 和 decode 有不同调度，Q/K/V 的加载、score、softmax、PV 和写回都围绕硬件流水与 buffer 复用组织过。

## 候选做到了什么

student 最终交出的是一份可运行的通用实现。它按 `(batch, kv head, q block)` 拆分 unit，Q 和 K 做 matmul 得到 score，加 scale 和 causal mask，做在线 max、sum、rescale，再让 P 和 V 做 matmul，最后写回输出。

```text
按 (batch, kv head, q block) 拆分 unit
  → Q 与 K 做 matmul，得到 score
  → 加 scale，加 causal mask
  → 在线 max / sum / rescale
  → P 与 V 做 matmul
  → 写回输出
```

代码里有两种布局。`_build_batched` 用 rank-3 batched cube，`_build_flat` 用 rank-2 flat cube。它用 `pypto.loop(..., parallel=True)` 并行 unit，对 Q/K/V 做 per-axis view，score 和 softmax 留在 FP32，online rescale 之后再把 P cast 回输入 dtype 做 PV matmul。

这份实现有正确的 GQA 分组、causal mask、decode 和 prefill 共用路径，也按 shape 做了 JIT 特化。它最终 **20/20 通过官方正确性**。M0 gate 的 device geomean 是 **40.2399959 ms**。

## 差距在哪里

与参照的 case 级对照很直接。差距很大。

| case | 参照 | 候选 M0 | 现象 |
|---|---|---|---|
| `gqa_8` | 166.85 微秒 | **2245.98 ms** | 约 1.35 万倍慢 |
| `gqa_10` | 184.09 微秒 | **617.09 ms** | 约 3350 倍慢 |
| `gqa_9` | 130.49 微秒 | **372.45 ms** | 约 2855 倍慢 |
| `gqa_18` | 11.24 微秒 | **1.37 ms** | 约 122 倍慢 |

差距不只出现在一个 case 上。几个 case 的差距在 55 到 122 倍。更多 case 落在数百到数千倍。`gqa_8` 超过一万三千倍。GQA 的问题出在整个执行架构和库实现的层级差上，换一个尾块 tile 参数救不回来。

这里还有一个口径要说明。候选的 40.24 ms 是 card 声明的 `cann_kernel_details` 窗口，包含 PyPTO 动态执行的控制行。按已经认证的 KB，三行合计约为 compute row 的 2.35 到 2.6 倍。即便只取 compute row，候选仍然比参照慢两个数量级以上，口径差异解释不了主要差距。

## 第一个阻碍，通用 attention 架构

候选是“通用 online softmax 加两个 matmul”的结构。参照是专门的 fused attention。差距同时落在单个算子和整个调度方式上。

参照对 prefill 和 decode 走不同路径。候选把两种 regime 塞进同一个家族，靠 shape-specialized JIT 分开，但循环结构没有跟着变。prefill 的 Q block 很长，decode 的 Q block 往往只有 1，两者对 tile 和并行的要求不一样。

候选的 KV tile 循环里，每个 tile 都要做 matmul、mask、softmax、rescale 和 PV。预取、双缓冲和跨阶段重叠没有跟上来。参照会把这些阶段组织成适合硬件流水的形状，让数据搬入和计算尽量交叠。候选的循环能跑对，却常常在等上一段算完。

在 SwiGlu 里，一个 shape 特化就能改变数量级。GQA 需要的是 prefill 和 decode 分流、tile 和 pipeline 一起改、buffer 生命周期重排，甚至 persistent scheduling。这些都属于架构级改动，不在 M1 的局部 lever 里。

## 第二个阻碍，PyPTO 的布局和 lowering

GQA 的关键路径要频繁构造 cube operand。实际遇到的约束比文档里写的更硬。

| 约束 | 后果 |
|---|---|
| `pypto.view` 是逐轴 slice，不能当 reshape 用 | 中间轴的 run 长度不能直接当 N 轴 |
| rank-3 的 grouped K/V 要先用 `view(x, [N, 1, D])` 再 `reshape([1, N, D])` | 直接 view 会读到不同的窗口 |
| tile rank、vec tile shape 和 view 的 rank 要一致 | 不一致会在编译期报 `REGISTER_COPY` 或 `CAST` 尺寸不匹配 |
| UB 容量约 253,952 B | 生产环境里“先物化再切”的写法搬不进来 |
| Q/K/V 的 origin 可能是 4-D，cube 需要 2-D 或 3-D operand | 转换路径要逐项验证 |

这些约束直接决定某个 attention 分解能不能 lower。一个看起来合理的分解，可能卡在 view 的 rank 上，也可能在 UB 上超一点。agent 为此认证了两个 KB，`pypto-operand-tile-construction` 解决 rank-3 的 per-axis view 加 reshape，`pypto-dynamic-execution` 解决一次 JIT 调用对应多个 msprof task row 的读数问题。

这两个 KB 解决的是“能不能构造”。构造出来以后有没有库级性能，是另一回事。

## 第三个阻碍，探索成本

GQA 每试一次都要走一遍完整路径。先写 probe，再等 JIT 编译，做单合同验证，修正布局，合并成完整 kernel，最后跑 20-case 官方评测。

20 个 case 各自按 shape 特化，JIT 编译成本很高。某些 probe 要编译几十分钟，接近 7200 秒的 job 上限。一次全量评测本身也要几分钟到几十分钟。几个 probe 或编译失败，就能吃掉整个外层循环的预算。

SwiGlu 可以把搜索集中在两个 case 上。GQA 必须在 20 个 case 上同时保持正确性，每个 case 的最优数据流还不一样。agent 很难在有限预算里从一份通用实现走到库级实现。

## 尝试过程

GQA 的接口和数值 contract 是一点点磨出来的。最早是 4/8，后来到 16/17，再到 18/18，最终 M0 smoke 做到 21/21。

M0 形成了 batched 和 flat 两种在线 softmax 路径，20/20 正确。第一次 M0 gate 因为计时配置不一致被判无效，card 采集 `kernel_details`，adapter 却去读 trace_view。把口径统一到 `cann_kernel_details` 以后，才得到真实的 40.24 ms。

外层诊断把问题归到 G3 的硬件接口和执行模型，没有把它当成精度失败。随后 campaign 认证了 operand construction 和 dynamic execution 两个 KB，并把它们用到候选里。M1 继续试 tile 和布局。多次 probe 或编译失败，也有超时。这次没有形成性能突破。

最后 campaign 主动停止。没有 best confirmation。状态记为未完成优化。这个结果不能算优化成功。它证明后端能识别“正确但慢”的算子，也阻止了把 20/20 正确性误报成优化成功。

## 两次对照

| 维度 | SwiGlu | GQA |
|---|---|---|
| 初始点 | CANN-Bench 公开 PyPTO 实现 | 从接口桩自行编写 |
| 参照强度 | 公开 PyPTO 实现 | 华为强库 fused kernel |
| 关键收益来源 | 两个尾部 shape 的专用路径 | 未形成 |
| 优化粒度 | 访问模式、tile、unroll、分派 | 需要完整的 fused attention 架构 |
| 正确性 | 20/20 | 20/20 |
| 最终状态 | 1.20165×，3+3 确认 | 40.24 ms，未确认 |
| 可复用产出 | 两个 case 的优化方法 | 两个 NPU 接口 KB 和差距分析 |

## 读者能带走什么

优化几何平均时，先找尾部。一个 0.5 ms 的 case 掉到 0.06 ms，比十个 0.05 ms 的 case 各快一点更能改变总分。

访问模式、循环结构、参数，这三件事有先后。SwiGlu 先改数据怎样进入计算，再调 tile 和 unroll。顺序反过来，常常会在一个错误的循环结构里反复扫参数。

专用路径要有明确的不变量和回退边界。只对 `split_dim = 1` 且输出半维不超过 8 的 shape 生效，其他 shape 保持原路径。特化范围越窄，回归越容易查。

正确性通过只是进入优化阶段的入场券。GQA 20/20 但慢约 658 倍，说明 correctness 不能代表 architecture。一个 kernel 能算对，离能打还有很长的路。

强库参照需要架构级追赶。参照是 fused attention 时，在现有循环里扫 tile 通常只能得到局部最优。先把 prefill 和 decode 分开，再谈统一实现，顺序更合理。

测量错误要和 kernel 错误分开。GQA 的计时口径问题和动态 task row 问题都被显式记录，没有把测量问题当成 kernel 问题去修。这个习惯在长 campaign 里很值钱。

## 后续

如果继续 GQA，先按 prefill 和 decode、causal 和 non-causal、`D=128` 和 `D=256`、GQA 和 MQA 分成子家族。每个子家族先写一个只覆盖代表 case 的专用 kernel，再用官方全量 case 验证。进入 20-case 评测前，用官方 `kernel_details` 对单 case 做代价拆解，确认瓶颈在 QK、softmax、PV 还是搬运。persistent scheduling、双缓冲和 KV tile 预取要作为架构决策，不要留到 M1 再补。只有单个家族已经接近强库，才考虑统一实现。

SwiGlu 还可以继续找下一个几何平均尾部。新的薄维或极窄维 shape 可以建立受控专用路径，但每条路径都要带 no-go 条件和回退测试。比起平均地扫所有参数，按 case 贡献找下一处更划算。

## 相关文档

- PyPTO 后端架构 [pypto-backend-technical-report-zh.md](pypto-backend-technical-report-zh.md)
- 验证状态与完整证据索引 [ascend-agent-validation.md](ascend-agent-validation.md)
- NPU 接口 KB [../hw_kb/pypto/ascend950/index.md](../hw_kb/pypto/ascend950/index.md)
