# GQA 的编译时间阻塞,与怎样绕开它

> 对应分支 `feat/pypto-ascend`,平台 Ascend 950PR、CANN 9.2、PyPTO 0.3.0、CANN-Bench 2.0.0
>
> 这份文档只谈 GQA campaign 为什么迭代不起来,以及可以怎么改。优化结果见
> [pypto-optimization-case-studies-zh.md](pypto-optimization-case-studies-zh.md)。

GQA campaign 最终停在"20/20 正确但慢两个数量级"。差距本身是架构级的,这一点案例文档已经说清楚。
但 campaign 之所以没机会去追那个差距,原因更朴素:**每问一个架构问题的代价太高,预算在得到答案之前就花完了**。
这篇文档想把这件事从"编译很慢"这种印象,变成一个可以算的数,再给出几条能落地的改法。

## 1. 代价结构

GQA 的 20 个公开 case 覆盖 prefill、decode、MTP 三种 regime,`D` 取 128 或 256,causal 与
non-causal,fp16 与 bf16,`B` 从 2 到 128,`S_kv` 到 2048。候选实现按 shape 做 JIT 特化。

这里有一个乘法关系,是整个问题的根源:**特化键的粒度决定一次 job 要编译多少次**。
按 `problem.json` 里 20 个 case 的真实 axes 算:

| 特化键 | 一次 job 的编译次数 |
|---|---|
| 完整 shape(GQA 当前的做法) | **20** |
| regime + D + causal + dtype | 10 |
| regime + D + causal | 6 |
| regime + D | **5** |
| 仅 regime | 3 |

而 20 个 case 按 `(regime, D)` 分组,分布是很不均匀的:

| regime | D | case 数 | case |
|---|---|---|---|
| prefill | 128 | 11 | `gqa_1..4`, `gqa_7..10`, `gqa_18..20` |
| prefill | 256 | 2 | `gqa_5`, `gqa_6` |
| decode | 128 | 4 | `gqa_11..14` |
| decode | 256 | 1 | `gqa_15` |
| mtp | 128 | 2 | `gqa_16`, `gqa_17` |

也就是说,**20 次编译里有 15 次在回答结构上已经被别的 case 回答过的问题**。
一个 `(prefill, D=128)` 的 tile 决策,在 11 个 case 上编译 11 遍,得到的架构信息和编译 1 遍是一样的。

案例文档记录的现象(某些 probe 编译几十分钟、接近 7200 秒 job 上限、一次全量评测几分钟到几十分钟)
在这个结构下就不意外了。阻塞不是"PyPTO 编译慢"这一件事,而是**慢 × 20**。

## 2. 三个独立的放大因子

把成本拆开,有三处各自可以单独处理。

**特化粒度。** 上面那张表就是它。这是乘数最大的一项,也是唯一能一次拿掉 4 倍的。
它属于 kernel 设计,不属于 harness。

**问题与评测的粒度绑死。** 之前没有办法只问一部分 case。想知道"把 KV tile 从 128 改成 256
对 prefill 有没有用",唯一的途径是跑一次完整 20-case 评测。架构探索的代价因此等于验收的代价,
而这两件事的性质完全不同:前者要快、可以不精确,后者要全、必须可信。

**成本不可见。** 这一项最隐蔽。一次 job 回来只有一个墙钟时间,没有任何地方记录这段时间花在
JIT 还是设备上。所以外层诊断看不到"这一轮是编译受限",只能看到"又超时了"。
`cann_bench` 里没有 cost 字段,`IterationRecord` 里也没有。一个编译受限的 campaign 和一个
kernel 很慢的 campaign,在轨迹里长得一样。于是 teacher 无法把"缩小 job"当成一个决策来做,
因为它拿不到支持这个决策的证据。

## 3. 本次改了什么

这一轮只改了后两项,因为它们在 harness 里,可以在没有 NPU 的环境下验证。
第一项是 kernel 设计,留给下一次 campaign,但下面第 4 节给了做法。

**成本进入证据。** adapter 现在记录 `cann_bench.cost`:evaluator 墙钟秒数、这次 job 的预算、
请求的 case 数。它在任何结局下都会写出来——包括超时和崩溃的 job,因为那恰恰是最需要知道
时间去哪了的时候。`profiling.benchmark_digest` 把它带进 teacher 的 diagnose 视图,
所以"编译受限"从此是一条可以被读到的测量,不是一个印象。

cost 只记墙钟,不假装拆出了编译和设备各自的占比。要拆需要一个编译钩子,那得先在硬件上
probe 过、按 KB 的准入条件认证过才能写进来。这里不提前写一个看起来精确的数。

**超时不再丢掉现场。** `gpu_entry.sh` 原来直接 `--signal=KILL`,一个编译了四十分钟的 probe
被杀掉之后什么都不剩。现在先发 SIGTERM、30 秒后才 KILL,adapter 有机会把日志和 cost 行刷出来;
超时消息也会说明这段预算是编译和设备共享的。决定要不要缩小 job,靠的就是这条记录。

**可以只问一部分 case。** adapter 接受 `FK_CANN_CASES`,把它翻成 evaluator 的 case 过滤参数,
于是一次 scout job 只编译几个代表 case。GQA 卡片声明了 5 个:

```json
"scout_cases": ["gqa_1", "gqa_5", "gqa_8", "gqa_11", "gqa_16"]
```

一个 `(regime, D)` 家族一个代表,外加 M0 差距分析里最差的 `gqa_8`。5 次编译,不是 20 次。
`gqa_16` 是测试帮我补上的——第一版 scout 集漏了 mtp regime,而一个漏掉某个 regime 的 scout 集
比付全量代价更糟:它会用不适用的 shape 去给一个架构决策定价。

两条约束让 scout 不会变成一种廉价的虚报方式。

卡片必须声明自己这个 CANN-Bench build 的 case 过滤参数(`case_filter_flag`)。adapter 不猜
厂商 CLI 合同;没有声明时,一个被要求缩小的 job 会被**拒绝**,而不是悄悄跑完 20 个 case 再
用"4 个 case"的名义报回来。GQA 卡片现在把这个字段留空,因为这个 flag 还没在硬件上验证过。

scout 结果带着 `workload_scope` 进入 eval JSON,而 `loop/parse.py` 本来就拒绝任何带
`workload_scope` 的读数作为 gate。这里复用的是 CuTe DSL group card 已经在用的那一条守卫,
不是另写一条——另写一条就会和它漂移。所以 scout 对 teacher 是证据,对 gate 永远不是。

## 4. 下一次 campaign 的做法

**先按子家族拆,再谈统一实现。** 按 `(regime, D)` 建 5 个特化入口,host 侧 O(1) 分派。
这是 4 倍,而且它同时解决另一个问题:prefill 的 Q block 很长、decode 的 Q block 往往只有 1,
两者对 tile 和并行的要求本来就不一样,硬塞进同一个循环结构是案例文档里记的第一个阻碍。
拆开之后每个家族可以有自己的循环结构,而不只是自己的 tile 常数。

先做 `(prefill, D=128)`。11 个 case,占 20 个里的一半以上,几何平均里权重最大,
`gqa_8`、`gqa_9`、`gqa_10` 这几个最差的 case 都在里面。

**用 scout 定价,用全量验收。** 架构问题走 5-case scout job;只有要宣布一个 KEEP 或者过一个
gate 的时候才付 20-case 的代价。这件事现在有机制支持了,但要先在硬件上确认
`case_filter_flag`,并把它作为一条 KB 记录下来。

**进 20-case 之前先做单 case 代价拆解。** 用官方 `kernel_details` 在一个代表 case 上确认瓶颈
在 QK、softmax、PV 还是搬运。按已认证的 `pypto-dynamic-execution`,一次 PyPTO 调用对应多个
msprof task row,三行合计约为 compute row 的 2.35 到 2.6 倍,所以读的时候要按 task type 分,
不能数行数。

**把 persistent scheduling、双缓冲和 KV tile 预取当 M0 决策。** 案例文档已经指出这一点,
这里只补一句理由:它们都要改循环结构,而循环结构在 M1 是改不动的。留到 M1 再补,
就只能在一个错误的循环结构里扫 tile 参数。

**编译时间本身值得 probe 一次。** PyPTO 的编译结果能不能跨 job 复用,`clean_build` 擦掉
`build/` 和 `dist/` 的时候有没有连带擦掉可复用的缓存,这两个问题现在没有答案。
`clean_build` 的语义是对的——每次 job 必须构建当前提交,不能复用上一个候选的 wheel——
但"构建当前提交"和"丢掉一个按内容寻址的 JIT 缓存"是两件事,值得分开确认。
这要在硬件上 probe,probe 过了再按 KB 准入条件写进 `hw_kb/pypto/<arch>/`。

## 5. 没有解决的部分

特化粒度还是 20。这一轮只是让缩小 job 成为可能,并没有改 kernel。

`case_filter_flag` 还是空的,所以 scout 在硬件上还跑不起来。这是故意的:
填一个没验证过的 flag,比留空更糟。

cost 只有墙钟。编译和设备的占比要靠一个认证过的编译钩子,现在没有。

这些都不影响已有行为:scout 默认关闭,不声明 `FK_CANN_CASES` 的 job 和改动之前一样跑全量。

## 相关文档

- 两次真实优化尝试 [pypto-optimization-case-studies-zh.md](pypto-optimization-case-studies-zh.md)
- PyPTO 后端架构 [pypto-backend-technical-report-zh.md](pypto-backend-technical-report-zh.md)
- NPU 接口 KB [../hw_kb/pypto/ascend950pr/index.md](../hw_kb/pypto/ascend950pr/index.md)
