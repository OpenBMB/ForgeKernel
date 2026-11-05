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
被杀掉之后什么都不剩。现在 shell 先发 SIGTERM、30 秒后才 KILL,但那只是兜底:
adapter 自己会按 `TASK_TIMEOUT - 预留` 停住 evaluator(默认比 job 预算早 30 到 120 秒),
并先对它自己的进程组发 SIGTERM 再 KILL——`kernel_eval` 会对每个 case 起子进程,
只杀父进程会留下一批还占着 NPU 的进程。停下之后 adapter 写出结构化 partial eval:
墙钟、预算、停止原因、`workload_scope`(如果是 scout)、以及已经产出的日志与 partial report。
里面没有任何正确性或性能读数——一个没跑完的 job 没有读数,只有代价。
adapter 自己如果被外层 TERM,也会走同一条路,而不是按默认动作直接退出。

**可以只问一部分 case。** adapter 接受 `FK_CANN_CASES`。`scout` 取卡片声明的代表集,
也可以直接写 id(`gqa_8` 单 case 探针,或 `gqa_1,gqa_8`)。GQA 卡片声明了 5 个:

```json
"scout_cases": ["gqa_1", "gqa_5", "gqa_8", "gqa_11", "gqa_16"]
```

一个 `(regime, D)` 家族一个代表,外加 M0 差距分析里最差的 `gqa_8`。5 次编译,不是 20 次。
`gqa_16` 是测试帮我补上的——第一版 scout 集漏了 mtp regime,而一个漏掉某个 regime 的 scout 集
比付全量代价更糟:它会用不适用的 shape 去给一个架构决策定价。

怎么做到的要紧,因为它决定了这个机制是否真的可用。**上游的 CANN-Bench 没有 case 集合参数**:
`kernel_eval.cli` 只有 `--case-id` 且 `type=int`,一次一个 case;`ForgeKernel` 原来的
`case_filter_flag` 字段实际上无 flag 可填。所以 subset 改成用它**本来就支持**的那条路径表达:
在 `CANN_BENCH_DIR` 下建一个隔离的 task root(`.forgekernel-scout/`),把官方 task 目录镜像进去
——`proto.yaml`、`golden.py`、`cases.csv`、`desc.md` 全部 symlink 回官方文件,只重写
`<root>/tasks/level4/gqa/cases.yaml`,让它里面只剩选中的 case,再把官方
`tasks/metadata`(和 `suite_manifest.json`)link 到 `<root>/tasks/` 下。
`resolve_task_dir` 把 `<root>/tasks` 当 bench root,`CannCaseLoader` 只扫到选中的 case,
而 case id(`level4/gqa_5`)、input_shape、dtype、`baseline_perf_us`/`t_hw_us` 与全量 loader
完全一致。一个 job、一次构建、5 次 case 编译,不是 5 次 job。
全量 job 仍然用官方 `task_dir`,一个字节都不动,也不建任何临时目录。

两条约束让 scout 不会变成一种廉价的虚报方式。

第一,`all`/`full`/`gate`/`ref`/`reference`/`confirm` 与不声明 `FK_CANN_CASES` 都还是完整
declared set;`gpu_entry.sh` 的 gate task(`all`/`correctness`/`perf`/...)会主动 unset 继承来的
`FK_CANN_CASES`,所以一个跑偏的环境变量不可能把 gate 缩小。未知 id、重复 id、写不出代表集的
`scout` 一律 fail closed,不会“静默跑全量再用 4 个 case 的名义报回来”。报告里的 case 集如果和
请求的不一致,adapter 直接报 harness error,不定价任何东西。

第二,scout 结果带着 `workload_scope` 进入 eval JSON,而 `loop/parse.py` 本来就拒绝任何带
`workload_scope` 的读数作为 gate。这里复用的是 CuTe DSL group card 已经在用的那一条守卫,
不是另写一条——另写一条就会和它漂移。所以 scout 对 teacher 是证据,对 gate 永远不是。

teacher/recon 侧的入口只有一个:`notes/experiment.json` 里写 `"cases"`。
`skills/ascend-profile/ascend_probe.py` 把它翻成 `FK_CANN_CASES` 传给 evaluator,
并同时把 `FK_CANN_EVAL_BUDGET` 设成比自己 timeout 早 60 秒。scout 实验**不会**打印
`PROBE PASS`:它打印 `PROBE SCOUT` 和带 `"certifies": false` 的 `ASCEND-PRICE-JSON`。
一个 5-case 读数没有资格认证一个 lever,只有全量 gate 有。
`bash gpu_entry.sh scout [ids]` 是同一条路的直调入口。

## 4. 下一次 campaign 的做法

**先按子家族拆,再谈统一实现。** 按 `(regime, D)` 建 5 个特化入口,host 侧 O(1) 分派。
这是 4 倍,而且它同时解决另一个问题:prefill 的 Q block 很长、decode 的 Q block 往往只有 1,
两者对 tile 和并行的要求本来就不一样,硬塞进同一个循环结构是案例文档里记的第一个阻碍。
拆开之后每个家族可以有自己的循环结构,而不只是自己的 tile 常数。

先做 `(prefill, D=128)`。11 个 case,占 20 个里的一半以上,几何平均里权重最大,
`gqa_8`、`gqa_9`、`gqa_10` 这几个最差的 case 都在里面。

**用 scout 定价,用全量验收。** 架构问题走 5-case scout job;只有要宣布一个 KEEP 或者过一个
gate 的时候才付 20-case 的代价。机制已经就位(`FK_CANN_CASES=scout` / `gpu_entry.sh scout` /
probe spec 里的 `"cases"`),但要先在硬件上确认两件事:一次 scout job 的评测时间确实接近
1/4,以及 `baseline_perf_us`/`t_hw_us` 与全量 loader 完全一致。确认之后写成一条 KB。

三层代价阶梯,不要跳着用:最差的单 case(`"cases": "gqa_8"`,一次编译)用来快速否证一个
猜想——它只能杀方向,不能证明方向;5-case scout(五次编译)给一个还想活下去的架构决策定价;
20-case gate 才是宣布 KEEP 的地方。前面两层都是“买/杀下一次全量”的输入,不是替代品。

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

scout 改的是 job 的 case 集,没改 PyPTO 的 JIT 键:一次 scout job 之内,5 个 case 仍然是
5 次编译。跨 job 的编译缓存能不能复用,没有硬件证据,这里不声称。

`baseline_perf_us`/`t_hw_us` 的等价性只在 CPU 上用官方 loader 验证过(隔离 root 与全量 root
逐字段相等);昇腾机上还要再核对一次真实 `metadata/<hardware>.json`。这是唯一的机制性风险:
adapter 从官方 bench root 逐级向上找 `metadata/`(和 `suite_manifest.json`)并 link 过去;
如果某台机器的 baseline 放在 `CANN_BENCH_DIR` 之外,隔离 root 会拿不到 baseline,scout 的
`speedup` 会退化成缺失——adapter 不会伪造它,而是让 `validate_pypto_scout.py cases` 报 DIFFERS。

cost 只有墙钟。编译和设备的占比要靠一个认证过的编译钩子,现在没有。

这些都不影响已有行为:scout 默认关闭,不声明 `FK_CANN_CASES` 的 job 和改动之前一样跑全量。

## 相关文档

- 两次真实优化尝试 [pypto-optimization-case-studies-zh.md](pypto-optimization-case-studies-zh.md)
- PyPTO 后端架构 [pypto-backend-technical-report-zh.md](pypto-backend-technical-report-zh.md)
- NPU 接口 KB [../hw_kb/pypto/ascend950pr/index.md](../hw_kb/pypto/ascend950pr/index.md)
