# Ascend 公开算子复现记录

本目录用于复现 CANN-Bench `official-tasks 2.0.0` 的公开算子。当前先跑通官方已公开 Triton-Ascend 源码的 **Exp** 和 **SwiGlu**。

## 结论与边界

- 两个公开实现均已在 **Ascend 950PR** 上通过全部 20 个公开用例，并由 CANN-Bench 官方 profiler 产出性能数据。
- 当前机器没有 950DT；这里的结果是 **950PR 本地对照**，不是 ForgeKernel 950DT 原数值的精确复现。
- `public_triton/cann_bench/exp.py` 和 `swi_glu.py` 与 CANN-Bench 公开示例逐字一致；它们不等于 ForgeKernel 榜单最佳 submission。
- ForgeKernel 对应公开 job 的 submission/artifact API 当前返回 `401 login required`。要精确复跑原 submission，还需平台登录权限和 950DT runner。

## 固定任务口径

榜单对应任务包：

- benchmark：`official-tasks`
- version：`2.0.0`
- source commit：`059cb61d41b279ad021160517bd2a5f6b7a05e01`
- 公开实现：`examples/triton_ascend_cann_example/cann_bench/{exp.py,swi_glu.py}`

必须固定该 commit。后续 CANN-Bench commit 修改了 `tasks/metadata/950dt.json` 和 `950pr.json`，混用会改变基线和分数。

本目录仅做两项打包调整：

1. `__init__.py` 只导出 Exp/SwiGlu，避免官方五算子示例中无关的 Mish 导入阻塞；
2. `setup.py` 修改包描述。两个 kernel 文件未改。

## 2026-11-03 实测结果

本地环境：单张 `Ascend950PR_9579`（物理卡 1）、CANN 9.2.0、Python 3.12.15、PyTorch 2.13.0+cpu、torch_npu 2.13.0.rc1、Triton 3.6.0、Triton-Ascend 3.6.0.post2。评测参数为 warmup=3、repeat=5、单卡单进程。

| 算子 | 本地硬件/实现 | 正确性 | 本地分数 | 本地几何平均加速比 | 原始报告 |
|---|---|---:|---:|---:|---|
| Exp（首轮） | 950PR / 公开 Triton | 20/20 | **91.4797** | **2.8142×** | [JSON](./results/exp-950pr.json) |
| Exp（脚本复跑） | 950PR / 公开 Triton | 20/20 | **91.5540** | **2.8214×** | [JSON](./results/exp-950pr-rerun.json) |
| SwiGlu | 950PR / 公开 Triton | 20/20 | **69.3473** | **0.7550×** | [JSON](./results/swiglu-950pr.json) |

Exp 两次完整 20-case 运行的几何平均加速比相差约 **0.26%**，脚本入口复跑通过。

对应 ForgeKernel 950DT 公开榜单参照如下。**这些数字使用不同硬件和 baseline metadata，不与上表直接作同条件比较。**

| 算子 | 950DT 分数 | 950DT 几何平均加速比 | 正确性 | 公开 job |
|---|---:|---:|---:|---|
| Exp | 88.3509 | 3.6188× | 20/20 | `job_c87985ad15af` |
| SwiGlu | 74.4489 | 1.4174× | 20/20 | `job_f5abb0ee99e6` |

SwiGlu 公开示例在部分非最后维拆分/大 outer-grid 用例上明显慢于 950PR baseline；逐 case 数据保留在 JSON 中。这里不以修改 kernel 的方式掩盖该差异。

## 复跑

### 1. 准备精确任务 worktree

```bash
cd /mnt/home/dataset/shw/Workspace
git clone https://gitcode.com/cann/cann-bench.git
git -C cann-bench fetch --depth 1 origin 059cb61d41b279ad021160517bd2a5f6b7a05e01
git -C cann-bench worktree add --detach \
  /mnt/home/dataset/shw/Workspace/cann-bench-2.0.0 \
  059cb61d41b279ad021160517bd2a5f6b7a05e01
```

### 2. 准备隔离 Python 环境

需要已有匹配的 CANN、torch、torch_npu。当前环境从 DeepInfer conda 环境继承二进制包，再将额外依赖安装到独立 venv：

```bash
PY=/mnt/home/dataset/shw/miniconda3/envs/deepinfer/bin/python
VENV=/mnt/home/dataset/shw/Workspace/.venv-cannbench
"$PY" -m venv --system-site-packages "$VENV"
export PATH="$VENV/bin:$PATH"

python -m pip install attrs tornado expecttest 'ruamel.yaml>=0.18' \
  pandas psutil decorator build wheel cmake
python -m pip install \
  --index-url https://triton-ascend.osinfra.cn/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  'triton-ascend==3.6.0.post2'

cd /mnt/home/dataset/shw/Workspace/cann-bench-2.0.0/src/cann_bench_utils
PYTHON="$VENV/bin/python" bash build.sh --clean
"$VENV/bin/python" -m pip install --no-deps dist/cann_bench_utils-*.whl
```

检查 NPU backend，而不只是检查 import：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python -c \
  'from triton.runtime import driver; print(driver.active.get_current_target())'
# 应包含 backend='npu'
```

### 3. 运行官方评测器

```bash
cd /mnt/home/dataset/shw/Workspace/ForgeKernel
./repro/ascend/run_public.sh Exp 1
./repro/ascend/run_public.sh SwiGlu 1
```

第二个参数是物理 NPU 号；脚本将其写入 `ASCEND_RT_VISIBLE_DEVICES`，评测器内部使用逻辑卡 0。可用环境变量覆盖路径：

- `CANN_BENCH_DIR`
- `PYTHON`
- `CANN_BENCH_RESULTS_ROOT`
- `CANN_BENCH_RUNTIME_ROOT`

脚本默认拒绝非榜单任务 commit；`ALLOW_TASK_REF_MISMATCH=1` 仅用于明确知道自己在比较另一任务口径的情况。

## 950DT 精确复现还缺什么

1. 登录 CANNBench，下载原 job 的 submission 与 artifacts：
   - `https://cannbench.com/api/jobs/job_c87985ad15af/submission/download`
   - `https://cannbench.com/api/jobs/job_f5abb0ee99e6/submission/download`
2. 核对原 submission 源码、任务 ref、runner harness 和环境；
3. 在 CANNBench 950DT runner 上重交，比较逐 case 时间、通过率、HAP 分数和几何平均加速比。

当前 `xrun` 在本机作业 shell 启动前以 120 退出，因此上述本地运行是在 `xinfo` 显示 idle 的物理卡 1 上直接执行。系统根分区 `/` 已满，复跑脚本将临时文件、profiler 输出、Triton cache 和报告重定向到数据盘。
