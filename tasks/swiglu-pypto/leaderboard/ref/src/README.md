# PyPTO CANN Example

基于 PyPTO (Python-to-NPU compiler) 的 Ascend NPU 算子示例，通过 `@pypto.frontend.jit` 编译算子，打包为 `cann_bench` Python 包供 CANN Bench 评测。

## 目录结构

```
pypto_example/
├── cann_bench/                 # Python包
│   ├── __init__.py             # 顶层 thin forwarder（聚合所有算子入口）
│   └── swi_glu/                # 算子子包：SwiGlu (L1)
│       ├── __init__.py         # from .dispatcher import swi_glu
│       ├── dispatcher.py       # 多 class (ndim,dtype) 分发器
│       └── c1/ ~ c8/           # 每类 (ndim, dtype) 签名的 kernel
│           └── swi_glu_impl.py # @pypto.frontend.jit kernel + host wrapper
├── build.sh                    # 构建脚本
├── setup.py                    # 打包配置
└── README.md
```

## 多算子隔离约定

每个算子独占一个 `cann_bench/<op>/` 子包，内含：
- `__init__.py`：从 dispatcher 导出入口函数
- `dispatcher.py`：多 class 分发器（按 `(ndim, dtype)` 签名路由；单 class 算子可省略，直接在 `__init__.py` 中导出 wrapper）
- `c{N}/<op>_impl.py`：每个 class 的 `@pypto.frontend.jit` kernel + host wrapper（`c{N}/` 不是 Python 包，通过 `package_data` 打包、由 dispatcher 按文件路径懒加载）

顶层 `cann_bench/__init__.py` 为每个算子写一个 thin forwarder，评测框架通过 `dir(cann_bench)` 扫描到所有算子接口。新增算子时只需：
1. 新建 `cann_bench/<op>/` 子包（复制上述结构）
2. 顶层 `__init__.py` 追加一个 forwarder
3. `setup.py` 的 `packages` 和 `package_data` 追加该子包

## 算子说明

**SwiGlu** (SwiGLU 激活函数)：`output = SiLU(x0) * x1`，其中 `x0, x1 = input.chunk(2, dim)`。

- 类别：Elementwise (L1)
- 支持 dtype：float16, float32, bfloat16
- 支持 2D~5D 输入，通过 8 个 class (c1~c8) 覆盖不同 (ndim, dtype) 签名

## 构建方法

```bash
bash build.sh           # 构建 wheel 包到 dist/
```

## 评测方法

### 前置条件

1. PyPTO 已安装（`pip install pypto`）
2. NPU 设备可用（Ascend 910B 系列）

### 评测 SwiGlu 算子

```bash
# 使用直接 Python CLI（支持 --perf-metric-strategy）
PYTHONPATH=src python -m kernel_eval.cli eval \
  --bench-name cann \
  --task-dir tasks/level1/swi_glu \
  --source-dir examples/pypto_example \
  --device-id 0 \
  --perf-metric-strategy trace_view
```

**重要：PyPTO 算子必须使用 `--perf-metric-strategy trace_view`**，从 PyPTO 生成的 `trace_view.json` 中提取 `aicore_e2e` 作为性能数据，而非默认的 `kernel_details.csv` 中的 host 端 wall-clock 时间（后者会包含 Python/框架开销，导致耗时偏高、评分偏低）。

### 报告输出位置

报告默认保存到 `<project_root>/reports/`，包含三种格式：
- `{eval_code}.json` — 完整结构化数据（含 per-case elapsed_us、accuracy、trace_view 指标）
- `{eval_code}.md`   — Markdown 摘要（概览表 + 每算子详情表）
- `{eval_code}.html` — 独立可视化报告

可通过 `--reports-dir` 指定输出目录。

### 评测流程

```
python -m kernel_eval.cli eval --source-dir examples/pypto_example
  │
  ├─ build.sh → cann_bench-1.0.0-py3-none-any.whl
  ├─ pip install --no-deps cann_bench-1.0.0-py3-none-any.whl
  ├─ import cann_bench → 扫描接口: swi_glu
  ├─ 匹配 tasks/level1/swi_glu 中的算子定义
  │
  └─ 逐用例评测:
      ├─ 加载 cases.yaml 用例（20个用例）
      ├─ 生成输入数据
      ├─ 执行 golden 参考（CPU fp64）
      ├─ 执行 AI 算子 (NPU + Profiler)
      ├─ 精度对比（MERE/MARE）
      ├─ 性能：trace_view.json → aicore_e2e
      └─ 性能评分（SOL-Score）
```

## 调用链

```
cann_bench/__init__.py
  → swi_glu() thin forwarder        # 入口
  → swi_glu/__init__.py             # from .dispatcher import swi_glu
  → swi_glu/dispatcher.py           # 按 (ndim, dtype) 签名匹配 c1~c8
  → swi_glu/c1/swi_glu_impl.py      # 实际 kernel
```

## PyPTO Kernel 结构

每个 `swi_glu/c{1-8}/swi_glu_impl.py` 包含两部分：

| 部分 | 职责 | 示例 |
|------|------|------|
| **Kernel 定义** | `@pypto.frontend.jit` 装饰的 JIT 编译函数 | `swi_glu_kernel_npu(input, output, split_dim)` |
| **Host wrapper** | 分配输出内存，调用 JIT | `swi_glu_impl(input, dim=-1)` |

PyPTO kernel 通过 `pypto.view()` 分块读取，`pypto.cast()` 做精度转换，`pypto.assemble()` 写回输出。

JIT 装饰器参数：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
```

## 多 Class 分发机制

`cann_bench/swi_glu/dispatcher.py` 中的 `_CLASSES` 表将输入 tensor 的 `(ndim, dtype)` 签名路由到对应 class 的 kernel。dispatcher 通过 `importlib` 按文件路径懒加载对应 class 的 `swi_glu_impl.py`，入口函数查找顺序为 `swi_glu` → `swi_glu_wrapper`：

| Class | ndim | dtype |
|-------|------|--------|
| c1    | 2D   | float16 |
| c2    | 2D   | float32 |
| c3    | 2D   | bfloat16 |
| c4    | 3D   | bfloat16 |
| c5    | 4D   | float32 |
| c6    | 5D   | float16 |
| c7    | 3D   | float32 |
| c8    | 5D   | float32 |

## trace_view vs kernel_details 性能数据差异

| 指标来源 | elapsed_us 含义 | 典型值 (case1) | 对应评分 |
|----------|----------------|---------------|---------|
| `kernel_details`（默认） | Host 端 wall-clock（含 Python+框架开销） | ~157us | ~53.8 |
| `trace_view`（--perf-metric-strategy trace_view） | ASCI Core E2E 纯计算时间 (`aicore_e2e`) | ~80us | ~61.8 |

`trace_view` 排除了 Python 调度和框架开销，更准确反映 kernel 在 AI Core 上的真实执行性能，因此评分更高。
