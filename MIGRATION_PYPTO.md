# ForgeKernel PyPTO / Ascend 950PR 迁移交接

> **模型路由更正（2026-11-04）**：当前 PyPTO 任务固定 `provider="deepseek"`，
> 师生均为 `deepseek-flash[1m]` / `effort="max"`，使用官方
> `https://api.deepseek.com/anthropic` 和 `.env` 的 `API_KEY`。
> 旧的 `TEACHER_BASEURL` 曾仍指向 Kimi；只改模型名并没有切换服务商。
> 本机已删除该旧配置并停止旧 GQA 的配额重试，两侧实际启动命令均通过 DeepSeek 连通性检查。
> **旧 run 快照不会随 task 修改，应新建 run；不要把旧结果重新标为全 DeepSeek。**
> 详见 `docs/ascend-agent-validation.md` 的 provider-routing incident。
>
> **全 DeepSeek 最新结果**：SwiGlu 新 run `swiglu-pypto_261104-210250-b244` 已
> 以 `stop=target` 完成，最终 3+3 确认 student 0.0547384 ms vs fresh ref
> 0.0657764 ms，**1.20165×**。Level-4 GQA 新 run
> `gqa-pypto_261104-210250-eee5` 已完成 20/20 正确 M0，但 device geomean
> **40.24 ms**，远慢于强库 0.06116 ms；经多轮 G3 probe/编译循环后仍未收敛，
> 已主动停止并标记 `status=stopped`，没有 best confirmation。
> 全 DeepSeek 期间没有出现 402/配额错误。论文的 910C/CANN 1.1.2 hosted
> 结果无法在 950PR/CANN 2.0.0 上严格复现。

> **2026-11-04 增量**：当前分支已增加官方全量数据与原始报告归档、真实 msprof 导出、
> Ascend 专用 profiler/prompt 路由、密封 PyPTO 参照晋升、三次确认与双侧噪声检查。
> `tasks/swiglu-pypto-g2/` 是上一代 confirmed best 生成的 blackbox 下一代任务；
> `tasks/gqa-pypto/` 是 Level-4 华为强库参照任务。通用 probe `valid_shape_assemble`
> 已在真实 NPU 认证入库（job 1793761707），并明确保留 padding no-go。
> 这些能力不由旧的 timing-only 说明替代；详细实证见 `docs/ascend-agent-validation.md`。

**给下一位 agent 的第一句话：先完整阅读本文件，再检查 `repro/ascend/check_remote_env.sh` 和新机器的 Claude 沙箱，最后恢复 `tasks/swiglu-pypto`。不要从头重新设计后端。**

## 1. 当前进度与未完成事项

目标是在 Ascend 950PR 上运行 ForgeKernel 的 PyPTO 自动优化闭环：

```text
teacher 读取公开 PyPTO 参照并蒸馏 milestone
  → student 修改自己的 kernel.py
  → CANN-Bench 构建 + 20-case 正确性 + trace_view 性能
  → ForgeKernel KEEP / REVERT
  → best confirmation + report.md
```

截至 2026-11-03，已完成：

- `language="pypto"`、`arch="ascend950"`、PyPTO KB/prompt overlay；
- CANN-Bench 冻结任务卡和独立参照树；
- CANN-Bench 单文件候选打包、旧 wheel/旧包清理、官方报告转 ForgeKernel eval JSON；
- NPU 本地/SSH 作业环境、逻辑卡 0 映射、per-GPU 锁；
- 候选手工验证：**20/20 正确，AI-Core 几何平均 0.06455 ms**；
- 自动 run 的参照 baseline 已测量并归档：**0.0651787338 ms，20 workloads**；
- PyPTO 相关及相邻回归测试：**139 passed**；
- `forge new tasks/swiglu-pypto --dry-run` 通过。

首次端到端闭环已于 2026-11-03 在新控制机 + SSH `9502` 上完成：

- 参照 baseline：20/20，device 几何平均 **0.0666214970 ms**；
- teacher 首次蒸馏完成，`milestone v1` 校验为 `violations=none`；
- student 产生真实 commit `a9acbe276483b97cca8ed9cc16e4dae572bef25b`，20/20 正确并 **KEEP**，首轮 device 几何平均 **0.0664000291 ms**；
- 停止前同期复测确认达到目标，最终 best confirmation 为 **0.0662125262 ms**，同期参照为 **0.0669795575 ms**，median-vs-median 比值 **1.011584×**；
- run 以 `status=done`、`stop=target`、`loop_rc=0` 结束，`best_confirmation.json`、`report.md` 和 `best/kernel.py` 均已生成并核验。

成功 run（仅在本机 `.artifacts/`，未提交 git）：

```text
.artifacts/swiglu-pypto_261103-210743-f995
```

本次 `ratio=0.5` 的目标是验证完整闭环；1.011584× 来自同一 9502 上的单次最终 student/ref 测量，不应解读为已证实的稳定显著提速。旧 run `.artifacts/swiglu-pypto_261103-084626-4e7a` 在 teacher 启动前停止；当前没有并发 agent 作业需要保留。

## 2. 两台机器的职责

### 新机器：agent 控制面

运行：

- git 仓库中的 `forge`；
- Claude Code teacher/student；
- 任务、workspace、git commit 和报告；
- `.env` 中的模型配置。

不需要安装 CANN，也不需要 NPU。

### 当前机器：950PR 执行面

保留并运行：

- Ascend 950PR、CANN 9.2、PyPTO 0.3.0；
- CANN-Bench 2.0.0 精确 worktree；
- 独立 `.venv-cannbench`；
- 所有构建、正确性、trace_view 性能评测。

控制面通过 SSH 使用 ForgeKernel 的 `FK_DELIVERY=ssh` 提交作业。物理卡 1 在作业内映射为逻辑卡 0，不依赖当前有故障的 `xrun`。

## 3. 当前 950PR 机器已准备好的路径

| 用途 | 路径 |
|---|---|
| ForgeKernel 仓库 | `/mnt/home/dataset/shw/Workspace/ForgeKernel` |
| CANN-Bench 2.0.0 worktree | `/mnt/home/dataset/shw/Workspace/cann-bench-2.0.0` |
| CANN-Bench commit | `059cb61d41b279ad021160517bd2a5f6b7a05e01` |
| NPU Python | `/mnt/home/dataset/shw/Workspace/.venv-cannbench/bin/python` |
| PyPTO 源码阅读目录 | `/mnt/home/dataset/shw/Workspace/.venv-cannbench/lib/python3.12/site-packages/pypto` |
| PTO ISA | `/home/shw/Workspace/pto-isa` |
| CANN | `/mnt/home/dataset/shw/Ascend/cann-9.2.0` |
| 远程 source 环境脚本 | `repro/ascend/remote_env.sh` |
| SSH 专用工作目录 | `/mnt/home/dataset/shw/Workspace/forgekernel-remote` |
| NPU 环境自检 | `repro/ascend/check_remote_env.sh` |

远程 workdir 只能用于 ForgeKernel：runner 会替换其下的任务树，不要放其他数据。

当前软件版本：Python 3.12.15、torch 2.13.0+cpu、torch_npu 2.13.0.rc1、PyPTO 0.3.0、Triton-Ascend 3.6.0.post2。

在 950PR 机器上可再次验证：

```bash
cd /mnt/home/dataset/shw/Workspace/ForgeKernel
./repro/ascend/check_remote_env.sh
```

该命令检查 CANN-Bench commit、隔离环境 PyPTO import 和一次真实 NPU tensor 运算。

## 4. 新机器必须准备什么

### 4.1 Claude 沙箱与 CLI

需要 Python ≥ 3.11、git、tmux、Claude Code，以及可正常工作的 `bubblewrap` / `socat`：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
npm install -g @anthropic-ai/claude-code
claude --version
bwrap --unshare-all --die-with-parent --ro-bind / / true
```

最后一条必须返回 0。不要通过把 `sandbox.failIfUnavailable` 改为 `false` 来绕过。

### 4.2 安全复制 `.env`

`.env` 和 `local.env` 都被 gitignore，**不会随仓库迁移**。将原 `.env` 通过安全渠道复制到新机器仓库根目录，不要提交、不要在日志中打印。

每个启动 agent 的 shell 都执行：

```bash
cd <新机器上的 ForgeKernel>
set -a
source .env
set +a
export ANTHROPIC_API_KEY="$API_KEY"
export ANTHROPIC_AUTH_TOKEN="$API_KEY"
export ANTHROPIC_BASE_URL="$BASEURL"
```

任务已经固定使用 `.env` 中的 `MODEL=deepseek-flash`。

### 4.3 同步 PyPTO 源码阅读目录

student 离线运行，必须在**新机器本地**读取 PyPTO/PTO ISA 源码。参照本身已经随任务提交，不需要额外下载。

最简单的做法是保持原绝对路径，从 950PR 机器同步：

```bash
# 在可 ssh 到 950PR 的新机器上执行；NPU_HOST 换成连接别名。
mkdir -p /mnt/home/dataset/shw/Workspace/.venv-cannbench/lib/python3.12/site-packages
ssh "$NPU_HOST" 'tar -C /mnt/home/dataset/shw/Workspace/.venv-cannbench/lib/python3.12/site-packages -cf - pypto' \
  | tar -C /mnt/home/dataset/shw/Workspace/.venv-cannbench/lib/python3.12/site-packages -xf -

mkdir -p /home/shw/Workspace
ssh "$NPU_HOST" 'tar -C /home/shw/Workspace -cf - pto-isa' \
  | tar -C /home/shw/Workspace -xf -
```

如果新机器不能使用相同路径，修改：

```text
tasks/swiglu-pypto/leaderboard/card/loop.toml.example
  → [loop] dsl_source_dirs
```

已有 run 的 `operators/swiglu-pypto/loop.toml` 也需同步修改。这里是**源码阅读路径**，不是要求新机器安装 CANN。

## 5. 配置新机器连回 950PR

用户会提供 SSH 地址/别名。先确认非交互登录：

```bash
ssh <REMOTE_HOST> true
ssh <REMOTE_HOST> /mnt/home/dataset/shw/Workspace/ForgeKernel/repro/ascend/check_remote_env.sh
```

然后：

```bash
cp local.env.pypto.remote.example local.env
# 编辑 local.env，仅替换 REMOTE_HOST；路径已经按当前 950PR 机器填好。
```

关键配置：

```toml
FK_DELIVERY=ssh
FK_GPU=1
REMOTE_WORKDIR=/mnt/home/dataset/shw/Workspace/forgekernel-remote
REMOTE_ENV_FILE=/mnt/home/dataset/shw/Workspace/ForgeKernel/repro/ascend/remote_env.sh
```

不要把 agent 的 Anthropic key 写进 remote_env.sh 或提交到 git。

## 6. 启动前检查

在新机器仓库根目录：

```bash
# 配置和任务快照
.venv/bin/forge new tasks/swiglu-pypto --dry-run

# 可选：PyPTO 及相邻回归测试，不需要 NPU
.venv/bin/python -m pytest -q \
  tests/test_pypto.py tests/test_window_contract.py \
  tests/test_timing_audit.py tests/test_sched_live_tools.py
```

`forge` 也可直接使用 `.venv/bin/forge`，或把 `.venv/bin` 放进 PATH。

## 7. 恢复还是新建 run

### 情况 A：`.artifacts/` 也迁移了

恢复原 run，已有 baseline 不会重复测量：

```bash
forge run .artifacts/swiglu-pypto_261103-084626-4e7a
```

如果远程路径变化，先更新该 run 的 `operators/*/submit.toml`；正常 `forge run` 会从新 `local.env` 重写它们。

### 情况 B：只迁移了 git 仓库

`.artifacts` 被 gitignore，重新创建：

```bash
forge new tasks/swiglu-pypto
forge run <命令打印出的 .artifacts/swiglu-pypto_... 路径>
```

首次会重新测量公开 PyPTO 参照，约 8 分钟；后续才是 teacher 和 student。建议用 tmux：

```bash
tmux new -s forge-pypto
forge run <run-dir>
# 另一个 shell：forge status <run-dir>
```

不要同时在两台机器上恢复同一个 run。

## 8. 如何确认真的端到端成功

日志必须依次出现：

1. 参照 baseline 测量与归档；
2. `meta/distill` teacher 成功写出 `milestone.md`；
3. student turn 产生真实 commit；
4. CANN-Bench 20-case 正确性和 trace_view 性能；
5. `student/status.jsonl` 出现 KEEP/REVERT；
6. `best_confirmation.json`；
7. `.artifacts/<run>/report.md` 和 `best/kernel.py`。

关键文件：

```text
<run>/operators/swiglu-pypto/student/status.jsonl
<run>/operators/swiglu-pypto/meta_status.jsonl
<run>/operators/swiglu-pypto/best_confirmation.json
<run>/report.md
<run>/best/kernel.py
```

当前任务目标是参照的 0.5× 吞吐门槛，仅用于先验证完整闭环。初始候选是公开 PyPTO 实现的单文件合并版，**不是 agent 从零生成的结果**；后续 commit 才是 agent 的优化工作。

## 9. 移植代码索引

| 文件 | 作用 |
|---|---|
| `src/forgekernel/card_template/eval/cann_bench.py` | 调用官方 CANN-Bench、清理旧包/wheel、转换 20-case trace_view JSON |
| `src/forgekernel/card_template/pypto_gpu_entry.sh` | NPU 作业入口 |
| `scripts/init_pypto_swiglu_task.py` | 从公开 PyPTO c1–c8 生成冻结任务与单文件候选 |
| `tasks/swiglu-pypto/` | 完整任务、候选、teacher-only 参照和 loop 配置 |
| `src/forgekernel/task.py` | `pypto`、`ascend950`、`cann-bench` suite |
| `src/forgekernel/taskcard.py` | 任务树、oracle/ref 路径和 NPU submit 环境 |
| `src/forgekernel/submit/direct.py` | `ASCEND_RT_VISIBLE_DEVICES` 与 CUDA 分流 |
| `src/forgekernel/loop/lang_mode.py` | PyPTO KB/prompt/provenance 路由 |
| `meta/pypto/student_prompt.md` | PyPTO 学生约束与构建协议 |
| `hw_kb/pypto/ascend950/index.md` | 初始硬件知识库 |
| `tests/test_pypto.py` | 转换、任务、设备环境、语言路由测试 |
| `repro/ascend/remote_env.sh` | 950PR 作业环境，不含密钥 |
| `local.env.pypto.remote.example` | 新控制机的 SSH 配置模板 |

## 10. 常见坑

- **CANN-Bench commit 不可换**：后续 commit 改了 baseline metadata，分数会不可比。
- **PyPTO 必须使用 `trace_view`**：默认 host wall-clock 不是本任务的 device 指标。
- **每轮必须清理旧包和 wheel**：否则 import 失败或旧候选会污染下一轮。
- **参照和学生是不同 git 树**：学生 job 不包含 teacher-only 参照。
- **没有 NCU**：当前运行是有意的 timing-only，不要伪造 Ascend 版 NCU 指标。
- **根分区曾接近满**：远程环境已把临时文件、cache 和日志导向数据盘，不要改回 `/tmp` 长期写入。
- **密钥不在 git**：`.env`、`local.env` 需要单独安全迁移。
- **用户级 Claude 配置可能覆盖 `.env`**：本机 `~/.claude/settings.json` 的旧 `ANTHROPIC_BASE_URL` 曾让 agent 连接 `127.0.0.1` 而失败。成功 run 使用 `local.env` 中的独立 `CLAUDE_CONFIG_DIR=/data/shwstone/Workspace/forgekernel-claude`，没有改动全局网关配置，也没有放宽 sandbox。
