#!/usr/bin/env python3
"""Self-evolution: each generation's best kernel becomes the next generation's reference.

    python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]

A generation is an ordinary run (`forge new`, then `forge run`). When it ends with a confirmed
best kernel faster than its reference by more than the noise floor, the next generation is
written as a new task, ``tasks/<op>-g<k>/``: the same task.toml, description and shapes, the
best kernel copied into its reference/ (so the new task carries everything it needs and can be
committed and rerun elsewhere), and a target ``--step`` times the new reference's speed.

The chain stops only when the hardware gives no more: at a generation whose gain is within the
noise floor, or one that ended without a confirmed best. The noise floor is the spread of the
generation's own same-job confirmation ratios, never below NOISE_FLOOR_MIN (the loop's plateau
margin). ``--max-gen`` caps the number of generations; there is no cap by default.

CuTe DSL references stay source-readable to the teacher only. CANN-Bench
references instead remain sealed blackbox packages: the student receives only
a signature stub, the teacher measures them, and fresh two-sided confirmation
is required before promotion. No reference source crosses the boundary.

Only the public commands are used; nothing here is special to the harness.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import tomllib
import tempfile
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgekernel import launch  # noqa: E402
from forgekernel.runs import list_runs  # noqa: E402
from forgekernel.task import (TaskError, dump_task_toml, load_task,  # noqa: E402
                              read_task_toml)

REF_DIR = "reference"
# == forgekernel.loop.parse.detect_plateau's default: below this, two readings of one
# kernel are not told apart, however tight a handful of confirmation jobs happened to be
NOISE_FLOOR_MIN = 0.015


def latest_run(task_name: str):
    runs = [r for r in list_runs() if r.task_name == task_name]
    return max(runs, key=lambda r: r.record().get("created", "")) if runs else None


def noise_floor(best: dict) -> float:
    """The relative gain a generation must clear: the spread of its same-job ratios
    ((max - min) / min), never below NOISE_FLOOR_MIN."""
    ratios = best.get("ratios") or []
    def factor(values):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in values):
            return math.inf
        return max(values) / min(values) if len(values) > 1 else 1.0
    if ratios:
        spread = factor(ratios) - 1.0
    else:
        # Separate jobs are NOT paired ratios. Both sides' spread contributes.
        spread = factor(best.get("samples_ms", [])) * factor(best.get("ref_samples_ms", [])) - 1
    return max(NOISE_FLOOR_MIN, spread)


def promotion_problem(task, best: dict) -> str | None:
    if task.language != "pypto":
        return None
    if best.get("ratio_basis") != "cross-job-median-vs-median":
        return "PyPTO promotion needs a fresh cross-job reference confirmation"
    requested = max(3, best.get("requested", 0))
    if (len(best.get("samples_ms", [])) < requested
            or len(best.get("ref_samples_ms", [])) < requested):
        return "PyPTO promotion needs at least three complete student and reference jobs"
    if best.get("rejected_jobs") or best.get("ref_workloads_mismatch"):
        return "confirmation contains a failed job or workload mismatch"
    gain = best.get("speedup")
    if not isinstance(gain, (float, int)) or not math.isfinite(gain):
        return "confirmed speedup is unavailable"
    if gain <= 1 + noise_floor(best):
        return "confirmed gain is within the measured noise floor"
    return None


def _next_pypto(task, best: dict, dest: Path, step: float) -> Path:
    if not task.leaderboard or task.leaderboard.suite != "cann-bench":
        raise ValueError("PyPTO promotion requires a frozen CANN-Bench task")
    if problem := promotion_problem(task, best):
        raise ValueError(problem)
    ignore = shutil.ignore_patterns(".git", ".jobs", ".exec", ".cann_results", "__pycache__",
                                    "*.pyc", "*.local.toml", "build", "dist", "*.egg-info")
    shutil.copytree(task.dir, dest, ignore=ignore)
    cfg = read_task_toml(dest)
    card = dest / cfg["leaderboard"]["card"]
    ref = dest / "leaderboard" / "ref"
    if ref.exists():
        shutil.rmtree(ref)  # only the new, private destination; never the original task
    ref.mkdir(parents=True)
    for name in ("eval", "gpu_entry.sh", "window.toml", ".gitignore", ".gitattributes"):
        source = card / name
        if source.is_dir():
            shutil.copytree(source, ref / name)
        elif source.exists():
            shutil.copy2(source, ref / name)
    source = ref / "src"
    source.mkdir()
    for name in ("build.sh", "setup.py", "cann_bench"):
        p = card / name
        if p.is_dir():
            shutil.copytree(p, source / name, ignore=ignore)
        else:
            shutil.copy2(p, source / name)
    (source / "cann_bench" / "kernel_impl.py").unlink(missing_ok=True)
    (source / "kernel.py").write_text(best["kernel"], encoding="utf-8")
    # A blackbox teacher can read the candidate. Seeding it with the promoted
    # body would defeat sealing even though that body was our previous output.
    problem = json.loads((card / "eval/leaderboard/problem.json").read_text())
    function = next((n for n in ast.parse(best["kernel"]).body
                     if isinstance(n, ast.FunctionDef) and n.name == problem["entry"]), None)
    if function is None:
        raise ValueError("promoted kernel must define the card's public entry function")
    function.decorator_list = []
    function.body = [ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                                          args=[ast.Constant("implement the milestone")], keywords=[]))]
    ast.fix_missing_locations(function)
    (card / "kernel.py").write_text("from __future__ import annotations\nimport torch\nimport pypto\n\n"
                                    + ast.unparse(function) + "\n", encoding="utf-8")
    conf = json.loads((ref / "eval/cann_task.json").read_text())
    conf["source_dir"] = "src"
    (ref / "eval/cann_task.json").write_text(json.dumps(conf, indent=2) + "\n")
    cfg["leaderboard"]["ref"] = "leaderboard/ref"
    cfg.pop("reference", None)
    cfg["target"] = {**cfg.get("target", {}), "ratio": float(step)}
    cfg["target"].pop("ms", None)
    (dest / "task.toml").write_text(dump_task_toml(cfg))
    loop_file = card / "loop.toml.example"
    loop = tomllib.loads(loop_file.read_text())
    loop["loop"].update(oracle_mode="blackbox", ref_name="promoted previous-generation kernel",
                         confirm_jobs=3, profiler_backend="msprof")
    loop["loop"].pop("oracle", None)
    from forgekernel.taskcard import _toml_value
    loop_file.write_text("[loop]\n" + "\n".join(
        f"{k} = {_toml_value(v)}" for k, v in loop["loop"].items()) + "\n")
    with (card / "README.md").open("a") as f:
        f.write("\n## Generation disclosure (supersedes initial seed description)\n"
                "The campaign reference is the previous confirmed best, sealed at run installation.\n"
                "The candidate starts from a signature-only stub; no reference body crosses the boundary.\n")
    (dest / "promotion.json").write_text(json.dumps({
        "parent_task": task.name, "parent_sha": best["sha"],
        "speedup": best["speedup"], "noise_floor": noise_floor(best),
        "ratio_basis": best["ratio_basis"], "samples_ms": best["samples_ms"],
        "ref_samples_ms": best["ref_samples_ms"], "disclosure": "blackbox",
        "student_seed": "signature-only stub; promoted body is teacher/student sealed",
    }, indent=2) + "\n")
    from forgekernel.paths import template_dir
    for tree in (card, ref):
        for name in ("cann_bench.py", "ascend_profiler.py"):
            shutil.copy2(template_dir() / "eval" / name, tree / "eval" / name)
        shutil.copy2(template_dir() / "pypto_gpu_entry.sh", tree / "gpu_entry.sh")
    load_task(dest)
    return dest


def next_generation(task, best: dict, dest: Path, step: float) -> Path:
    """tasks/<op>-g<k>/: the task with ``best`` as its reference and target ``step``x."""
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it or start the chain elsewhere")
    if not math.isfinite(step) or step <= 1:
        raise ValueError("generation target must be finite and greater than 1")
    if task.language == "pypto":
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".promote-", dir=dest.parent) as tmp:
            staged = Path(tmp) / dest.name
            _next_pypto(task, best, staged, step)
            staged.rename(dest)
        return dest
    if task.language != "cutedsl":
        raise ValueError("self-evolution needs a Python kernel to seal as the next reference "
                         f"(this task's language is {task.language!r})")
    dest.mkdir(parents=True)
    for name in ("description.md", "shapes.json"):
        shutil.copy2(task.dir / name, dest / name)
    if (task.dir / "inputs").is_dir():
        shutil.copytree(task.dir / "inputs", dest / "inputs")
    (dest / REF_DIR).mkdir()
    (dest / REF_DIR / "kernel.py").write_text(best["kernel"], encoding="utf-8")
    cfg = read_task_toml(task.dir)
    cfg["reference"] = {"run": f"{REF_DIR}/kernel.py:run", "source": REF_DIR}
    cfg["target"] = {**cfg.get("target", {}), "ratio": float(step)}
    cfg["target"].pop("ms", None)
    (dest / "task.toml").write_text(dump_task_toml(cfg), encoding="utf-8")
    load_task(dest)                       # the new generation must be a valid task
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="self_evolve.py", description=__doc__.split("\n\n")[0])
    p.add_argument("task", help="the first generation's task directory")
    p.add_argument("--step", type=float, default=1.05,
                   help="each new generation's target, x its reference's speed")
    p.add_argument("--max-gen", type=int, default=None,
                   help="stop after this many generations (default: no cap)")
    args = p.parse_args(argv)
    task_dir = Path(args.task).resolve()
    base = re.sub(r"-g\d+$", "", task_dir.name)
    for gen in itertools.count(1):
        try:
            task = load_task(task_dir)
        except TaskError as e:
            sys.stderr.write(f"[self-evolve] {e}\n")
            return 2
        print(f"[self-evolve] generation {gen}: {task_dir}", flush=True)
        rc = launch.cmd_new([str(task_dir)])
        run = latest_run(task.name) if rc == 0 else None
        if run is not None:
            rc = launch.cmd_run([str(run.dir)])
        best = launch.best_of(run, task) if run is not None else None
        if best is None or not best.get("kernel"):
            print(f"[self-evolve] generation {gen} ended without a confirmed best kernel "
                  f"(loop exit {rc}): the chain stops here", flush=True)
            return 0 if gen > 1 else 1
        if problem := promotion_problem(task, best):
            print(f"[self-evolve] {problem}: the chain stops here", flush=True)
            return 0
        gain = best.get("speedup")
        floor = noise_floor(best)
        print(f"[self-evolve] generation {gen}: best {best['sha'][:12]}, "
              + (f"{gain:.4f}x its reference" if gain else "speedup unknown")
              + f" (noise floor {floor:.1%})", flush=True)
        if gain is None or gain <= 1.0 + floor:
            print(f"[self-evolve] gain within the {floor:.1%} noise floor: the chain stops here",
                  flush=True)
            return 0
        if args.max_gen is not None and gen >= args.max_gen:
            print(f"[self-evolve] reached --max-gen {args.max_gen}", flush=True)
            return 0
        task_dir = next_generation(task, best, task_dir.parent / f"{base}-g{gen + 1}", args.step)
        print(f"[self-evolve] next generation: {task_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
