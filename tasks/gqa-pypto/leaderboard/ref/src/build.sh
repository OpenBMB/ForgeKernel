#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build dist cann_bench.egg-info cann_bench/kernel_impl.py
cp kernel.py cann_bench/kernel_impl.py
python setup.py bdist_wheel
