# tinygrad Testing Guide

> Testing strategy, how to run tests, and how to write them.
>
> Last updated: 2025-04-15

---

## Overview

tinygrad has a comprehensive test suite of 331 files organized by scope and hardware requirements. Tests use Python's `unittest` framework with `pytest` as the runner, `hypothesis` for property-based testing, and `numpy`/`torch` as reference implementations.

**Key configuration** (from `pyproject.toml`):
- Test timeout: 300 seconds per test
- Test paths: `test/`
- Excluded from test discovery: `extra/`, `.hypothesis/`, `.git/`

---

## Test Categories

### `test/backend/` — Per-Backend Tests (40 files)

Tests that run on **every** hardware backend. These are the most important tests — they verify that operations produce correct results regardless of device.

| File | What it tests |
|------|--------------|
| `test_ops.py` | All tensor operations (largest test file, ~196KB) |
| `test_tensor.py` | Core Tensor class functionality |
| `test_jit.py` | JIT compilation and caching |
| `test_nn.py` | Neural network layers (Conv2d, Linear, BatchNorm, etc.) |
| `test_linearizer.py` | Kernel linearization and code generation |
| `test_schedule.py` | Operation scheduling and fusion |
| `test_dtype.py` | Data type handling, casting, precision |
| `test_symbolic_*.py` | Symbolic shape computation |
| `test_multitensor.py` | Multi-device tensor operations |

Run on a specific backend:
```bash
DEV=CPU python3 -m pytest test/backend/test_ops.py
DEV=CUDA python3 -m pytest test/backend/test_ops.py
DEV=METAL python3 -m pytest test/backend/test_ops.py
```

### `test/null/` — Backend-Independent Tests (57 files)

Tests that use the NULL device — no real hardware needed. These test the compiler, scheduler, pattern matcher, and symbolic engine.

| File | What it tests |
|------|--------------|
| `test_uops.py` | UOp representation and manipulation |
| `test_pattern_matcher.py` | PatternMatcher and UPat |
| `test_uop_symbolic.py` | Symbolic simplification |
| `test_schedule.py` | Schedule optimization |
| `test_device.py` | Device abstraction |
| `test_llm_*.py` | LLM utilities |

```bash
python3 -m pytest test/null/
```

### `test/unit/` — Focused Unit Tests (36 files)

Single-concern tests for specific components.

| File | What it tests |
|------|--------------|
| `test_gradient.py` | Automatic differentiation |
| `test_attention.py` | Attention mechanism implementations |
| `test_function.py` | @function decorator |
| `test_conv.py` | Convolution operations |
| `test_assign.py` | Tensor assignment |
| `test_schedule_cache.py` | Schedule caching |
| `test_helpers.py` | Helper utilities |

```bash
python3 -m pytest test/unit/
```

### `test/external/` — Integration & Benchmark Tests (83 files)

Integration tests that test real-world scenarios, plus benchmarks and fuzzers.

| Pattern | What it tests |
|---------|--------------|
| `external_test_*.py` | Functional integration tests |
| `external_benchmark_*.py` | Performance benchmarks |
| `fuzz_*.py` | Symbolic and shape fuzzing |
| `mlperf_*/` | MLPerf training benchmarks |
| `process_replay/` | Kernel comparison against master |

```bash
# Run integration tests
python3 -m pytest test/external/external_test_example.py

# Run fuzz tests
python3 -m pytest test/external/fuzz_symbolic.py
```

### `test/amd/` — AMD-Specific Tests (19 files)

Tests for AMD hardware instruction encoding/decoding.

```bash
# Requires AMD hardware or MOCKGPU
MOCKGPU=1 python3 -m pytest test/amd/
```

### `test/device/` — Device-Specific Tests (4 files)

Tests for specific device features:
- `test_amd_llvm.py` — AMD LLVM backend
- `test_hcq.py` — Hardware Command Queue
- `test_metal.py` — Metal-specific features
- `test_ocl.py` — OpenCL-specific features

### `test/models/` — End-to-End Model Tests (9 files)

Full model inference/training tests:
- `test_bert.py`, `test_mnist.py`, `test_efficientnet.py`
- `test_whisper.py`, `test_rnnt.py`
- `test_onnx.py`, `test_train.py`, `test_end2end.py`

### `test/opt/` — Optimization Tests (3 files)

- `test_gen_float4.py` — Float4 vectorization
- `test_kernel_opts.py` — Kernel optimization correctness
- `test_tensor_cores.py` — Tensor core utilization

### `test/speed/` — Performance Tests (4 files)

Speed benchmarks comparing tinygrad against torch and measuring device throughput.

### `test/mockgpu/` — Mock GPU Implementations

Mock implementations of GPU hardware for testing without real devices:
- `am/` — Abstract Machine mock
- `amd/` — AMD emulator + disassembler
- `cuda/` — CUDA mock
- `nv/` — NVIDIA mock

---

## Running Tests

### Quick Commands

```bash
# Quick sanity test (what pre-commit runs first)
python3 -m pytest test/test_tiny.py

# Single test file
python3 -m pytest test/backend/test_ops.py

# Single test class
python3 -m pytest test/backend/test_ops.py::TestOps

# Single test method
python3 -m pytest test/backend/test_ops.py::TestOps::test_add

# Run in parallel (6 workers)
python3 -m pytest -n=6 test/backend/test_ops.py

# With verbose output
python3 -m pytest -v test/backend/test_ops.py

# Stop on first failure
python3 -m pytest -x test/backend/test_ops.py
```

### Running on Specific Backends

```bash
DEV=CPU python3 -m pytest test/backend/
DEV=CUDA python3 -m pytest test/backend/
DEV=METAL python3 -m pytest test/backend/
DEV=AMD python3 -m pytest test/backend/
DEV=CL python3 -m pytest test/backend/
```

### Running What CI Runs

The pre-commit `tests` hook runs:
```bash
OMP_NUM_THREADS=1 SKIP_SLOW_TEST=1 PYTHONPATH="." python3 -m pytest -n=6 \
  test/backend/test_ops.py \
  test/backend/test_schedule.py \
  test/unit/test_assign.py \
  test/backend/test_tensor.py \
  test/backend/test_jit.py \
  test/unit/test_schedule_cache.py \
  test/null/test_pattern_matcher.py \
  test/null/test_uop_symbolic.py \
  test/unit/test_helpers.py
```

### Useful Environment Variables for Testing

| Variable | Effect |
|----------|--------|
| `SKIP_SLOW_TEST=1` | Skip tests marked as slow |
| `CHECK_OOB=1` | Enable out-of-bounds checking |
| `SPEC=1` | Enable UOp specification validation |
| `CAPTURE_PROCESS_REPLAY=1` | Record kernels for process replay |
| `MOCKGPU=1` | Use mock GPU instead of real hardware |
| `OMP_NUM_THREADS=1` | Limit CPU threads (avoids test interference) |

---

## Process Replay

Process replay is a CI mechanism that ensures refactors don't change generated kernels.

### How It Works

1. CI records all generated kernels when running tests
2. Kernels are compared against the master branch
3. If any kernel differs, CI fails

### When It Applies

- PR titles containing `[pr]` trigger process replay
- Indicates the PR is a refactor/speedup with no intended behavior change

### Running Locally

```bash
# Record kernels
CAPTURE_PROCESS_REPLAY=1 python3 -m pytest test/backend/test_ops.py

# Compare against master
python3 test/external/process_replay/process_replay.py
```

See `test/external/process_replay/README.md` for full details.

---

## Writing Tests

### Basic Test Structure

```python
import unittest
from tinygrad import Tensor, dtypes
import numpy as np

class TestMyFeature(unittest.TestCase):
  def test_basic(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = (a + b).numpy()
    np.testing.assert_allclose(result, [5, 7, 9])

  def test_with_dtype(self):
    a = Tensor([1.0, 2.0], dtype=dtypes.float16)
    result = a.sum().numpy()
    np.testing.assert_allclose(result, 3.0, atol=1e-2)

if __name__ == "__main__":
  unittest.main()
```

### Using Hypothesis for Property-Based Testing

```python
from hypothesis import given, strategies as st

class TestArithmetic(unittest.TestCase):
  @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=100))
  def test_sum_matches_numpy(self, data):
    t = Tensor(data)
    np.testing.assert_allclose(t.sum().numpy(), np.sum(data), rtol=1e-5)
```

### Using torch as Reference

```python
import torch

class TestOps(unittest.TestCase):
  def test_matmul_matches_torch(self):
    np_a = np.random.randn(4, 8).astype(np.float32)
    np_b = np.random.randn(8, 4).astype(np.float32)

    tiny_result = (Tensor(np_a) @ Tensor(np_b)).numpy()
    torch_result = (torch.tensor(np_a) @ torch.tensor(np_b)).numpy()

    np.testing.assert_allclose(tiny_result, torch_result, atol=1e-5)
```

### Guidelines

- Use `np.testing.assert_allclose` with appropriate `atol`/`rtol` for float comparisons
- Test on the default device; `test/backend/` tests run on all devices automatically
- Mark known-failing tests with `@unittest.expectedFailure`
- Mark slow tests so they can be skipped with `SKIP_SLOW_TEST=1`
- Put test helpers in `test/helpers.py`
- New tests for a feature should go in the most specific category that applies
