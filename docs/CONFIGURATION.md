# tinygrad Configuration Reference

> Every environment variable, ContextVar, config file, and feature flag.
>
> Last updated: 2025-04-15

---

## Environment Variables / ContextVars

All configuration in tinygrad is done through `ContextVar` instances (defined in `helpers.py`), which read from environment variables and can be temporarily overridden with `Context(VAR=value)`.

### Core Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `DEV` | (auto-detected) | Target device. Syntax: `device[:renderer[:arch]]`. Examples: `AMD`, `NV:CUDA:sm_70`, `USB+AMD:LLVM` |
| `DEBUG` | `0` | Debug output level (0-7). See [Debug Levels](#debug-levels) below |
| `JIT` | `1` (macOS x86: `2`) | JIT mode: 0=disabled, 1=enabled with graphs, 2=enabled without graphs |
| `JIT_BATCH_SIZE` | `32` | Maximum number of kernels in a JIT graph batch |
| `BEAM` | `0` | BEAM search width for kernel optimization. 0=use heuristics |
| `NOOPT` | `0` | Disable all kernel optimizations |

### Data Types

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_FLOAT` | `FLOAT32` | Default float dtype. Set to `HALF`, `BFLOAT16`, `FLOAT64`, etc. |
| `FLOAT16` | `0` | Use float16 for image operations |
| `ALLOW_TF32` | `0` | Enable TensorFloat-32 tensor cores on Ampere+ GPUs |
| `EMULATED_DTYPES` | `""` | Comma-separated list of dtypes to emulate (cast to nearest supported type) |

### Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `0` | Enable 2D image-specific optimizations |
| `USE_TC` | `1` | Enable tensor core detection and usage |
| `TC_SELECT` | `-1` | Select specific tensor core variant (-1=auto) |
| `TC_OPT` | `0` | Tensor core optimization level |
| `AMX` | `0` | Enable Apple AMX (Advanced Matrix Extensions) |
| `WINO` | `0` | Enable Winograd convolution |
| `TRANSCENDENTAL` | `1` | Use hardware transcendental functions (sin, cos, exp, log) |
| `DEVECTORIZE` | `1` | Enable devectorization pass in codegen |
| `SPLIT_REDUCEOP` | `1` | Enable reduction operation splitting |
| `NOLOCALS` | `0` | Disable local (shared) memory usage in kernels |
| `FUSE_OPTIM` | `0` | Fuse optimizer operations (concatenate all params into one buffer) |
| `DISABLE_FAST_IDIV` | `0` | Disable fast integer division optimization |
| `CORRECT_DIVMOD_FOLDING` | `0` | Use correct div/mod folding (slower but more precise) |
| `USE_ATOMICS` | `0` | Use atomic operations for reductions |
| `PCONTIG` | `0` | Enable partial contiguous optimization in rangeify |

### Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `NO_MEMORY_PLANNER` | `0` | Disable the memory planner (TLSF allocator) |
| `MAX_BUFFER_SIZE` | `0` | Maximum buffer size in bytes (0=unlimited) |
| `MAX_KERNEL_BUFFERS` | `0` | Maximum buffers per kernel (0=unlimited) |
| `LRU` | `1` | Enable LRU cache for compiled kernels |

### Debugging & Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `VIZ` | `0` | Enable computation graph visualization (opens web UI) |
| `PROFILE` | `abs(VIZ)` | Enable profiling (automatically on when VIZ is on) |
| `SPEC` | `1` | UOp specification validation level (0=off, 1=basic, 2=strict) |
| `CHECK_OOB` | `0` | Enable out-of-bounds checking |
| `VALIDATE_WITH_CPU` | `0` | Validate GPU results by re-running on CPU |
| `TRACEMETA` | `1` | Track and display metadata in debug output |
| `DEBUG_RANGEIFY` | `0` | Debug the rangeify pass |

### Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHELEVEL` | `2` | Disk cache level for compiled kernels |
| `IGNORE_BEAM_CACHE` | `0` | Ignore cached BEAM search results |
| `SCACHE` | `1` | Enable schedule cache |
| `CCACHE` | `1` | Enable compiler cache (ccache for C compilation) |

### Multi-Device

| Variable | Default | Description |
|----------|---------|-------------|
| `RING` | `1` | Use ring topology for multi-GPU communication |
| `ALL2ALL` | `0` | Use all-to-all communication pattern |
| `ALLREDUCE_CAST` | `1` | Cast during all-reduce operations |

### Process Control

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPTURING` | `1` | Enable capture mode for process replay |
| `CAPTURE_PROCESS_REPLAY` | `0` | Record kernels for process replay comparison |
| `ALLOW_DEVICE_USAGE` | `1` | Allow real device usage (set to 0 for testing) |
| `CPU_COUNT` | (auto) | Number of CPU threads |
| `NULL_ALLOW_COPYOUT` | `0` | Allow copyout from NULL device |
| `OMP_NUM_THREADS` | (system) | OpenMP thread count (not a ContextVar, standard env var) |

### Backend-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_PATH` | `/usr/local/cuda` | CUDA toolkit path for headers |
| `WEBGPU_BACKEND` | (auto) | Force WebGPU backend: `WGPUBackendType_Metal`, `WGPUBackendType_Vulkan`, etc. |
| `OPTIM_DTYPE` | `float32` | Optimizer internal dtype |
| `CONST_LR` | `0` | Use constant learning rate (no Tensor wrapper) |

### Compiler Internals (in `uop/ops.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACK_MATCH_STATS` | `2 if VIZ else 0` | Track pattern matcher statistics |
| `REWRITE_STACK_LIMIT` | `250000` | Maximum rewrite stack depth before error |

### Backend-Specific Profiling (in device runtimes)

| Variable | Default | Description |
|----------|---------|-------------|
| `SQTT` | `abs(VIZ)>=2` | Enable AMD SQTT (System Queue Trace) profiling |
| `SQTT_ITRACE_SE_MASK` | `0b11` | AMD SQTT shader engine mask |
| `SQTT_LIMIT_SE` | `0` | AMD SQTT limit shader engines |
| `SQTT_SIMD_SEL` | `0` | AMD SQTT SIMD selection |
| `SQTT_TOKEN_EXCLUDE` | `0` | AMD SQTT token exclusion mask |
| `PMC` | `abs(VIZ)>=2` | Enable AMD Performance Monitor Counters |
| `PMA` | `abs(VIZ)>=2` | Enable NVIDIA Performance Monitoring Architecture |

### Internal / Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `TUPLE_ORDER` | `1` | Enable tuple ordering optimization |
| `OPENPILOT_HACKS` | `0` | Enable OpenPilot compatibility hacks |

---

## Debug Levels

| Level | Output |
|-------|--------|
| `DEBUG=1` | Devices in use |
| `DEBUG=2` | Kernel timing, memory, bandwidth per kernel |
| `DEBUG=3` | Applied kernel optimizations |
| `DEBUG=4` | Generated kernel source code |
| `DEBUG=5` | UOp intermediate representation |
| `DEBUG=6` | Linearized UOp sequence |
| `DEBUG=7` | Generated assembly code |

Usage:
```bash
DEBUG=4 python3 my_script.py
```

Or in code:
```python
from tinygrad import Context
with Context(DEBUG=4):
  result = model(x).realize()
```

---

## Config Files

### `pyproject.toml`

The main project configuration file. Defines:
- Package metadata (name, version, dependencies)
- Build system (setuptools)
- Optional dependency groups (`linting`, `testing_minimal`, `testing_unit`, `testing`, `docs`)
- Tool configuration for mypy, pytest, ruff, mutmut

### `.pre-commit-config.yaml`

Pre-commit hooks that run on `git commit`:
1. **ruff** — Static linting
2. **tiny** — Quick sanity tests (`test/test_tiny.py`)
3. **mypy** — Type checking
4. **example** — Test all device backends
5. **tests** — Comprehensive test suite (parallel)

Skip hooks: `SKIP=tests,example git commit -m "msg"`

### `.pylintrc`

Pylint configuration:
- 2-space indentation
- 150-char line length
- 1000 max module lines
- Many checks disabled (conventions, refactoring)

### `.coveragerc`

Coverage configuration:
- Source: `tinygrad/`
- Branch coverage: enabled

### `mkdocs.yml`

Documentation site configuration:
- MkDocs with Material theme
- Auto-generated API docs via mkdocstrings
- Dark/light mode support

### `opencode.json`

OpenCode IDE integration:
- Formatter: disabled
- LSP: disabled

---

## Using Context

ContextVars can be set three ways:

### 1. Environment Variable
```bash
DEBUG=4 BEAM=2 python3 script.py
```

### 2. Context Manager
```python
from tinygrad import Context
with Context(DEBUG=4, BEAM=2):
  result = model(x).realize()
```

### 3. Decorator
```python
from tinygrad import Context

@Context(DEBUG=0)
def quiet_function():
  return Tensor.rand(3, 3).realize()
```

Context managers nest properly — inner contexts override outer ones, and values are restored on exit.
