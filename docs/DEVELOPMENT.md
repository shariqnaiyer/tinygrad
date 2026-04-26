# tinygrad Development Guide

> How to develop, test, lint, and contribute to tinygrad.
>
> Last updated: 2025-04-15

---

## Getting Started

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e '.[linting,testing]'
pre-commit install
```

---

## Running tinygrad

### Basic Usage

```bash
# Run a script
python3 examples/beautiful_mnist.py

# With debug output (levels 1-7)
DEBUG=2 python3 examples/beautiful_mnist.py

# On a specific device
DEV=CPU python3 examples/beautiful_mnist.py
DEV=CUDA python3 examples/beautiful_mnist.py

# See generated kernel code
DEBUG=4 python3 -c "from tinygrad import Tensor; Tensor.rand(32,32).realize()"

# Visualize computation graph
VIZ=1 python3 -c "from tinygrad import Tensor; (Tensor.rand(4,4) + Tensor.rand(4,4)).realize()"
```

### BEAM Search Optimization

```bash
# Use BEAM search with 2 candidates for kernel optimization
BEAM=2 python3 examples/beautiful_mnist.py

# Higher values try more configurations (slower compile, potentially faster kernels)
BEAM=8 python3 my_model.py
```

---

## Code Style

tinygrad uses a distinctive code style:

- **2-space indentation** (not 4)
- **150-character line length** (not 80 or 120)
- **Compact style** — multiple statements on one line where readable
- **No code golf** — don't delete newlines just to reduce line count
- **Target Python 3.11+**

### Linting

```bash
# Run ruff linter
python3 -m ruff check .

# Run mypy type checker
python3 -m mypy

# Run pylint
python3 -m pylint tinygrad/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

The project uses 5 pre-commit hooks (configured in `.pre-commit-config.yaml`):

| Hook | What it does |
|------|-------------|
| `ruff` | Static linting with ruff |
| `tiny` | Runs `test/test_tiny.py` — quick sanity tests |
| `mypy` | Type checking |
| `example` | Tests all device backends work |
| `tests` | Comprehensive test suite (6 parallel jobs, skips slow tests) |

Skip specific hooks when needed:
```bash
SKIP=tests,example git commit -m "quick fix"
```

---

## Testing

### Test Organization

Tests are organized into categories in the `test/` directory:

| Directory | Scope | Description |
|-----------|-------|-------------|
| `test/backend/` | Per-backend | Tests that run on **each** hardware backend (40 files) |
| `test/null/` | Backend-independent | Tests using the NULL device — no hardware needed (57 files) |
| `test/unit/` | Focused | Single-concern unit tests (36 files) |
| `test/external/` | Integration | Integration tests, benchmarks, and fuzz tests (83 files) |
| `test/amd/` | AMD-specific | AMD hardware instruction tests (19 files) |
| `test/device/` | Device-specific | Tests for specific device features (4 files) |
| `test/models/` | End-to-end | Full model tests (9 files) |
| `test/opt/` | Optimization | Kernel optimization tests (3 files) |
| `test/speed/` | Performance | Speed benchmarks (4 files) |
| `test/mockgpu/` | Mock | Mock GPU implementations for hardware-free testing |

### Running Tests

```bash
# Run the quick sanity test
python3 -m pytest test/test_tiny.py

# Run a specific test file
python3 -m pytest test/backend/test_ops.py

# Run a specific test class or method
python3 -m pytest test/backend/test_ops.py::TestOps::test_add

# Run all tests
python3 -m pytest test/

# Run tests in parallel (6 workers)
python3 -m pytest -n=6 test/backend/test_ops.py

# Run with a specific backend
DEV=CPU python3 -m pytest test/backend/test_ops.py
DEV=CUDA python3 -m pytest test/backend/test_ops.py

# Skip slow tests
SKIP_SLOW_TEST=1 python3 -m pytest test/

# Run with out-of-bounds checking
CHECK_OOB=1 python3 -m pytest test/

# Run what pre-commit runs
OMP_NUM_THREADS=1 SKIP_SLOW_TEST=1 python3 -m pytest -n=6 \
  test/backend/test_ops.py test/backend/test_schedule.py \
  test/unit/test_assign.py test/backend/test_tensor.py \
  test/backend/test_jit.py test/unit/test_schedule_cache.py \
  test/null/test_pattern_matcher.py test/null/test_uop_symbolic.py \
  test/unit/test_helpers.py
```

### Process Replay Tests

Process replay compares generated kernels between your PR and the master branch. If your PR is a refactor (no behavior change), add `[pr]` to the PR title:

```
fix: improve matmul scheduling [pr]
```

This tells CI to verify that generated kernels match master exactly. See `test/external/process_replay/` for details.

### Writing Tests

- Use Python's `unittest` framework (tests extend `unittest.TestCase`)
- Use `hypothesis` for property-based testing (generates random test inputs)
- Use `numpy` and `torch` as reference implementations for numerical validation
- Test helpers are in `test/helpers.py`
- Tests should not be brittle — avoid exact float comparisons (use `np.testing.assert_allclose`)

---

## Debugging

### DEBUG Levels

| Level | Output |
|-------|--------|
| `DEBUG=1` | Devices in use |
| `DEBUG=2` | Kernel timing, memory usage, bandwidth (per kernel) |
| `DEBUG=3` | Applied kernel optimizations |
| `DEBUG=4` | Generated kernel source code |
| `DEBUG=5` | UOp intermediate representation |
| `DEBUG=6` | Linearized UOp sequence |
| `DEBUG=7` | Generated assembly code |

### Visualization

```bash
# Launch the computation graph visualizer
VIZ=1 python3 your_script.py
# Opens a web browser with an interactive DAG viewer
```

### Profiling

```bash
# Profile kernel execution
PROFILE=1 python3 your_script.py

# Convert to Perfetto format for visualization
python3 extra/perfetto/to_perfetto.py
# Open perfetto.html in browser
```

### SPEC Validation

```bash
# Enable UOp specification checking (slower but catches bugs)
SPEC=1 python3 your_script.py

# Stricter validation (also validates rendering)
SPEC=2 python3 your_script.py
```

---

## Building Documentation

```bash
# Install docs dependencies
python3 -m pip install -e '.[docs]'

# Serve docs locally with auto-reload
mkdocs serve -w tinygrad/
# or
./serve_docs.sh

# Build static site
mkdocs build --strict
```

The docs use MkDocs with the Material theme. API docs are auto-generated from docstrings using `mkdocstrings`.

---

## Line Count Enforcement

tinygrad enforces a maximum of ~24,000 lines in the core library. The `sz.py` script counts lines and tokens:

```bash
# Check current line count
python3 sz.py

# CI enforces this limit
MAX_LINE_COUNT=24000 python3 sz.py
```

The count excludes `runtime/autogen/` (auto-generated FFI bindings) and `viz/assets/` (third-party JS libraries).

---

## Contributing Guidelines

### What gets merged

- **Bug fixes** with regression tests
- **Bounty solutions** — see the [bounty spreadsheet](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing)
- **Features** with regression tests (API should match torch/numpy where possible)
- **Clear-win refactors** that improve readability or reduce complexity
- **Tests and fuzzers** that find real bugs
- **Dead code removal** from core `tinygrad/` folder

### What gets closed

- Code golf (reducing lines without improving readability)
- Docs/whitespace changes from non-core contributors
- Unbenched "speedups"
- Complex or large-diff PRs (break them into smaller PRs)
- Changes to `extra/` unless the current code is broken

### PR Guidelines

- Keep PRs small and focused
- Include regression tests for bug fixes
- Benchmark any claimed speedups
- Use `[pr]` in PR title for refactors that shouldn't change kernel output
- Consider the line count tradeoff — a 3-line feature has a lower bar than a 300-line feature

### Branch Strategy

- `master` is the main development branch
- PRs are made against `master`
- CI runs on every push to master and on all PRs
