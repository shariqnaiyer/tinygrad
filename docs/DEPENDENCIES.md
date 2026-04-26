# tinygrad Dependencies

> Every dependency, what it does, and why it's included.
>
> Last updated: 2025-04-15

---

## Runtime Dependencies

**tinygrad has zero runtime dependencies.** The core library (`tinygrad/`) runs with only Python 3.11+ standard library. All listed dependencies below are optional.

---

## Optional Dependency Groups

Defined in `pyproject.toml` under `[project.optional-dependencies]`.

### `linting` — Code Quality Tools

Install: `pip install -e '.[linting]'`

| Package | Version | Purpose |
|---------|---------|---------|
| `pylint` | latest | Static code analysis (custom rules in `.pylintrc`) |
| `mypy` | `==1.19.1` | Static type checking (strict settings in `pyproject.toml`) |
| `typing-extensions` | latest | Backported typing features for mypy |
| `pre-commit` | latest | Git hook framework for automated checks on commit |
| `ruff` | `==0.14.10` | Fast Python linter/formatter (replaces flake8/isort/black) |
| `numpy` | latest | Required by mypy for numpy stub types |
| `typeguard` | latest | Runtime type checking (enabled with `TYPED=1` env var) |

### `testing_minimal` — Core Test Dependencies

Install: `pip install -e '.[testing_minimal]'`

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | latest | Reference implementation for numerical correctness validation |
| `torch` | `==2.9.1` | Reference implementation for operation semantics and gradients |
| `pytest` | latest | Test runner |
| `pytest-xdist` | latest | Parallel test execution (`-n=6`) |
| `pytest-timeout` | latest | Test timeout enforcement (300s default) |
| `pytest-split` | latest | Split tests across CI jobs |
| `hypothesis` | `>=6.148.9` | Property-based testing (generates random test inputs) |
| `z3-solver` | `<4.15.4` | SMT solver for symbolic validation in `uop/validate.py`. Version pinned due to segfault in 4.15.4 when creating many z3.Context() |

### `testing_unit` — Unit Test Dependencies

Install: `pip install -e '.[testing_unit]'`

Includes everything in `testing_minimal`, plus:

| Package | Version | Purpose |
|---------|---------|---------|
| `tqdm` | latest | Progress bars for long-running test operations |
| `safetensors` | latest | SafeTensors format for model state save/load testing |
| `tabulate` | latest | Table formatting in test output |
| `openai` | latest | Used in certain test utilities |
| `gguf` | `>=0.18` | GGUF format for LLM weight loading tests |

### `testing` — Full Test Suite Dependencies

Install: `pip install -e '.[testing]'`

Includes everything in `testing_unit`, plus:

| Package | Version | Purpose |
|---------|---------|---------|
| `pillow` | latest | Image processing for vision model tests |
| `onnx` | `==1.19.0` | ONNX model format for import/export testing |
| `onnx2torch` | latest | Convert ONNX models to PyTorch for comparison |
| `onnxruntime` | latest | ONNX runtime for reference inference |
| `opencv-python` | latest | Computer vision operations for model tests |
| `transformers` | latest | Hugging Face transformers for model loading tests |
| `sentencepiece` | latest | Tokenizer for LLM tests |
| `tiktoken` | latest | OpenAI tokenizer for LLM tests |
| `blobfile` | latest | Cloud storage access for model weights |
| `librosa` | latest | Audio processing for Whisper tests |
| `numba` | `>=0.55` | JIT compilation (required by librosa) |
| `networkx` | latest | Graph algorithms used in some model tests |
| `nibabel` | latest | Medical imaging format (NIfTI) for kits19 dataset |
| `bottle` | latest | Lightweight web framework for viz server tests |
| `capstone` | latest | Disassembler for assembly output validation |
| `pycocotools` | latest | COCO dataset evaluation metrics |
| `boto3` | latest | AWS SDK for downloading datasets |
| `pandas` | latest | Data analysis for benchmark result processing |
| `influxdb3-python` | latest | InfluxDB client for benchmark logging |

### `docs` — Documentation Build Dependencies

Install: `pip install -e '.[docs]'`

| Package | Version | Purpose |
|---------|---------|---------|
| `mkdocs` | latest | Static site generator for documentation |
| `mkdocs-material` | latest | Material theme for MkDocs (dark/light mode, search) |
| `mkdocstrings[python]` | latest | Auto-generate API docs from Python docstrings |
| `markdown-callouts` | latest | Callout/admonition markdown extension |
| `markdown-exec[ansi]` | latest | Execute code blocks in docs and embed output |
| `black` | latest | Code formatter (used by mkdocstrings for signature formatting) |
| `numpy` | latest | Required for some doc examples |

---

## System Dependencies

These are not Python packages but system libraries/tools required by specific backends:

| Dependency | Backend | Purpose |
|------------|---------|---------|
| clang / gcc | CPU | C compiler for CPU kernels |
| LLVM | CPU (LLVM mode) | LLVM JIT compilation |
| CUDA toolkit | CUDA, NV | NVIDIA GPU runtime and compiler |
| ROCm / HIP | AMD, HIP | AMD GPU runtime |
| Metal framework | METAL | Apple GPU (macOS only, bundled with OS) |
| OpenCL ICD | CL | OpenCL runtime (`ocl-icd-opencl-dev` on Ubuntu) |
| Dawn | WEBGPU | WebGPU implementation |

---

## Build System

| Tool | Purpose |
|------|---------|
| `setuptools` | Python package build backend (specified in `[build-system]`) |
| `pip` | Package installer |

---

## Version Constraints Summary

| Package | Constraint | Reason |
|---------|-----------|--------|
| Python | `>=3.11` | Uses 3.11+ features (match statements, type unions) |
| mypy | `==1.19.1` | Pinned for consistent CI behavior |
| ruff | `==0.14.10` | Pinned to avoid rule changes between versions |
| torch | `==2.9.1` | Pinned for test reproducibility |
| onnx | `==1.19.0` | Pinned for compatibility |
| hypothesis | `>=6.148.9` | Minimum version for required features |
| z3-solver | `<4.15.4` | 4.15.4 has segfault bug |
| numba | `>=0.55` | Required by librosa; upper bound handled by uv |
| gguf | `>=0.18` | Minimum for required GGUF features |
