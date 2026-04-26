# tinygrad Documentation Plan

> Last updated: 2025-04-15

## 1. Project Overview

**tinygrad** (v0.12.0) is an end-to-end deep learning stack maintained by tiny corp (George Hotz). It sits between PyTorch and micrograd, offering:

- **Tensor library** with autograd
- **IR and compiler** that fuse and lower kernels (UOp-based)
- **JIT + graph execution**
- **nn / optim / datasets** for real training
- **20+ hardware backends**: CPU, CUDA, Metal, AMD, NV, QCOM, OpenCL, WebGPU, DSP, etc.

The core is intentionally tiny (~24,000 lines enforced by CI) and hackable.

---

## 2. Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ (core), JavaScript (viz), C/C++ (thunder kernels), Swift (usbgpu installer), CUDA/HIP/PTX/WGSL (generated) |
| Package Manager | pip / setuptools (pyproject.toml) |
| Build System | setuptools, mkdocs (docs) |
| Test Framework | pytest (with xdist, timeout, split, hypothesis) |
| Type Checking | mypy 1.19.1 |
| Linting | ruff 0.14.10, pylint |
| Pre-commit | pre-commit (5 hooks: ruff, tiny, mypy, example, tests) |
| CI/CD | GitHub Actions (test.yml, docs.yml, autogen.yml, benchmark.yml, python-publish.yml) |
| Documentation | MkDocs + Material theme + mkdocstrings |
| Deployment | GitHub Pages (docs), PyPI (package) |
| License | MIT |

---

## 3. Architecture Summary

The tinygrad compilation pipeline has 4 stages:

```
Tensor API  →  Scheduler  →  Codegen/Lowering  →  Runtime Execution
(lazy UOp      (fuse ops,    (optimize, render    (compile, run on
 graph)         schedule)     to target code)       hardware)
```

### Core Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `tinygrad/tensor.py` | User-facing Tensor API, lazy computation graph | tensor.py |
| `tinygrad/uop/` | UOp IR system - the intermediate representation | ops.py, spec.py, symbolic.py, upat.py |
| `tinygrad/mixin/` | Operation mixins (elementwise, reduce, movement) | elementwise.py, reduce.py, movement.py |
| `tinygrad/dtype.py` | Type system (DType, PtrDType, ImageDType) | dtype.py |
| `tinygrad/device.py` | Device abstraction, Buffer, Compiled | device.py |
| `tinygrad/gradient.py` | Automatic differentiation | gradient.py |
| `tinygrad/function.py` | @function decorator for graph capture | function.py |
| `tinygrad/callify.py` | Transform computation to CALL nodes | callify.py |
| `tinygrad/schedule/` | Schedule creation, indexing, memory planning | __init__.py, rangeify.py, indexing.py, memory.py |
| `tinygrad/codegen/` | Code generation pipeline + optimization | __init__.py, gpudims.py, simplify.py |
| `tinygrad/codegen/opt/` | Kernel optimization (BEAM search, heuristics, tensor cores) | search.py, heuristic.py, tc.py |
| `tinygrad/codegen/late/` | Late-stage passes (linearize, devectorize, expand) | linearizer.py, devectorizer.py, expander.py |
| `tinygrad/renderer/` | Target-specific code generators | cstyle.py, ptx.py, wgsl.py, llvmir.py, nir.py, amd/ |
| `tinygrad/engine/` | JIT compilation and schedule execution | jit.py, realize.py |
| `tinygrad/runtime/` | Hardware runtimes, compilers, allocators | ops_*.py, support/, graph/, autogen/ |
| `tinygrad/nn/` | Neural network layers, optimizers, state management | __init__.py, optim.py, state.py, onnx.py |
| `tinygrad/apps/` | Applications (LLM inference) | llm.py |
| `tinygrad/viz/` | Web-based computation graph visualization | serve.py, index.html |

### Supporting Directories

| Directory | Purpose |
|-----------|---------|
| `extra/` | Supplementary tools: models, datasets, drivers, benchmarks, profiling, PyTorch backend |
| `extra/models/` | Pre-built models (BERT, Llama, ResNet, CLIP, ViT, etc.) |
| `extra/datasets/` | Dataset loaders (ImageNet, SQuAD, LibriSpeech, etc.) |
| `extra/gemm/` | GEMM benchmarks and optimized implementations |
| `extra/thunder/` | High-performance kernel library (Kittens) for AMD/CUDA/Metal |
| `examples/` | Example applications (LLMs, Stable Diffusion, YOLO, training scripts, MLPerf) |
| `test/` | Test suite (331 files across backend/, null/, unit/, external/, amd/, device/, models/) |

---

## 4. Existing Documentation Audit

### Complete / Well-Documented (keep as-is, augment if needed)
| File | Status | Notes |
|------|--------|-------|
| `docs/developer/speed.md` | **COMPLETE** | Excellent technical depth on 4 speed aspects |
| `docs/developer/hcq.md` | **COMPLETE** | Comprehensive with code examples |
| `docs/runtime.md` | **COMPLETE** | Good runtime reference |
| `docs/mnist.md` | **COMPLETE** | Well-structured tutorial |
| `docs/quickstart.md` | **COMPLETE** | Good beginner guide with training walkthrough |
| `docs/tinybox.md` | **COMPLETE** | Practical hardware setup guide |
| `docs/env_vars.md` | **COMPLETE** | Good environment variables reference |

### Partial / Needs Enhancement
| File | Status | What's Missing |
|------|--------|---------------|
| `docs/developer/developer.md` | **PARTIAL** | Very high-level, needs detailed explanations, code examples |
| `docs/developer/runtime.md` | **PARTIAL** | Structure exists but no implementation guide |
| `docs/developer/layout.md` | **PARTIAL** | Has auto-docs but no conceptual explanations |
| `docs/developer/am.md` | **PARTIAL** | Technical but lacks setup/usage examples |
| `docs/tinygpu.md` | **PARTIAL** | Good but needs troubleshooting section |

### Minimal / API-Reference Only (need conceptual content)
| File | Status | What's Missing |
|------|--------|---------------|
| `docs/developer/uop.md` | **MINIMAL** | Only auto-generated signatures, no concepts |
| `docs/nn.md` | **MINIMAL** | Auto-docs only, no examples or explanations |
| `docs/dtypes.md` | **MINIMAL** | Auto-docs only, no type system explanation |
| `docs/showcase.md` | **MINIMAL** | Just links, no descriptions or run instructions |
| `docs/tensor/index.md` | **MINIMAL** | Just class reference |
| `docs/tensor/properties.md` | **MINIMAL** | API reference only |
| `docs/tensor/creation.md` | **MINIMAL** | API reference only |
| `docs/tensor/movement.md` | **MINIMAL** | API reference only |
| `docs/tensor/elementwise.md` | **MINIMAL** | API reference only |
| `docs/tensor/ops.md` | **MINIMAL** | API reference only |

### Does Not Exist (must be created)
- `docs/README.md` - Master documentation hub
- `docs/ARCHITECTURE.md` - System architecture with diagrams
- `docs/SETUP.md` - Complete environment setup guide
- `docs/DEVELOPMENT.md` - Developer workflow guide
- `docs/API.md` - Consolidated public API reference with examples
- `docs/DATA_MODELS.md` - Data model documentation
- `docs/CONFIGURATION.md` - Complete configuration reference
- `docs/GLOSSARY.md` - Domain terminology
- `docs/DECISIONS.md` - Architectural decision records
- `docs/TROUBLESHOOTING.md` - Debugging and troubleshooting guide
- `docs/DEPENDENCIES.md` - Dependency documentation
- `docs/TESTING.md` - Testing strategy and guide
- `docs/FILE_STRUCTURE.md` - Annotated directory tree

---

## 5. Documentation Plan - Priority Order

### Priority 1: Foundation Documents (create first)

| # | Document | Type | Description |
|---|----------|------|-------------|
| 1 | `docs/README.md` | New | Master hub linking all docs, project overview, navigation |
| 2 | `docs/ARCHITECTURE.md` | New | System architecture, mermaid diagrams, data flow, compilation pipeline |
| 3 | `docs/FILE_STRUCTURE.md` | New | Annotated directory tree, every directory/file explained |
| 4 | `docs/GLOSSARY.md` | New | All domain terms (UOp, BEAM, HCQ, realize, kernel fusion, etc.) |

### Priority 2: Getting Started Documents

| # | Document | Type | Description |
|---|----------|------|-------------|
| 5 | `docs/SETUP.md` | New | Zero-to-running setup for macOS, Linux, Windows; each backend |
| 6 | `docs/DEVELOPMENT.md` | New | Dev workflow, testing, linting, building, pre-commit, CI |
| 7 | `docs/TESTING.md` | New | Test categories, running tests, process replay, writing tests |

### Priority 3: Reference Documents

| # | Document | Type | Description |
|---|----------|------|-------------|
| 8 | `docs/API.md` | New | Consolidated public API with usage examples |
| 9 | `docs/DATA_MODELS.md` | New | UOp, Tensor, Buffer, DType, Ops, ProgramSpec field-by-field |
| 10 | `docs/CONFIGURATION.md` | New | All env vars, ContextVars, config files, feature flags |
| 11 | `docs/DEPENDENCIES.md` | New | Every dependency with purpose and version info |

### Priority 4: Deep Dive Documents

| # | Document | Type | Description |
|---|----------|------|-------------|
| 12 | `docs/DECISIONS.md` | New | Why lazy evaluation, UOp IR, pattern matching, no nn.Module, etc. |
| 13 | `docs/TROUBLESHOOTING.md` | New | Common errors, DEBUG levels walkthrough, debugging workflow |

### Priority 5: Audit and Enhancement Pass

| # | Document | Type | Description |
|---|----------|------|-------------|
| 14 | Review loop | Audit | Re-walk all files, check accuracy, improve clarity, add examples |

---

## 6. Estimated Passes

- **Pass 1**: Create all Priority 1-3 documents (foundation + getting started + reference)
- **Pass 2**: Create Priority 4 documents (deep dives)
- **Pass 3**: Audit and enhancement loop (completeness, accuracy, clarity, consistency)
- **Pass 4+**: Iterative refinement until no meaningful changes remain

---

## 7. Documentation Standards

- Every doc file starts with: `# Title`, one-line summary, `> Last updated: YYYY-MM-DD`
- Use GitHub-flavored Markdown
- Use mermaid for diagrams where applicable
- Use exact function names, param names, types — no hand-waving
- Cross-reference between docs using relative links
- Include practical code examples, not just API signatures
- Flag anything uncertain with `<!-- TODO: verify -->`
- Preserve all existing good documentation — augment, don't overwrite
