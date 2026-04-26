# Commenting Log

> Tracks each commenting pass: what was reviewed, changed, and what needs attention.

---

## Pass 1 — Comprehensive Commenting (2025-04-15)

### Summary
- **88 files modified** across the entire tinygrad core library
- **2,861 lines of comments/docstrings added**, 175 lines replaced
- **89 of 103 non-autogen Python files** now have module-level docstrings (86% coverage)
- **0 lines of code behavior changed** — comments only

### Files Commented (by package)

**Root level (8 files):**
- `__init__.py` — module docstring, TYPED comment, Variable alias
- `tensor.py` — module docstring (35 lines), all_tensors + _apply_map_to_tensors
- `device.py` — module docstring, _Device/Buffer/MultiBuffer/Allocator/Compiler/Compiled docstrings
- `helpers.py` — module docstring, Context/ContextVar/Target/GlobalCounters docstrings, ContextVar block comments, fetch/diskcache docstrings
- `dtype.py` — comprehensive: module doc, all class docs, all function docs, type constant comments
- `gradient.py` — comprehensive: module doc, all functions, gradient rule comments (calculus identities)
- `function.py` — comprehensive: module doc, _function class, decorator, all methods, 4-phase comments
- `callify.py` — comprehensive: module doc, AllocCtx, all 12 functions, phase labels

**uop/ (8 files):**
- `__init__.py` — comprehensive: module doc, FastEnum, Ops enum (all ~91 values commented), GroupOp (all 14 groups)
- `ops.py` — module docstring, UOpMetaClass docstring with deduplication explanation
- `symbolic.py` — module docstring (simplification rules overview)
- `upat.py` — module docstring (pattern compilation)
- `spec.py` — module docstring (validation rules, tensor/kernel/program specs)
- `decompositions.py` — module docstring (transcendental + dtype decompositions)
- `validate.py` — module docstring (Z3 bounds checking)
- `divandmod.py` — module docstring (symbolic div/mod optimization)

**mixin/ (6 files):**
- `__init__.py` — module docstring (4 primitive categories), OpMixin docstring
- `elementwise.py` — module docstring (operations list, fusion explanation)
- `movement.py` — module docstring (6 core movement ops, zero-copy explanation)
- `reduce.py` — module docstring (accumulation dtype handling)
- `creation.py` — module docstring
- `dtype.py` — module docstring

**schedule/ (6 files):**
- `__init__.py` — module docstring (3-stage pipeline), key function docstrings
- `rangeify.py` — module docstring (6-step pipeline), key function docstrings
- `indexing.py` — module docstring, BufferizeOpts/IndexingContext docs
- `memory.py` — module docstring (TLSF), memory_plan_rewrite docstring
- `multi.py` — module docstring, alu_multi/reduce_multi docstrings
- `allreduce.py` — module docstring (naive/ring/all2all strategies)

**codegen/ (11 files):**
- `__init__.py` — module docstring (6-phase pipeline), full_rewrite_to_sink docstring
- `gpudims.py` — module docstring, function docstrings
- `simplify.py` — module docstring (4 pattern matchers)
- `opt/__init__.py` — module docstring, Opt class docstring
- `opt/heuristic.py` — module docstring (9-step priority)
- `opt/search.py` — module docstring (BEAM loop), beam_search docstring
- `opt/tc.py` — module docstring (TensorCore fields + architectures)
- `opt/postrange.py` — module docstring, Scheduler/apply_opts docstrings
- `late/linearizer.py` — module docstring (priority toposort)
- `late/devectorizer.py` — module docstring (7 pattern matchers)
- `late/expander.py` — module docstring (3 pattern matchers)

**engine/ (2 files):**
- `jit.py` — comprehensive: module docstring, TinyJit, graph batching functions, replay logic
- `realize.py` — comprehensive: module docstring, all Runner classes, ExecItem, run_schedule, inline comments

**renderer/ (10 files):**
- `__init__.py` — module docstring, Renderer/ProgramSpec/Estimates class docstrings
- `cstyle.py` — module docstring (CStyleLanguage + ClangJITRenderer)
- `ptx.py` — module docstring (PTX ISA, register naming)
- `llvmir.py` — module docstring (LLVM IR, AMX/MFMA)
- `wgsl.py` — module docstring (WGSL constraints)
- `nir.py` — module docstring (Mesa NIR, 3 subclasses)
- `amd/__init__.py` — module docstring (ISA detection)
- `amd/dsl.py` — module docstring (Reg/Inst hierarchy)
- `amd/elf.py` — module docstring (ELF packer)
- `amd/generate.py` — module docstring (XML->Python codegen)

**nn/ (6 files):**
- `__init__.py` — module docstring, undocumented method docstrings
- `optim.py` — module docstring (optimizer architecture), method docstrings
- `state.py` — module docstring, TensorIO/get_parameters/zip_extract docstrings
- `datasets.py` — module docstring, mnist/cifar docstrings
- `onnx.py` — module docstring
- `torch.py` — module docstring

**runtime/ops_*.py (16 files):** All 16 backend files have module + class docstrings

**runtime/support/ (15 files):** All 15 support files have module docstrings; HCQSignal class docstring added

### What was NOT commented (intentional):
- `runtime/autogen/` (77 files) — auto-generated FFI bindings, not human-maintained
- `viz/` — web UI files (HTML/JS/Python server)
- `apps/llm.py` — application code, not core library

### Audit notes:
- All 88 modified files verified to still import cleanly
- No code behavior changes (confirmed by agents running syntax checks)
- All existing comments preserved
- 2-space indent and 150-char line style maintained throughout
