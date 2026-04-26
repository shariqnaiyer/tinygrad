# tinygrad Commenting Execution Plan

> Ordered plan for adding comments to every source file.
>
> Last updated: 2025-04-15

---

## Priority 1: Root-Level Core (7 files) — Entry points and foundational abstractions

| File | Current State | Lines | What to Add |
|------|--------------|-------|-------------|
| `tinygrad/__init__.py` | none | 12 | Module docstring explaining public API surface |
| `tinygrad/tensor.py` | minimal (91/173 fns) | ~2200 | Module docstring, class docstring, undocumented methods, key algorithm comments |
| `tinygrad/device.py` | minimal (3/66 fns) | ~450 | Module docstring, _Device class, Buffer class, Compiled base, all methods |
| `tinygrad/dtype.py` | minimal (1/64 fns) | ~350 | Module docstring, DType/PtrDType/ImageDType, type system explanation |
| `tinygrad/helpers.py` | minimal (0/125 fns) | ~700 | Module docstring, ContextVar, Context, GlobalCounters, all utility functions |
| `tinygrad/gradient.py` | none (1/6 fns) | ~150 | Module docstring, compute_gradient, backward pass logic |
| `tinygrad/function.py` | none (0/7 fns) | ~110 | Module docstring, @function decorator, _function class |
| `tinygrad/callify.py` | none (2/13 fns) | ~150 | Module docstring, transform_to_call, CALL node construction |

## Priority 2: UOp Package (8 files) — The IR everything flows through

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/uop/__init__.py` | none | Module docstring, Ops enum value-by-value comments, GroupOp |
| `tinygrad/uop/ops.py` | minimal (5/220 fns) | Module docstring, UOp class, UOpMetaClass, PatternMatcher, key algorithms |
| `tinygrad/uop/symbolic.py` | minimal (1/18 fns) | Module docstring, simplification rules, symbolic arithmetic |
| `tinygrad/uop/upat.py` | none | Module docstring, UPat class, pattern compilation |
| `tinygrad/uop/spec.py` | none | Module docstring, validation rules |
| `tinygrad/uop/decompositions.py` | none | Module docstring, transcendental decomposition functions |
| `tinygrad/uop/validate.py` | none | Module docstring, Z3-based validation |
| `tinygrad/uop/divandmod.py` | none | Module docstring, symbolic div/mod |

## Priority 3: Mixin Package (6 files) — Tensor operation definitions

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/mixin/__init__.py` | none | Module docstring, OpMixin class |
| `tinygrad/mixin/elementwise.py` | none (78/118 fns have docs) | Module docstring, undocumented methods |
| `tinygrad/mixin/movement.py` | none (26/38 fns) | Module docstring, undocumented reshape/permute/pad logic |
| `tinygrad/mixin/reduce.py` | none (5/7 fns) | Module docstring, reduction strategy |
| `tinygrad/mixin/creation.py` | none (4/5 fns) | Module docstring |
| `tinygrad/mixin/dtype.py` | none (8/12 fns) | Module docstring |

## Priority 4: Schedule Package (6 files) — Scheduling and fusion

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/schedule/__init__.py` | none | Module docstring, create_schedule, kernel dependency graph |
| `tinygrad/schedule/rangeify.py` | none | Module docstring, tensor-to-loop conversion |
| `tinygrad/schedule/indexing.py` | none | Module docstring, buffer address computation |
| `tinygrad/schedule/memory.py` | none | Module docstring, TLSF allocator |
| `tinygrad/schedule/multi.py` | none | Module docstring, multi-device scheduling |
| `tinygrad/schedule/allreduce.py` | none | Module docstring, all-reduce patterns |

## Priority 5: Codegen Package (11 files) — Code generation pipeline

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/codegen/__init__.py` | none | Module docstring, full_rewrite_to_sink pipeline |
| `tinygrad/codegen/gpudims.py` | none | Module docstring, GPU dimension mapping |
| `tinygrad/codegen/simplify.py` | none | Module docstring, range simplification |
| `tinygrad/codegen/opt/__init__.py` | none | Module docstring, Opt dataclass |
| `tinygrad/codegen/opt/heuristic.py` | none | Module docstring, hand-coded heuristics |
| `tinygrad/codegen/opt/postrange.py` | none | Module docstring, apply_opts |
| `tinygrad/codegen/opt/search.py` | none | Module docstring, BEAM search |
| `tinygrad/codegen/opt/tc.py` | none | Module docstring, tensor core detection |
| `tinygrad/codegen/late/linearizer.py` | none | Module docstring, linearization |
| `tinygrad/codegen/late/devectorizer.py` | none | Module docstring, vectorization |
| `tinygrad/codegen/late/expander.py` | none | Module docstring, op expansion |

## Priority 6: Engine Package (3 files) — JIT and execution

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/engine/__init__.py` | none | Module docstring |
| `tinygrad/engine/jit.py` | none | Module docstring, TinyJit class, graph batching |
| `tinygrad/engine/realize.py` | none | Module docstring, Runner classes, CompiledRunner |

## Priority 7: Renderer Package (11 files) — Code generators

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/renderer/__init__.py` | none | Module docstring, Renderer base, ProgramSpec, Estimates |
| `tinygrad/renderer/cstyle.py` | minimal | Module docstring, C code generation |
| `tinygrad/renderer/ptx.py` | none | Module docstring, PTX generation |
| `tinygrad/renderer/llvmir.py` | minimal | Module docstring, LLVM IR generation |
| `tinygrad/renderer/wgsl.py` | none | Module docstring, WGSL generation |
| `tinygrad/renderer/nir.py` | none | Module docstring, NIR generation |
| `tinygrad/renderer/amd/__init__.py` | none | Module docstring |
| `tinygrad/renderer/amd/dsl.py` | minimal | Module docstring |
| `tinygrad/renderer/amd/elf.py` | none | Module docstring |
| `tinygrad/renderer/amd/generate.py` | minimal | Module docstring |
| `tinygrad/renderer/amd/sqtt.py` | partial | Enhance existing docs |

## Priority 8: NN Package (6 files) — Neural network modules

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/nn/__init__.py` | none (11 classes have docs) | Module docstring, undocumented methods |
| `tinygrad/nn/optim.py` | none (4/5 classes) | Module docstring, optimizer internals |
| `tinygrad/nn/state.py` | minimal | Module docstring, safe_save/safe_load |
| `tinygrad/nn/datasets.py` | none | Module docstring, fetch_mnist, fetch_cifar |
| `tinygrad/nn/onnx.py` | minimal | Module docstring, ONNX import logic |
| `tinygrad/nn/torch.py` | none | Module docstring |

## Priority 9: Runtime Backends (16 files) — Hardware implementations

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/runtime/ops_cpu.py` | none | Module docstring, CPUDevice, multi-threaded execution |
| `tinygrad/runtime/ops_cuda.py` | none | Module docstring, CUDADevice, NVRTC compilation |
| `tinygrad/runtime/ops_nv.py` | minimal | Module docstring, direct NVIDIA driver access |
| `tinygrad/runtime/ops_metal.py` | none | Module docstring, MetalDevice |
| `tinygrad/runtime/ops_amd.py` | minimal | Module docstring, direct AMD driver access |
| `tinygrad/runtime/ops_hip.py` | none | Module docstring, HIP runtime |
| `tinygrad/runtime/ops_cl.py` | none | Module docstring, OpenCL |
| `tinygrad/runtime/ops_webgpu.py` | none | Module docstring, WebGPU/WGSL |
| `tinygrad/runtime/ops_qcom.py` | minimal | Module docstring, Qualcomm Adreno |
| `tinygrad/runtime/ops_dsp.py` | minimal | Module docstring, Qualcomm DSP |
| `tinygrad/runtime/ops_python.py` | none | Module docstring, pure Python reference |
| `tinygrad/runtime/ops_npy.py` | none | Module docstring |
| `tinygrad/runtime/ops_disk.py` | none | Module docstring, mmap storage |
| `tinygrad/runtime/ops_tinyfs.py` | none | Module docstring |
| `tinygrad/runtime/ops_null.py` | none | Module docstring |
| `tinygrad/runtime/ops_rdma.py` | none | Module docstring |

## Priority 10: Runtime Support (15+ files)

| File | Current State | What to Add |
|------|--------------|-------------|
| `tinygrad/runtime/support/hcq.py` | minimal | Module docstring, HCQ abstraction |
| `tinygrad/runtime/support/memory.py` | minimal | Module docstring, TLSF allocator |
| `tinygrad/runtime/support/compiler_*.py` | none | Module docstrings for each compiler |
| `tinygrad/runtime/support/c.py` | none | Module docstring, libc FFI |
| `tinygrad/runtime/support/system.py` | none | Module docstring, system calls |
| `tinygrad/runtime/support/elf.py` | none | Module docstring, ELF loading |
| `tinygrad/runtime/support/objc.py` | none | Module docstring, ObjC bridge |
| `tinygrad/runtime/support/usb.py` | minimal | Module docstring, USB comm |

## SKIP: Auto-generated files

All files in `tinygrad/runtime/autogen/` (77 files) — these are machine-generated FFI bindings and should not be manually commented.
