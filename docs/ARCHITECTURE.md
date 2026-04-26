# tinygrad Architecture

> High-level system architecture of the tinygrad deep learning framework.
>
> Last updated: 2025-04-15

---

## Overview

tinygrad is structured as a **4-stage compilation pipeline** that transforms lazy tensor operations into optimized hardware-specific code. Every computation flows through the same path regardless of target device.

```
User Code (Tensor API)
        |
        v
  [1] Lazy Graph Construction   (tensor.py, mixin/, uop/ops.py)
        |
        v
  [2] Scheduling & Fusion       (schedule/, engine/jit.py)
        |
        v
  [3] Codegen & Optimization    (codegen/, codegen/opt/, codegen/late/, renderer/)
        |
        v
  [4] Runtime Execution         (engine/realize.py, runtime/ops_*.py)
```

---

## Stage 1: Lazy Graph Construction

**Files**: `tensor.py`, `mixin/`, `uop/ops.py`, `dtype.py`, `gradient.py`

When you write `a + b` or `x.reshape(3, 4)`, nothing executes. Instead, tinygrad builds a **directed acyclic graph (DAG)** of `UOp` nodes representing the computation.

### Tensor

The `Tensor` class (`tensor.py`) is the user-facing API. Each Tensor wraps a single `UOp` node (`self.uop`) that represents its value in the computation graph. The Tensor class itself is thin — most operations are defined in **mixin classes**:

| Mixin | File | Operations |
|-------|------|------------|
| `ElementwiseMixin` | `mixin/elementwise.py` | `+`, `-`, `*`, `/`, `log`, `exp`, `relu`, comparisons |
| `ReduceMixin` | `mixin/reduce.py` | `sum`, `prod`, `max`, `min`, `mean`, `var`, `std` |
| `MovementMixin` | `mixin/movement.py` | `reshape`, `permute`, `expand`, `pad`, `shrink`, `flip` |
| `CreationMixin` | `mixin/creation.py` | `const_like`, `cast`, `full_like`, `zeros_like` |
| `DTypeMixin` | `mixin/dtype.py` | `dtype`, `element_size`, `float`, `half` |

### UOp (Micro-Operation)

`UOp` (`uop/ops.py`) is the **single intermediate representation (IR)** used throughout tinygrad. Every computation — from high-level tensor operations down to hardware instructions — is expressed as a UOp graph.

```python
@dataclass
class UOp:
  op: Ops           # Operation type (70+ variants in Ops enum)
  dtype: DType       # Data type of the result
  src: tuple[UOp, ...]  # Source operands (edges in the DAG)
  arg: Any           # Operation-specific argument (const value, axis, etc.)
  tag: Any           # Metadata tag
```

The `Ops` enum (in `uop/__init__.py`) defines all operations, ordered by toposort priority:

| Category | Operations |
|----------|-----------|
| Defines | `DEFINE_VAR`, `BIND`, `SPECIAL`, `DEFINE_LOCAL`, `DEFINE_REG` |
| Structure | `NOOP`, `PARAM`, `CALL`, `SINK`, `AFTER`, `GROUP`, `GEP`, `VECTORIZE`, `TUPLE`, `GETTUPLE` |
| Load/Store | `INDEX`, `LOAD`, `STORE` |
| Tensor Core | `WMMA`, `SHAPED_WMMA` |
| Unary | `CAST`, `BITCAST`, `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIPROCAL`, `NEG`, `TRUNC` |
| Binary | `ADD`, `MUL`, `SHL`, `SHR`, `IDIV`, `MAX`, `MOD`, `CMPLT`, `CMPNE`, `CMPEQ`, `XOR`, `OR`, `AND`, `SUB`, `FDIV`, `POW` |
| Ternary | `WHERE`, `MULACC` |
| Control Flow | `BARRIER`, `RANGE`, `IF`, `END`, `ENDIF` |
| Constants | `VCONST`, `CONST` |
| Tensor Graph | `CONTIGUOUS`, `DETACH`, `BUFFER`, `COPY`, `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| Reduce | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |

UOps are **deduplicated** (cached via `UOpMetaClass`) — identical computations produce the same object.

### Pattern Matching

Graph transformations are driven by `PatternMatcher` with `UPat` (UOp pattern) rules (`uop/upat.py`). This is the primary mechanism for optimization and lowering throughout the compiler:

```python
# Example: fold x * 1 -> x
pm = PatternMatcher([
  (UPat(Ops.MUL, src=[UPat.var("x"), UPat.cvar("c")]), lambda x, c: x if c.arg == 1 else None),
])
```

### Automatic Differentiation

Gradient computation (`gradient.py`) operates on the UOp graph. `compute_gradient()` traverses the forward graph and constructs backward pass UOps using chain rule composition. The `@function` decorator (`function.py`) captures computation graphs for differentiation and precompilation.

---

## Stage 2: Scheduling & Fusion

**Files**: `schedule/`, `engine/jit.py`

When `.realize()` or `.numpy()` is called, the lazy UOp graph must be converted into executable units.

### Schedule Creation

The scheduler (`schedule/__init__.py`) takes the UOp graph and:

1. **Identifies kernel boundaries** — determines which operations can be fused into a single kernel
2. **Creates dependency graphs** — tracks data dependencies between kernels using `AFTER` nodes
3. **Linearizes execution order** — topological sort of kernels respecting dependencies

### Rangeification

`schedule/rangeify.py` converts tensor-level operations into loop-level operations:

- Tensor element access becomes loop `RANGE` nodes
- Reductions become `REDUCE` loops
- Output writes become `STORE` operations
- Movement ops (`RESHAPE`, `PERMUTE`, `PAD`, `EXPAND`) are lowered into index expressions

### Indexing

`schedule/indexing.py` computes buffer addresses:

- Maps multi-dimensional tensor indices to flat buffer offsets
- Handles strides from `PERMUTE`, offsets from `SHRINK`, padding from `PAD`
- Generates the actual `INDEX` UOps used in `LOAD`/`STORE`

### Memory Planning

`schedule/memory.py` plans buffer allocation:

- Tracks buffer lifetimes
- Suballocates internal temporary buffers
- Uses a TLSF (Two-Level Segregated Fit) allocator for efficient memory reuse

### Multi-Device

`schedule/multi.py` and `schedule/allreduce.py` handle multi-GPU execution:

- Tensor sharding across devices
- All-reduce communication patterns
- `MultiBuffer` wraps per-device buffers

### JIT (Just-In-Time Compilation)

`engine/jit.py` implements `TinyJit`:

1. On the first call, records all kernels executed
2. On subsequent calls, **replays** the recorded kernels without re-scheduling
3. Groups kernels into hardware **graphs** for batched execution (CUDA graphs, Metal command buffers, HCQ graphs)
4. Extracts variable bindings so shapes can change between runs

---

## Stage 3: Code Generation & Optimization

**Files**: `codegen/`, `codegen/opt/`, `codegen/late/`, `renderer/`

Each scheduled kernel goes through a multi-pass lowering and optimization pipeline.

### Pipeline

The main entry point is `full_rewrite_to_sink()` in `codegen/__init__.py`:

```
Kernel UOp Graph
      |
      v
[Early Passes]  pm_mops, pm_syntactic_sugar, pm_store_ranges
      |          Lower movement ops, normalize stores
      v
[Load Collapse]  pm_load_collapse
      |           Reduce indexing overhead
      v
[Range Splitting] pm_split_ranges, pm_flatten_range
      |            Split complex loops
      v
[Symbolic]       sym, symbolic_simple
      |           Algebraic simplification (x*0=0, x+0=x, etc.)
      v
[Range Simplify]  pm_simplify_ranges
      |            Merge and simplify loop ranges
      v
[Optimization]   apply_opts (hand-coded heuristics or BEAM search)
      |           UPCAST, UNROLL, LOCAL, GROUP_REDUCE decisions
      v
[Expander]       expander, pm_pre_expander, pm_group_for_reduce
      |           Expand complex ops into primitives
      v
[Local Buffers]  pm_add_buffers_local, rangeify_codegen
      |           Add shared memory, registers
      v
[Devectorize]    devectorize, load_store_folding
      |           Vectorization, load/store optimization
      v
[Linearize]      linearize, pm_add_control_flow
      |           Topological sort with priority ordering
      v
[GPU Dims]       pm_add_gpudims
      |           Map loops to GPU threads/blocks
      v
[Render]         Target-specific code generation
```

### Optimization (codegen/opt/)

Two strategies for kernel optimization:

1. **Hand-coded heuristics** (`heuristic.py`): Fast default optimizer that applies reasonable UPCAST/UNROLL/LOCAL decisions based on tensor dimensions and device capabilities

2. **BEAM search** (`search.py`): Exhaustive search over kernel configurations. Set `BEAM=N` to try N candidates. Measures actual execution time to pick the fastest variant

3. **Tensor core utilization** (`tc.py`): Detects matmul patterns and maps them to hardware tensor cores (NVIDIA WMMA, AMD Matrix Fused Multiply-Add)

### Rendering

Renderers convert the optimized UOp graph into target-specific code:

| Renderer | File | Output |
|----------|------|--------|
| `ClangJITRenderer` | `renderer/cstyle.py` | C code (compiled via clang) |
| `CPULLVMRenderer` | `renderer/llvmir.py` | LLVM IR (compiled via LLVM) |
| `PTXRenderer` | `renderer/ptx.py` | NVIDIA PTX assembly |
| `LVPRenderer` | `renderer/nir.py` | NVIDIA IR (via LVP) |
| `WGSLRenderer` | `renderer/wgsl.py` | WebGPU WGSL shaders |
| AMD renderers | `renderer/amd/` | AMD RDNA/CDNA ISA (native assembly via ELF) |

Each renderer implements a mapping from UOp operations to target-specific instructions/syntax.

---

## Stage 4: Runtime Execution

**Files**: `engine/realize.py`, `runtime/ops_*.py`, `runtime/support/`, `runtime/graph/`

### Execution Flow

`engine/realize.py` orchestrates execution:

1. `CompiledRunner` wraps a compiled kernel (`ProgramSpec`) with its runtime program
2. `Runner.exec()` allocates output buffers and launches the kernel
3. `BufferCopy` / `BufferXfer` handle data movement between devices
4. `update_stats()` records timing and FLOP counts for `DEBUG >= 2` output

### Device Abstraction

`device.py` defines the device system:

- `_Device` singleton discovers available backends by scanning `runtime/ops_*.py` files
- `Device[name]` returns a `Compiled` instance (cached per device string)
- `Compiled` provides: `Allocator` (memory), `Compiler` (code compilation), `Renderer` (code generation), and `graph` (batched execution)

### Hardware Backends

Each `ops_*.py` file implements a device:

| Backend | File | Hardware |
|---------|------|----------|
| `ops_cpu.py` | CPU | Multi-threaded CPU via clang/LLVM |
| `ops_cuda.py` | CUDA | NVIDIA GPUs via NVRTC |
| `ops_nv.py` | NV | NVIDIA GPUs via direct driver access |
| `ops_amd.py` | AMD | AMD GPUs via direct driver access (ROCm) |
| `ops_hip.py` | HIP | AMD GPUs via HIP runtime |
| `ops_metal.py` | METAL | Apple GPUs via Metal |
| `ops_cl.py` | CL | Any GPU via OpenCL |
| `ops_webgpu.py` | WEBGPU | Browsers via WebGPU/WGSL |
| `ops_qcom.py` | QCOM | Qualcomm GPUs (Adreno) |
| `ops_dsp.py` | DSP | Qualcomm DSP |
| `ops_python.py` | PYTHON | Pure Python reference |
| `ops_npy.py` | NPY | NumPy-backed (for data loading) |
| `ops_disk.py` | DISK | Memory-mapped file storage |

### HCQ (Hardware Command Queue)

`runtime/support/hcq.py` provides a unified abstraction for direct hardware access (used by NV, AMD, QCOM backends):

- **Command queues** for kernel dispatch and memory operations
- **Signals** for synchronization between queues
- **HCQGraph** for batched multi-kernel execution
- Bypasses runtime libraries (CUDA/HIP) for lower latency

### Graph Execution

`runtime/graph/` implements hardware-specific graph execution:

- `cuda.py` — CUDA graphs
- `metal.py` — Metal command buffers
- `hcq.py` — HCQ-based graph execution

---

## Buffer Lifecycle

```
Tensor.rand(3, 3)          # Creates UOp graph (no memory allocated)
      |
      v
.realize()                  # Triggers scheduling + codegen + execution
      |
      v
Buffer.allocate()           # Device memory allocated
      |
      v
CompiledRunner.__call__()   # Kernel executes, writes to buffer
      |
      v
.numpy()                    # Copies buffer to CPU, returns numpy array
      |
      v
(garbage collected)         # Buffer freed when Tensor goes out of scope
```

---

## Key Design Patterns

### Single IR (UOp)

Unlike frameworks with separate IRs for different levels (e.g., XLA HLO -> LLVM IR), tinygrad uses a **single `UOp` graph** from tensor operations down to hardware instructions. Graph transformations via `PatternMatcher` progressively lower and optimize the representation.

### Lazy Evaluation

All tensor operations are lazy — they build UOp graphs without executing. This enables:
- **Kernel fusion**: sequential operations are merged into single kernels
- **Dead code elimination**: unrealized computations are never executed
- **Whole-program optimization**: the compiler sees the full computation graph

### Pattern-Matching Rewrites

Instead of procedural compiler passes, tinygrad uses declarative `PatternMatcher` rules:
- Each pass is a set of `(UPat, replacement)` pairs
- `graph_rewrite()` applies patterns bottom-up or top-down until convergence
- This makes passes composable, testable, and easy to extend

### Mixin Composition

The Tensor class uses mixins instead of a deep inheritance hierarchy. Operations are grouped by category (elementwise, reduction, movement) and mixed in. This keeps the Tensor class focused on graph construction while operations are defined declaratively.

---

## Data Flow Example

Here's what happens when you run:

```python
from tinygrad import Tensor
a = Tensor([1, 2, 3])
b = (a + 1).sum()
result = b.numpy()
```

1. `Tensor([1, 2, 3])` — creates a `BUFFER` UOp backed by Python data, wraps in a Tensor
2. `a + 1` — creates `ADD(LOAD(buffer_a), CONST(1))` UOp, wraps in new Tensor (lazy)
3. `.sum()` — creates `REDUCE_AXIS(add_uop, axis=0)` UOp, wraps in new Tensor (lazy)
4. `.numpy()` — triggers:
   - **Schedule**: fuses ADD + REDUCE into one kernel, creates `STORE` to output buffer
   - **Codegen**: lowers to a loop that loads from input, adds 1, accumulates sum, stores result
   - **Optimize**: decides loop dimensions, vectorization
   - **Render**: generates C code (CPU) or PTX (CUDA) or Metal shader, etc.
   - **Compile**: JIT compiles the generated code
   - **Execute**: runs the compiled kernel
   - **Copy**: transfers result from device buffer to NumPy array
5. Returns `numpy.array([9])`
