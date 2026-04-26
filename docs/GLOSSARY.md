# tinygrad Glossary

> Domain-specific terms, acronyms, and conventions used in the tinygrad codebase.
>
> Last updated: 2025-04-15

---

## Core Concepts

### BEAM / BEAM Search
An optimization strategy (`BEAM=N` env var) that tries N different kernel configurations and picks the fastest. Each configuration varies UPCAST, UNROLL, LOCAL, and GROUP_REDUCE dimensions. The search measures actual execution time rather than estimating. See `codegen/opt/search.py`.

### Buffer
A chunk of device memory with a dtype and size. Buffers are allocated lazily — only when a Tensor is realized. The `Buffer` class in `device.py` wraps the actual device-specific memory allocation. See also: MultiBuffer.

### Codegen
The code generation pipeline that transforms optimized UOp graphs into target-specific source code (C, PTX, WGSL, LLVM IR, AMD assembly). The main entry point is `full_rewrite_to_sink()` in `codegen/__init__.py`.

### Compiled
The base class for device implementations in `device.py`. A `Compiled` instance provides a `Renderer` (code generation), `Compiler` (code compilation), `Allocator` (memory management), and optionally a `graph` (batched execution).

### ContextVar
A global variable managed by `helpers.py` that can be temporarily overridden using `Context(VAR=value)` or the `@Context(VAR=value)` decorator. Examples: `DEBUG`, `BEAM`, `IMAGE`, `JIT`.

### DEBUG
Environment variable controlling debug output verbosity (1-7). Higher values show more detail: 1=device info, 2=kernel timing, 3=optimizations, 4=generated code, 5=UOps IR, 6=linearized UOps, 7=assembly.

### DEV
Environment variable specifying the target device. Supports a triple syntax: `DEV=device:renderer:arch` (e.g., `DEV=AMD:LLVM:gfx950`). Can also specify interface: `DEV=USB+AMD`.

### DType
Data type specification. Includes base types (`float32`, `int32`, `bool`), special types (`ImageDType` for 2D image formats, `PtrDType` for pointer types with address space), and vector types (`dtypes.float32.vec(4)`). Defined in `dtype.py`.

### Kernel Fusion
The process of combining multiple tensor operations into a single GPU kernel. Enabled by lazy evaluation — when you write `(a + b).relu()`, both the addition and ReLU execute in one kernel rather than two separate ones. The scheduler determines fusion boundaries.

### Lazy Evaluation
Operations on Tensors don't execute immediately. Instead, they build a UOp computation graph. Execution happens only when `.realize()`, `.numpy()`, `.item()`, or `.tolist()` is called. This enables kernel fusion and whole-graph optimization.

### MultiBuffer
A buffer sharded across multiple devices for multi-GPU execution. Wraps per-device `Buffer` instances. Used with `Tensor.shard()` for data or model parallelism.

### Ops
The enum (`uop/__init__.py` and `uop/ops.py`) defining all 70+ operation types used in UOp graphs. Grouped as: buffer ops, math ops, unary ops, reduce ops, movement ops, control flow, structure ops, and definition ops.

### PatternMatcher
The primary mechanism for UOp graph transformations. A `PatternMatcher` holds a list of `(UPat, replacement_function)` rules. `graph_rewrite()` applies these rules bottom-up or top-down until convergence. Used throughout the compiler for optimization, lowering, and rendering.

### Realize / .realize()
The act of executing all pending lazy operations to produce concrete values in device memory. Calling `.realize()` triggers scheduling, code generation, compilation, and kernel execution.

### Renderer
A class that converts optimized UOp graphs to target-specific code. Each hardware backend has its own renderer: `ClangJITRenderer` (C), `PTXRenderer` (NVIDIA PTX), `WGSLRenderer` (WebGPU), etc.

### Runner
An executable unit in `engine/realize.py`. `CompiledRunner` wraps a compiled program; `BufferCopy` and `BufferXfer` handle data movement. Runners are the final step before hardware execution.

### Scheduler / Schedule
The component that converts a lazy UOp graph into a sequence of executable kernels. It determines kernel boundaries (which ops fuse together), resolves data dependencies, and produces a linearized execution order.

### Tensor
The user-facing class (`tensor.py`) for all computation. Wraps a `UOp` node representing its value. Supports PyTorch-like API: `Tensor.rand(3, 3)`, `a + b`, `x.relu()`, `y.sum()`. All operations are lazy.

### TinyJit / @TinyJit
JIT compiler decorator (`engine/jit.py`). On the first call, records all kernels. On subsequent calls, replays them without re-scheduling. Groups kernels into hardware graphs (CUDA graphs, Metal command buffers) for minimal launch overhead.

### UOp (Micro-Operation)
The single intermediate representation used throughout tinygrad. A `UOp` is a node with `op` (Ops enum), `dtype`, `src` (tuple of input UOps), `arg`, and `tag`. All computation — from tensor ops to hardware instructions — is represented as UOp DAGs.

### UPat (UOp Pattern)
A pattern for matching UOp nodes, used with `PatternMatcher`. Supports matching by op type, dtype, source patterns, and argument predicates. Defined in `uop/upat.py`.

---

## Hardware & Runtime

### HCQ (Hardware Command Queue)
A unified abstraction (`runtime/support/hcq.py`) for direct hardware access. Provides command queues, signals for synchronization, and graph execution. Used by NV, AMD, and QCOM backends to bypass runtime libraries for lower latency.

### AM Driver
tinygrad's userspace AMD GPU driver (`runtime/support/am/`). Manages GPU boot, virtual memory, compute and SDMA queues directly, without relying on the ROCm userspace stack.

### HIP
AMD's GPU programming interface (Heterogeneous-computing Interface for Portability). The `ops_hip.py` backend uses HIP via the ROCm runtime.

### NVRTC
NVIDIA Runtime Compilation library. Used by `ops_cuda.py` to compile CUDA/PTX code at runtime.

### PTX
Parallel Thread Execution — NVIDIA's virtual instruction set architecture. tinygrad generates PTX assembly directly via `renderer/ptx.py`.

### WGSL
WebGPU Shading Language. tinygrad generates WGSL shaders via `renderer/wgsl.py` for browser-based execution.

### WMMA
Warp Matrix Multiply Accumulate — hardware tensor core operations. tinygrad detects matmul patterns and maps them to WMMA instructions for accelerated matrix multiplication.

### Graph Execution
Batched kernel launch (CUDA graphs, Metal command buffers, HCQ graphs). Instead of launching kernels one at a time, the JIT groups them into a single graph that is launched as a unit, reducing CPU-side overhead.

---

## Optimization

### UPCAST
An optimization that processes multiple elements per thread by unrolling inner dimensions. For example, UPCAST a dimension of size 4 means each thread handles 4 elements.

### UNROLL
Loop unrolling — expanding a loop body multiple times to reduce loop overhead and enable instruction-level parallelism.

### LOCAL
An optimization that tiles work into local/shared memory. Maps operations to GPU local (shared) memory for faster access within a workgroup.

### GROUP_REDUCE
An optimization for reduction operations that splits the reduction across local workgroup threads, using shared memory for intermediate results.

### Linearize / Linearization
The process of converting a UOp DAG into a linear sequence of instructions. Assigns priorities to operations and produces a topologically sorted order suitable for code emission.

### Tensor Core
Hardware matrix multiply units (NVIDIA Tensor Cores, AMD Matrix cores). `codegen/opt/tc.py` detects patterns suitable for tensor core acceleration.

---

## Codegen Pipeline Terms

### Movement Ops (mops)
Operations that change tensor shape/layout without changing data: `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`, `STRIDE`. These are lowered into index expressions during codegen.

### Rangeify
The process of converting tensor-level operations into loop-level operations with explicit `RANGE` (for) nodes, `REDUCE` (accumulation) nodes, and `STORE` (output write) nodes.

### Load Collapse
An optimization that reduces indexing overhead by collapsing indirect loads (tensor indexing) into simpler address computations.

### Range Splitting
Breaking complex multi-dimensional loops into simpler forms that can be individually optimized.

### Expander
A codegen pass that expands high-level operations (like WMMA) into their primitive instruction sequences.

### Devectorizer
A codegen pass that handles vectorization (packing multiple operations into SIMD instructions) and optimizes load/store patterns.

---

## Data & Training

### SafeTensors
The standard weight format used by tinygrad (`nn/state.py`). A simple, safe file format for storing tensors, originally from Hugging Face. Preferred over pickle for security.

### Process Replay
A CI mechanism that compares generated kernels between a PR and the master branch. If a PR claims to be a refactor (marked with `[pr]` in the title), generated kernels must match exactly. Defined in `test/external/process_replay/`.

### ONNX
Open Neural Network Exchange format. tinygrad can import ONNX models via `nn/onnx.py` for inference.

---

## Project Conventions

### sz.py
The line count enforcer. Counts tokens and lines in Python/JS files, excluding autogen and viz/assets. CI enforces a maximum of ~24,000 lines to keep the project small.

### SPEC
Environment variable for enabling UOp specification validation (`SPEC=1` or `SPEC=2`). When enabled, every UOp created is validated against specification rules in `uop/spec.py`.

### VIZ
Environment variable that enables computation graph visualization (`VIZ=1`). Opens a web-based viewer showing the UOp graph at each compilation stage.

### @function
A decorator (`function.py`) that captures a computation graph inside a function body. Extracts `PARAM` nodes for learnable parameters. Used for precompilation and functional-style model definitions.

### GroupOp
Named groups of related `Ops` values used in pattern matching (e.g., `GroupOp.ALU` matches all arithmetic/logic operations). Defined in `uop/__init__.py`.
