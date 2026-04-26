# tinygrad Architectural Decisions

> Key design decisions inferred from the codebase and their rationale.
>
> Last updated: 2025-04-15

---

## 1. Single IR (UOp) for Everything

**Decision**: Use one intermediate representation (`UOp`) from tensor operations down to hardware instructions, rather than multiple IRs at different abstraction levels.

**Rationale**: Most ML compilers (XLA, TVM, MLIR) use a hierarchy of IRs — high-level graph IR, loop IR, hardware IR. tinygrad uses a single `UOp` DAG that is progressively lowered through pattern-matching rewrites. This means:
- Less code: no translation layers between IR levels
- Easier debugging: the same data structure is used everywhere, so tools (VIZ, printing) work at every stage
- Composable passes: any `PatternMatcher` can operate on any UOp graph regardless of lowering stage

**Trade-off**: UOp nodes are more complex individually (the `Ops` enum has 70+ entries) because they must represent both high-level operations (REDUCE_AXIS) and low-level operations (DEFINE_LOCAL, SPECIAL).

---

## 2. Lazy Evaluation

**Decision**: All tensor operations build a computation graph; nothing executes until `.realize()` or `.numpy()` is called.

**Rationale**: Laziness enables **kernel fusion** — the compiler sees the full computation graph before generating code, so it can fuse `(a + b).relu()` into a single kernel instead of two separate ones. This is critical for GPU performance where kernel launch overhead is significant.

**Why not eager like PyTorch**: PyTorch spends enormous engineering effort making eager dispatch fast. tinygrad avoids this complexity by being lazy. The `TinyJit` decorator provides an "eager-feeling" API by replaying recorded kernels on subsequent calls.

---

## 3. Pattern-Matching Rewrites Over Procedural Passes

**Decision**: Compiler transformations are expressed as declarative `PatternMatcher` rules rather than procedural tree-walking passes.

**Rationale**:
- **Declarative**: Each optimization rule is a self-contained `(pattern, replacement)` pair, making them easy to understand, test, and compose
- **Convergent**: `graph_rewrite()` applies rules until no more match (fixed point), so passes don't need to worry about ordering
- **Extensible**: Adding a new optimization is adding a rule, not modifying a visitor

**Trade-off**: Pattern matching can be slower than hand-written visitors for very large graphs, and understanding the interaction between many rules requires global reasoning.

---

## 4. No nn.Module

**Decision**: Neural network models are plain Python classes with `__call__`. There is no base `Module` class to inherit from.

**Rationale**: PyTorch's `nn.Module` provides parameter tracking, state dict management, and hook infrastructure. tinygrad achieves the same with simpler mechanisms:
- `nn.state.get_parameters(obj)` recursively finds all `Tensor` attributes — no registration needed
- `nn.state.get_state_dict(obj)` builds a state dict by walking attribute names
- Models are just classes with Tensor attributes; `__call__` defines the forward pass

This keeps models lightweight and avoids framework-specific base class ceremony.

---

## 5. Zero Runtime Dependencies

**Decision**: The core `tinygrad` package has no pip dependencies. It runs on pure Python 3.11+.

**Rationale**: This makes tinygrad trivially installable and embeddable. No dependency conflicts, no version pinning issues, no supply chain concerns. The library generates and compiles code using system tools (clang, CUDA toolkit, Metal framework) that are already present on target machines.

**Trade-off**: Some functionality that libraries like numpy provide must be reimplemented (e.g., struct packing, type handling). numpy is used only in the test suite and optional features.

---

## 6. Line Count Enforcement

**Decision**: CI enforces a maximum of ~24,000 lines in the core `tinygrad/` folder (via `sz.py`).

**Rationale**: tinygrad's core value proposition is being small and hackable. The line count limit is a forcing function that:
- Prevents feature creep and unnecessary abstraction
- Forces contributors to find simpler solutions
- Keeps the entire codebase readable by one person
- Makes dead code removal a valued contribution

**How it works**: `sz.py` counts tokens and lines in Python/JS files, excluding `runtime/autogen/` (generated FFI bindings) and `viz/assets/` (third-party JS).

---

## 7. Direct Hardware Access (HCQ)

**Decision**: For NVIDIA and AMD, tinygrad bypasses the vendor runtime libraries (CUDA Runtime, HIP) and talks directly to hardware via ioctl/driver interfaces.

**Rationale**: Runtime libraries add overhead — function call chains, internal synchronization, driver version coupling. Direct hardware access via HCQ (Hardware Command Queue) provides:
- Lower kernel launch latency
- Fine-grained control over command queues and synchronization
- Independence from vendor runtime versions
- Ability to implement custom features (e.g., graph execution) without vendor support

**Trade-off**: More code to maintain per hardware platform. Breakage risk when GPU driver internals change.

---

## 8. Auto-Generated FFI Bindings

**Decision**: Foreign function interface bindings (for CUDA, Metal, OpenCL, HIP, etc.) are auto-generated from C headers rather than hand-written.

**Rationale**:
- **Correctness**: Generated bindings match the actual C API exactly
- **Maintainability**: When APIs change, re-run the generator instead of manually updating
- **Completeness**: Can bind the entire API surface without selective omission

The `autogen.yml` CI workflow regenerates bindings and checks for drift. The bindings live in `runtime/autogen/` and are excluded from line count enforcement.

---

## 9. SafeTensors Over Pickle

**Decision**: The standard weight format is SafeTensors, not Python pickle.

**Rationale**: Pickle is a security risk — loading a pickle file can execute arbitrary code. SafeTensors is a simple, safe format (from Hugging Face) that stores tensors as named byte arrays with metadata. tinygrad's `nn/state.py` implements `safe_save()` and `safe_load()`.

---

## 10. Functional Over Object-Oriented Operations

**Decision**: Stateless operations (relu, softmax, conv2d) are methods on Tensor, not separate classes.

**Rationale**: In tinygrad, you write `x.conv2d(w, b)` or `x.relu()` rather than creating `nn.ReLU()` instances. Classes exist only when they need to hold state (e.g., `nn.Conv2d` stores weight and bias tensors, but delegates to `Tensor.conv2d`). This is more Pythonic and reduces the number of concepts a user must learn.

---

## 11. Multi-Stage Kernel Optimization

**Decision**: Kernel optimization uses either hand-coded heuristics (default) or BEAM search (opt-in), rather than a fixed optimization strategy.

**Rationale**:
- **Heuristics** (`codegen/opt/heuristic.py`): Fast, good-enough optimization for most cases. Applies UPCAST, UNROLL, LOCAL, and GROUP_REDUCE decisions based on tensor dimensions and device capabilities.
- **BEAM search** (`codegen/opt/search.py`): For maximum performance, tries N configurations and measures actual execution time. This is slow (compilation + profiling per candidate) but finds better solutions than heuristics for unusual shapes.

The two-tier approach means development iteration is fast (heuristics), while production can use BEAM for peak performance.

---

## 12. Mixin Composition for Tensor

**Decision**: Tensor operations are defined in mixin classes (`ElementwiseMixin`, `ReduceMixin`, `MovementMixin`, etc.) that are composed into the Tensor class via multiple inheritance.

**Rationale**: This keeps the main `tensor.py` focused on core Tensor mechanics (creation, realization, autograd) while operations are organized by category in separate files. Each mixin is ~100-200 lines and self-contained.

---

## Apparent Tech Debt / Oddities

These are patterns in the code that appear intentional but may surprise newcomers:

- **Compact code style**: 2-space indent, 150-char lines, multiple statements per line. This is a deliberate stylistic choice, not laziness.
- **`extra/` is loosely maintained**: The README explicitly says code outside core `tinygrad/` is not well-tested. `extra/` is a grab-bag of tools, drivers, and experiments.
- **`runtime/autogen/` is large**: Auto-generated FFI bindings can be thousands of lines. They're excluded from line count.
- **Tests use `torch` as reference**: This creates a large test dependency, but ensures semantic compatibility with the de facto standard.
- **`weakref` used extensively**: UOp caching and Tensor tracking use weak references to avoid memory leaks in long-running programs.
