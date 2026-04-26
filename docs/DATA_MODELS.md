# tinygrad Data Models

> Core data structures and their fields, relationships, and usage.
>
> Last updated: 2025-04-15

---

## UOp

**File**: `tinygrad/uop/ops.py`

The universal intermediate representation. Every computation in tinygrad is a DAG of UOp nodes.

```python
class UOp:
  op: Ops                    # Operation type (Ops enum)
  dtype: DType               # Result data type
  src: tuple[UOp, ...]      # Source operands (parent nodes in DAG)
  arg: Any                   # Operation-specific argument
  tag: Any                   # Metadata tag
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `op` | `Ops` | The operation this node represents (e.g., `Ops.ADD`, `Ops.LOAD`, `Ops.RANGE`) |
| `dtype` | `DType` | Data type of the output (e.g., `dtypes.float32`, `dtypes.int32`) |
| `src` | `tuple[UOp, ...]` | Ordered tuple of input UOp nodes. Empty for leaf nodes (constants, buffers) |
| `arg` | `Any` | Operation-specific data: constant values, axis indices, shape tuples, kernel info |
| `tag` | `Any` | Optional metadata tag for debugging/tracking |

**Key behaviors:**
- **Deduplicated**: identical UOps (same op, dtype, src, arg, tag) return the same object via `UOpMetaClass` cache
- **Immutable**: once created, fields cannot change; use `.replace()` to create modified copies
- **Lazy graph**: UOps form a DAG; execution happens only when the graph is scheduled

**Key methods:**

| Method | Description |
|--------|-------------|
| `toposort()` | Topological sort of the UOp subgraph |
| `substitute(map)` | Replace nodes according to a mapping |
| `simplify()` | Apply algebraic simplifications |
| `render()` | Render as a human-readable string |
| `replace(src=..., arg=...)` | Create a copy with modified fields |
| `const(dtype, value)` | Create a constant UOp |
| `sink(*srcs)` | Create a SINK node collecting multiple outputs |
| `new_buffer(device, size, dtype)` | Create a new buffer UOp |

---

## Ops Enum

**File**: `tinygrad/uop/__init__.py`

Defines all 70+ operation types. Key categories:

### 1. Defines / Special
| Op | Description |
|----|-------------|
| `DEFINE_VAR` | Define a symbolic variable (pointer to outside the kernel) |
| `BIND` | Bind a variable to a value |
| `SPECIAL` | GPU dimension (threadIdx, blockIdx) |
| `DEFINE_LOCAL` | Allocate local/shared memory |
| `DEFINE_REG` | Allocate register |

### 2. Non-op UOps
| Op | Description |
|----|-------------|
| `NOOP` | No operation |
| `PARAM` | Kernel parameter (buffer pointer) |
| `CALL` | Kernel function call |
| `PROGRAM`, `LINEAR`, `SOURCE`, `BINARY` | Renderer stages |

### 3. Structure
| Op | Description |
|----|-------------|
| `SINK` | Collect multiple outputs |
| `AFTER` | Dependency ordering (pass src[0] through, ensure consumers run after src[1:]) |
| `GROUP` | Merge things together (NOOP) |
| `GEP` | Get element pointer (vector element extraction) |
| `VECTORIZE` | Vector creation |
| `TUPLE` | Tuple of values |
| `GETTUPLE` | Extract from tuple |

### 4. Load/Store
| Op | Description |
|----|-------------|
| `INDEX` | Buffer index computation (pointer arithmetic) |
| `LOAD` | Load from buffer at index |
| `STORE` | Store to buffer at index |

### 5. Math - Unary
| Op | Description |
|----|-------------|
| `CAST`, `BITCAST` | Type conversion, type reinterpretation |
| `EXP2`, `LOG2`, `SIN` | Transcendental functions |
| `SQRT`, `RECIPROCAL`, `NEG`, `TRUNC` | Square root, reciprocal, negation, truncation |

### 6. Math - Binary
| Op | Description |
|----|-------------|
| `ADD`, `MUL`, `SUB`, `FDIV`, `POW` | Arithmetic |
| `SHL`, `SHR`, `IDIV`, `MOD` | Integer operations |
| `MAX` | Maximum |
| `CMPLT`, `CMPNE`, `CMPEQ` | Comparisons |
| `XOR`, `OR`, `AND` | Bitwise logic |
| `THREEFRY` | Random number generation |

### 7. Math - Ternary
| Op | Description |
|----|-------------|
| `WHERE` | Conditional select (condition, true_val, false_val) |
| `MULACC` | Fused multiply-accumulate |
| `WMMA`, `SHAPED_WMMA` | Tensor core matrix multiply |

### 8. Control Flow
| Op | Description |
|----|-------------|
| `BARRIER` | Thread synchronization barrier |
| `RANGE` | For-loop |
| `IF` | Conditional execution |
| `END`, `ENDIF` | End of scoped block |
| `CONST`, `VCONST` | Scalar and vector constants |

### 9. Tensor Graph Only (not in programs)
| Op | Description |
|----|-------------|
| `CONTIGUOUS` | Ensure contiguous layout |
| `DETACH` | Detach from gradient graph |
| `BUFFER`, `BUFFER_VIEW`, `COPY` | Buffer management |
| `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` | The 6 core movement ops |
| `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` | Reduction operations |

**Note**: The order of Ops in the enum controls toposort priority. Ops that don't exist in rendered programs (tensor graph ops, movement ops, reduce ops) have the highest enum values.
| `DEFINE_LOCAL` | Local/shared memory allocation |
| `DEFINE_REG` | Register allocation |
| `SPECIAL` | Special GPU dimensions (threadIdx, blockIdx) |

---

## GroupOp

**File**: `tinygrad/uop/__init__.py`

Named sets of Ops used for pattern matching:

| Group | Members |
|-------|---------|
| `GroupOp.Unary` | EXP2, LOG2, SIN, SQRT, RECIPROCAL, NEG, TRUNC |
| `GroupOp.Binary` | ADD, MUL, IDIV, MAX, MOD, CMPLT, CMPNE, CMPEQ, XOR, SHL, SHR, OR, AND, THREEFRY, SUB, FDIV, POW |
| `GroupOp.Ternary` | WHERE, MULACC |
| `GroupOp.ALU` | Union of Unary, Binary, Ternary |
| `GroupOp.Elementwise` | ALU + CAST, BITCAST |
| `GroupOp.Movement` | RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP |
| `GroupOp.Defines` | PARAM, DEFINE_LOCAL, DEFINE_REG |
| `GroupOp.Irreducible` | CONST, DEFINE_VAR, SPECIAL, RANGE |
| `GroupOp.Buffer` | LOAD, STORE, CONST, DEFINE_VAR |
| `GroupOp.Commutative` | ADD, MUL, MAX, CMPNE, CMPEQ, XOR, AND, OR |
| `GroupOp.Associative` | ADD, MUL, AND, OR, MAX |
| `GroupOp.Idempotent` | OR, AND, MAX |
| `GroupOp.Comparison` | CMPLT, CMPNE, CMPEQ |
| `GroupOp.UnsafePad` | RECIPROCAL, LOG2, EXP2, IDIV, POW |

---

## DType

**File**: `tinygrad/dtype.py`

Data type specification.

```python
@dataclass(frozen=True)
class DType:
  priority: int      # Type promotion priority (higher = wider)
  itemsize: int      # Size in bytes
  name: str          # Human-readable name (e.g., "float")
  fmt: str | None    # struct.pack format character
  count: int         # Vector width (1 for scalars)
```

**Key properties:**

| Property | Description |
|----------|-------------|
| `itemsize` | Size in bytes of one element |
| `name` | Type name string |
| `fmt` | Format string for `struct.pack` (None for types without one) |
| `count` | Vector width (1 for scalar, N for vecN) |
| `scalar()` | Returns the scalar base type |
| `vec(n)` | Creates a vector type of width n |
| `min` | Minimum representable value |
| `max` | Maximum representable value |

**Specialized subtypes:**

### PtrDType

Pointer type with address space information:

```python
@dataclass(frozen=True)
class PtrDType(DType):
  base: DType           # Pointed-to element type
  addrspace: AddrSpace  # Memory address space (GLOBAL, LOCAL, REG)
  size: int             # Number of elements (-1 for unknown)
```

### ImageDType

Image format type for 2D-specific optimizations:

```python
@dataclass(frozen=True)
class ImageDType(DType):
  shape: tuple[int, ...]  # Image dimensions
```

---

## Buffer

**File**: `tinygrad/device.py`

A chunk of device memory.

```python
@dataclass
class Buffer:
  device: str           # Device string (e.g., "CUDA", "METAL")
  size: int             # Number of elements
  dtype: DType          # Element type
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `allocate(data=None)` | Allocate device memory, optionally copying data |
| `copyin(src)` | Copy data from host to device |
| `copyout(dest)` | Copy data from device to host |
| `nbytes` | Total size in bytes (`size * dtype.itemsize`) |
| `_buf` | Raw device-specific buffer handle |

### MultiBuffer

Sharded buffer across multiple devices:

```python
class MultiBuffer:
  bufs: tuple[Buffer, ...]    # Per-device buffers
  axis: int | None            # Shard axis
```

---

## Tensor

**File**: `tinygrad/tensor.py`

The user-facing computation handle.

```python
class Tensor:
  uop: UOp                    # The lazy computation graph node
  requires_grad: bool | None  # Gradient tracking
  grad: Tensor | None         # Accumulated gradient (after backward)
```

**Key properties:**

| Property | Description |
|----------|-------------|
| `shape` | Tuple of dimension sizes |
| `dtype` | Element data type |
| `device` | Device string |
| `uop` | Underlying UOp graph node |
| `requires_grad` | Whether to track gradients |
| `grad` | Gradient tensor (populated after `.backward()`) |

---

## ProgramSpec

**File**: `tinygrad/renderer/__init__.py`

Specification of a compiled kernel program.

```python
@dataclass(frozen=True)
class ProgramSpec:
  name: str                    # Kernel function name
  src: str                     # Generated source code
  device: str                  # Target device
  uops: tuple[UOp, ...]       # Linearized UOps
  global_size: list[int] | None  # GPU grid size
  local_size: list[int] | None   # GPU workgroup size
  estimates: Estimates         # FLOP/memory estimates
  vars: tuple[Variable, ...]   # Symbolic variables
```

---

## Estimates

**File**: `tinygrad/renderer/__init__.py`

Performance estimates for a kernel.

```python
@dataclass(frozen=True)
class Estimates:
  ops: sint   # Number of FLOPs
  lds: sint   # Bytes accessed in loads/stores
  mem: sint   # Total unique bytes accessed
```

---

## KernelInfo

**File**: `tinygrad/uop/ops.py`

Metadata for a kernel during codegen.

| Field | Description |
|-------|-------------|
| `local_dims` | Number of local (workgroup) dimensions |
| `upcasted` | Number of upcasted dimensions |
| `dont_use_locals` | Whether to avoid local memory |

---

## ExecItem

**File**: `tinygrad/engine/realize.py`

A scheduled execution unit (kernel call, buffer copy, etc.).

Contains a `Runner` and its argument `Buffer` list. This is the fundamental unit passed to the execution engine.

---

## Relationships

```
Tensor  --wraps-->  UOp  --has-->  Ops (operation type)
                     |              DType (data type)
                     |              tuple[UOp] (sources)
                     |
                     +--buffer-->  Buffer  --on-->  Compiled (device)
                     
ProgramSpec  --contains-->  UOps (linearized)
             --targets-->   Device
             --estimates--> Estimates

Runner  --wraps-->  ProgramSpec  --compiled-->  device program
        --executes with-->  list[Buffer]
```
