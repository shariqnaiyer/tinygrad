"""Defines the Ops enum and GroupOp class — the two most fundamental types in tinygrad.

Every computation in tinygrad is represented as a DAG of UOp nodes. Each UOp has exactly one Ops member
as its `.op`, making Ops the universal instruction set: from high-level tensor operations (RESHAPE, REDUCE_AXIS)
down to low-level codegen primitives (RANGE, LOAD, STORE, ALU ops). The compiler pipeline progressively
rewrites UOps with high-level Ops into UOps with lower-level Ops until only hardware-renderable ops remain.

GroupOp provides named frozensets of Ops used by UPat pattern matching. Instead of matching `{Ops.ADD, Ops.MUL, ...}`
everywhere, patterns match `GroupOp.Binary` or `GroupOp.ALU`. GroupOp also encodes algebraic properties
(Commutative, Associative, Idempotent) that the symbolic simplifier exploits.

The INTEGER VALUE of each Ops member matters: it controls toposort ordering within a UOp graph.
Lower-numbered ops are placed earlier in the topological sort. This is why defines come first (they must
exist before anything uses them) and consts come late (they can be placed near their consumers).
"""
# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
from enum import auto, IntEnum, Enum

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  """Custom IntEnum base that fixes two problems with stdlib IntEnum.

  1. __str__: Python's IntEnum.__str__ returns "ClassName.MEMBER" in 3.10 but just the int in 3.11+.
     FastEnum always uses the Enum-style "ClassName.MEMBER" string, which is what the viz and debug output expect.
  2. auto() uniqueness: _generate_next_value_ ensures auto() values are globally unique across ALL FastEnum
     subclasses (not just within one class). This prevents collisions if multiple FastEnum subclasses coexist,
     and is essential because the integer values double as toposort priority keys.
  """
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  """The universal op enum — every UOp node has exactly one of these as its `.op`.

  Ops are numbered by `auto()` in a deliberate order that matches the desired toposort: definitions first,
  then structural ops, then memory ops, then math, then control flow, then consts (late so they sit near
  consumers), and finally tensor-graph-only ops that never appear in rendered programs.

  Each UOp node has: op (Ops), dtype (DType), src (tuple[UOp,...]), arg (Any).
  The comments below describe what `src` and `arg` typically hold for each op.
  """

  # ============================================================
  # ** 1 -- defines/special: things that must exist before the kernel body **
  # These get the lowest toposort numbers so they appear at the top of any linearized program.
  # ============================================================

  # DEFINE_VAR: a symbolic variable with known bounds, used for dynamic shapes.
  #   arg=(name:str, min:int, max:int), src=(), dtype=int.
  # BIND: pairs a DEFINE_VAR with a concrete runtime value (a CONST). src=(DEFINE_VAR, CONST).
  #   Used by the JIT to substitute actual values for symbolic vars at execution time.
  DEFINE_VAR = auto(); BIND = auto()

  # SPECIAL: represents a GPU grid/block dimension index (like CUDA's threadIdx.x / blockIdx.x).
  #   arg=(axis_prefix:str, axis_index:int) where prefix is 'g' (global/grid), 'l' (local/block), or 'i' (item).
  #   src=(size:UOp,) giving the dimension's extent. Unlike RANGE, SPECIAL is not a loop — it's an implicit parallel index.
  SPECIAL = auto()

  # DEFINE_LOCAL: allocates shared/local memory (e.g. __shared__ in CUDA, __local in OpenCL).
  #   arg=name:str, dtype=PtrDType with addrspace=LOCAL, no src.
  # DEFINE_REG: allocates register-file storage (used on AMD for explicit register management).
  #   arg=name:str, dtype=PtrDType with addrspace=REG, no src.
  DEFINE_LOCAL = auto(); DEFINE_REG = auto()

  # ============================================================
  # ** 2 -- non-op / structural UOps: graph plumbing that isn't rendered as math **
  # These organize the UOp graph but don't produce arithmetic instructions.
  # ============================================================

  # NOOP: a pass-through that forwards src[0] unchanged. Used as a placeholder (e.g. when shape/device is absent)
  #   and to mark nodes the scheduler must always run. Rewrite rules often collapse NOOPs away.
  # REWRITE_ERROR: injected by the pattern matcher when a rewrite rule throws an exception.
  #   Captures the error message in arg for visualization/debugging. Never appears in valid programs.
  NOOP = auto(); REWRITE_ERROR = auto()

  # PARAM: a pointer to a kernel argument in global memory (the "define global" of the kernel).
  #   arg=slot:int (which argument position), dtype=PtrDType. src carries shape/device/bounds metadata.
  #   This is what LOAD/STORE index into to access buffers passed to the kernel.
  # CALL: invokes a compiled kernel or scheduled operation.
  #   src=(program_or_linear, *buffer_args), represents "run this kernel with these buffers".
  PARAM = auto(); CALL = auto()

  # --- Renderer / compilation pipeline ops ---
  # PROGRAM: the top-level wrapper for a compiled kernel. src=(SINK, DEVICE, LINEAR[, SOURCE, BINARY]).
  # LINEAR: an ordered list of UOps forming the kernel body (the linearized program). src=(*uops,).
  # SOURCE: human-readable source code. arg=str (e.g. the CUDA/OpenCL/Metal source text).
  # BINARY: compiled machine code. arg=bytes (the compiled binary blob).
  PROGRAM = auto(); LINEAR = auto(); SOURCE = auto(); BINARY = auto()

  # SINK: the root of a kernel graph — collects all STORE ops. src=(*stores,), arg=KernelInfo.
  #   Every kernel has exactly one SINK; it's the "return" of the computation.
  # AFTER: ordering primitive. Passes src[0] through but forces toposort to place it after src[1:].
  #   Used to enforce execution order between independent operations (e.g. store-before-load dependencies).
  # GROUP: a NOOP that merges multiple srcs into one node, purely for graph structure.
  # BEAM: wraps a SINK to request beam search optimization. arg=beam_width:int.
  SINK = auto(); AFTER = auto(); GROUP = auto(); BEAM = auto()

  # GEP (Get Element Pointer): extracts element(s) from a vector register.
  #   src=(vector_uop,), arg=tuple[int,...] of indices to extract. Named after LLVM's GEP instruction.
  # VECTORIZE: packs scalar UOps into a single vector register. src=(*scalars,), dtype=scalar_type.vec(N).
  GEP = auto(); VECTORIZE = auto()

  # TUPLE: groups multiple return values from a kernel into one node. src=(*values,).
  # GETTUPLE: extracts one element from a TUPLE. src=(tuple_uop,), arg=index:int.
  #   Used when a CALL returns multiple buffers and you need to select one.
  TUPLE = auto(); GETTUPLE = auto()

  # ============================================================
  # ** 3 -- load/store: memory access operations **
  # These must come before math ops in toposort so loads are ready before the ALU uses their values.
  # ============================================================

  # INDEX: computes a pointer + offset address, like pointer arithmetic (ptr[idx]).
  #   src=(buffer_ptr, offset_idx[, gate:bool]). The optional gate enables conditional/masked access.
  #   Similar to ADD but type-aware: operates on pointer types, not integers.
  INDEX = auto()

  # LOAD: reads a value from memory at an INDEX. src=(index_uop,) or (index_uop, barrier).
  # STORE: writes a value to memory. src=(index_uop, value[, gate]). The optional gate makes the store conditional.
  # Load before math in toposort ensures values are available before the ALU operations that consume them.
  LOAD = auto(); STORE = auto()

  # ============================================================
  # ** 4 -- math: the ALU operations that do actual computation **
  # These are the ops that become arithmetic instructions in the rendered kernel.
  # Grouped into Unary/Binary/Ternary by arity; GroupOp.ALU is their union.
  # ============================================================

  # --- Tensor core / matrix ops (not elementwise — they operate on tiles) ---
  # WMMA: Warp Matrix Multiply Accumulate (hardware tensor core instruction).
  #   src=(A_vec, B_vec, C_acc_vec), arg=8-tuple of (dims, device, thread_count, ...).
  #   Produces a vector result. This is the low-level op after lowering.
  # SHAPED_WMMA: the high-level shaped version of WMMA that still has tensor shapes.
  #   src=(A, B, accumulator), same arg format. Gets lowered to CONTRACT + WMMA during codegen.
  WMMA = auto(); SHAPED_WMMA = auto()

  # --- UnaryOps: one input, one output, elementwise ---
  # CAST: dtype conversion (e.g. float32->float16). BITCAST: reinterpret bits as different type (no conversion).
  # The rest are standard math: EXP2 (2^x), LOG2 (log base 2), SIN, SQRT, RECIPROCAL (1/x), NEG (-x), TRUNC (round toward zero).
  # tinygrad uses base-2 exp/log because GPUs natively support them; base-e is synthesized from these.
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  # --- BinaryOps: two inputs, one output, elementwise ---
  # Standard arithmetic: ADD, MUL, SHL (shift left), SHR (shift right), IDIV (integer division), MAX, MOD.
  # Comparisons: CMPLT (<), CMPNE (!=), CMPEQ (==). These produce bool dtype. No CMPLE/GT — those are derived via CMPLT+swap.
  # Bitwise: XOR, OR, AND.
  # THREEFRY: the Threefry2x32 counter-based PRNG hash. src=(counter, key). This IS an ALU op that some backends implement natively.
  # SUB, FDIV (float division), POW: "sugar" ops that most backends decompose (SUB->ADD+NEG, FDIV->MUL+RECIPROCAL, POW->EXP2+LOG2+MUL).
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  # --- TernaryOps: three inputs ---
  # WHERE: conditional select. src=(condition, true_val, false_val). Like numpy.where / C's ternary `?:`.
  # MULACC: fused multiply-accumulate. src=(a, b, c) computes a*b+c. Maps to hardware FMA instructions.
  WHERE = auto(); MULACC = auto()

  # ============================================================
  # ** 5 -- control flow / consts / custom: loops, branches, literals, and escape hatches **
  # Consts are intentionally late in the toposort so they sit near their consumers in the linearized program.
  # ============================================================

  # --- Control flow ---
  # BARRIER: a GPU synchronization barrier (e.g. __syncthreads). src=(*ops_to_wait_for,). Forces all threads
  #   in a workgroup to reach this point before any proceed. Essential between shared memory write and read.
  # RANGE: a loop induction variable. src=(end,) or src=(start, end). arg=(range_id, axis_type).
  #   This is tinygrad's loop — the body is everything between RANGE and its matching END in the toposort.
  # IF: conditional branch. src=(condition:bool, gate_src). Begins a conditional block.
  # END: closes a RANGE loop or wraps a CALL for scheduling. src=(body_result, range_uop) for loops.
  # ENDIF: closes an IF block. src=(if_uop,).
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()

  # --- Constants (late in toposort so they're placed close to their consumers) ---
  # VCONST: a vector constant. arg=tuple of scalar values, dtype=scalar.vec(N).
  # CONST: a scalar constant. arg=the value (int/float), dtype=the type.
  #   In the tensor graph, CONST has src=(UNIQUE, DEVICE) to carry identity and placement info.
  VCONST = auto(); CONST = auto()

  # --- Escape hatches for custom codegen ---
  # CUSTOM/CUSTOMI: inject literal strings into generated code. arg=str.
  #   CUSTOM is a statement (own line), CUSTOMI is an inline expression that can be used in other expressions.
  CUSTOM = auto(); CUSTOMI = auto()

  # INS: a raw machine instruction for assembly-level backends (e.g. AMD GCN/RDNA).
  #   arg=instruction data. Used inside LINEAR when the renderer produces assembly instead of source code.
  INS = auto()

  # ============================================================
  # ** 6 -- ops that don't exist in rendered programs **
  # These only live in the tensor graph or the scheduler. They are rewritten away before codegen.
  # Having the highest enum values means they sort to the end of any toposort.
  # ============================================================

  # --- Identity / device tags (metadata nodes in the tensor graph) ---
  # UNIQUE: carries a globally-unique integer so two independently created constants aren't CSE'd together.
  #   arg=int (unique id). Created by UOp.unique(). Without this, `Tensor(1.0) + Tensor(1.0)` would merge into one node.
  # DEVICE: tags which device a tensor lives on. arg=device_str (e.g. "CUDA", "CPU", "AMD").
  UNIQUE = auto(); DEVICE = auto()

  # LUNIQUE: "local unique" — same purpose as UNIQUE but scoped to a single function call.
  #   When a function template is instantiated, UNIQUE nodes are rewritten to LUNIQUE so each call site
  #   gets its own numbering without polluting the global unique counter.
  LUNIQUE = auto()

  # --- Scheduler-influencing ops (control how the tensor graph gets broken into kernels) ---
  # CONTIGUOUS: forces a tensor to be materialized into a contiguous buffer (triggers a kernel boundary).
  # CONTIGUOUS_BACKWARD: marks that the backward pass should insert a CONTIGUOUS.
  # DETACH: breaks the autograd tape — gradient computation stops here. Passes src[0] through in forward.
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # --- Buffer / memory management ops ---
  # BUFFERIZE: wraps a computation that needs to be stored into a buffer. src=(value, *range_dims), arg=options.
  #   This is the bridge between "compute" and "memory" — it says "this value needs a buffer to live in".
  # COPY: cross-device data transfer. src=(source_tensor, DEVICE). Triggers a DMA copy.
  # BUFFER: an allocated memory buffer. src=(UNIQUE, DEVICE), arg=size. The fundamental storage primitive.
  # BUFFER_VIEW: a zero-copy view into an existing BUFFER (e.g. for reshape that doesn't move data). src=(BUFFER,), arg=(size,offset).
  # MSELECT/MSTACK: multi-buffer select/stack for operations across sharded tensors.
  # CUSTOM_FUNCTION: escape hatch for operations implemented outside the normal compiler (e.g. graph execution, enc/dec).
  BUFFERIZE = auto(); COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto(); CUSTOM_FUNCTION = auto()

  # --- The 6 core movement ops (tensor graph only — rewritten to index math during scheduling) ---
  # These describe lazy view transformations that don't move data. They get lowered into INDEX arithmetic.
  # RESHAPE: change shape without moving data (if contiguous). PERMUTE: transpose axes. EXPAND: broadcast.
  # PAD: add zero-padding. SHRINK: slice/crop. FLIP: reverse along axes.
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()
  MULTI = auto()  # MULTI: shards a tensor across multiple devices. Semantically a movement op — it changes where data lives.

  # --- Reduce ops ---
  # REDUCE_AXIS: high-level reduce in the tensor graph. src=(input,), arg=(reduce_op:Ops, axes:tuple[int,...]).
  #   Gets lowered to RANGE loops + REDUCE during scheduling.
  # REDUCE: low-level reduce inside a kernel. src=(value, *range_uops). The actual accumulation loop body.
  # ALLREDUCE: cross-device reduction (e.g. sum across GPUs). src=(value, DEVICE), arg=reduce_op.
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto()

  # --- Expander / vectorization ops (used during codegen's expand/contract passes) ---
  # UNROLL: marks a dimension to be unrolled (expanded) during the expander pass. src=(body,), arg=unroll_axes.
  # CONTRACT: the inverse of UNROLL — packs unrolled scalars back into a vector. src=(body,), arg=contract_axes.
  # VCAT: concatenates vectors (like VECTORIZE but for already-vector inputs). Lowered to GEP + VECTORIZE.
  # PTRCAT: concatenates pointer-typed vectors for vectorized memory access patterns.
  UNROLL = auto(); CONTRACT = auto(); VCAT = auto(); PTRCAT = auto()

class GroupOp:
  """Named sets of Ops used for efficient pattern matching in UPat rewrite rules.

  Instead of writing `UPat({Ops.ADD, Ops.MUL, Ops.IDIV, ...})` in every pattern, you write `UPat(GroupOp.Binary)`.
  This is both more readable and easier to maintain — adding a new binary op means updating one place.

  The algebraic property sets (Commutative, Associative, Idempotent) are used by the symbolic simplifier
  to decide which optimizations are safe. For example, `a+b` can become `b+a` only because ADD is in Commutative.
  """

  # --- Arity-based groups: ops classified by number of inputs ---
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  # ALU = all arithmetic/logic ops. The core set that renderers must implement.
  ALU = set.union(Unary, Binary, Ternary)

  # Elementwise: ops where each output element depends only on the corresponding input element(s).
  # Includes ALU + type conversions. Used to determine which ops can be fused into the same kernel.
  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})

  # Defines: ops that define addressable memory regions (kernel args, shared memory, registers).
  # These are the "pointers" that INDEX operates on to compute memory addresses.
  Defines = {Ops.PARAM, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}

  # Irreducible: leaf nodes that can't be simplified further in symbolic math.
  # The symbolic simplifier stops recursing when it hits these.
  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}

  # Movement: the 6 lazy view ops in the tensor graph. These become index math, never rendered as instructions.
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  # Buffer: ops that interact with memory in the rendered kernel. Used to identify memory-touching nodes.
  Buffer = {Ops.LOAD, Ops.STORE, Ops.CONST, Ops.DEFINE_VAR}

  # --- Algebraic property sets: used by the symbolic simplifier for safe rewrites ---

  # BinaryOps that can be flipped: f(a,b) = f(b,a). Enables canonical ordering of operands.
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c)). Enables flattening nested expressions.
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  # Enables simplification: `x AND x` -> `x`, `MAX(x,x)` -> `x`.
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # These can change the dtype to bool (output is always bool regardless of input dtype).
  Comparison = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}

  # Ops where f(0) != 0, making them unsafe to use with PAD (which introduces zeros at boundaries).
  # e.g. RECIPROCAL(0)=inf, LOG2(0)=-inf. The scheduler must avoid padding reductions that include these.
  UnsafePad = {Ops.RECIPROCAL, Ops.LOG2, Ops.EXP2, Ops.IDIV, Ops.POW}

  # Every op in the enum — useful for "match anything" patterns.
  All = set(Ops)
