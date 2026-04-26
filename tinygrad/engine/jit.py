"""TinyJit: record-replay JIT compiler for tinygrad.

On the first call to a @TinyJit-decorated function, the JIT runs the function normally (cnt=0, "ignore" pass).
On the second call (cnt=1, "capture" pass), it intercepts all kernel dispatches via the `capturing` list,
recording them as a flat sequence of UOps (a "linear"). On all subsequent calls (cnt>=2, "replay" pass),
it replays the captured kernels directly -- skipping Python-level scheduling entirely -- and optionally
batches them into hardware graphs (CUDA graphs, Metal command buffers, HCQ graphs) for minimal CPU overhead.

The two-pass warmup (ignore then capture) exists because the first call may trigger one-time work like
autotuning or lazy weight loading that should not be baked into the cached program.
"""
from typing import TypeVar, Generic, Callable, cast, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, JIT_BATCH_SIZE, dedup, unwrap, pluralize, VIZ
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, sym_infer, Ops, buffers, track_rewrites, graph_rewrite
from tinygrad.engine.realize import ExecItem, capturing, BufferCopy, BufferXfer, EncDec, CompiledRunner, Runner, Estimates
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.schedule import linear_to_schedule
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  """Splits a captured linear into kept (repeating) and one-time kernels.

  Walks the linear's schedule items and separates them into two groups:
  - kept: items whose buffers overlap with `needed` (i.e., they touch JIT inputs or their transitive dependents).
          These are the kernels that must run on every JIT replay.
  - onetime: items that only produce intermediates not depended on by any kept item (e.g., initial weight transforms).
             These are run once at capture time and discarded.

  The split propagates transitively: if a kernel's output feeds into a kept kernel, it also becomes kept.

  Args:
    linear: The full captured linear (Ops.LINEAR UOp) containing all schedule items.
    needed: Set of buffer UOps corresponding to the JIT's dynamic inputs.

  Returns:
    A tuple of (kept_linear, onetime_linear) -- two LINEAR UOps partitioning the original.
  """
  kept, onetime = [], []
  for si in linear.src:
    # collect all buffers that this schedule item reads or writes
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      # transitively mark this item's buffers as needed so downstream dependents are also kept
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def create_graph_call(batch:list[UOp], input_buffers:set[Buffer]) -> UOp:
  """Wraps a batch of schedule items into a single graph-dispatched UOp.

  Takes a list of compatible kernels and produces a CUSTOM_FUNCTION UOp with arg="graph" that,
  when lowered, becomes a GraphRunner -- a single hardware graph call (e.g., one CUDA graph launch)
  that executes all the batched kernels together with minimal CPU dispatch overhead.

  Args:
    batch: List of schedule item UOps to be batched into one graph.
    input_buffers: Set of buffers that are dynamic JIT inputs (so the graph knows which buffers may change between calls).

  Returns:
    A CALL UOp wrapping the graph, ready to be inserted into the linear.
  """
  def bufs_for(b): return b.buffer.bufs if isinstance(b.buffer, MultiBuffer) else [b.buffer]

  # identify which buffers in this batch are dynamic inputs (they change between JIT calls)
  input_list = dedup(b for si in batch for b in si.src[1:] if b.op in (Ops.BUFFER, Ops.BUFFER_VIEW) and not input_buffers.isdisjoint(bufs_for(b)))
  cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp(Ops.LINEAR, src=tuple(batch)), *input_list), arg="graph")
  return cf.call(*input_list, metadata=tuple(m for si in batch for m in si.arg.metadata))

def graph_split_rewrite(linear:UOp, input_buffers:set[Buffer], max_batch_size:int=0) -> UOp:
  """Rewrites a linear by grouping consecutive graph-compatible kernels into batched graph calls.

  Walks the linear's schedule items in order and greedily accumulates runs of kernels that can share
  a single hardware graph (same device, supported op types). When a kernel breaks compatibility
  (different device, unsupported op, or batch size limit hit), the accumulated batch is flushed into
  a create_graph_call. The batch size limit doubles after each flush to allow larger graphs as
  confidence grows.

  Kernels that are not graph-compatible (e.g., host-side buffer views) pass through ungraphed.

  Args:
    linear: The memory-planned linear to rewrite.
    input_buffers: Dynamic JIT input buffers (passed through to create_graph_call).
    max_batch_size: Initial max kernels per graph (0 = unlimited). Doubles after each flush.

  Returns:
    A new LINEAR UOp where consecutive compatible kernels are replaced by graph call UOps.
  """
  new_src: list[UOp] = []
  current_batch: list[UOp] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size, new_src
    # don't bother graphing a single kernel unless forced -- the overhead isn't worth it
    if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current_batch)
    else:
      new_src.append(create_graph_call(current_batch, input_buffers))
      # exponential backoff: allow the next graph batch to be twice as large
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels")
    current_batch, current_batch_devs = [], []

  for si in linear.src:
    # BUFFER_VIEW items are host-side metadata ops with no GPU work -- skip them entirely
    if si.src[0].op is Ops.BUFFER_VIEW: continue

    devs = [Device[x] for x in (si.device if isinstance(si.device, tuple) else (si.device,))]
    graph_t = graph_class(devs[0]) if devs[0].graph is not None else None

    # can this kernel be graphed at all?
    can_graph = graph_t is not None and graph_t.supports_exec_item(devs, si)
    # can it extend the current batch? (same graph type, compatible devices, within size limit)
    can_extend = can_graph and graph_t is not None and (not current_batch_devs or graph_t.supports_exec_item(current_batch_devs, si)) \
      and (max_batch_size == 0 or len(current_batch) < max_batch_size)
    # if we can't extend, flush whatever we've accumulated so far before starting fresh
    if not can_extend and current_batch: flush_batch()

    # append this si and update devs
    (current_batch if can_graph else new_src).append(si)
    current_batch_devs = dedup(current_batch_devs + devs) if can_graph else []
  if current_batch: flush_batch()
  return linear.replace(src=tuple(new_src))

def jit_cache_bufs(jit_cache:list[ExecItem]):
  """Yields all Buffers referenced by a JIT cache, recursing into nested GraphRunners.

  GraphRunners contain their own sub-list of ExecItems (the kernels they batch), so this walks
  the full tree to find every buffer the JIT touches -- needed for allocation and deallocation.
  """
  for ei in jit_cache:
    for b in ei.bufs:
      if b is not None: yield b
    # GraphRunners wrap sub-caches of the kernels they batch -- recurse into them
    if isinstance(ei.prg, GraphRunner): yield from jit_cache_bufs(ei.prg.jit_cache)

@track_rewrites(lambda linear,held_bufs,input_buffers=None,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_buffers:list[Buffer]|None=None) -> list[ExecItem]:
  """Lowers a captured linear UOp into executable ExecItems, applying memory planning and graph batching.

  This is the main compilation pipeline for a captured JIT trace:
  1. Memory plan rewrite: assigns buffer memory layout, reusing allocations where possible.
  2. Graph split rewrite (unless JIT>=2): groups compatible kernels into hardware graph batches.
  3. Lower: converts the rewritten schedule items into ExecItems with compiled programs.

  Args:
    linear: The captured LINEAR UOp containing all schedule items from the JIT capture pass.
    held_bufs: Buffer UOps that must not be freed (inputs, outputs, model weights).
    input_buffers: Dynamic input Buffers whose identity changes between JIT calls.

  Returns:
    List of ExecItems ready for execution (the JIT cache).
  """
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")
  # step 1: optimize memory layout -- reuse allocations for non-overlapping lifetimes
  linear = memory_plan_rewrite(linear, held_bufs)
  # step 2: batch consecutive compatible kernels into hardware graphs (JIT>=2 disables this for debugging)
  if JIT < 2: linear = graph_split_rewrite(linear, set(input_buffers or []), max_batch_size=JIT_BATCH_SIZE.value)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View graphed linear")
  # step 3: lower each schedule item into an ExecItem with a compiled program
  return [ei.lower() for ei in linear_to_schedule(linear)]

class GraphException(Exception):
  """Raised when hardware graph construction fails (e.g., unsupported op in a graph batch)."""

class JitError(Exception):
  """Raised for JIT-specific errors: input mismatches, non-Tensor returns, const inputs, etc."""

def _check_no_non_tensor_return(ret):
  """Validates that a JIT function's return value contains only Tensors (or None).

  Non-Tensor return values would be stale on replay since they're captured once and reused.
  This catches that mistake early with a clear error rather than letting silent bugs through.
  """
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev):
  """Unwraps a device's graph class from a possible functools.partial (used when graph constructors carry default args)."""
  return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

def get_input_replace(jit_cache: list[ExecItem], input_buffers:list[Buffer],
                      orig_valid_positions: dict[int, set[int]]|None = None) -> dict[tuple[int, int], int]:
  """Builds a mapping from (exec_item_index, buf_slot_index) -> input_buffer_index.

  This tells the JIT replay loop which buffer slots in which ExecItems need to be swapped out
  with the caller's new input buffers on each call. Without this map, the JIT would always
  operate on the same buffers captured during the recording pass.

  Args:
    jit_cache: The compiled list of ExecItems from the capture pass.
    input_buffers: The list of dynamic input Buffers (order matches the caller's args).
    orig_valid_positions: Optional filter to prevent aliasing bugs -- only slots that were
        actual inputs during the original capture are eligible for replacement.

  Returns:
    Dict mapping (exec_item_index, buffer_slot_index) to the index in input_buffers.
  """
  input_replace: dict[tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_buffers:
        # filter out positions that weren't valid inputs in the original capture (prevents aliasing bugs)
        if orig_valid_positions is not None and i not in orig_valid_positions.get(id(ji), set()): continue
        input_replace[(j,i)] = input_buffers.index(a)
  return input_replace

class GraphRunner(Runner):
  """Executes a batch of kernels as a single hardware graph dispatch.

  Wraps a list of ExecItems into a device-specific graph (e.g., CUDA graph, Metal command buffer)
  so they can be launched together with a single CPU-side call. This eliminates per-kernel launch
  overhead which dominates small-kernel workloads like transformer inference.

  The constructor can be called in two modes:
  - With a `linear` UOp: lowers the linear into ExecItems internally (used during JIT capture).
  - With explicit `jit_cache`/`input_replace`: reconstitutes from a previously captured state (used for pickle/rebuild).

  Subclasses (per backend) override exec() to implement the actual graph launch.
  """
  def __init__(self, linear:UOp|None, input_buffers:list[Buffer]|None,
               jit_cache:list[ExecItem]|None=None, input_replace:dict[tuple[int,int],int]|None=None):
    # TODO: captured jit as linear?
    if linear is not None:
      jit_cache = [ei.lower() for ei in linear_to_schedule(linear.src[0])]
      for b in jit_cache_bufs(jit_cache): b.ensure_allocated()
      input_replace = get_input_replace(jit_cache, input_buffers) if input_buffers else {}
    self.jit_cache, self.input_replace = unwrap(jit_cache), input_replace or {}

    # maps for resolving symbolic (variable-dependent) values at replay time
    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}      # exec_item_idx -> [(var_position, vars_list_idx)]
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}  # exec_item_idx -> (global_dim_idx, local_dim_idx) into symbolic_dims
    self.launch_dims_base:dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}  # exec_item_idx -> (global_size, local_size) for non-symbolic dims

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    # collect all CompiledRunner exec items to analyze their symbolic dependencies
    crs = [(ji, ji.prg) for ji in self.jit_cache if isinstance(ji.prg, CompiledRunner)]
    # vars that are neither fixed at capture time nor runtime-resolved -- these must be provided at each call
    self.vars = sorted({v.expr for ji,p in crs for v in p.p.vars if v.expr not in ji.fixedvars | p.p.runtimevars})
    # unique symbolic launch dimension tuples (deduplicated since many kernels share the same symbolic shape)
    self.symbolic_dims = dedup([tuple(d) for _,p in crs if (d:=p.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for _,p in crs if (d:=p.p.global_size) and is_sym_dim(d)])

    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    # build per-kernel replacement tables for symbolic variables and launch dimensions
    estimates = Estimates()
    for j,ji in enumerate(self.jit_cache):
      assert ji.prg is not None
      estimates += ji.prg.estimates
      if isinstance(ji.prg, CompiledRunner):
        # record which variable slots in this kernel need to be resolved from var_vals at replay time
        if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(ji.prg.p.vars) if v.expr not in ji.fixedvars | ji.prg.p.runtimevars]):
          self.var_vals_replace[j] = replace

        # record which kernels have symbolic (variable-dependent) global/local launch dimensions
        global_dim_idx, local_dim_idx = find_symbolic_dim(ji.prg.p.global_size), find_symbolic_dim(ji.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert ji.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(ji.prg.p.global_size), tuple(ji.prg.p.local_size))

    # used in MultiGraphRunner. tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)

    assert self.jit_cache[0].prg is not None
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), self.jit_cache[0].prg.device.split(":")[0], estimates.simplify())

  def __reduce__(self): return self.__class__, (None, None, self.jit_cache, self.input_replace)

  def updated_vars(self, var_vals: dict[str, int]):
    """Yields (exec_item_idx, var_position, concrete_value) for all symbolic variables that need updating.

    Called by backend graph implementations before replay to patch variable values into each kernel.
    """
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    """Yields (exec_item_idx, global_size, local_size) for kernels with symbolic launch dimensions.

    Resolves symbolic dimension expressions (e.g., containing Variable("n")) to concrete integers
    using the provided var_vals, then yields the resolved sizes for each affected kernel.
    """
    # resolve all unique symbolic dimension tuples once, then look up by index
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    """Tracks buffer access ranges to detect read-after-write and write-after-write hazards in MultiGraphRunner.

    Uses interval-based dependency tracking on the underlying raw buffer (by base._buf id) to handle
    suballocated buffers correctly -- two Buffers sharing the same raw allocation but at different offsets
    only conflict if their byte ranges actually overlap.

    Args:
      bufs: List of Buffers accessed by this kernel.
      write: Indices into `bufs` that are written (the rest are read-only).
      new_dependency: An opaque dependency token (e.g., a graph node) for the current access.

    Returns:
      List of dependency tokens that this access must wait on before executing.
    """
    # phase 1: collect all dependencies we need to wait on (WAW + RAW hazards)
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      # any overlapping write is a WAW or RAW hazard
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      # if we're writing, any overlapping read is a WAR hazard
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    # phase 2: update dependency maps -- punch out our write range from existing entries
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      if i in write:
        # our write supersedes any prior accesses to this range -- split existing intervals around our write
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for st,en,dep in dmap[key]:
            if st < min(s, en): kept.append((st, min(s, en), dep))
            if max(e, st) < en: kept.append((max(e, st), en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

  @staticmethod
  def _all_devs(batch_devs:list[Compiled], new_call:UOp) -> list[Compiled]:
    """Returns the deduplicated union of the current batch's devices and the new call's buffer devices."""
    return dedup(batch_devs + [Device[x] for b in new_call.src[1:] if b.op is not Ops.BIND
                 for x in (b.device if isinstance(b.device, tuple) else (b.device,))])

  @staticmethod
  def supports_exec_item(batch_devs:list[Compiled], new_call:UOp) -> bool:
    """Returns True if new_call can be added to a single-device graph batch (compute ops on exactly one device)."""
    return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM) and len(GraphRunner._all_devs(batch_devs, new_call)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  """GraphRunner variant for backends that support multi-device graphs (e.g., HCQ).

  Unlike GraphRunner which requires all kernels on the same device, MultiGraphRunner allows
  kernels across multiple devices of the same type (e.g., multiple CUDA GPUs). It also supports
  COPY ops for inter-device transfers within the graph.
  """
  @staticmethod
  def supports_exec_item(batch_devs:list[Compiled], new_call:UOp) -> bool:
    # Devices must be the same type (e.g., all CUDADevice) but can be different ordinals
    return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM, Ops.COPY) and len(dedup([type(d) for d in GraphRunner._all_devs(batch_devs, new_call)])) == 1

def get_out_buffers_for_ei(ei:ExecItem) -> list[Buffer]:
  """Returns the list of Buffers that this ExecItem writes to (output-only, excluding in-place).

  Used to track which buffers are produced by which ExecItem, enabling read-after-write hazard
  detection when JIT inputs alias JIT outputs (e.g., feeding a model's output back as input).
  """
  if isinstance(ei.prg, CompiledRunner): return [cast(Buffer, ei.bufs[out]) for out in ei.prg.p.outs if out not in ei.prg.p.ins]
  if isinstance(ei.prg, (BufferCopy, BufferXfer, EncDec)): return [cast(Buffer, ei.bufs[0])]
  if isinstance(ei.prg, GraphRunner): return dedup([b for inner in ei.prg.jit_cache for b in get_out_buffers_for_ei(inner)])
  return []

def update_depends(depends:set[Buffer|None], jit_cache:list[ExecItem]):
  """Propagates buffer dependencies forward through the JIT cache.

  If any of an ExecItem's input buffers are in the depends set, its output buffers are added too.
  This is used by free_intermediates to find all buffers transitively reachable from None (unassigned input slots),
  which identifies the intermediate buffers that can be safely deallocated.
  """
  for ei in jit_cache:
    if any(b in depends for b in ei.bufs): depends.update(get_out_buffers_for_ei(ei))

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  """The frozen result of a JIT capture pass -- everything needed to replay the computation.

  After the capture pass (cnt=1) records all kernels, this dataclass holds the compiled ExecItems,
  the buffer replacement map, and the original return value (with Tensor refs pointing into the
  JIT cache's output buffers). On each replay call, it swaps in new input buffers, handles
  read-after-write hazards for aliased inputs, and runs the cached kernels.

  Attributes:
    ret: The original return value from the captured function (Tensors reference the JIT cache's output buffers).
    jit_cache: The compiled list of ExecItems to replay.
    input_replace: Map from (exec_item_idx, buf_slot) to input_buffer_idx -- tells replay which slots to swap.
    extra_view_inputs: View buffers derived from inputs that also need swapping (offset, device, size, dtype info).
    expected_names: Positional/keyword arg names from capture, validated on replay to catch signature mismatches.
    expected_input_info: Per-input (view_uop, variables, dtype, device) tuples, validated on replay for shape/type safety.
  """
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: list[ExecItem]
  input_replace: dict[tuple[int, int], int]
  extra_view_inputs: list[tuple[int, int, str, int, DType]]
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  def __reduce__(self):
    # TODO: free_intermediates here?
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs, self.expected_names, self.expected_input_info)

  def __post_init__(self):
    """Initializes replay state: mutable copies of cache/replace maps and read-after-write hazard tables."""
    self._jit_cache: list[ExecItem] = self.jit_cache
    self._input_replace: dict[tuple[int, int], int] = self.input_replace
    self._first_run = True
    self._needs_rebuild = False
    # precompute read-after-write hazard detection:
    # _output_to_writer: buffer -> index of the ExecItem that writes it
    self._output_to_writer = {b: j for j, ei in enumerate(self.jit_cache) for b in get_out_buffers_for_ei(ei)}
    # _input_to_max_reader: input_buffer_idx -> latest ExecItem index that reads this input.
    # if an input is also an output (aliasing), we need to copy it before the writer overwrites it.
    self._input_to_max_reader: dict[int, int] = {}
    for (j, i), idx in self.input_replace.items():
      # only buffers that were different during capture but alias at jit time (e.g. feeding output back as input) need the copy.
      if self.jit_cache[j].bufs[i] not in get_out_buffers_for_ei(self.jit_cache[j]):
        self._input_to_max_reader[idx] = max(self._input_to_max_reader.get(idx, -1), j)
    self._clear_inputs()

  def _clear_inputs(self):
    """Nulls out all dynamic input buffer slots in the JIT cache after each run.

    This prevents stale buffer references from persisting between calls and ensures that
    unreferenced input buffers can be garbage collected.
    """
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  def free_intermediates(self):
    """Deallocates all intermediate buffers (those not externally visible as inputs or outputs).

    Walks the JIT cache to find buffers reachable from None-slotted (cleared) input positions,
    which are the intermediates. Also frees their backing arenas if all views are deallocated.
    After freeing, reinitializes replay state and marks graphs for rebuild on next run.
    """
    # start from None (cleared input slots) and propagate forward to find all intermediate buffers
    depends: set[Buffer|None] = set([None])
    update_depends(depends, self.jit_cache)
    # find memory arenas backing these intermediates (suballocated buffers share arenas)
    arenas = {b._base for b in depends if b is not None and b._base is not None}
    to_free = {b for b in depends if b is not None} | {b for b in jit_cache_bufs(self.jit_cache) if b._base in arenas}
    for b in to_free:
      if hasattr(b, '_buf'): b.deallocate()
    for a in arenas:
      if a.allocated_views == 0 and a.is_allocated(): a.deallocate()
    # reinitialize so next run re-allocates intermediates and rebuilds graphs
    self.__post_init__()
    self._needs_rebuild = True

  # jit exec
  def __call__(self, input_buffers:list[Buffer], var_vals:dict[str, int]) -> ReturnType:
    """Replays the captured JIT computation with new input buffers and variable values.

    This is the hot path -- called on every JIT invocation after capture. It patches input buffers
    into the cached ExecItems and runs them, avoiding all Python-level scheduling overhead.

    Args:
      input_buffers: New input Buffers for this call (same order as during capture).
      var_vals: Concrete values for symbolic variables (e.g., dynamic batch size).

    Returns:
      The original return value (Tensors now backed by updated output buffers).
    """
    # reconstruct view-based inputs (e.g., slices of input tensors) from the new base buffers
    for idx, offset, device, size, dtype in self.extra_view_inputs:
      input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())

    # copy aliased inputs to prevent read-after-write hazard:
    # if an input buffer is also written by the JIT (output fed back as input), and a later kernel reads it,
    # we must copy it now before the writer overwrites it
    for i, ib in enumerate(input_buffers):
      if (writer := self._output_to_writer.get(ib)) is not None and self._input_to_max_reader.get(i, -1) >= writer:
        input_buffers[i] = Buffer(ib.device, ib.size, ib.dtype).ensure_allocated().copyin(ib.as_memoryview())

    # patch new input buffers into the cached ExecItems at the recorded positions
    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # after free_intermediates, re-allocate all intermediate buffers on first run
    if self._first_run:
      for b in jit_cache_bufs(self.jit_cache): b.ensure_allocated()
    # rebuild hardware graphs if intermediates were freed and reallocated (buffer addresses changed)
    if self._needs_rebuild:
      for ei in self.jit_cache:
        if isinstance(ei.prg, GraphRunner): ei.prg = type(ei.prg)(None, None, ei.prg.jit_cache, ei.prg.input_replace)
    self._first_run = self._needs_rebuild = False

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    for ei in self._jit_cache: ei.run(var_vals, jit=True)
    # null out input slots to avoid holding references and allow GC
    self._clear_inputs()
    return self.ret

def _prepare_jit_inputs(args, kwargs):
  """Extracts and normalizes all Tensor inputs from a JIT call's args/kwargs.

  This is the shared input-preparation logic used by all three TinyJit phases (ignore, capture, replay).
  It finds all Tensor arguments (including those inside shallow containers like lists/dicts), realizes
  any lazy tensors, extracts their underlying Buffers, unbinds symbolic variables, and builds the
  expected_input_info fingerprint used to validate that replay inputs match the capture signature.

  Returns:
    A tuple of (input_buffers, var_vals, names, expected_input_info) where:
    - input_buffers: flat list of concrete Buffers backing all input Tensors.
    - var_vals: dict mapping variable expression strings to their current concrete values.
    - names: list of argument names/positions for the Tensor inputs (for mismatch detection).
    - expected_input_info: per-input (view, variables, dtype, device) tuples used to validate replay args.
  """
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  # ensure all input tensors are realized (have concrete buffers) before we extract them
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  # flatten multi-device tensors (sharded across GPUs) into individual per-device UOps
  input_uops: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.base.op is Ops.CONST for u in input_uops):
    raise JitError("JIT inputs cannot be const, create a buffer with .contiguous()")
  input_buffers: list[Buffer] = flatten([b.bufs if isinstance(b, MultiBuffer) else [b] for u in input_uops if (b:=u.base.realized) is not None])
  if len(set(input_buffers)) != len(input_buffers): raise JitError("duplicate inputs to JIT")
  # unbind symbolic variables from each input's view UOp, producing a shape-only UOp + variable bindings
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buffers, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  """JIT compiler decorator that records and replays tinygrad kernel executions.

  Lifecycle (controlled by self.cnt):
    cnt=0 ("ignore"):  Runs the function normally. This warmup pass handles one-time work like
                        autotuning (BEAM search) or lazy weight loading that shouldn't be cached.
    cnt=1 ("capture"):  Runs the function again, but with `capturing` active so all Tensor.realize()
                        calls route their schedule items to self._linears. After the function returns,
                        the captured linears are merged, memory-planned, optionally pruned, batched into
                        hardware graphs, and frozen into a CapturedJit.
    cnt>=2 ("replay"):  Skips the function entirely. Validates that inputs match the capture signature,
                        then delegates to CapturedJit.__call__ which patches in new buffers and replays.

  Can be constructed from a function (normal use) or from a pre-captured CapturedJit (deserialization).

  Args:
    fxn: The function to JIT compile (None when loading from pickle).
    captured: A pre-captured JIT state (None for fresh JIT, set when loading from pickle).
    prune: If True, splits captured kernels into repeating vs. one-time and only caches the repeating ones.
  """
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    # if loaded from a CapturedJit (no function), start at cnt=2 to go straight to replay mode
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_linear(self, linear:UOp, var_vals:dict[str, int]):
    """Called by the realize machinery during capture (cnt=1) to record each dispatched linear."""
    self._linears.append(linear)

  def reset(self):
    """Resets the JIT to its initial state, allowing re-capture (e.g., after model changes)."""
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    """Pickle support: serializes only the captured state, not the original function."""
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> list[ExecItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> dict[tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    """Main entry point: dispatches to ignore, capture, or replay based on self.cnt."""
    input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # === PHASE 0: IGNORE === run normally without recording (warmup pass for autotuning, lazy init, etc.)
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # === PHASE 1: CAPTURE === run the function while intercepting all kernel dispatches
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      # register ourselves so realize() routes schedule items to our add_linear method
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buffers)} inputs")

      # merge all captured linears into a single flat sequence of schedule items
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      # optionally prune: separate one-time kernels (e.g., weight preprocessing) from repeating kernels
      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, {k for k,v in buffers.items() if isinstance(v, Buffer) and v in set(input_buffers)})
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        # run one-time kernels immediately and discard them from the JIT cache
        for ei in (si.lower() for si in linear_to_schedule(onetime_linear)):
          for b in ei.bufs: cast(Buffer, b).ensure_allocated()
          ei.run(var_vals, jit=True)

      # compile the repeating kernels: memory plan, batch into graphs, lower to ExecItems
      held_bufs = set(buffers) | {t.uop.buf_uop for t in get_parameters(ret) if t.uop.buf_uop.op is Ops.BUFFER}
      with Context(BEAM=getenv("JITBEAM", BEAM.value)):
        jit_cache = jit_lower(big_linear, held_bufs, input_buffers)

      # track inputs that are views (slices) of other input buffers -- they need separate replacement entries
      # TODO: eventually expected_buffers should live in ExecItem
      extra_view_inputs: list[tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      # build the buffer replacement map so replay knows which slots to swap with new inputs
      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # run the captured program for the first time (this is the capture call's actual execution)
      for ei in jit_cache: ei.run(var_vals)

      # freeze everything into a CapturedJit for future replay
      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, expected_input_info)
    elif self.cnt >= 2:
      # === PHASE 2+: REPLAY === skip the function entirely, just swap buffers and re-run cached kernels
      assert self.captured is not None
      # validate that the caller's inputs match what was captured (same arg names, shapes, dtypes, devices)
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
