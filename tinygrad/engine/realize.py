"""Execution engine: takes a schedule of kernels/copies and runs them on hardware.

This is the final step in tinygrad's lazy-evaluation pipeline. The scheduler
(engine/schedule.py) figures out *what* to compute; the codegen turns that
into device code; and this module actually launches those programs, copies
buffers between devices, and records performance statistics.

Key abstractions:
  Runner          -- base class for anything that can be .exec()'d on buffers.
  CompiledRunner  -- wraps a compiled ProgramSpec and launches it via the device runtime.
  BufferCopy      -- host-mediated copy between two devices.
  BufferXfer      -- peer-to-peer transfer when both devices share an allocator.
  ExecItem        -- a single scheduled unit: an AST + its buffer arguments + a Runner.
  run_schedule    -- top-level entry point that drains a list of ExecItems.
"""
from typing import cast, Callable
import time, pprint, random, itertools, math
from dataclasses import dataclass, replace, field
from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, NOOPT, all_int, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import DEVECTORIZE, time_to_str, VALIDATE_WITH_CPU, cpu_profile, PROFILE, ProfilePointEvent, cpu_events, prod, unwrap
from tinygrad.helpers import EMULATED_DTYPES
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer
from tinygrad.device import Device, Buffer
from tinygrad.renderer import ProgramSpec, Estimates
from tinygrad.codegen import get_program

# **************** Stat ****************

def update_stats(display_name:str, device:str, estimates:Estimates, var_vals:dict[str, int], et:float|None, buf_count:int,
                 jit=False, metadata:tuple[Metadata, ...]=(), first_run=False):
  """Record execution stats in GlobalCounters and optionally print a DEBUG>=2 summary line.

  This is called after every kernel launch and buffer copy so that cumulative
  FLOP / memory / time numbers stay accurate. The printed line (when DEBUG>=2)
  is the primary way to profile tinygrad from the command line.

  Args:
    display_name: Human-readable kernel name shown in the debug line.
    device: Device string (e.g. "METAL", "CUDA:0") for the header column.
    estimates: Estimated FLOPs, load/store bytes, and unique memory bytes.
    var_vals: Concrete values for any symbolic variables in the estimates.
    et: Wall-clock execution time in seconds, or None if timing was skipped.
    buf_count: Number of buffer arguments (printed as "arg N").
    jit: True when replaying from the JIT cache (header colored magenta).
    metadata: Tensor-level operation metadata for tracing.
    first_run: True on the very first invocation of this Runner (header colored green).
  """
  # --- accumulate into the global counters that power .numpy() timing info ---
  GlobalCounters.kernel_count += 1
  GlobalCounters.global_ops += (op_est:=sym_infer(estimates.ops, var_vals))
  GlobalCounters.global_mem += (mem_est:=sym_infer(estimates.mem, var_vals))
  if et is not None: GlobalCounters.time_sum_s += et
  if DEBUG >= 2:
    lds_est = sym_infer(estimates.lds, var_vals)
    header_color = 'magenta' if jit else ('green' if first_run else None)
    ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
    # derive throughput rates; guard against et==0 with 1e-20 to avoid division by zero
    flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
    flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
    mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
      colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
    print(f"{colored(f'*** {device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
      f" {display_name+' '*(46-ansilen(display_name))} arg {buf_count:2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
      ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
      f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in metadata] if metadata else ''}")

# **************** Runners ****************

class Runner:
  """Base class for anything that can execute on a list of Buffers.

  Subclasses implement __call__ to perform the actual work (kernel launch,
  buffer copy, etc.). The Runner also carries metadata used by update_stats
  (display_name, device, estimates) and tracks whether this is the first
  invocation (for coloring the DEBUG line green on first run).
  """
  def __init__(self, display_name:str, device:str, estimates=Estimates()):
    self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None) -> float|None:
    """Convenience wrapper that defaults var_vals to {} and forwards to __call__."""
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    raise NotImplementedError("override this")

def optimize_local_size(_prg:Callable, global_size:list[int], rawbufs:list[Buffer]) -> list[int]:
  """Brute-force search for the fastest local (workgroup) size on this device.

  Enumerates all valid power-of-two local sizes (up to MAX_WORKGROUP=1024),
  runs each candidate twice with wall-clock timing, and returns the fastest.
  This is a fallback used only when the renderer says it needs a local size
  but the compiler didn't pick one (local_size is None).

  Args:
    _prg: The compiled runtime program callable.
    global_size: The GPU grid dimensions before dividing by local size.
    rawbufs: Buffers to pass to the kernel; buf[0] may be aliased for safety.

  Returns:
    The local_size list that yielded the lowest execution time.
  """
  # if the output buffer appears in the inputs, allocate a scratch buffer so timing runs don't corrupt real data
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  # build candidate local sizes: every power-of-two up to MAX_WORKGROUP that fits each dimension
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try:
      return _prg(*[x._buf for x in test_rawbuffers],global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)],
                  local_size=local_size, wait=True)
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

class CompiledRunner(Runner):
  """Runner backed by a compiled GPU/CPU kernel (ProgramSpec).

  The constructor compiles the source (if not already cached), loads the
  binary into the device runtime, and stores the callable as self._prg.
  __call__ then launches the kernel with the correct grid/block dimensions.
  """
  def __init__(self, p:ProgramSpec, prg=None):
    if DEBUG >= 3 and p.applied_opts: print(p.applied_opts)
    if DEBUG >= 4: print(p.src)
    # compile source -> binary if we don't already have the lib bytes
    if p.lib is None:
      with cpu_profile(TracingKey(f"compile {p.name}", (p.function_name,)), "TINY"):
        p = replace(p, lib=Device[p.device].compiler.compile_cached(p.src))
    self.p:ProgramSpec = p
    assert self.p.lib is not None
    if DEBUG >= 7: Device[p.device].compiler.disassemble(self.p.lib)
    # instantiate the device-specific runtime handle (e.g. a Metal or CUDA program object)
    self._prg = Device[p.device].runtime(p.function_name, self.p.lib, *p.aux, runtimevars=p.runtimevars) if prg is None else prg
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p,)  # pickle support: re-compile from ProgramSpec on load

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None, wait=False, timeout:int|None=None) -> float|None:
    """Launch the compiled kernel on its device.

    Resolves symbolic grid dimensions, auto-tunes the local size if needed,
    and dispatches via the device runtime.

    Args:
      rawbufs: Concrete Buffers matching the kernel's argument list.
      var_vals: Concrete values for symbolic variables (e.g. batch size).
      wait: If True, synchronize and return wall-clock execution time.
      timeout: Optional timeout in ms forwarded to the runtime.

    Returns:
      Wall-clock seconds if wait=True, else None.
    """
    if var_vals is None: var_vals = {}
    global_size, local_size = self.p.launch_dims(var_vals)
    # auto-tune local size on first launch when the renderer needs one but codegen didn't choose
    if Device[self.p.device].renderer.has_local and local_size is None and all_int(self.p.global_size):
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)  # cache the tuned sizes for future launches
    # dispatch: pass raw device handles, resolved grid dims, and concrete variable values
    return self._prg(*[x._buf for x in rawbufs], global_size=tuple(global_size), local_size=tuple(local_size) if local_size else None,
                     vals=tuple(var_vals[k.expr] if k.expr not in self.p.runtimevars else None for k in self.p.vars), wait=wait, timeout=timeout)

class ViewOp(Runner):
  """No-op runner for BUFFER_VIEW: asserts that a view's base pointer matches the source buffer.

  Buffer views share underlying memory (base + offset), so there is nothing
  to execute -- this just validates the aliasing invariant at runtime.
  """
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  """Host-mediated copy between two device buffers (possibly on different devices).

  The default path reads source bytes into a CPU memoryview and writes them
  to the destination allocator. Specialized fast paths exist for DISK sources
  (readinto / copy_from_disk) to avoid a full CPU-side allocation.
  """
  def __init__(self, total_sz, dest_device, src_device):
    sz = f"{total_sz/1e6:7.2f}M" if total_sz >= 1e6 else f"{total_sz:8d}"
    name = f"{type(self).__name__[6:].lower()} {sz}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    # fastest path: DMA directly from a DISK fd into a device buffer (requires page-aligned size >= 4096)
    disk_supports_fast_copyout = src.device.startswith("DISK") and getattr(src.allocator.dev, 'fd', None) is not None
    if disk_supports_fast_copyout and hasattr(dest.allocator, 'copy_from_disk') and src.nbytes >= 4096 and dest.allocator.supports_copy_from_disk:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif isinstance(src.device, str) and src.device.startswith(("DISK", "TINYFS")) and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path: readinto the dest's backing buffer directly, avoiding an extra CPU allocation
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      # generic fallback: materialize source as a memoryview on CPU, then copyin to dest device
      dest.copyin(src.as_memoryview(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    """Execute the copy and optionally time it by synchronizing the destination device."""
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()  # ensure the copy is complete before measuring wall-clock time
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  """Peer-to-peer device transfer (e.g. GPU-to-GPU without staging through the host).

  Only used when both devices share an allocator that advertises
  supports_transfer=True. Falls back to BufferCopy otherwise.
  """
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

class EncDec(Runner):
  """Runner for hardware HEVC encode/decode operations (Ops.CUSTOM_FUNCTION "encdec").

  Used by video-oriented backends that expose on-device codec hardware.
  """
  def __init__(self, cf:UOp, total_sz:int, device:str):
    self.shape, self.pos_var = tuple(s.arg for s in cf.src if s.op is Ops.CONST), cf.variables()[0].expr
    name = f"enc/dec {total_sz/1e6:7.2f}M, HEVC" if total_sz >= 1e6 else f"enc/dec {total_sz:8d}, HEVC"
    super().__init__(colored(name, "yellow"), device, Estimates(lds=total_sz, mem=total_sz))
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    st = time.perf_counter()
    rawbufs[0].allocator._encode_decode(rawbufs[0]._buf, rawbufs[1]._buf, rawbufs[2]._buf,
                                        [x._buf for x in rawbufs[3:]], self.shape, var_vals[self.pos_var])
    if wait:
      Device[rawbufs[0].device].synchronize()
      return time.perf_counter() - st

# **************** method cache ****************

method_cache: dict[tuple[str, type, bytes, tuple, bool], CompiledRunner] = {}  # avoids re-compiling the same AST for the same device+context
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  """Look up or compile a CompiledRunner for the given AST and device.

  Uses a two-level cache keyed by (device, compiler type, AST, context flags):
    - ckey: exact device match (e.g. "CUDA:0")
    - bkey: base-device match (e.g. "CUDA"), so the same binary can be shared
      across device indices that use the same compiler.

  Args:
    device: Target device string (e.g. "METAL", "CUDA:1").
    ast: The Ops.SINK / Ops.PROGRAM / Ops.BEAM root UOp to compile.

  Returns:
    A ready-to-call CompiledRunner.
  """
  # TODO: this should be all context relevant to rendering
  context = (NOOPT.value, DEVECTORIZE.value, EMULATED_DTYPES.value)
  # first try an exact-device hit (fast path for repeated launches on the same device)
  ckey = (device, type(Device[device].compiler), ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  # then try a base-device hit: same binary, just stamp a new device string on it
  bkey = (device.split(":")[0], type(Device[device].compiler), ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device))
  else:
    # cold miss: full codegen + compile
    prg: ProgramSpec = get_program(ast, Device[device].renderer)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

# Pattern-match the schedule AST root op to create the right Runner subclass.
# NOTE: ctx is the list of Buffers associated with this ExecItem.
si_lowerer = PatternMatcher([
  # compiled kernel: SINK / PROGRAM / BEAM all go through get_runner -> CompiledRunner
  (UPat((Ops.SINK, Ops.PROGRAM, Ops.BEAM), name="sink"), lambda ctx,sink: get_runner(ctx[0].device, sink)),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: ViewOp(ctx[0])),
  # COPY: prefer peer-to-peer xfer if the allocator supports it, else fall back to host-staged copy
  (UPat(Ops.COPY), lambda ctx: (BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(alc:=Device[ctx[0].device].allocator, '_transfer') and alc.supports_transfer and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device))),
  (UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="cf"), lambda ctx,cf: EncDec(cf, ctx[0].nbytes, ctx[0].device)),
  # graph: delegate to the device's graph executor (e.g. CUDA graphs)
  (UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="cf"), lambda ctx,cf: Device[cf.device if isinstance(cf.device,str) else cf.device[0]].graph(cf, ctx))
])

@dataclass
class ExecItem:
  """A single scheduled execution unit: an AST paired with its buffer arguments.

  The lifecycle is:  schedule -> ExecItem(ast, bufs) -> .lower() -> .run()
    lower(): pattern-matches the AST root to produce the right Runner subclass.
    run():   allocates buffers (if needed), launches the Runner, and records stats.

  Attributes:
    ast: The root UOp for this scheduled operation (SINK, COPY, BUFFER_VIEW, etc.).
    bufs: Buffers that this operation reads/writes (may contain None for not-yet-allocated).
    metadata: Tensor-level tracing info (source file, op name) for profiling.
    fixedvars: Symbolic variable values baked in at schedule time.
    prg: The Runner created by lower(); None until lower() is called.
  """
  ast: UOp
  bufs: list[Buffer|None] = field(default_factory=list)
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  prg: Runner|None = None

  def lower(self):
    """Populate self.prg by pattern-matching the AST to a Runner subclass via si_lowerer."""
    if self.prg is not None: return self  # already lowered (e.g. from JIT cache)
    try: self.prg = cast(Runner, si_lowerer.rewrite(self.ast, self.bufs))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {self.ast.op}")
        print("tensor operations:")
        pprint.pprint(self.metadata, indent=2)
      raise e
    return self

  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    """Lower (if needed), allocate buffers, launch the kernel, and record stats.

    Args:
      _var_vals: Runtime symbolic variable values; merged with self.fixedvars.
      wait: If True, synchronize and return wall-clock time.
      jit: If True, skip ensure_allocated (the JIT guarantees buffers are ready).
      do_update_stats: If False, suppress the GlobalCounters / DEBUG output update.

    Returns:
      Wall-clock seconds if wait=True (or DEBUG>=2), else None.
    """
    if self.prg is None: self.lower()
    assert self.prg is not None
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    # reorder bufs to match the kernel's declared argument order (p.globals)
    _bufs = [self.bufs[i] for i in self.prg.p.globals] if isinstance(self.prg, CompiledRunner) else self.bufs
    # JIT path: buffers are pre-allocated; normal path: lazily allocate now
    bufs = [unwrap(x) for x in _bufs] if jit else [unwrap(x).ensure_allocated() for x in _bufs]
    if PROFILE:
      payload = {"metadata":self.metadata, "var_vals":var_vals, "bufs":[b.trace_num for b in bufs], "name":self.prg.display_name}
      payload["outputs"], payload["inputs"] = (self.prg.p.outs, self.prg.p.ins) if isinstance(self.prg, CompiledRunner) else ([0], [1])
      cpu_events.append(ProfilePointEvent(self.prg.device, "exec", len(cpu_events), payload))
    # actually launch the kernel / copy / etc. -- DEBUG>=2 forces wait so we can report timing
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      update_stats(self.prg.display_name, self.prg.device, self.prg.estimates, var_vals, et, len(bufs), jit, self.metadata, self.prg.first_run)
      self.prg.first_run = False
    return et

# **************** main run function ****************

capturing: list = []  # hooks for graph capture (e.g. CUDA graphs): classes with an add_linear method

def run_schedule(schedule:list[ExecItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  """Drain a schedule list by lowering and executing each ExecItem in order.

  This is the main entry point called from Tensor.realize() after the
  scheduler has produced a list of ExecItems. Each item is popped, lowered
  to a Runner, and executed.

  When VALIDATE_WITH_CPU is set, every SINK kernel is additionally re-run
  on the CPU and the outputs are compared with numpy allclose -- useful for
  debugging correctness on new backends.

  Args:
    schedule: Mutable list of ExecItems; will be drained (emptied) in place.
    var_vals: Symbolic variable values shared across all items in the schedule.
    do_update_stats: Forwarded to ExecItem.run() to control stat recording.
  """
  while len(schedule):
    ei = schedule.pop(0).lower()
    sink = ei.ast.src[0] if ei.ast.op is Ops.BEAM else ei.ast  # unwrap BEAM wrapper to get the actual SINK
    if VALIDATE_WITH_CPU and sink.op is Ops.SINK:
      # --- VALIDATE_WITH_CPU path: run on both GPU and CPU, then compare outputs ---
      # snapshot GPU buffer contents into CPU mirrors before the kernel mutates them
      bufs = [b for b in ei.bufs if b is not None]
      nb: list[Buffer|None] = [Buffer("CPU", b.size, b.dtype) for b in bufs]
      for cpu_b, gpu_b in zip(nb, bufs):
        if cpu_b is not None and gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_memoryview())

      # run on GPU (the real execution)
      ei.run(var_vals, do_update_stats=do_update_stats)

      # re-run the same kernel on CPU with the mirrored buffers
      # NOTE: this assumes the output is buffer 0 -- true for all current SINK kernels
      ExecItem(sink, nb, ei.metadata, ei.fixedvars).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      assert nb[0] is not None
      np.testing.assert_allclose(bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      ei.run(var_vals, do_update_stats=do_update_stats)
