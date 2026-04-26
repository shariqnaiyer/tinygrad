"""Transforms the lazy UOp tensor graph into CALL nodes suitable for JIT execution.

This module is the bridge between tinygrad's lazy tensor world and the scheduled kernel world. When a user
calls .realize() or .schedule(), the lazy UOp graph (which describes *what* to compute) passes through here
to be wrapped into executable CALL nodes (which describe *how* to run it). The pipeline works in four phases:

  1. **Tagging** (add_tags): Walk the tensor graph bottom-up, assigning integer tags to nodes that need their
     own buffer allocations (CONTIGUOUS, AFTER+STORE, COPY-from-disk, requested bases). Tags act as stable
     identifiers that survive subsequent graph rewrites.
  2. **Early transform** (pm_early_transform_tensor_graph): Restructure the graph while preserving tags --
     lower precompiled CALLs, collapse movement ops into BUFFER_VIEWs, convert tagged nodes into explicit
     STORE+AFTER pairs so every realized value has a concrete buffer target.
  3. **Finalize** (pm_finalize_call): Strip tags and build the buffer_map (original lazy UOp -> allocated
     buffer UOp), collecting the list of assignments that the final CALL must execute.
  4. **Parameterize** (pm_replace_buf): Replace concrete BUFFER / BUFFER_VIEW / BIND nodes with position-
     indexed PARAMs so the CALL body is device-agnostic and cacheable across invocations.

The public entry point is `transform_to_call`, which runs all four phases and returns the CALL UOp plus
the buffer_map needed to update live Tensor objects.
"""
from dataclasses import dataclass, field
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, GroupOp, graph_rewrite, track_rewrites
from tinygrad.helpers import VIZ, pluralize, all_int

@dataclass
class AllocCtx:
  """Mutable context threaded through every PatternMatcher phase to accumulate allocation state.

  This single object is the shared scratchpad for the entire callify pipeline.  Pattern callbacks
  mutate it as a side-effect while also returning rewritten UOps.

  Attributes:
    uop_list: Ordered registry of tagged UOps. The index becomes the tag integer, so
        ``uop_list[tag]`` recovers the original node after arbitrary graph rewrites.
    buffer_map: Maps original lazy UOps to their allocated buffer UOps. This is the
        "becomes map" returned to the caller so live Tensors can be updated.
    bases: Set of base UOps that the caller explicitly requested to be realized
        (i.e. the SINK sources). Membership here causes a node to receive a tag.
    assigns: Collected AFTER / COPY nodes that form the body of the final CALL -- every
        realized value appears here exactly once.
    replacements: Ordered list of concrete BUFFER/BIND UOps that were replaced by PARAMs.
        Position ``i`` in this list corresponds to PARAM index ``i``, and the list is
        passed to ``UOp.call()`` to bind arguments at execution time.
  """
  uop_list: list[UOp] = field(default_factory=list)
  buffer_map: dict[UOp, UOp] = field(default_factory=dict)
  bases: set[UOp] = field(default_factory=set)
  assigns: list[UOp] = field(default_factory=list)
  replacements: list[UOp] = field(default_factory=list)

def tag_uop(ctx:AllocCtx, x:UOp):
  """Assign a stable integer tag to a UOp that needs its own buffer allocation.

  Tags are indices into ``ctx.uop_list`` so later phases can recover the original node even after
  the graph has been heavily rewritten. Returns None (no rewrite) if the node is already tagged.
  """
  if x.tag is not None: return None
  ctx.uop_list.append(x)
  # tag is a tuple so multiple tags can be merged (e.g. when COPY is folded into AFTER)
  return x.replace(tag=(len(ctx.uop_list)-1,))

def disk_copy_is_buffer(ctx:AllocCtx, u:UOp):
  """Handle COPY ops that touch non-compute devices (DISK, TINYFS, NPY, PYTHON).

  Two distinct cases:
    - *To* disk/tinyfs: the copy's result is a disk-resident buffer, so we immediately
      register it in buffer_map (no kernel needed, just a host-side memcpy).
    - *From* a creation source (NPY, DISK, PYTHON, TINYFS): the data must be materialized
      into a real compute buffer, so we tag the COPY for later allocation.
  """
  # copies to disk are replaced with the disk buffer
  to_disk = isinstance(u._device, str) and u._device.startswith(("DISK", "TINYFS"))
  if to_disk: ctx.buffer_map[u] = UOp.new_buffer(u.device, u.shard_size, u.dtype).reshape(u.max_shard_shape)
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0]._device, str) and any(u.src[0]._device.startswith(x) for x in ["NPY", "DISK", "PYTHON", "TINYFS"])
  if from_creation: return tag_uop(ctx, u)

def apply_after(ctx:AllocCtx, u:UOp):
  """Register an AFTER node in the buffer_map, pointing through to the underlying buffer.

  AFTER wraps a buffer with ordering dependencies (its extra srcs are the STOREs it must wait for).
  We peel through any nested AFTERs to find the real buffer underneath, then record the mapping
  so that any Tensor referencing this AFTER will resolve to the concrete buffer.
  """
  base = u.src[0]
  while base.op is Ops.AFTER: base = base.src[0]
  ctx.buffer_map[u] = base

# CONTIGUOUS and AFTER+STORE + parents are the only nodes that get updated
# Phase 1 PatternMatcher: bottom-up walk that assigns integer tags to nodes needing buffers.
# This is "read-only" w.r.t. graph structure -- it only adds tags and populates buffer_map.
add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  # When a COPY result is immediately stored (AFTER(_, STORE(_, COPY))), the COPY doesn't need its own
  # allocation -- merge the COPY's tag into the AFTER so the buffer is allocated for the AFTER instead.
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(name="dest"), UPat(Ops.COPY, name="c")))), name="a"),
   lambda a,c,dest: a.replace(src=(a.src[0], a.src[1].replace(src=(dest, c.rtag(())))), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE)), name="x"), tag_uop),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat(Ops.CONTIGUOUS, name="x"), tag_uop),
  # tag any node the caller explicitly asked to realize (i.e. it was a SINK source)
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx.bases else None),
])

def _buffer_like(u:UOp) -> UOp:
  """Create a fresh buffer UOp with the same device/dtype/shape as ``u``.

  Handles MULTI (sharded) tensors by attaching the axis annotation so downstream
  code knows how the buffer is partitioned across devices.
  """
  buffer = UOp.new_buffer(u.device, u.shard_size, u.dtype).reshape(u.max_shard_shape).shrink_to(u.shard_shape)
  if isinstance(u.device, tuple) and u.axis is not None: buffer = buffer.multi(u.axis)
  return buffer

def replace_contig_with_store_after(u:UOp):
  """Lower a CONTIGUOUS node into an explicit BUFFER + STORE + AFTER triple.

  CONTIGUOUS is the lazy graph's way of saying "I need a real buffer here." This function
  allocates that buffer and wires up the STORE (write the value) and AFTER (ordering barrier),
  carrying the original tag forward so the finalize phase can trace back to the source.
  """
  # can't allocate a buffer without a device (e.g., inside a CALL function body with only PARAMs)
  if u._device is None: return None
  # if size is 0, remove the contig
  if u.size == 0: return u.src[0]
  # no real contig for DISK/TINYFS tensors, they are left alone
  if isinstance(u._device, str) and u._device.startswith(("DISK", "TINYFS")): return u.rtag(None)
  buf = _buffer_like(u)
  # buf.store(src) writes the value; buf.after(store) adds an ordering dependency so readers wait
  return buf.after(buf.store(u.src[0])).rtag(u.tag)

def replace_store_after_with_contig(u:UOp, src:UOp):
  """Undo AFTER+STORE when the store target is not a real BUFFER (e.g. a view or expression).

  If the AFTER's underlying base is already a BUFFER, the store is valid and we leave it. Otherwise
  the STORE was speculative, so we revert it back to a CONTIGUOUS and let a later pass handle allocation.
  """
  assigned_to = u
  # walk through BITCAST/AFTER wrappers to find the actual destination
  while assigned_to.op in {Ops.BITCAST, Ops.AFTER}: assigned_to = assigned_to.src[0].base
  if assigned_to.op is not Ops.BUFFER: return src.contiguous(tag=u.tag)

def _make_buffer_view(src:UOp) -> UOp|None:
  """If movement ops on src collapse to a contiguous range, return BUFFER_VIEW.reshape(src.shape). Otherwise None."""
  if (offset := src.contiguous_view_offset()) is None: return None
  buf = src.base
  if buf.op is Ops.BUFFER_VIEW: offset, buf = offset + buf.arg[1], buf.src[0]
  return UOp(Ops.BUFFER_VIEW, src.dtype, (buf,), (src.numel(), offset)).reshape(src.shape)

def contiguous_mops_to_view(c:UOp, src:UOp):
  """Optimize CONTIGUOUS(movement_ops(BUFFER)) into a zero-copy BUFFER_VIEW when possible.

  If a chain of movement ops (reshape, permute, shrink, etc.) on a buffer collapses to a
  contiguous byte range, we can represent it as a BUFFER_VIEW (offset + length) instead of
  allocating a new buffer and copying. This is the key optimization that avoids unnecessary
  memcpys for slices and reshapes.

  Falls back to None (no rewrite) when:
    - The base is not a buffer, or the device allocator lacks offset support.
    - The shape is symbolic (offsets can't be computed statically).
    - The movement ops don't collapse to a contiguous range.
  """
  buf = src.base
  if buf.op not in {Ops.BUFFER, Ops.BUFFER_VIEW}: return None
  # simple reshape of a buffer/view is already identity -- no optimization needed
  if src.op is Ops.RESHAPE and src.src[0].op in {Ops.BUFFER, Ops.BUFFER_VIEW}: return None

  # no symbolic shape
  if not all_int(c.shape): return None

  # check if view is supported -- the allocator must expose _offset for sub-buffer addressing
  from tinygrad.device import Device
  if isinstance(c.device, str):
    if not hasattr(Device[c.device].allocator, "_offset"): return None
  elif not all(hasattr(Device[d].allocator, "_offset") for d in c.device): return None

  # for MULTI tensors, use multi_pm to resolve per-shard movement ops, then create BUFFER_VIEW on the resolved result
  if not isinstance(c.device, str):
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(src, multi_pm, name="multi_buffer_view")
    if resolved.op is not Ops.MULTI: return None
    if (view := _make_buffer_view(resolved.src[0])) is None: return None
    return view.multi(resolved.arg).contiguous(tag=c.tag)

  # NOTE: this contiguous is removed because this BUFFER_VIEW/RESHAPE has_buffer_identity
  if (view := _make_buffer_view(src)) is None: return None
  return view.contiguous(tag=c.tag)

def transform_precompiled_call(c:UOp) -> UOp|None:
  """Expand a precompiled CALL into a fully wired CALL with explicit output buffers.

  Precompiled CALLs (e.g. custom_kernel) arrive with a TUPLE body listing the outputs but no
  output buffers. This function:
    1. Allocates a fresh buffer for each output.
    2. Rewrites the CALL body to STORE into those buffers via PARAMs.
    3. Wraps each output buffer in AFTER(buffer, new_call) so readers wait for execution.
    4. Returns a TUPLE of the realized outputs.
  """
  if not c.arg.precompile: return None
  if c.src[0].op is Ops.SINK: return None
  assert c.src[0].op is Ops.TUPLE, f"expected TUPLE body for precompiled call, got {c.src[0].op}"
  # ensure all input buffers are contiguous (materialized) before passing to the kernel
  input_buffers = tuple(x.contiguous() if x.op not in {Ops.AFTER, Ops.BIND} else x for x in c.src[1:])

  # add the outputs to the call -- one fresh buffer per TUPLE element
  srcs = c.src[0].src
  resolved = [c.gettuple(i) for i in range(len(srcs))]
  outs = tuple(_buffer_like(r) for r in resolved)
  # PARAMs inside the CALL body reference these output buffers by index (after the inputs)
  targets = [o.param_like(len(c.src)-1+i).shrink_to(s.shape) for i,(o,s) in enumerate(zip(outs, srcs))]
  fxn = UOp.sink(*[t.after(t.store(s)) for t,s in zip(targets, srcs)])

  # create the new thing for the big graph -- inputs + output buffers as CALL sources
  new_call = c.replace(src=(fxn, *input_buffers, *outs), tag=None)
  # each output buffer depends on the CALL completing before it can be read
  rets = tuple(o.after(new_call) for o in outs)

  # if the CALL has symbolic shapes, shrink the max-sized output to the actual symbolic shape
  # NOTE: must use resolved shapes from the CALL (which substitutes PARAMs with external args), not raw body shapes
  rets = tuple(r.shrink_to(rs.shape) for r,rs in zip(rets, resolved))

  return UOp.maketuple(*rets)

# Phase 2 PatternMatcher: restructure the tensor graph while maintaining tagged identifiers.
# IMPORTANT: adding rules here is bad -- they all run before the schedule cache, so new rules
# increase the uncached work on every schedule() call.
pm_early_transform_tensor_graph = PatternMatcher([
  # transform precompiled CALLs -- allocate output buffers and wire up STORE+AFTER
  (UPat(Ops.CALL, name="c"), transform_precompiled_call),

  # resolve TUPLE+GETTUPLE (for precompiled calls) -- simple constant-fold
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # CONTIGUOUS(MOPS(BUFFER/BUFFER_VIEW)) -> CONTIGUOUS(BUFFER_VIEW) when movement ops collapse to contiguous range
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="c"), contiguous_mops_to_view),

  # any tagged non-CONTIGUOUS/AFTER/STORE node needs a CONTIGUOUS wrapper so it gets a buffer later;
  # untagged nodes just get their tag cleared (they were tagged only for propagation purposes)
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.AFTER, Ops.STORE}, name="x"),
   lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag else x.replace(tag=None)),
  # if an AFTER already points at a contiguous buffer, the wrapping CONTIGUOUS is redundant -- absorb its tag
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.AFTER, name="a"),), name="c"),
   lambda a,c: a.replace(tag=(a.tag or ())+(c.tag or ())) if a.src[0].has_buffer_identity() else None),
  # replace AFTER+STORE with CONTIGUOUS when target is not a buffer
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat(name="src")))), name="u"), replace_store_after_with_contig),
  # final CONTIGUOUS lowering: allocate a buffer and emit STORE+AFTER
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_store_after),
  # DETACH/CONTIGUOUS_BACKWARD are autograd artifacts with no runtime meaning -- strip them
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
])

def untag_and_append(ctx:AllocCtx, x:UOp):
  """Strip tags from AFTER nodes and finalize the buffer_map entries they represent.

  For each tag on the node, look up the original UOp from phase 1 and record the rewritten
  buffer (peeling through AFTER wrappers) in buffer_map. This is how the lazy Tensor world
  learns which concrete buffer it should point to after scheduling.
  """
  if x.tag is None: return None
  ret = x.replace(tag=None)
  for t in x.tag:
    original_uop: UOp = ctx.uop_list[t]
    # peel through AFTERs to find the concrete buffer that was allocated
    replace_uop = ret
    while replace_uop.op is Ops.AFTER: replace_uop = replace_uop.src[0]
    ctx.buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  if ret.op is not Ops.AFTER: ctx.assigns.append(ret)  # AFTER gets appended by append_after
  return ret

def append_after(ctx:AllocCtx, x:UOp):
  """Collect AFTER nodes into the assigns list so they appear in the final CALL body."""
  ctx.assigns.append(x)

def replace_input_buffer(ctx:AllocCtx, b:UOp):
  """Replace a concrete BUFFER/BUFFER_VIEW/BIND with a positional PARAM for cache-key normalization.

  The CALL body must be device- and allocation-agnostic so that identical computation graphs
  (differing only in which physical buffers they use) hit the same schedule cache entry. PARAMs
  are positional placeholders -- the actual buffers are supplied at call time via ``UOp.call()``.
  """
  ctx.replacements.append(b)
  return UOp.param(len(ctx.replacements)-1, b.dtype, b.shape, b._device,
                   b._min_max if b.op is Ops.BIND else None, b.src[0].arg[0] if b.op is Ops.BIND else None)

# Phase 3 PatternMatcher: strip tags, build buffer_map, and collect the assigns list.
pm_finalize_call = PatternMatcher([
  # untag_and_append runs first for tagged AFTERs (registers buffer_map + collects assigns)
  (UPat(Ops.AFTER, name="x"), untag_and_append),
  # untagged AFTERs still need to appear in the assigns list
  (UPat(Ops.AFTER, name="x"), append_after),
  # disk/tinyfs COPYs are collected as assigns even though they have no AFTER wrapper
  (UPat(Ops.COPY, name="x"), lambda ctx,x: append_after(ctx,x) if isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS")) else None),
  # remove unique from const. TODO: this is copied in function.py
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE, name="d")), name="b"), lambda b,d: b.replace(src=(d,))),
])

# Phase 4 PatternMatcher: replace concrete identities with positional PARAMs for cacheability.
# Runs bottom-up so that unreferenced BUFFERs (e.g. BUFFER_VIEW's parent) are naturally excluded.
pm_replace_buf = PatternMatcher([
  # replace BUFFER with PARAM for cache key normalization
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # replace BUFFER_VIEW with PARAM. this rewrite is bottom up so BUFFERs we don't need won't be in the input
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.BUFFER),), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), replace_input_buffer),
])

@track_rewrites(lambda _,ret: f"Callify {pluralize('Buffer', len(ret[1]))}")
def transform_to_call(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  """Public entry point: transform a lazy tensor SINK into a callable CALL UOp.

  This orchestrates the four-phase pipeline described in the module docstring.

  Args:
    big_sink: A SINK UOp whose sources are the lazy tensors the caller wants to realize.

  Returns:
    A tuple of:
      - The CALL UOp ready for scheduling / JIT execution.
      - A buffer_map dict (original lazy UOp -> concrete buffer UOp) so callers can
        update live Tensor objects to point at their newly-allocated buffers.
  """
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  # Phase 1 setup: identify which SINK sources need realization (skip consts, existing buffers, vars, etc.)
  dont_realize = {Ops.CONST, Ops.BUFFER, Ops.BIND, Ops.DEFINE_VAR, Ops.AFTER}
  ctx = AllocCtx(bases=set([x.multibase for x in big_sink.src if x.base.op not in dont_realize]))

  # Phase 1: bottom-up tag assignment. "read-only" -- only adds tags and populates buffer_map via side-effects.
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=ctx, bottom_up=True, name="number the uops")

  # Phase 2: structural rewrites (CONTIGUOUS lowering, BUFFER_VIEW optimization, precompiled CALL expansion).
  # Tags must be maintained through these rewrites so phase 3 can trace back to originals.
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, name="early transform tensor graph")

  # Phase 3: finalize -- strip tags, populate buffer_map, collect the assigns list.
  graph_rewrite(big_sink, pm_finalize_call, ctx=ctx, name="finalize call")
  # Phase 4: replace concrete BUFFER/BIND with PARAMs, then wrap in a CALL with the actual buffers as args.
  ret = graph_rewrite(UOp.sink(*ctx.assigns), pm_replace_buf, ctx=ctx, bottom_up=True, name="replace bufs").call(*ctx.replacements)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret, ctx.buffer_map
