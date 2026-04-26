"""Automatic differentiation (backpropagation) for tinygrad's UOp graph.

This module constructs the backward pass by walking the forward UOp DAG in reverse
topological order and applying the chain rule at each node.  Gradient rules for
element-wise, reduction, and shape operations are expressed as PatternMatcher
entries in ``pm_gradient``.  Multi-output CALL nodes (compiled sub-graphs) get
their own backward CALL via ``call_gradient``.

Key concepts:
  - ``ctx`` always refers to the upstream gradient (grad_output) flowing into a node.
  - Each rule returns a tuple of gradients, one per source operand, or ``None``
    for operands that do not need a gradient (e.g. the predicate of WHERE).
  - ``compute_gradient`` is the top-level driver: it walks the graph, dispatches
    to ``pm_gradient`` or ``call_gradient``, and accumulates gradients with ``+``.
"""
from typing import cast
import math, dataclasses, itertools
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata, graph_rewrite
from tinygrad.helpers import argsort
from tinygrad.dtype import sum_acc_dtype

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  """Compute the gradient of a REDUCE_AXIS node (sum, max, or product reduction).

  A reduction collapses one or more axes, so the gradient must be broadcast back
  to the original (pre-reduction) shape before being applied.

  Args:
    ctx: Upstream gradient (has the reduced shape).
    ret: The forward REDUCE_AXIS UOp whose gradient we are computing.
    op: The reduction operation (ADD for sum, MAX, or MUL for product).

  Returns:
    A 1-tuple containing the gradient w.r.t. the reduction's input tensor.
  """
  # Restore reduced dims so the gradient can be broadcast back to the input shape.
  def broadcast_to_input(x): return x.reshape(x.shape+(1,)*(len(ret.src[0].shape)-len(x.shape))).expand(ret.src[0].shape)
  # d/dx sum(x) = 1, so just broadcast the upstream gradient unchanged.
  if op == Ops.ADD: return (broadcast_to_input(ctx),)
  if op == Ops.MAX:
    # d/dx max(x): gradient flows only to elements equal to the max.
    # When multiple elements tie for max, the gradient is split equally among them.
    assert ret.op is Ops.REDUCE_AXIS, "only works on REDUCE_AXIS"
    mask = ret.src[0].eq(broadcast_to_input(ret)).cast(ctx.dtype)
    count = mask._rop(Ops.ADD, ret.arg[1])  # count ties along the reduced axes
    return ((mask/broadcast_to_input(count)) * broadcast_to_input(ctx),)
  # d/dx prod(x) = prod(x) / x_i  for each element x_i  (via the product rule).
  if op == Ops.MUL: return (broadcast_to_input(ctx * ret) / ret.src[0],)

def _compact_params(body:UOp, all_args:tuple[UOp, ...]) -> tuple[UOp, tuple[UOp, ...]]:
  """Remove unused PARAMs from body and renumber the survivors contiguously.

  The backward body may reference only a subset of the forward CALL's
  arguments.  This function strips the unused PARAMs so the backward CALL
  receives the minimal set of arguments, reducing unnecessary data movement.

  Args:
    body: The UOp sub-graph representing the backward computation.
    all_args: The full argument tuple (forward args + grad args + optional forward outputs).

  Returns:
    A (compacted_body, compacted_args) tuple where PARAM slots are renumbered
    0..N-1 and only the actually-referenced args are kept.
  """
  used = sorted({p.arg: p for p in body.toposort() if p.op is Ops.PARAM}.items())
  return body.substitute({p: p.replace(arg=j) for j,(_, p) in enumerate(used)}, walk=True), tuple(all_args[i] for i,_ in used)

def call_gradient(ctx:UOp, k:UOp, needed:set[int]) -> tuple[UOp|None, ...]:
  """Construct the backward pass for a CALL (compiled sub-graph) node.

  There are two paths:
    1. **Custom grad_fxn** -- if the CALL was created via ``Tensor.custom_kernel``
       with an explicit ``grad_fxn``, delegate directly to it.
    2. **Automatic** -- recursively call ``compute_gradient`` on the CALL's body,
       then package the per-parameter gradients into a single backward CALL.

  Args:
    ctx: Upstream gradient (may be a TUPLE for multi-output CALLs).
    k: The forward CALL UOp.  ``k.src[0]`` is the body (a TUPLE of outputs),
       ``k.src[1:]`` are the call arguments.
    needed: Indices (into k.src[1:]) of arguments whose gradients are actually
            required, so we can skip unnecessary backward computation.

  Returns:
    Gradient tuple aligned with ``k.src`` (None for the body slot, then one
    entry per argument -- UOp gradient or None if not needed/not differentiable).
  """
  fxn, args = k.src[0], k.src[1:]
  # --- Path 1: user-supplied gradient function (e.g. custom flash-attention kernels) ---
  if k.arg.grad_fxn is not None:
    if ctx.op is Ops.TUPLE:
      real = [g for g in ctx.src if g.op is not Ops.NOOP]  # filter out unused outputs
      return (None,) + (k.arg.grad_fxn(*real, call=k) if len(real) > 1 else k.arg.grad_fxn(real[0], k))
    return (None,) + k.arg.grad_fxn(ctx, k)
  # --- Path 2: automatic differentiation through the CALL body ---
  assert fxn.op is Ops.TUPLE, f"expected TUPLE body for gradient, got {fxn.op}"
  # Map each PARAM slot to its UOp so we know which params to differentiate w.r.t.
  params = {x.arg:x for x in fxn.toposort(enter_calls=False) if x.op == Ops.PARAM}
  # Build symbolic root_grad: each output's upstream gradient becomes a PARAM in the backward body.
  # NOOP entries mark outputs whose gradients are not needed (dead ends).
  grad_args = ctx.src
  root_grad = UOp(Ops.TUPLE, src=tuple(UOp(Ops.NOOP) if g.op is Ops.NOOP else g.param_like(len(args)+i) for i,g in enumerate(grad_args)))
  grads = compute_gradient(fxn, root_grad, set(params.values()))
  # For precompiled calls, substitute forward outputs with params so intermediates aren't recomputed
  # (the forward outputs are already materialized and can be passed in as extra args).
  fwd_subs = {src: src.param_like(len(args)+len(grad_args)+i) for i, src in enumerate(fxn.src)} if k.arg.precompile else {}
  fwd_outs = tuple(k.gettuple(i) for i in range(len(fxn.src))) if k.arg.precompile else ()
  # Collect only the gradient sub-graphs we actually need, compact unused params, and emit a single backward CALL.
  grad_bodies = [(i, grads[p]) for i in needed if (p:=params.get(i)) is not None and p in grads]
  bwd_body = UOp.maketuple(*(gb for _, gb in grad_bodies)).substitute(fwd_subs, walk=True)
  bwd_body, compact_args = _compact_params(bwd_body, (*args, *grad_args, *fwd_outs))
  # TODO: is this okay here?
  from tinygrad.function import pm_transform_unique_const
  bwd_body = graph_rewrite(bwd_body, pm_transform_unique_const, ctx=(None, itertools.count(0)))
  bwd_call = bwd_body.call(*compact_args, name=(k.arg.name or "")+"_backward", precompile=k.arg.precompile_backward)
  # Map original arg indices to their position in the backward CALL's output tuple.
  gb_map = {i: idx for idx, (i, _) in enumerate(grad_bodies)}
  return (None,) + tuple(bwd_call.gettuple(gb_map[i]) if i in gb_map else None for i in range(len(args)))

# ------------------------------------------------------------------
# Gradient rules table (PatternMatcher).
#
# Each entry maps a forward-pass UOp pattern to a lambda that returns
# the gradient(s) w.r.t. that UOp's source operands.
#
# Convention:
#   ctx  = upstream gradient (grad_output) -- always provided by the framework.
#   ret  = the forward UOp that produced the value being differentiated.
#   None = "no gradient for this operand" (e.g. non-differentiable inputs).
# ------------------------------------------------------------------
pm_gradient = PatternMatcher([
  # --- Unary element-wise ops ---
  # d/dx cast(x) = cast(ctx): just pass the gradient through with the input dtype.
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  # d/dx (1/x) = -1/x^2.  Since ret = 1/x, this simplifies to -ctx * ret^2.
  (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  # d/dx sin(x) = cos(x) * ctx.  cos(x) = sin(pi/2 - x).
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  # d/dx log2(x) = 1 / (x * ln(2)).
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  # d/dx 2^x = 2^x * ln(2).  Since ret = 2^x, this is ret * ln(2) * ctx.
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  # d/dx sqrt(x) = 1 / (2*sqrt(x)).  Since ret = sqrt(x), this is ctx / (2*ret).
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  # Comparisons are non-differentiable (step functions); gradient is None for both operands.
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),

  # --- Binary element-wise ops ---
  # d/dx (a + b): gradient passes through to both operands unchanged.
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  # d/db b^e = e * b^(e-1);  d/de b^e = b^e * ln(b).  Special-cased for b==0 to avoid NaN.
  (UPat(Ops.POW, name="ret", src=(UPat.var("b"), UPat.var("e"))), lambda ctx, ret, b, e:
    (ctx * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), ctx * b.eq(0).where((e<0).where(ret.const_like(-math.inf), 0), ret*b.log2()*math.log(2.0)))),
  # d/dx max(x, y): gradient goes to the larger input; ties split 50/50.
  (UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)), (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)))),
  # d/dx (a * b) = (b * ctx, a * ctx)  -- standard product rule.
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  # d/dx where(p, a, b): gradient flows to a where p is true, to b otherwise.  Predicate p gets None.
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),

  # --- Reductions (sum / max / prod over axes) ---
  (UPat(Ops.REDUCE_AXIS, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0])),

  # --- Shape / memory layout ops (these just rearrange data, so the gradient is the inverse rearrangement) ---
  # contiguous: identity for gradients.
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  # contiguous_backward: force the *gradient* to be contiguous (used to control memory layout in backward).
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  # reshape: reverse the reshape to recover the input shape.
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  # expand: the inverse of broadcast is summation over the broadcast axes (to collect duplicated gradients).
  (UPat(Ops.EXPAND, name="ret"), lambda ctx, ret:
    (ctx.cast(sum_acc_dtype(ctx.dtype))._rop(Ops.ADD, tuple(i for i,(s,n) in enumerate(zip(ret.src[0].shape, ret.shape)) if s!=n))
     .cast(ctx.dtype), None)),
  # pad: inverse is shrink -- discard the padding region's gradients.
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  # shrink: inverse is pad -- zero-fill the regions that were cropped away.
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  # permute: inverse permutation undoes the axis reordering.
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  # flip: flipping is its own inverse.
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),

  # --- Device / multi-device ops ---
  # copy: move the gradient back to the source device.
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device), None)),
  # multi (sharded tensor): shard the gradient the same way the forward tensor was sharded.
  (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),

  # --- Structural ops ---
  # tuple: unpack the gradient tuple into per-element gradients.
  (UPat(Ops.TUPLE), lambda ctx: ctx.src),
  # NOTE: this is only correct when the KERNEL has a single output
  (UPat(Ops.AFTER), lambda ctx: (ctx, ctx)),
  # there's no gradient for bitcast
  (UPat(Ops.BITCAST), lambda: (None,)),
])

def _deepwalk(root:UOp, targets:set[UOp]) -> tuple[list[UOp], dict[UOp, bool]]:
  """Find every node on a path from ``root`` down to any target, in topological order.

  This is the "reachability" phase of backprop: we only need to compute
  gradients for nodes that actually influence a target parameter.  Nodes
  behind a DETACH op are excluded so that ``Tensor.detach()`` correctly
  stops gradient flow.

  Args:
    root: The loss / output UOp at the top of the graph.
    targets: The set of UOps (typically PARAMs) we want gradients for.

  Returns:
    A (walk, in_target_path) tuple where ``walk`` is the topologically-sorted
    list of gradient-relevant nodes (excluding DETACH) and ``in_target_path``
    maps every visited UOp to whether it lies on a path to a target.
  """
  # Top-down pass: mark each node True if any of its children reach a target.
  in_target_path: dict[UOp, bool] = {}
  root.topovisit(lambda u: any(in_target_path[x] or x in targets for x in u.src), in_target_path)
  # Exclude DETACH nodes (gradient-stopping barriers) and nodes not on any target path.
  return [node for node in in_target_path if node.op is not Ops.DETACH and in_target_path[node]], in_target_path

def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
  """Walk the forward graph in reverse topological order and accumulate gradients via the chain rule.

  This is the core backpropagation driver.  It processes every node on the
  path from ``root`` to ``targets``, applies the matching gradient rule
  (from ``pm_gradient`` or ``call_gradient``), and sums contributions when
  multiple paths converge on the same node.

  Args:
    root: The forward-pass output UOp (e.g. the scalar loss).
    root_grad: The initial gradient seeding ``root`` (typically 1.0 for a scalar loss).
    targets: The set of UOps whose gradients we ultimately want (e.g. model parameters).

  Returns:
    A dict mapping each reachable UOp to its accumulated gradient UOp.
    Only UOps on a path to a target will have entries.
  """
  walk, in_target_path = _deepwalk(root, targets)
  grads: dict[UOp, UOp] = {root: root_grad}
  # Iterate in reverse topological order (from outputs toward inputs) so every
  # node's upstream gradient is fully accumulated before we propagate through it.
  for t0 in reversed(walk):
    if t0 not in grads or grads[t0].op is Ops.NOOP: continue  # skip nodes with no gradient (dead paths)
    # GETTUPLE: a multi-output CALL's individual outputs are accessed via GETTUPLE.
    # Instead of immediately propagating, accumulate each output's gradient into a
    # TUPLE on the parent CALL so we can process the CALL once with all output grads.
    if t0.op is Ops.GETTUPLE:
      k = t0.src[0]  # the CALL
      assert k.op is Ops.CALL and k.src[0].op is Ops.TUPLE
      n_outputs = len(k.src[0].src)
      prev = grads[k].src if k in grads else tuple(UOp(Ops.NOOP) for _ in range(n_outputs))
      grads[k] = UOp.maketuple(*(prev[i] + grads[t0] if i == t0.arg and prev[i].op is not Ops.NOOP else
                                 grads[t0] if i == t0.arg else prev[i] for i in range(n_outputs)))
      continue
    # CALL: compute backward for a compiled sub-graph, passing which params actually need gradients.
    if t0.op is Ops.CALL:
      needed = {i for i, arg in enumerate(t0.src[1:]) if arg in targets or in_target_path.get(arg, False)}
      lgrads:tuple[UOp|None, ...]|None = call_gradient(grads[t0], t0, needed)
    else:
      # Element-wise / shape ops: look up the gradient rule in the PatternMatcher table.
      lgrads = cast(tuple[UOp|None, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    # Distribute the computed local gradients to each input operand, accumulating
    # (summing) when a node receives gradients from multiple consumers.
    for k,v in zip(t0.src, lgrads):
      if v is None: continue  # non-differentiable operand (e.g. WHERE predicate)
      if k in grads and grads[k].op is not Ops.NOOP: grads[k] = grads[k] + v  # multi-path accumulation (sum)
      else: grads[k] = v
      # Propagate forward metadata (e.g. layer names) to the new backward UOps
      # so profiling / debugging tools can associate backward ops with their forward origin.
      if len(forward_metadata:=all_metadata.get(t0, ())):
        backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        # we add the backward metadata to everything new in the graph
        for bw_uop in v.toposort(lambda x: x not in (t0, *t0.src, grads[t0])):
          all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata
  return grads
