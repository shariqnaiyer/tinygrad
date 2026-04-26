"""Linearizer: topological sort of UOps with priority ordering for code emission.

After all optimisation and lowering passes, the kernel is still a DAG of UOps.  The
linearizer flattens it into a sequential instruction list suitable for rendering to source
code.  The ordering matters for register pressure and cache locality.

linearize() implements a priority-aware topological sort:
  1. Assign each UOp a priority based on its op type and nesting depth (run_count):
       - PARAM/DEFINE_VAR/DEFINE_LOCAL go first (declarations).
       - LOAD is placed early (prefetch data).
       - STORE is placed late (write-back only when values are ready).
       - RANGE is placed late and END early, so that loop bodies stay compact.
  2. Sort UOps by (run_count, priority) to get the "ideal" ordering.
  3. Re-toposort using a min-heap that always pops the node closest to its ideal position,
     respecting data dependencies.

CFGContext + pm_add_control_flow handle ordering between sibling ranges that share a parent
scope but have cross-dependencies (e.g. a store in one range feeds into a load in the next).
It adds explicit control-flow edges so the linearizer respects the required order.

pm_split_ends breaks multi-range END nodes into individual per-range ENDs (one RANGE per END).
"""
import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize(sink:UOp) -> list[UOp]:
  """Flatten the UOp DAG into a priority-ordered instruction list for code emission."""
  # this is a toposort with priority
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    for s in u.src: out_degree[s] += 1

    # we place UOps with higher run_counts later
    run_count = prod([int(r.vmax)+1 for r in u.ranges])

    # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
    extra = None
    match u.op:
      # the order and placement of these defines is important
      case Ops.PARAM: priority, extra = -20, u.arg
      case Ops.DEFINE_VAR: priority, extra = -19, u.arg
      case Ops.DEFINE_LOCAL: priority = -18
      case Ops.DEFINE_REG: priority = -17
      case Ops.LOAD: priority = -1    # place loads early
      case Ops.STORE: priority = 1    # place stores late
      case Ops.RANGE: priority = 5    # placing RANGE is good
      case Ops.END: priority = -5     # placing END is bad
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, priority, extra)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ())))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in u.src:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  newlst = newlst[::-1]

  if getenv("DEBUG_LINEARIZE"):
    for i,u in enumerate(newlst):
      print(f"{i:4d} {str(u.op):20s} {multirange_str(u.ranges, color=True, pad=10)} {priorities[u]}")
  return newlst

class CFGContext:
  def __init__(self, sink:UOp):
    # there are 3 relationships between ranges:
    # nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
    # dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
    # independent, endrange y is not a dependency of endrange x
    # everything is nested inside the sink
    deps: dict[UOp, dict[UOp, None]] = {}
    nesting: dict[UOp, UOp] = {}
    for u in sink.toposort():
      # get the deps from the src
      deps[u] = {}
      for s in u.src: deps[u] |= deps[s]

      if u.op in (Ops.END, Ops.SINK):
        nesting |= {x:u for x in deps[u] if x.op is Ops.END and (u.op is Ops.SINK or u.src[1] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END): deps[u][u] = None

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # ranges that have dependencies on other siblings need to be scheduled after them
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[1]] + order, order)
      for x,y in zipped:
        # TODO: this can happen! it causes infinite loop in shufflenet
        assert y.src[1] not in x.backward_slice_with_self
        self.edges[y.src[1]] = x

pm_add_control_flow = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
])

def do_split_ends(e:UOp):
  ret = e.src[0]
  for r in sorted(UOp.sink(*e.src[1:]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])