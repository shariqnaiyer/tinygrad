"""Kernel optimization primitives: the vocabulary of transformations that BEAM search and heuristics apply.

Each optimization is represented as an Opt(op, axis, arg) triple.  The optimizer builds a
Scheduler (see postrange.py) wrapping the kernel AST, then calls scheduler.apply_opt(Opt(...))
for each chosen action.  The Opt dataclass is intentionally tiny and serialisable so that
optimisation sequences can be cached to disk and replayed.

OptOps enum members and what they do to the kernel shape:
  TC          -- Apply a tensor-core (WMMA) pattern to the specified axis triple (N, M, K).
  UPCAST      -- Duplicate a GLOBAL/LOCAL/LOOP axis into an UPCAST axis (register-level unrolling of output dims).
  UNROLL      -- Duplicate a REDUCE/GROUP_REDUCE axis into an UNROLL axis (register-level unrolling of reduce dims).
  LOCAL       -- Split a GLOBAL/LOOP axis, moving the inner portion to LOCAL (shared-memory tiling).
  THREAD      -- Split a LOOP axis, moving the inner portion to THREAD (CPU thread parallelism).
  GROUP       -- Split a REDUCE axis bottom-up into GROUP_REDUCE (cooperative reduction across local threads).
  GROUPTOP    -- Like GROUP but splits top-down (outer portion becomes GROUP_REDUCE).
  NOLOCALS    -- Mark the kernel as not using local memory (forces all dims to GLOBAL on devices that support locals).
  PADTO       -- Pad an axis to a multiple of `arg`, masking out-of-bounds elements.
  SWAP        -- Swap the order of two GLOBAL axes (changes memory access pattern / coalescing).
"""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UNROLL = auto(); LOCAL = auto(); THREAD = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  """A single kernel optimization action.

  Attributes:
    op:   Which transformation to apply (see OptOps).
    axis: The logical axis index to transform (interpretation depends on op; None for NOLOCALS).
    arg:  Transformation parameter -- e.g. the split factor for UPCAST/LOCAL, or a (tc_select, tc_opt, use_tc)
          tuple for TC.  0 means "use the full axis size".
  """
  op: OptOps
  axis: int|None = None
  arg: int|tuple|None = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, arg={self.arg})"

class KernelOptError(Exception): pass
def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)
