"""Implements the @function decorator that captures a computation graph inside a function body.

The decorator traces a user-written function (which builds Tensor operations) into a UOp graph,
extracts PARAM nodes for learnable parameters and other inputs, supports optional precompilation
and caching of forward/backward passes, and wraps everything in a CALL UOp.  This is the
functional-style alternative to defining models as nn.Module-like classes: you write a plain
Python function, decorate it with @function, and tinygrad lifts the body into a reusable,
compilable graph fragment.

Key ideas:
  - **Graph capture**: the decorated function runs with device access disabled (ALLOW_DEVICE_USAGE=0)
    so that tensor operations build a lazy UOp graph instead of executing immediately.
  - **Parameter extraction**: every Tensor/UOp argument is deduplicated and turned into a PARAM
    placeholder via ``UOp.param_like``.  BUFFERs, BINDs, and certain CONTIGUOUS nodes that remain
    in the graph after substitution are discovered as *implicit* inputs by ``pm_ctx``.
  - **UNIQUE -> LUNIQUE rewrite**: constants tagged with ``Ops.UNIQUE`` (which carry a global
    identity) are rewritten to ``Ops.LUNIQUE`` so each call site can receive its own numbering.
  - **CALL node**: the final graph is wrapped in ``UOp.call(...)`` which bundles the body, all
    explicit + implicit inputs, and compilation options into a single CALL UOp.
"""
import functools, itertools, time
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv, DEBUG
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

def add_to_ctx(ctx, x:UOp):
  """Register an implicit input discovered during the graph rewrite pass.

  Called by ``pm_ctx`` when a BUFFER, BIND, or qualifying CONTIGUOUS/AFTER node is found in the
  traced graph that was *not* among the explicit arguments.  The node is converted to a PARAM
  placeholder (assigned the next available slot) and the original UOp is appended to the running
  list so it can be passed as an argument to the CALL node later.

  Args:
    ctx: A two-element tuple ``(call_uops_list, lunique_counter)``.  ``call_uops_list`` accumulates
         every discovered input; its current length determines the PARAM slot index.
    x: The UOp node to replace with a PARAM placeholder.

  Returns:
    A new PARAM UOp that takes the slot ``len(ctx[0])`` at the time of the call.
  """
  # slot index == current length *before* we append, giving each input a unique position
  ret = x.param_like(len(ctx[0]))
  ctx[0].append(x)
  return ret

# Ops.UNIQUE nodes carry a globally-unique identity so that two independently created constants
# with the same value are still distinct in the graph.  Inside a @function body we need a *local*
# numbering (LUNIQUE) so the captured graph can be instantiated multiple times without colliding.
pm_transform_unique_const = PatternMatcher([
  # transform unique consts to LUNIQUE
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="x"),
   lambda ctx,x: x.replace(src=(UOp(Ops.LUNIQUE, arg=next(ctx[1])), x.src[1]))),
])

# Pattern matcher that discovers implicit inputs and normalizes UNIQUE->LUNIQUE.
# Rule 1: any BUFFER or BIND node becomes an implicit parameter (always captured).
# Rule 2: AFTER/CONTIGUOUS nodes are captured only when they reference a real BUFFER
#         but are NOT already derived from a PARAM (i.e. not already an explicit arg).
# The LUNIQUE rewrite from pm_transform_unique_const is merged in so a single bottom-up
# pass handles both jobs.
pm_ctx = PatternMatcher([
  (UPat((Ops.BUFFER, Ops.BIND), name="x"), add_to_ctx),
  (UPat((Ops.AFTER, Ops.CONTIGUOUS), name="x"),
   lambda ctx,x: add_to_ctx(ctx,x) if not x.op_in_backward_slice_with_self(Ops.PARAM) and x.op_in_backward_slice_with_self(Ops.BUFFER) else None),
])+pm_transform_unique_const

ReturnType = TypeVar('ReturnType')
class _function(Generic[ReturnType]):
  """Callable wrapper produced by the ``@function`` decorator.

  ``_function`` is the object that replaces the original function after decoration.  When called
  it traces the wrapped function to build a UOp computation graph, replaces every input Tensor
  with a PARAM placeholder, discovers any implicit inputs (e.g. model weights that were closed
  over rather than passed as arguments), and emits a single ``Ops.CALL`` UOp that can later be
  scheduled and compiled.

  The class tracks ``depth`` as a class variable so nested ``@function`` calls can be detected
  (important for debugging output indentation and future nested-call support).

  Relationship to the Tensor/UOp system:
    - Each Tensor carries a ``.uop`` that is the lazy graph node representing its value.
    - ``_function.__call__`` runs the user code, collects the resulting UOps, rewrites them into
      a self-contained graph with PARAM inputs, and wraps the result in ``UOp.call()``.
    - The CALL UOp is what the scheduler later lowers into device kernels.
  """
  depth = 0
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool, precompile_backward:bool, allow_implicit:bool, grad_fxn:Callable|None):
    """Initialize the function wrapper with tracing and compilation options.

    Args:
      fxn: The user-defined function to be traced.
      precompile: If True, eagerly compile the forward-pass kernels when the CALL is created.
      precompile_backward: If True, eagerly compile the backward-pass kernels as well.
      allow_implicit: If False, raise an error when the function body references BUFFERs that
          were not passed as explicit arguments (e.g. closed-over weight tensors).
      grad_fxn: Optional custom gradient function.  When provided it overrides the default
          autograd behavior for the backward pass of this CALL node.
    """
    self.fxn = fxn
    self.precompile = precompile
    self.precompile_backward = precompile_backward
    self.allow_implicit = allow_implicit
    self.grad_fxn = grad_fxn

  def __get__(self, obj, objtype=None):
    """Descriptor protocol: bind ``self`` as a bound method when accessed on an instance.

    This lets ``@function`` work on class methods -- when ``obj.method()`` is called, Python's
    descriptor protocol invokes ``__get__`` which returns a partial with ``obj`` pre-filled as
    the first argument, mirroring normal bound-method semantics.
    """
    return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    """Trace the wrapped function and return the result as Tensor(s) backed by a CALL UOp.

    The call proceeds in four phases:

    1. **Collect explicit inputs** -- walk ``args``/``kwargs`` with ``get_state_dict`` to find
       every Tensor and UOp, then deduplicate them.  Each unique input gets a slot index.
    2. **Trace the function body** -- execute ``self.fxn`` with device access disabled so all
       tensor operations build a lazy graph instead of running on hardware.
    3. **Substitute explicit inputs** -- replace every known input UOp in the traced graph with
       a PARAM placeholder carrying the corresponding slot index.
    4. **Discover implicit inputs** -- run ``pm_ctx`` over the remaining graph to find leftover
       BUFFERs/BINDs and convert them to additional PARAMs.  If ``allow_implicit`` is False and
       any implicit BUFFERs are found, raise an error.

    Finally the rewritten graph is wrapped in ``UOp.call()`` to produce a CALL node, and the
    result is re-wrapped in Tensor(s) for the caller.

    Args:
      *args: Positional arguments forwarded to the wrapped function.
      **kwargs: Keyword arguments forwarded to the wrapped function.

    Returns:
      A single Tensor or a tuple of Tensors whose underlying UOps are CALL outputs.

    Raises:
      RuntimeError: If the function returns a non-Tensor type, or if implicit buffers are
          detected and ``allow_implicit`` is False.
    """
    st = time.perf_counter()

    # Phase 1: extract all Tensor/UOp leaves from args and kwargs using the state-dict walker
    params = get_state_dict((args, kwargs), tensor_type=(Tensor, UOp)).values()

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup([(t.uop if isinstance(t, Tensor) else t) for t in params])

    # Phase 2: trace the function body -- ALLOW_DEVICE_USAGE=0 prevents real device calls,
    # so every Tensor operation just extends the lazy UOp graph.
    # The DEVICE_IN_FUNCTION_BUG env var exists as an escape hatch for debugging.
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      _function.depth += 1
      ret = self.fxn(*args, **kwargs)
      _function.depth -= 1

    # Normalize the return value to a single UOp (wrapping tuples in a TUPLE node)
    if isinstance(ret, Tensor):
      uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
      uret = UOp.maketuple(*[x.uop for x in ret])
    else:
      raise RuntimeError(f"function return type {type(ret)} not supported")

    # Phase 3: replace known (explicit) input UOps with PARAM placeholders.
    # Each PARAM gets the same slot index as the deduped position in call_uops.
    subs = {}
    for i,x in enumerate(call_uops): subs[x] = x.param_like(i)
    uret = uret.substitute(subs)

    # add contiguous to call_uops
    #call_uops = [x.contiguous() for x in call_uops]

    # Phase 4: discover implicit inputs -- BUFFERs that were closed over rather than passed in.
    # pm_ctx walks bottom-up, converting each remaining BUFFER/BIND/qualifying CONTIGUOUS into a
    # new PARAM and appending the original UOp to call_uops.  The LUNIQUE counter (itertools.count)
    # is shared across the whole rewrite so each UNIQUE const gets a distinct local id.
    num_explicit = len(call_uops)
    uret = graph_rewrite(uret, pm_ctx, (call_uops, itertools.count(0)), bottom_up=True, name="get_implicit_inputs")
    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__
    # Guard: if the caller opted out of implicit captures, surface them as an explicit error
    if not self.allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op is Ops.BUFFER]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.size}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    # assign output
    #pbuffer = uret.param_like(len(call_uops))
    #assigned = pbuffer.assign(uret).sink()
    #buffer = UOp.new_buffer(pbuffer.device, pbuffer.size, pbuffer.dtype).reshape(uret.shape)
    #call = assigned.call(*call_uops, buffer, name=name)
    #ret = buffer.after(call)

    # Build the CALL UOp: bundles the rewritten graph body + all input UOps + compilation flags
    fret = uret.call(*call_uops, grad_fxn=self.grad_fxn, name=name, precompile=self.precompile,
                     precompile_backward=self.precompile_backward)

    if DEBUG >= 2:
      #signature = [(x._shape, x.dtype, x._device) for x in call_uops]
      print("  "*_function.depth+f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}") # with sig {signature}")

    # Unwrap the CALL result back into user-facing Tensor(s) by pulling each element from the TUPLE
    if isinstance(ret, tuple):
      return cast(ReturnType, tuple(Tensor(fret.gettuple(i)) for i in range(len(ret))))
    else:
      return cast(ReturnType, Tensor(fret.gettuple(0)))

# overload signatures support both @function and @function(precompile=True) syntax
@overload
def function(fxn:Callable[..., ReturnType], *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> _function[ReturnType]: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> Callable[[Callable[..., ReturnType]], _function[ReturnType]]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None):
  """Decorator that lifts a plain Python function into a captured UOp computation graph.

  Supports two calling conventions so it works as both a bare decorator and a decorator factory::

      @function                              # bare -- fxn is the decorated callable
      def forward(x: Tensor) -> Tensor: ...

      @function(precompile=True)             # factory -- returns a decorator, fxn is None here
      def forward(x: Tensor) -> Tensor: ...

  When called, the resulting ``_function`` object traces the body, extracts PARAM nodes for every
  input, optionally discovers implicit (closed-over) buffers, and emits a CALL UOp that the
  scheduler can later compile into device kernels.

  Args:
    fxn: The function to decorate, or None when using the ``@function(...)`` factory form.
    precompile: Eagerly compile forward-pass kernels at trace time rather than deferring to
        the scheduler.
    precompile_backward: Eagerly compile backward-pass kernels as well.
    allow_implicit: Allow the function body to reference BUFFERs that were not passed as
        explicit arguments (e.g. model weights captured via closure).  When False (default),
        any such buffers cause a RuntimeError.
    grad_fxn: Optional custom gradient function that overrides default autograd for this call.

  Returns:
    A ``_function`` wrapper (if ``fxn`` is provided), or a single-argument decorator that
    produces one (if ``fxn`` is None).
  """
  # When used as @function(precompile=True), fxn is None -- return a decorator that will
  # receive the actual function on the next call.
  if fxn is None:
    return lambda f: _function(f, precompile=precompile, precompile_backward=precompile_backward,
                               allow_implicit=allow_implicit, grad_fxn=grad_fxn)
  # When used as bare @function, fxn is already the decorated callable
  return _function(fxn, precompile=precompile, precompile_backward=precompile_backward,
                   allow_implicit=allow_implicit, grad_fxn=grad_fxn)
