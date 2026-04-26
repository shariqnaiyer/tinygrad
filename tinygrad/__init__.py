"""tinygrad: a tiny deep learning framework with a built-in compiler.

This is the public API surface of tinygrad. Everything a user needs is re-exported here:
  - Tensor: the core lazy tensor class (all computation goes through this)
  - TinyJit: JIT decorator that records + replays kernel execution for speed
  - function: decorator that captures a computation graph for differentiation/precompilation
  - UOp / Variable: the intermediate representation and symbolic variables (advanced/internal)
  - dtypes: data type constants (float32, float16, int32, bool, etc.)
  - Device: singleton for device management (Device.DEFAULT, Device["CUDA"], etc.)
  - GlobalCounters: tracks kernel count, FLOPs, memory, and timing across all operations
  - fetch: utility for downloading files/URLs with caching
  - Context: context manager for temporarily overriding ContextVars (DEBUG, BEAM, etc.)
  - getenv: typed environment variable access with defaults

Usage:
  from tinygrad import Tensor, dtypes, Device
  x = Tensor.rand(3, 3)
  print((x @ x).numpy())
"""
import os
# TYPED=1 enables runtime type checking via typeguard — useful for catching type errors during development
# but adds overhead, so it's off by default
if int(os.getenv("TYPED", "0")):
  from typeguard import install_import_hook
  install_import_hook(__name__)
from tinygrad.tensor import Tensor                                    # noqa: F401
from tinygrad.engine.jit import TinyJit                               # noqa: F401
from tinygrad.function import function                                # noqa: F401
from tinygrad.uop.ops import UOp
Variable = UOp.variable  # convenience alias: Variable is just UOp.variable (creates symbolic shape variables)
from tinygrad.dtype import dtypes                                     # noqa: F401
from tinygrad.helpers import GlobalCounters, fetch, Context, getenv   # noqa: F401
from tinygrad.device import Device                                    # noqa: F401
