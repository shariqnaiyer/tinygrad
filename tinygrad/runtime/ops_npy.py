"""NumPy-backed pseudo-device for data loading. Not a real compute backend -- it has no renderer or program runner.
Buffers are numpy uint8 arrays; the allocator supports copyout (GPU->host) but not compute. This device exists so
that numpy data can be loaded into tinygrad's buffer system and then transferred to an actual compute device.

Key classes:
  NpyDevice    -- Compiled device with an empty renderer list and no program runner.
  NpyAllocator -- Allocator that wraps numpy arrays and exposes them as contiguous memoryviews.
"""
import numpy as np
from tinygrad.helpers import flat_mv
from tinygrad.device import Compiled, Allocator

class NpyAllocator(Allocator['NpyDevice']):
  def _alloc(self, size:int, options=None) -> np.ndarray: return np.empty(size, dtype=np.uint8)
  def _as_buffer(self, src:np.ndarray) -> memoryview: return flat_mv(np.require(src, requirements='C').data)
  def _copyout(self, dest:memoryview, src:np.ndarray): dest[:] = self._as_buffer(src)

class NpyDevice(Compiled):
  """NumPy pseudo-device for data ingestion. Has no renderer or program runner -- exists solely to wrap numpy arrays
  as tinygrad buffers so they can be transferred to a real compute device."""
  def __init__(self, device:str): super().__init__(device, NpyAllocator(self), [], None)
