# tinygrad Public API Reference

> Consolidated reference for the tinygrad public API with usage examples.
>
> Last updated: 2025-04-15

---

## Package Exports

The main `tinygrad` package exports these symbols:

```python
from tinygrad import Tensor      # The tensor class
from tinygrad import TinyJit     # JIT compilation decorator
from tinygrad import function    # @function decorator for graph capture
from tinygrad import dtypes      # Data type constants
from tinygrad import Device      # Device singleton
from tinygrad import UOp         # Micro-operation IR node
from tinygrad import Variable    # Symbolic variable (UOp.variable)
from tinygrad import GlobalCounters  # Kernel count, FLOP tracking
from tinygrad import fetch       # URL/file fetching utility
from tinygrad import Context     # Context manager for ContextVars
from tinygrad import getenv      # Environment variable access
from tinygrad import nn          # Neural network subpackage
```

---

## Tensor

The primary class for all computation. Wraps a lazy UOp computation graph.

### Creating Tensors

```python
from tinygrad import Tensor, dtypes

# From Python data
t = Tensor([1, 2, 3])
t = Tensor([[1.0, 2.0], [3.0, 4.0]])

# From numpy
import numpy as np
t = Tensor(np.array([1, 2, 3]))

# Factory methods
Tensor.zeros(3, 3)                          # 3x3 zeros
Tensor.ones(2, 4)                           # 2x4 ones
Tensor.full((2, 3), fill_value=7.0)         # 2x3 filled with 7.0
Tensor.eye(3)                               # 3x3 identity
Tensor.arange(0, 10, step=2)               # [0, 2, 4, 6, 8]
Tensor.rand(4, 4)                           # uniform [0, 1)
Tensor.randn(4, 4)                          # normal(0, 1)
Tensor.uniform(3, 3, low=-1, high=1)        # uniform [-1, 1)
Tensor.kaiming_uniform(128, 64)             # Kaiming initialization
Tensor.empty(4, 4)                          # uninitialized

# With specific dtype and device
Tensor.zeros(3, 3, dtype=dtypes.float16, device="CUDA")
```

For full list see [Tensor Creation](tensor/creation.md).

### Properties

```python
t = Tensor.rand(3, 4, 5)

t.shape        # (3, 4, 5)
t.dtype        # dtypes.float32
t.device       # "METAL" (or whatever your default is)
t.requires_grad  # None, True, or False
t.numel()      # 60
t.ndim         # 3
```

### Realizing Tensors

All operations are lazy. Use these to trigger execution:

```python
t = Tensor.rand(3, 3) + 1

t.realize()    # Execute computation, keep on device
t.numpy()      # Execute and copy to numpy array
t.item()       # Execute and return Python scalar (for 0-d tensors)
t.tolist()     # Execute and return Python list
```

### Elementwise Operations

```python
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

# Arithmetic
a + b          # [5, 7, 9]
a - b          # [-3, -3, -3]
a * b          # [4, 10, 18]
a / b          # [0.25, 0.4, 0.5]
a ** 2         # [1, 4, 9]

# Unary
a.neg()        # [-1, -2, -3]
a.log()        # natural log
a.exp()        # exponential
a.sqrt()       # square root
a.abs()        # absolute value
a.reciprocal() # 1/x

# Activations
a.relu()
a.sigmoid()
a.tanh()
a.leaky_relu(neg_slope=0.01)
a.gelu()
a.silu()       # aka swish
a.softmax()
a.log_softmax()

# Comparison
a < b          # element-wise boolean
a == b
a != b
a.where(b, c)  # conditional select
a.maximum(b)
a.minimum(b)
a.clip(0, 1)   # clamp to range

# Casting
a.float()       # to float32
a.half()        # to float16
a.int()         # to int32
a.cast(dtypes.bfloat16)
```

For full list see [Tensor Elementwise](tensor/elementwise.md).

### Reduction Operations

```python
t = Tensor.rand(3, 4)

t.sum()              # scalar sum of all elements
t.sum(axis=0)        # sum along axis 0 -> shape (4,)
t.sum(axis=(0, 1))   # sum along multiple axes
t.prod()             # product
t.max()              # maximum
t.min()              # minimum
t.mean()             # mean
t.var()              # variance
t.std()              # standard deviation
t.argmax(axis=-1)    # index of max along axis
t.argmin(axis=-1)    # index of min along axis
```

### Movement / Shape Operations

```python
t = Tensor.rand(2, 3, 4)

t.reshape(6, 4)       # change shape (same number of elements)
t.view(6, 4)          # same as reshape
t.flatten()            # flatten to 1D
t.flatten(1)           # flatten from dim 1 onward -> (2, 12)
t.permute(2, 0, 1)    # reorder dimensions -> (4, 2, 3)
t.transpose(0, 2)     # swap two dims -> (4, 3, 2)
t.T                    # transpose last two dims
t.unsqueeze(0)         # add dim at position 0 -> (1, 2, 3, 4)
t.squeeze()            # remove dims of size 1
t.expand(2, 3, 8)      # broadcast (must be compatible)
t.repeat(2, 1, 1)      # tile along dims -> (4, 3, 4)
t.pad(((0,0),(1,1),(0,0)))  # pad with zeros
t.shrink(((0,1),(0,2),(0,4)))  # slice/crop
t.flip(0)              # reverse along axis
t.contiguous()         # ensure contiguous memory layout

# Slicing (Python-style)
t[0]                   # first element along dim 0
t[:, 1:3]              # slice dims
t[..., :2]             # ellipsis for remaining dims
```

For full list see [Tensor Movement](tensor/movement.md).

### Complex Operations

```python
# Matrix operations
a = Tensor.rand(4, 8)
b = Tensor.rand(8, 4)
c = a @ b              # matrix multiply -> (4, 4)
c = a.matmul(b)        # same as @
c = a.dot(b)           # same as @

# Convolution
x = Tensor.rand(1, 3, 32, 32)   # (batch, channels, height, width)
w = Tensor.rand(16, 3, 3, 3)    # (out_channels, in_channels, kH, kW)
y = x.conv2d(w, stride=1, padding=1)  # -> (1, 16, 32, 32)

# Concatenation and stacking
a, b = Tensor.rand(3, 4), Tensor.rand(3, 4)
Tensor.cat(a, b, dim=0)    # -> (6, 4)
Tensor.stack([a, b], dim=0)  # -> (2, 3, 4)

# Loss functions
logits = Tensor.rand(4, 10)
labels = Tensor([2, 4, 3, 7])
loss = logits.sparse_categorical_crossentropy(labels)
loss = logits.cross_entropy(labels)
```

For full list see [Tensor Ops](tensor/ops.md).

### Autograd

```python
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad.numpy())  # [2.0, 4.0, 6.0]
```

### Multi-Device

```python
from tinygrad import Tensor

# Shard a tensor across devices
t = Tensor.rand(8, 8)
sharded = t.shard(("CUDA:0", "CUDA:1"), axis=0)
# Each device gets half the rows

# Move between devices
t_gpu = t.to("CUDA")
t_cpu = t_gpu.to("CPU")
```

---

## TinyJit

JIT compiler that records and replays kernel execution.

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def fast_fn(x):
  return (x @ x).relu().realize()

# First call: records kernels
result = fast_fn(Tensor.rand(64, 64))

# Subsequent calls: replays kernels (much faster)
result = fast_fn(Tensor.rand(64, 64))
```

**Requirements:**
- Decorated function must call `.realize()` on outputs
- Input shapes must be the same across calls (or use symbolic shapes)
- Function should be pure (no side effects)

**JIT modes** (set via `JIT` env var):
- `JIT=0` — disabled
- `JIT=1` — enabled with graph batching (default)
- `JIT=2` — enabled without graph batching

---

## Device

Singleton for device management.

```python
from tinygrad import Device

Device.DEFAULT          # Current default device (e.g., "METAL")
Device["CUDA"]          # Get specific device instance
Device["CUDA:1"]        # Get specific GPU

# List available devices
list(Device.get_available_devices())

# Change default via context
from tinygrad import Context
with Context(DEV="CPU"):
  t = Tensor.rand(3, 3)  # created on CPU
```

---

## dtypes

Data type constants.

```python
from tinygrad import dtypes

# Float types
dtypes.float64    # 64-bit float
dtypes.float32    # 32-bit float (default)
dtypes.float16    # 16-bit float (half)
dtypes.bfloat16   # brain float 16
dtypes.float      # alias for float32

# Integer types
dtypes.int64
dtypes.int32
dtypes.int16
dtypes.int8
dtypes.uint64
dtypes.uint32
dtypes.uint16
dtypes.uint8

# Other
dtypes.bool       # boolean
dtypes.default_float  # configurable default (DEFAULT_FLOAT env var)
```

---

## Neural Networks (nn)

### Layers

```python
from tinygrad import nn

nn.Linear(in_features, out_features, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.Conv1d(in_channels, out_channels, kernel_size, ...)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, ...)
nn.BatchNorm(num_features, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1)
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
nn.InstanceNorm(num_features, eps=1e-5, affine=True)
nn.Embedding(vocab_size, embed_size)
nn.LSTMCell(input_size, hidden_size, bias=True)
```

### Defining Models

tinygrad has no `nn.Module`. Models are just classes with `__call__`:

```python
class MyModel:
  def __init__(self):
    self.l1 = nn.Linear(784, 128)
    self.l2 = nn.Linear(128, 10)

  def __call__(self, x):
    return self.l2(self.l1(x).relu())
```

### Optimizers

SGD, Adam, and AdamW are factory functions (not classes) that return configured optimizer instances. LAMB and LARS are classes.

```python
from tinygrad.nn.optim import SGD, Adam, AdamW, LAMB, LARS

opt = Adam(nn.state.get_parameters(model), lr=0.001)

with Tensor.train():
  opt.zero_grad()
  loss = model(x).sparse_categorical_crossentropy(y).backward()
  opt.step()
```

### State Management

```python
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, get_parameters

# Get all parameters
params = get_parameters(model)

# Save model
state_dict = get_state_dict(model)
safe_save(state_dict, "model.safetensors")

# Load model
state_dict = safe_load("model.safetensors")
load_state_dict(model, state_dict)
```

---

## Context

Context manager for temporarily setting ContextVars.

```python
from tinygrad import Context

# As context manager
with Context(DEBUG=4, BEAM=2):
  result = model(x).realize()

# As decorator
@Context(DEBUG=0)
def quiet_fn():
  return Tensor.rand(3, 3).realize()
```

---

## GlobalCounters

Tracks execution statistics.

```python
from tinygrad import GlobalCounters

GlobalCounters.reset()
result = model(x).realize()

print(GlobalCounters.kernel_count)  # number of kernels executed
print(GlobalCounters.global_ops)    # total FLOPs
print(GlobalCounters.global_mem)    # total bytes accessed
print(GlobalCounters.time_sum_s)    # total execution time
print(GlobalCounters.mem_used)      # current memory usage
```
