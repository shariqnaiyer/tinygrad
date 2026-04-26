# tinygrad Troubleshooting

> Common errors, debugging workflow, and solutions.
>
> Last updated: 2025-04-15

---

## Debugging Workflow

### Step 1: Increase DEBUG Level

```bash
# See what device is being used
DEBUG=1 python3 script.py

# See kernel timing and memory usage
DEBUG=2 python3 script.py

# See applied optimizations
DEBUG=3 python3 script.py

# See generated kernel source code
DEBUG=4 python3 script.py

# See UOp intermediate representation
DEBUG=5 python3 script.py
```

### Step 2: Check Device

```python
from tinygrad import Device
print(Device.DEFAULT)  # What device is tinygrad using?
print(list(Device.get_available_devices()))  # What's available?
```

### Step 3: Validate Results

```bash
# Compare GPU results against CPU
VALIDATE_WITH_CPU=1 python3 script.py

# Enable out-of-bounds checking
CHECK_OOB=1 python3 script.py

# Enable strict UOp validation
SPEC=2 python3 script.py
```

### Step 4: Visualize

```bash
# Launch computation graph visualizer
VIZ=1 python3 script.py
```

---

## Common Errors

### "no usable devices"

```
RuntimeError: no usable devices
```

**Cause**: tinygrad couldn't find any supported hardware backend.

**Solutions**:
- Verify you have a supported device: `python3 -c "from tinygrad import Device; print(list(Device.get_available_devices()))"`
- On macOS: Metal should auto-detect. If not, check macOS version.
- On Linux with NVIDIA: ensure CUDA toolkit or NVIDIA drivers are installed
- On Linux with AMD: ensure ROCm or amdgpu driver is present
- Force CPU: `DEV=CPU python3 script.py`

### "usage of device X disallowed"

```
AssertionError: usage of device CUDA disallowed
```

**Cause**: `ALLOW_DEVICE_USAGE=0` is set (common in test environments).

**Solution**: Unset or set `ALLOW_DEVICE_USAGE=1`.

### Shape Mismatch Errors

```
ValueError: inhomogeneous shape from ...
```

**Cause**: Creating a Tensor from a ragged (non-rectangular) Python list.

**Solution**: Ensure all sublists have the same length:
```python
# Bad
Tensor([[1, 2], [3, 4, 5]])

# Good
Tensor([[1, 2, 0], [3, 4, 5]])
```

### JIT Shape Mismatch

```
RuntimeError: JIT shape mismatch
```

**Cause**: Calling a `@TinyJit` function with different input shapes than the first call.

**Solution**: Use the same shapes across calls, or use symbolic shapes:
```python
@TinyJit
def fn(x):
  return (x + 1).realize()

fn(Tensor.rand(64, 64))   # First call records shape
fn(Tensor.rand(64, 64))   # OK: same shape
fn(Tensor.rand(32, 32))   # Error: different shape!
```

### CUDA/GPU Out of Memory

**Symptoms**: CUDA OOM, Metal allocation failure, or similar device memory errors.

**Solutions**:
- Reduce batch size
- Use `float16` to halve memory: `DEFAULT_FLOAT=HALF python3 script.py`
- Set memory limit: `MAX_BUFFER_SIZE=300000000 python3 script.py` (300MB)
- Ensure tensors are garbage collected (avoid holding references to intermediate results)

### Numerical Precision Issues

**Symptoms**: Results differ from PyTorch/numpy, NaN/inf values.

**Debugging**:
```bash
# Use float64 for maximum precision
DEFAULT_FLOAT=FLOAT64 python3 script.py

# Validate against CPU
VALIDATE_WITH_CPU=1 python3 script.py

# Disable TF32 (can cause precision loss on Ampere+)
ALLOW_TF32=0 python3 script.py
```

**Common causes**:
- `float16` operations losing precision — try `float32`
- `TF32` tensor cores (19-bit mantissa instead of 23-bit) — set `ALLOW_TF32=0`
- Reduction order differences between devices — expected within tolerance

### Slow Performance

**Debugging**:
```bash
# Check kernel count and timing
DEBUG=2 python3 script.py

# Are kernels being fused? Look for many small kernels
# Use BEAM search for better kernel optimization
BEAM=2 python3 script.py

# Make sure JIT is enabled (default)
JIT=1 python3 script.py
```

**Common causes**:
- JIT disabled or not warming up (first run is always slow)
- Too many small kernels (poor fusion) — check DEBUG=2 output
- Wrong device (running on CPU when GPU is available)
- BEAM=0 (heuristic optimization may not be optimal for your workload)

### Compilation Errors

**CUDA compilation failed**:
- Check `CUDA_PATH` points to valid CUDA installation
- Verify NVRTC is available: `python3 -c "import ctypes; ctypes.CDLL('libnvrtc.so')"`

**Metal compilation failed**:
- Ensure running on macOS with Metal-capable hardware
- Check macOS version is supported

**C compilation failed**:
- Ensure clang or gcc is installed and in PATH
- On macOS: `xcode-select --install`

### Import Errors

```
ModuleNotFoundError: No module named 'tinygrad'
```

**Solution**: Install tinygrad:
```bash
cd /path/to/tinygrad
python3 -m pip install -e .
```

Or set PYTHONPATH:
```bash
PYTHONPATH=/path/to/tinygrad python3 script.py
```

---

## Performance Tips

1. **Use TinyJit**: Wrap hot paths in `@TinyJit` — first call is slow, subsequent calls replay kernels
2. **Increase BEAM**: `BEAM=2` or higher finds better kernel configurations
3. **Batch operations**: Larger tensors amortize kernel launch overhead
4. **Use appropriate dtype**: `float16` is ~2x faster than `float32` on most GPUs
5. **Enable tensor cores**: `USE_TC=1` (default) for matmul/conv acceleration
6. **Profile**: `DEBUG=2` shows per-kernel timing and bandwidth to identify bottlenecks

---

## Getting Help

- **Discord**: [discord.gg/ZjZadyC7PK](https://discord.gg/ZjZadyC7PK) — ask in `#learn-tinygrad`
- **GitHub Issues**: [github.com/tinygrad/tinygrad/issues](https://github.com/tinygrad/tinygrad/issues)
- **Debug output**: Always include `DEBUG=4` output when reporting bugs
