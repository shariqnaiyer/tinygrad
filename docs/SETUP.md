# tinygrad Setup Guide

> Complete environment setup from zero for every platform and backend.
>
> Last updated: 2025-04-15

---

## Prerequisites

- **Python 3.11+** (required; 3.12 recommended)
- **git** (for source installation)
- **pip** (Python package manager)

---

## Basic Installation

### From Source (Recommended)

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### From git (Latest master)

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### From PyPI

```bash
python3 -m pip install tinygrad
```

### Verify Installation

```bash
# Check default device
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# Quick test
python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

---

## Platform-Specific Setup

### macOS

**CPU backend** works out of the box (uses clang).

**Metal backend** (recommended for Apple Silicon):
```bash
# Metal is auto-detected on macOS. Verify:
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
# Should print: METAL
```

**External GPU over USB4/Thunderbolt** (AMD/NVIDIA): See [tinygpu.md](tinygpu.md).

### Linux

**CPU backend** works out of the box with system clang/gcc.

**CUDA backend** (NVIDIA GPUs):
```bash
# Requires CUDA toolkit. Install from https://developer.nvidia.com/cuda-toolkit
# Or via package manager:
sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian

# Set CUDA path if not in default location:
export CUDA_PATH=/usr/local/cuda

# Verify:
DEV=CUDA python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

**NV backend** (NVIDIA direct driver access, lower latency than CUDA):
```bash
DEV=NV python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

**AMD backend** (AMD GPUs):
```bash
# Requires ROCm or direct AMD driver access
# For HIP backend:
DEV=HIP python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# For direct AMD backend:
DEV=AMD python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

**OpenCL backend** (any GPU with OpenCL support):
```bash
# Ubuntu/Debian:
sudo apt install ocl-icd-opencl-dev

DEV=CL python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

### Windows

**CPU backend** works with MSVC or clang.

**CUDA backend**:
```bash
# Install CUDA toolkit from https://developer.nvidia.com/cuda-toolkit
set DEV=CUDA
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

---

## Backend-Specific Setup

### WebGPU

```bash
# Install Dawn (WebGPU implementation)
# On macOS, Dawn uses Metal underneath

DEV=WEBGPU python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# Force a specific WebGPU backend:
WEBGPU_BACKEND=WGPUBackendType_Metal  # or Vulkan, DirectX, OpenGL
```

### Qualcomm (QCOM / DSP)

For Qualcomm Adreno GPUs and DSP:
```bash
DEV=QCOM python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
DEV=DSP python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

See `extra/dsp/` for DSP-specific setup and Docker environment.

---

## Installing Dependencies for Development

tinygrad has **zero runtime dependencies** — the core library runs with just Python 3.11+.

Optional dependency groups are defined in `pyproject.toml`:

```bash
# Linting tools (ruff, mypy, pylint, pre-commit)
python3 -m pip install -e '.[linting]'

# Minimal testing (numpy, torch, pytest, hypothesis, z3)
python3 -m pip install -e '.[testing_minimal]'

# Unit testing (adds safetensors, tqdm, gguf, tabulate)
python3 -m pip install -e '.[testing_unit]'

# Full testing (adds onnx, opencv, transformers, librosa, etc.)
python3 -m pip install -e '.[testing]'

# Documentation building
python3 -m pip install -e '.[docs]'
```

---

## Environment Variables

Key variables for controlling tinygrad behavior:

| Variable | Values | Description |
|----------|--------|-------------|
| `DEV` | `AMD`, `NV`, `CUDA`, `METAL`, `CL`, `CPU`, `WEBGPU`, etc. | Target device |
| `DEBUG` | `1`-`7` | Debug output verbosity |
| `BEAM` | integer | BEAM search width for kernel optimization |
| `JIT` | `0`, `1`, `2` | JIT mode: 0=disabled, 1=enabled (default), 2=enabled without graphs |
| `IMAGE` | `1` | Enable 2D image-specific optimizations |
| `VIZ` | `1` | Enable computation graph visualization |
| `DEFAULT_FLOAT` | `HALF`, `BFLOAT16`, etc. | Default float dtype (default: FLOAT32) |

See [Environment Variables](env_vars.md) for the complete list.

---

## Checking Your Setup

```bash
# See what device tinygrad will use
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# List all available devices
python3 -c "from tinygrad import Device; print(list(Device.get_available_devices()))"

# Run a quick benchmark
DEBUG=2 python3 -c "
from tinygrad import Tensor
a = Tensor.rand(1024, 1024)
b = Tensor.rand(1024, 1024)
c = (a @ b).realize()
"

# Run the minimal test suite
python3 -m pytest test/test_tiny.py
```
