# tinygrad Documentation

> Complete reference documentation for the tinygrad deep learning framework.
>
> Last updated: 2025-04-15

---

## What is tinygrad?

tinygrad is an end-to-end deep learning stack: a tensor library with autograd, an IR-based compiler with kernel fusion, JIT and graph execution, and neural network modules for real training. It supports 20+ hardware backends while staying intentionally small (~24,000 lines enforced by CI).

**Key properties:**

- **Lazy evaluation** — operations build a computation graph; nothing executes until `.realize()` or `.numpy()`
- **Kernel fusion** — the compiler automatically fuses sequences of operations into single kernels
- **UOp IR** — all computation is expressed as UOp (micro-operation) graphs, a single IR from tensor ops down to hardware instructions
- **Multi-backend** — CPU, CUDA, Metal, AMD, NV, QCOM, OpenCL, WebGPU, DSP, and more
- **Hackable** — the entire compiler, IR, and runtime are visible and modifiable

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0, 0, -2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

---

## Documentation Index

### Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start Guide](quickstart.md) | Build a working MNIST classifier from scratch |
| [MNIST Tutorial](mnist.md) | Step-by-step MNIST with TinyJit optimization |
| [Setup Guide](SETUP.md) | Environment setup from zero for every platform and backend |
| [Development Guide](DEVELOPMENT.md) | Dev workflow: running, testing, linting, contributing |

### Architecture & Design

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System architecture, compilation pipeline, data flow diagrams |
| [File Structure](FILE_STRUCTURE.md) | Annotated directory tree — every directory and key file explained |
| [Architectural Decisions](DECISIONS.md) | Why tinygrad is designed the way it is |
| [Glossary](GLOSSARY.md) | Domain-specific terms and concepts |

### API Reference

| Document | Description |
|----------|-------------|
| [Public API](API.md) | Consolidated API reference with usage examples |
| [Tensor](tensor/index.md) | Tensor class reference |
| [Tensor Creation](tensor/creation.md) | Factory methods: zeros, ones, rand, etc. |
| [Tensor Elementwise](tensor/elementwise.md) | Unary/binary operations, activations |
| [Tensor Movement](tensor/movement.md) | reshape, permute, expand, pad, shrink |
| [Tensor Ops](tensor/ops.md) | Reduction, convolution, matmul, loss functions |
| [Tensor Properties](tensor/properties.md) | shape, dtype, device, realize, numpy |
| [Data Types](dtypes.md) | DType system: float, int, bool, image, pointer types |
| [Neural Networks](nn.md) | Layers (Linear, Conv2d, BatchNorm), optimizers, state management |
| [Data Models](DATA_MODELS.md) | UOp, Buffer, DType, Ops enum, ProgramSpec — field by field |

### Configuration & Operations

| Document | Description |
|----------|-------------|
| [Configuration](CONFIGURATION.md) | Every env var, ContextVar, config file, and feature flag |
| [Environment Variables](env_vars.md) | Runtime behavior control (DEBUG, DEV, BEAM, etc.) |
| [Dependencies](DEPENDENCIES.md) | Every dependency: what it does, why it's there |
| [Testing Guide](TESTING.md) | Test categories, running tests, process replay, writing tests |
| [Troubleshooting](TROUBLESHOOTING.md) | Common errors, debugging workflow, DEBUG levels |

### Developer Internals

| Document | Description |
|----------|-------------|
| [Developer Overview](developer/developer.md) | Architecture overview for contributors |
| [UOp System](developer/uop.md) | UOp intermediate representation reference |
| [Memory Layout](developer/layout.md) | Memory layout and data flow internals |
| [Speed & Optimization](developer/speed.md) | Compile speed, execution speed, kernel optimization |
| [Runtime System](developer/runtime.md) | Compiled, Allocator, Program, Compiler abstractions |
| [HCQ (Hardware Command Queue)](developer/hcq.md) | HCQ runtime architecture and synchronization |
| [AM Driver](developer/am.md) | AMD userspace driver documentation |

### Hardware & Deployment

| Document | Description |
|----------|-------------|
| [Runtime Backends](runtime.md) | Available runtimes: NV, AMD, QCOM, Metal, CUDA, CL, CPU, WebGPU |
| [tinybox](tinybox.md) | tinybox hardware setup and management |
| [tinygpu](tinygpu.md) | External GPU over USB4/Thunderbolt on macOS |
| [Showcase](showcase.md) | Example projects built with tinygrad |

---

## Quick Links

- **GitHub**: [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
- **Docs Site**: [docs.tinygrad.org](https://docs.tinygrad.org/)
- **Discord**: [discord.gg/ZjZadyC7PK](https://discord.gg/ZjZadyC7PK)
- **PyPI**: `pip install tinygrad`
- **Community Tutorials**: [Di Zhu's tinygrad notes](https://mesozoic-egg.github.io/tinygrad-notes/)
- **Bounties**: [Bounty spreadsheet](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing)

---

## Installation

```bash
# Recommended: install from source
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .

# Check your default device
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

See [Setup Guide](SETUP.md) for detailed platform-specific instructions.
