# tinygrad File Structure

> Annotated directory tree explaining every directory and key file in the tinygrad project.
>
> Last updated: 2025-04-15

---

## Top Level

```
tinygrad/
├── tinygrad/          # Core library (~24,000 lines, enforced by CI)
├── extra/             # Supplementary tools, models, drivers, benchmarks
├── examples/          # Example applications and training scripts
├── test/              # Test suite (331 files)
├── docs/              # Documentation (MkDocs source)
├── .github/           # CI/CD workflows and actions
├── README.md          # Project README
├── pyproject.toml     # Package config, dependencies, tool settings
├── mkdocs.yml         # Documentation site configuration
├── .pre-commit-config.yaml  # Pre-commit hooks (ruff, mypy, tests)
├── .pylintrc          # Pylint configuration
├── .coveragerc        # Coverage settings
├── .gitignore         # Git ignore patterns
├── LICENSE            # MIT license
├── opencode.json      # OpenCode IDE config
├── serve_docs.sh      # Local docs server script
└── sz.py              # Line count enforcer (used in CI)
```

---

## Core Library: `tinygrad/`

The core library is the only code shipped in the pip package. Line count is enforced at ~24,000 lines by `sz.py` in CI.

```
tinygrad/
├── __init__.py              # Public exports: Tensor, TinyJit, function, dtypes, Device, UOp
├── tensor.py                # Tensor class - the user-facing API for all computation
├── device.py                # Device abstraction: _Device singleton, Buffer, Compiled base class
├── dtype.py                 # Type system: DType, PtrDType, ImageDType, dtypes constants
├── helpers.py               # Utilities: prod, getenv, Context, GlobalCounters, profiling, caching
├── gradient.py              # Automatic differentiation: compute_gradient, backward pass construction
├── function.py              # @function decorator for capturing computation graphs
├── callify.py               # Transform computation to CALL nodes for JIT execution
│
├── uop/                     # UOp intermediate representation system
│   ├── __init__.py          # Re-exports Ops enum and GroupOp
│   ├── ops.py               # UOp dataclass, Ops enum (70+ operations), UOpMetaClass, PatternMatcher
│   ├── spec.py              # UOp validation patterns and specification rules
│   ├── symbolic.py          # Symbolic simplification and constant folding
│   ├── upat.py              # UPat pattern matching and compilation
│   ├── validate.py          # Z3-based bounds checking and validation
│   ├── decompositions.py    # Decompose transcendental functions, bit manipulation
│   └── divandmod.py         # Symbolic division and modulo implementations
│
├── mixin/                   # Operation mixins that Tensor inherits
│   ├── __init__.py          # OpMixin base (combines all mixins)
│   ├── creation.py          # CreationMixin: const_like, cast, full_like, zeros_like
│   ├── dtype.py             # DTypeMixin: dtype queries, element_size, type conversion
│   ├── elementwise.py       # ElementwiseMixin: arithmetic, logic, activations
│   ├── movement.py          # MovementMixin: reshape, permute, shrink, pad, expand
│   └── reduce.py            # ReduceMixin: sum, prod, max, min, mean, var, std
│
├── schedule/                # Scheduling - converts tensor graph to executable plan
│   ├── __init__.py          # Schedule creation, kernel dependency graph, linearization
│   ├── rangeify.py          # Convert tensor ops to loop ranges (RANGE, REDUCE, STORE)
│   ├── indexing.py          # Buffer address computation from tensor indices
│   ├── memory.py            # Memory allocation planning (TLSF allocator)
│   ├── multi.py             # Multi-device and distributed computation
│   └── allreduce.py         # All-reduce communication pattern generation
│
├── codegen/                 # Code generation pipeline
│   ├── __init__.py          # full_rewrite_to_sink() - main codegen orchestration
│   ├── gpudims.py           # Map loop dimensions to GPU thread/block dimensions
│   ├── simplify.py          # Range simplification, load collapse, range splitting
│   │
│   ├── opt/                 # Kernel optimization
│   │   ├── __init__.py      # Opt dataclass for optimization actions
│   │   ├── heuristic.py     # Hand-coded optimization heuristics (default optimizer)
│   │   ├── postrange.py     # apply_opts() - orchestrates heuristic or BEAM
│   │   ├── search.py        # BEAM search over kernel configurations
│   │   └── tc.py            # Tensor core / WMMA detection and optimization
│   │
│   └── late/                # Late-stage codegen passes
│       ├── linearizer.py    # Topological sort of UOps with priority ordering
│       ├── devectorizer.py  # Vectorization, load/store folding, image support
│       └── expander.py      # Expand complex ops into primitives
│
├── renderer/                # Target-specific code generators
│   ├── __init__.py          # Renderer base class, ProgramSpec, Estimates
│   ├── cstyle.py            # C-style code generation (ClangJITRenderer)
│   ├── llvmir.py            # LLVM IR generation (CPULLVMRenderer)
│   ├── ptx.py               # NVIDIA PTX assembly generation
│   ├── nir.py               # NVIDIA IR / LVP renderer
│   ├── wgsl.py              # WebGPU WGSL shader generation
│   │
│   └── amd/                 # AMD-specific code generation
│       ├── __init__.py      # AMD renderer entry
│       ├── dsl.py           # AMD DSL for instruction generation
│       ├── elf.py           # ELF binary assembly for AMD kernels
│       ├── generate.py      # AMD kernel code generation
│       └── sqtt.py          # AMD SQTT profiling integration
│
├── engine/                  # Execution engine
│   ├── __init__.py
│   ├── jit.py               # TinyJit: JIT compilation, kernel caching, graph batching
│   └── realize.py           # Runner classes, CompiledRunner, BufferCopy, update_stats
│
├── runtime/                 # Hardware runtime implementations
│   ├── __init__.py
│   │
│   ├── ops_cpu.py           # CPU backend (multi-threaded via clang/LLVM)
│   ├── ops_cuda.py          # NVIDIA CUDA backend (via NVRTC)
│   ├── ops_nv.py            # NVIDIA direct driver access backend
│   ├── ops_amd.py           # AMD direct driver access backend
│   ├── ops_hip.py           # AMD HIP runtime backend
│   ├── ops_metal.py         # Apple Metal backend
│   ├── ops_cl.py            # OpenCL backend
│   ├── ops_webgpu.py        # WebGPU backend
│   ├── ops_qcom.py          # Qualcomm GPU (Adreno) backend
│   ├── ops_dsp.py           # Qualcomm DSP backend
│   ├── ops_python.py        # Pure Python reference backend
│   ├── ops_npy.py           # NumPy-backed backend (data loading)
│   ├── ops_disk.py          # Memory-mapped file storage backend
│   ├── ops_tinyfs.py        # TinyFS distributed storage backend
│   ├── ops_null.py          # Null device (for testing without hardware)
│   └── ops_rdma.py          # RDMA networking backend
│   │
│   ├── support/             # Runtime support libraries
│   │   ├── __init__.py
│   │   ├── hcq.py           # Hardware Command Queue abstraction
│   │   ├── memory.py        # TLSF memory allocator
│   │   ├── autogen.py       # FFI code auto-generation utilities
│   │   ├── c.py             # libc FFI wrappers
│   │   ├── objc.py          # Objective-C runtime bridge (macOS)
│   │   ├── system.py        # System calls (mmap, ioctl)
│   │   ├── elf.py           # ELF binary loading and JIT
│   │   ├── usb.py           # USB device communication
│   │   ├── amd.py           # AMD-specific runtime support
│   │   ├── compiler_amd.py  # AMD compiler (ROCm/comgr)
│   │   ├── compiler_cuda.py # CUDA compiler (NVRTC)
│   │   ├── compiler_cpu.py  # CPU compiler (clang/LLVM JIT)
│   │   ├── compiler_qcom.py # Qualcomm compiler
│   │   ├── compiler_mesa.py # Mesa/LVP compiler
│   │   ├── nv/              # NVIDIA device management
│   │   ├── mlx/             # Apple Metal device management
│   │   └── am/              # AMD device management (userspace driver)
│   │
│   ├── graph/               # Graph execution (batched kernel launch)
│   │   ├── __init__.py
│   │   ├── cuda.py          # CUDA graph execution
│   │   ├── metal.py         # Metal command buffer batching
│   │   └── hcq.py           # HCQ-based graph execution
│   │
│   └── autogen/             # Auto-generated FFI bindings (DO NOT EDIT)
│       ├── __init__.py
│       ├── libc.py          # libc bindings
│       ├── llvm.py          # LLVM C API bindings
│       ├── cuda.py          # CUDA driver API bindings
│       ├── nvrtc.py         # NVRTC compiler bindings
│       ├── hip.py           # HIP runtime bindings
│       ├── hsa.py           # HSA runtime bindings
│       ├── kfd.py           # KFD (AMD kernel fusion driver) bindings
│       ├── metal.py         # Metal framework bindings
│       ├── opencl.py        # OpenCL bindings
│       ├── webgpu.py        # WebGPU bindings
│       ├── am/              # AMD microarchitecture specs (CDNA, RDNA)
│       └── amd/             # AMD instruction encodings
│
├── nn/                      # Neural network modules
│   ├── __init__.py          # Layers: Linear, Conv2d, Conv1d, BatchNorm, LayerNorm, Embedding, etc.
│   ├── optim.py             # Optimizers: SGD, Adam, AdamW, LAMB, LARS
│   ├── state.py             # State dict save/load, SafeTensors format
│   ├── datasets.py          # Dataset loaders (MNIST, CIFAR)
│   ├── onnx.py              # ONNX model import
│   └── torch.py             # PyTorch model loading
│
├── apps/                    # Built-in applications
│   └── llm.py               # LLM inference (tokenizer, model loading, generation)
│
└── viz/                     # Computation graph visualization
    ├── serve.py             # HTTP server for visualization
    ├── index.html           # Web UI
    ├── js/                  # JavaScript (D3.js, Dagre layout)
    ├── assets/              # Third-party libraries
    ├── fetch_assets.sh      # Asset download script
    └── README               # Visualization documentation
```

---

## Extra: `extra/`

Supplementary tools not shipped in the pip package. Contains models, datasets, hardware drivers, benchmarks, and experimental features.

```
extra/
├── models/              # Pre-built ML models
│   ├── bert.py          # BERT (question answering)
│   ├── llama.py         # Llama LLM with rotary embeddings
│   ├── resnet.py        # ResNet CNN
│   ├── clip.py          # CLIP vision-language model
│   ├── vit.py           # Vision Transformer
│   ├── efficientnet.py  # EfficientNet
│   ├── convnext.py      # ConvNeXt
│   ├── inception.py     # Inception network
│   ├── t5.py            # T5 transformer
│   ├── transformer.py   # Generic transformer blocks
│   ├── mask_rcnn.py     # Mask R-CNN object detection
│   ├── retinanet.py     # RetinaNet object detection
│   ├── rnnt.py          # RNN-T speech recognition
│   ├── unet.py          # U-Net segmentation
│   └── unet3d.py        # 3D U-Net
│
├── datasets/            # Dataset loading utilities
│   ├── __init__.py      # MNIST/CIFAR loaders
│   ├── imagenet.py      # ImageNet dataset
│   ├── squad.py         # SQuAD Q&A dataset
│   ├── librispeech.py   # LibriSpeech audio
│   ├── wikipedia.py     # Wikipedia dataset
│   ├── openimages.py    # Open Images dataset
│   └── kits19.py        # Medical imaging dataset
│
├── gemm/                # GEMM benchmarks and optimized implementations
│   ├── simple_matmul.py # Reference implementations
│   ├── amd_*.py         # AMD-optimized GEMM (CDNA assembly)
│   ├── cuda_matmul.py   # CUDA GEMM
│   ├── metal_*.py       # Metal GEMM
│   ├── max_matmul.py    # Maximum performance GEMM
│   └── max_kernels/     # Pre-optimized CUDA kernels
│
├── thunder/             # High-performance kernel library (Kittens)
│   ├── amd/             # AMD flash attention, FP8 GEMM
│   ├── cuda/            # CUDA flash attention, matmul
│   ├── metal/           # Metal GEMM
│   └── tiny/            # Pure tinygrad flash attention
│
├── torch_backend/       # PyTorch integration backend
│   ├── backend.py       # Torch device registration for tinygrad
│   └── test_*.py        # Integration tests
│
├── amdpci/              # AMD PCI device interface and monitoring
├── hip_gpu_driver/      # AMD HIP ioctl interface and headers
├── nv_gpu_driver/       # NVIDIA ioctl interface and headers
├── qcom_gpu_driver/     # Qualcomm Adreno driver interface
├── dsp/                 # Qualcomm DSP compilation and runtime
├── usbgpu/              # USB GPU support (external GPUs on macOS)
│
├── perfetto/            # Perfetto profiling visualization
├── sqtt/                # AMD SQTT profiling integration
├── nv_pma/              # NVIDIA Performance Monitoring Architecture
│
├── hcqfuzz/             # HCQ fuzzing framework
├── optimization/        # Kernel optimization dataset tools
│
├── export_model.py      # Export models to WebGPU/CPU/CUDA
├── gradcheck.py         # Numerical gradient checking
├── introspection.py     # Memory introspection
├── lr_scheduler.py      # Learning rate schedulers
├── multitensor.py       # Multi-device tensor operations
├── training.py          # Generic training loop with JIT
└── bench_log.py         # Benchmark logging to InfluxDB
```

---

## Examples: `examples/`

Example applications demonstrating tinygrad capabilities.

```
examples/
├── beautiful_mnist.py       # MNIST training (98% in ~5 seconds)
├── beautiful_cifar.py       # CIFAR-10 training
├── beautiful_cartpole.py    # CartPole reinforcement learning
├── train_resnet.py          # ResNet training
├── gpt2.py                  # GPT-2 inference and training
├── llama.py                 # Llama model inference
├── llama3.py                # Llama 3 inference
├── mixtral.py               # Mixtral MoE inference
├── mamba.py                 # Mamba language model
├── whisper.py               # Whisper speech-to-text
├── stable_diffusion.py      # Stable Diffusion text-to-image
├── sdxl.py                  # Stable Diffusion XL
├── yolov8.py                # YOLOv8 object detection
├── compile_efficientnet.py  # Compile model to C
├── tinychat/                # WebGPU-based chat interface
├── webgpu/                  # Browser-based model demos
├── openpilot/               # Self-driving inference
├── mlperf/                  # MLPerf training submissions (v4.0-v6.0)
└── llm.c/                   # Port of Karpathy's llm.c
```

---

## Tests: `test/`

Test suite organized by scope and hardware requirements.

```
test/
├── README                   # Test organization documentation
├── helpers.py               # Shared test utilities
├── test_tiny.py             # Quick sanity tests (run by pre-commit)
│
├── backend/                 # Tests that run on EACH backend (40 files)
│   ├── test_ops.py          # Comprehensive operator tests (largest test file)
│   ├── test_tensor.py       # Core tensor operations
│   ├── test_jit.py          # JIT compilation tests
│   ├── test_nn.py           # Neural network layer tests
│   ├── test_linearizer.py   # Kernel linearization tests
│   ├── test_schedule.py     # Scheduling tests
│   └── test_dtype.py        # Data type handling tests
│
├── null/                    # Backend-independent tests (57 files)
│   ├── test_tensor.py       # Null backend tensor tests
│   ├── test_uops.py         # UOp representation tests
│   └── test_symbolic_*.py   # Symbolic computation tests
│
├── unit/                    # Focused unit tests (36 files)
│   ├── test_gradient.py     # Gradient computation
│   ├── test_attention.py    # Attention mechanism
│   └── test_function.py     # @function decorator
│
├── external/                # Integration and benchmark tests (83 files)
│   ├── external_test_*.py   # Functional integration tests
│   ├── external_benchmark_*.py  # Performance benchmarks
│   ├── fuzz_*.py            # Fuzzing tests
│   └── process_replay/      # Process replay utilities
│
├── amd/                     # AMD-specific tests (19 files)
│   └── hw/                  # AMD hardware instruction tests
│
├── device/                  # Device-specific tests (4 files)
├── models/                  # End-to-end model tests (9 files)
├── opt/                     # Optimization tests (3 files)
├── speed/                   # Performance tests (4 files)
├── mockgpu/                 # Mock GPU implementations for testing
└── web/                     # WebGPU tests
```

---

## CI/CD: `.github/`

```
.github/
├── workflows/
│   ├── test.yml             # Main test suite (25+ jobs across platforms)
│   ├── docs.yml             # Build and deploy docs to GitHub Pages
│   ├── autogen.yml          # Regenerate FFI bindings, check for drift
│   ├── benchmark.yml        # Performance benchmarks on self-hosted macOS
│   └── python-publish.yml   # Publish to PyPI on release
│
└── actions/
    ├── setup-tinygrad/      # Composite action: Python setup, deps, backend config
    └── process-replay/      # Process replay comparison with master
```
