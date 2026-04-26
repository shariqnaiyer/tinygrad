"""PyTorch compatibility shim: registers tinygrad as a torch backend device.

Importing this module installs the tinygrad backend into PyTorch's dispatch system (via extra.torch_backend.backend),
allowing `torch.device("tinygrad")` to route tensor operations through tinygrad. Requires an editable install (pip install -e .)
because the torch backend code lives in extra/ which is not included in wheel releases.
"""
# type: ignore
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
try: import extra.torch_backend.backend  # noqa: F401 # pylint: disable=unused-import
except ImportError as e: raise ImportError("torch frontend not in release\nTo fix, install tinygrad from a git checkout with pip install -e .") from e