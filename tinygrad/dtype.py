"""tinygrad's type system: DType, PtrDType, ImageDType, the dtypes constants, and type promotion/conversion utilities.

This is one of tinygrad's foundational modules -- nearly everything else depends on it. It defines:
  - DType: the core data-type descriptor (analogous to numpy's dtype, but much lighter).
  - PtrDType: a pointer-to-DType, used to represent buffers in device memory.
  - ImageDType: a specialized PtrDType for OpenCL image objects (GPU texture memory).
  - dtypes: a namespace class holding all concrete type constants (dtypes.float32, dtypes.int8, etc.)
    plus classification helpers (is_float, is_int, ...) and type groupings (floats, ints, ...).
  - Type promotion: a JAX-style lattice (promo_lattice) that defines how mixed-type operations resolve.
  - Conversion utilities: float<->fp8, float<->bf16, numpy/torch interop, and storage format helpers.
"""
from __future__ import annotations
from typing import Final, ClassVar, Callable, Literal
import math, struct, ctypes, functools
from dataclasses import dataclass, fields
from tinygrad.helpers import ceildiv, getenv, prod, round_up, OSX
from enum import Enum, auto

class ConstFloat(float):
  """Float subclass used as the canonical representation of float constants in the IR.

  Python's built-in float has two problems for use as dict keys / set members in a compiler:
    1. float('nan') != float('nan'), so NaN constants would never cache-hit.
    2. -0.0 == 0.0, so the compiler couldn't distinguish them even though they produce different bits.
  ConstFloat fixes both by comparing and hashing on the raw IEEE-754 bit pattern.

  Attributes:
    bits: The raw 64-bit IEEE-754 representation stored as an int, used for hashing and equality.
  """
  __slots__ = ('bits',)
  bits: int
  def __new__(cls, v:float):
    obj = super().__new__(cls, v)
    # store the raw IEEE-754 bits so we can distinguish -0.0 from 0.0 and make nan == nan
    obj.bits = struct.unpack('<Q', struct.pack('<d', v))[0]
    return obj
  def __eq__(self, other):
    if self is other: return True
    # NaN == NaN when both are float -- needed so NaN constants can be deduplicated in the IR cache
    if isinstance(other, float) and math.isnan(self) and math.isnan(other): return True
    return float.__eq__(self, other)
  def __hash__(self): return hash(self.bits)  # hash on bit pattern so -0.0 and 0.0 hash differently
  def __repr__(self): return f"ConstFloat({float.__repr__(self)})"
  def __str__(self): return float.__repr__(self)

class InvalidType:
  """Singleton sentinel representing an invalid/undefined value in the IR.

  Used in place of a real constant when an operation produces an undefined result (e.g., 0/0).
  It's a singleton so identity comparison (`is`) works, and it pickles correctly back to the same instance.
  Comparison operators are defined so InvalidType sorts consistently -- it compares as "not equal" to everything
  except itself, and as both less-than and greater-than any other value (for stable sorting in containers).
  """
  _instance: ClassVar[InvalidType|None] = None
  def __new__(cls):
    if cls._instance is None: cls._instance = object.__new__(cls)
    return cls._instance
  def __eq__(self, other): return self is other
  def __lt__(self, other): return self is not other
  def __gt__(self, other): return self is not other
  def __hash__(self): return id(self)
  def __repr__(self): return "Invalid"
  def __reduce__(self): return (InvalidType, ())  # unpickle returns the singleton
  def __format__(self, spec): return "Invalid"

# the global singleton -- use this instead of InvalidType() for clarity
Invalid = InvalidType()

# type aliases for constant values that can appear in the IR
PyConst = float|int|bool           # valid Python scalar constants
ConstType = PyConst|InvalidType    # PyConst + the Invalid sentinel

# struct format characters for each DType that has a native Python struct representation
FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']

# all DTypes should only be created once -- this metaclass enforces that
class DTypeMetaClass(type):
  """Metaclass that ensures each unique DType is only instantiated once (interning / flyweight pattern).

  DType equality is identity (`is`), which is only safe because this cache guarantees deduplication.
  The cache key is the positional args tuple passed to the DType constructor.
  """
  dcache: dict[tuple, DType] = {}
  def __call__(cls, *args, **kwargs):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
    return ret

class AddrSpace(Enum):
  """Memory address spaces for pointer types, matching GPU memory hierarchy.

  GLOBAL: main device memory (VRAM), accessible by all work items.
  LOCAL: shared memory within a work group (fast on-chip SRAM, e.g. __local in OpenCL, __shared__ in CUDA).
  REG: register-level storage, used for thread-private temporaries in codegen.
  """
  def __repr__(self): return str(self)
  GLOBAL = auto(); LOCAL = auto(); REG = auto()  # noqa: E702

@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
  """The core data-type descriptor in tinygrad, analogous to numpy.dtype but much lighter.

  Every tensor, buffer, and IR node carries a DType to describe its element type. DTypes are frozen
  dataclasses interned by DTypeMetaClass, so equality is identity (`a is b`) and they're hashable by id.

  Scalar types have count=1 and _scalar=None. Vector types (e.g. float32x4) have count>1 and _scalar
  pointing back to the scalar element type.

  Attributes:
    priority: Determines automatic upcasting order -- higher priority wins in type promotion.
    bitsize: Total size in bits (for vectors, this is scalar_bits * count).
    name: The C-level type name (e.g. "float", "int", "half"). Used in codegen.
    fmt: Python struct module format character, or None for types without a native struct format (bf16, fp8s).
    count: Vector width. 1 for scalar types, >1 for SIMD vector types (e.g. float4).
    _scalar: For vector types, points to the underlying scalar DType. None for scalar types.
  """
  priority: int  # this determines when things get upcasted
  bitsize: int
  name: str
  fmt: FmtStr|None
  count: int
  _scalar: DType|None
  @property
  def itemsize(self) -> int: return (self.bitsize + 7) // 8  # size in bytes, rounded up (e.g. bool is 1 bit -> 1 byte)
  @staticmethod
  def new(priority:int, bitsize:int, name:str, fmt:FmtStr|None): return DType(priority, bitsize, name, fmt, 1, None)
  def __reduce__(self): return type(self), tuple(getattr(self, f.name) for f in fields(self))
  def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self.scalar().name]}"+(f".vec({self.count})" if self.count != 1 else "")
  # __lt__ enables sorting -- used by least_upper_dtype to pick the minimum from the promotion set
  def __lt__(self, o:DType): return (self.priority, self.bitsize, self.name, self.fmt, self.count) < (o.priority, o.bitsize, o.name, o.fmt, o.count)
  @property
  def base(self): return self  # for DType, base is self; PtrDType overrides this to return the element type
  @property
  def vcount(self): return self.count  # PtrDType overrides to return `v` (the pointer vectorization count)
  @functools.cache  # pylint: disable=method-cache-max-size-none
  def vec(self, sz:int) -> DType:
    """Create a SIMD vector type from this scalar type (e.g. dtypes.float32.vec(4) -> float32x4)."""
    assert self.count == 1, f"can't vectorize {self} with size {sz}"
    if sz == 1 or self == dtypes.void: return self  # void doesn't vectorize, and sz=1 is scalar
    return DType(self.priority, self.bitsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz, self)
  def ptr(self, size=-1, addrspace=AddrSpace.GLOBAL) -> PtrDType:
    """Create a pointer type to this DType, representing a device buffer holding `size` elements."""
    return PtrDType(self.priority, self.bitsize, self.name, self.fmt, self.count, None, self, addrspace, 1, size)
  def scalar(self) -> DType: return self._scalar if self._scalar is not None else self
  def nbytes(self) -> int: raise RuntimeError("only ptr types have nbytes")
  @functools.cached_property
  def min(self):
    """The minimum representable value for this dtype (e.g. -2^31 for int32, -inf for floats, False for bool)."""
    if dtypes.is_int(self): return 0 if dtypes.is_unsigned(self) else -2**(self.scalar().bitsize-1)
    return -float("inf") if dtypes.is_float(self) else False
  @functools.cached_property
  def max(self):
    """The maximum representable value for this dtype (e.g. 2^31-1 for int32, inf for floats, True for bool)."""
    if dtypes.is_int(self): return 2**(self.scalar().bitsize)-1+self.min
    return float("inf") if dtypes.is_float(self) else True
  def const(self, val: tuple[ConstType, ...]|ConstType):
    """Canonicalize a Python constant into the appropriate type for this DType.

    Handles three key invariants:
      - NaN values are canonicalized to math.nan so they can be deduplicated in caches.
      - Float constants are wrapped in ConstFloat so -0.0 and 0.0 remain distinct.
      - Tuples are recursively mapped for vector types.

    Args:
      val: A Python scalar (float/int/bool), Invalid, or a tuple of them for vector types.

    Returns:
      The canonicalized constant: ConstFloat for floats, bool for bools, int for ints,
      Invalid passthrough, or a tuple for vector types.
    """
    if isinstance(val, tuple):
      assert len(val) == self.count, f"mismatch {val} {self}"
      return tuple(map(self.const, val))
    if isinstance(val, InvalidType): return val
    # NOTE: float('nan') != float('nan'), so we canonicalize here
    if isinstance(val, float) and math.isnan(val): val = math.nan
    # int is the default. wrap floats in ConstFloat to distinguish -0.0 from 0.0 in cache
    return ConstFloat(float(val)) if dtypes.is_float(self) else bool(val) if dtypes.is_bool(self) else int(val)

@dataclass(frozen=True, eq=False)
class PtrDType(DType):
  """A pointer-to-DType representing a device memory buffer.

  Used in the IR to type buffer arguments -- every Tensor's underlying buffer has a PtrDType.
  Inherits all DType fields (which describe the *element* type) and adds pointer-specific metadata.

  Attributes:
    _base: The element DType this pointer points to (e.g. dtypes.float32).
    addrspace: Which memory space the pointer lives in (GLOBAL, LOCAL, or REG).
    v: Pointer vectorization count -- how many pointers are packed together (for vectorized loads/stores).
    size: Number of elements in the buffer, or -1 for unbounded/unknown size.
  """
  _base: DType
  addrspace: AddrSpace
  v: int
  size: int = -1  # -1 is unlimited size
  @property
  def base(self): return self._base
  @functools.cache  # pylint: disable=method-cache-max-size-none
  def vec(self, sz:int) -> DType:
    """Vectorize the pointer itself (not the element type) -- used for vectorized memory access patterns."""
    assert self.v == 1, f"can't vectorize ptr {self} with size {sz}"
    if sz == 1: return self  # sz=1 is a scalar
    # ImageDType needs special handling to carry through the shape field
    if isinstance(self, ImageDType):
      return ImageDType(self.priority, self.bitsize, self.name, self.fmt, self.count, self, self._base, self.addrspace, sz, self.size, self.shape)
    return type(self)(self.priority, self.bitsize, self.name, self.fmt, self.count, self, self._base, self.addrspace, sz, self.size)
  def ptr(self, size=-1, addrspace=AddrSpace.GLOBAL) -> PtrDType: raise RuntimeError("can't make a pointer from a pointer")
  def nbytes(self) -> int:
    if self.size == -1: raise RuntimeError("can't get nbytes of a pointer with unlimited size")
    return self.size*self.itemsize
  @property
  def vcount(self): return self.v
  def __repr__(self):
    return f"{self.base.__repr__()}.ptr({self.size}{', '+str(self.addrspace) if self.addrspace != AddrSpace.GLOBAL else ''})" + \
      (f'.vec({self.v})' if self.v != 1 else '')

@dataclass(frozen=True, eq=False)
class ImageDType(PtrDType):
  """Specialized pointer type for OpenCL image objects (GPU texture memory).

  Images provide hardware-accelerated 2D spatial access and free interpolation on GPUs that support it.
  They always live in GLOBAL address space and carry an explicit (height, width) shape.
  The base type is always float32 (the hardware reads/writes float4 pixels), but the storage format
  can be half (imageh) or float (imagef).

  Attributes:
    shape: The (height, width) dimensions of the image. Pixel count = height * width, each pixel is 4 elements.
  """
  shape: tuple[int, ...] = ()   # shape of the Image
  def ptr(self, size=-1, addrspace=AddrSpace.GLOBAL) -> PtrDType:
    assert addrspace == AddrSpace.GLOBAL, "images can't be local"
    return self  # images are already pointers -- calling .ptr() is a no-op
  def __repr__(self): return f"dtypes.{self.name}({self.shape})" + (f'.vec({self.v})' if self.v != 1 else '')

  # for 1d images on macos, we need to round pitch up to 256 pixels to make CL happy
  @property
  def pitch(self): return (round_up(self.shape[1], 256) if OSX else self.shape[1]) * 4 * self.itemsize

  @staticmethod
  def valid_dims(ptr:PtrDType) -> list[tuple[int,int]]:
    """Find all (height, width) dimension pairs that can represent this buffer as a 2D image without pitch padding.

    Images store 4 elements per pixel (RGBA), so pxls = size // 4. The dimensions must satisfy OpenCL alignment
    constraints: pitch (width in bytes) must be aligned to IMAGE_PITCH_ALIGN, and total size must be aligned to
    IMAGE_BASE_ALIGN. Width cannot exceed the hardware maximum (16384).

    Args:
      ptr: The pointer type whose buffer we want to map as an image.

    Returns:
      A list of valid (height, width) pairs, or [] if the buffer can't be an image.
    """
    ALIGN, MAXW, pxls = getenv("IMAGE_PITCH_ALIGN", 256 if OSX else 64), 16384, ptr.size // 4
    # only half and float element types are supported, and the total pixel count must fit in a 2D image
    if ptr.base not in (dtypes.half, dtypes.float) or ptr.size > 4*MAXW*MAXW: return []
    # height=1 images just need to abide by alignment requirements in bytes, not pixels!
    if ptr.size % (ALIGN * 4) != 0: return [] if ptr.nbytes() % getenv("IMAGE_BASE_ALIGN", 64) != 0 or pxls > MAXW else [(1, pxls)]
    return [(pxls//ALIGN//k, ALIGN*k) for k in range(ceildiv(pxls//ALIGN, MAXW), min(pxls//ALIGN, MAXW//ALIGN)+1) if (pxls//ALIGN)%k == 0]

class dtypes:
  """Namespace holding all concrete DType constants and type classification helpers.

  This is the main entry point for working with types in tinygrad. Access types as dtypes.float32,
  dtypes.int8, etc. Classification methods (is_float, is_int, is_unsigned, is_bool) and type groupings
  (floats, ints, uints, sints) are also here.

  The type hierarchy by priority (lowest to highest):
    void (-1) -> weakint/bool (0) -> int8..uint64 (1-8) -> fp8s (9-10) -> float16 (11) -> bfloat16 (12) -> float32 (13) -> float64 (14)
  Priority determines automatic upcasting: when two types meet in an operation, the higher-priority type wins.
  """
  # --- type classification helpers ---
  # NOTE: static methods must be defined before the type constants below, otherwise `bool` in the
  # type hints would resolve to dtypes.bool (the DType) instead of the builtin bool.
  @staticmethod
  @functools.cache
  def is_float(x: DType) -> bool: return x.scalar() in dtypes.floats or isinstance(x, ImageDType)
  @staticmethod # static methods on top, or bool in the type info will refer to dtypes.bool
  @functools.cache
  def is_int(x: DType) -> bool: return x.scalar() in (dtypes.ints + (dtypes.weakint,))
  @staticmethod
  @functools.cache
  def is_unsigned(x: DType) -> bool: return x.scalar() in dtypes.uints
  @staticmethod
  def is_bool(x: DType) -> bool: return x.scalar() == dtypes.bool
  @staticmethod
  def from_py(x) -> DType:
    """Infer a DType from a Python scalar or nested list/tuple of scalars.

    Args:
      x: A Python bool, int, float, or a (possibly nested) list/tuple of them.

    Returns:
      The appropriate DType: bool->dtypes.bool, float->default_float, int->default_int.
      For containers, returns the max (highest priority) DType of all elements.
    """
    # NOTE: isinstance(True, int) is True, so bool must be checked before int
    if isinstance(x, bool): return dtypes.bool
    if isinstance(x, float): return dtypes.default_float
    if isinstance(x, int): return dtypes.default_int
    # put this in the last is faster because there are more items than lists/tuples to check
    if isinstance(x, (list, tuple)): return max(dtypes.from_py(xi) for xi in x) if x else dtypes.default_float
    raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")
  @staticmethod
  def finfo(dtype:DType) -> tuple[int, int]:
    """Return the (exponent_bits, mantissa_bits) for a floating-point DType.

    Args:
      dtype: A floating-point DType.

    Returns:
      Tuple of (exponent_bits, mantissa_bits). E.g. float32 -> (8, 23).

    Raises:
      ValueError: If dtype is not a float type.
    """
    if not dtypes.is_float(dtype): raise ValueError(f"{dtype} is not a floating point type")
    return {dtypes.float16: (5, 10), dtypes.bfloat16: (8, 7), dtypes.float32: (8, 23), dtypes.float64: (11, 52),
            dtypes.fp8e4m3: (4, 3), dtypes.fp8e5m2: (5, 2), dtypes.fp8e4m3fnuz: (4, 3), dtypes.fp8e5m2fnuz: (5, 2)}[dtype]

  # --- type constants ---
  # Each type is created via DType.new(priority, bitsize, c_name, struct_fmt).
  # Priority ordering determines promotion: higher priority = wins in mixed-type ops.
  # fmt is the Python struct module format char (None = no native struct support, needs special conversion).

  void: Final[DType] = DType.new(-1, 0, "void", None)       # no-type / unit type, used for side-effect-only ops
  weakint: Final[DType] = DType.new(0, 800, "weakint", None) # type of Python int literals before binding to a concrete int type;
                                                               # 800 bits so it's "big enough" to hold any Python int
  bool: Final[DType] = DType.new(0, 1, "bool", '?')

  # signed and unsigned integers, ordered by priority (signed < unsigned at each width)
  int8: Final[DType] = DType.new(1, 8, "signed char", 'b')
  uint8: Final[DType] = DType.new(2, 8, "unsigned char", 'B')
  int16: Final[DType] = DType.new(3, 16, "short", 'h')
  uint16: Final[DType] = DType.new(4, 16, "unsigned short", 'H')
  int32: Final[DType] = DType.new(5, 32, "int", 'i')
  uint32: Final[DType] = DType.new(6, 32, "unsigned int", 'I')
  int64: Final[DType] = DType.new(7, 64, "long", 'q')
  uint64: Final[DType] = DType.new(8, 64, "unsigned long", 'Q')
  _uint128: Final[DType] = DType.new(8, 128, "uint128", None)  # internal-only, used for wide intermediate results
  _uint256: Final[DType] = DType.new(8, 256, "uint256", None)  # internal-only, used for wide intermediate results

  # 8-bit floats: OCP (Open Compute Project) variants and FNUZ (Finite, No Unsigned Zero) variants
  fp8e4m3: Final[DType] = DType.new(9, 8, "float8_e4m3", None)        # OCP: 4-bit exponent, 3-bit mantissa, has inf/nan
  fp8e5m2: Final[DType] = DType.new(10, 8, "float8_e5m2", None)       # OCP: 5-bit exponent, 2-bit mantissa, has inf/nan
  fp8e4m3fnuz: Final[DType] = DType.new(9, 8, "float8_e4m3fnuz", None)  # FNUZ: no negative zero, 0x80 = NaN
  fp8e5m2fnuz: Final[DType] = DType.new(10, 8, "float8_e5m2fnuz", None) # FNUZ: no negative zero, 0x80 = NaN

  float16: Final[DType] = DType.new(11, 16, "half", 'e')
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType.new(12, 16, "__bf16", None)  # no struct fmt -- needs manual conversion via float_to_bf16
  float32: Final[DType] = DType.new(13, 32, "float", 'f')
  float64: Final[DType] = DType.new(14, 64, "double", 'd')

  # dtype aliases -- short names matching C/numpy conventions
  half = float16; float = float32; double = float64 # noqa: E702
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
  char = int8; short = int16; int = int32; long = int64 # noqa: E702

  # NOTE: these are image dtypes -- factory methods because each call creates a new ImageDType with a unique shape
  @staticmethod
  def imageh(shp): return ImageDType(100, 16, "imageh", 'e', 1, None, dtypes.float32, AddrSpace.GLOBAL, 1, prod(shp), shp)
  @staticmethod
  def imagef(shp): return ImageDType(100, 32, "imagef", 'f', 1, None, dtypes.float32, AddrSpace.GLOBAL, 1, prod(shp), shp)

  # defaults that can be overridden via DEFAULT_FLOAT env var (see below)
  default_float: ClassVar[DType] = float32
  default_int: ClassVar[DType] = int32

  # --- type groupings for classification ---
  fp8_ocp = (fp8e4m3, fp8e5m2)                                # OCP-standard 8-bit floats
  fp8_fnuz = (fp8e4m3fnuz, fp8e5m2fnuz)                       # FNUZ-variant 8-bit floats (AMD ROCm)
  fp8s = fp8_ocp + fp8_fnuz                                    # all 8-bit floats
  floats = fp8s + (float16, bfloat16, float32, float64)        # all floating-point types
  int8s = (uint8, int8)
  int16s = (uint16, int16)
  int32s = (uint32, int32)
  int64s = (uint64, int64)
  uints = (uint8, uint16, uint32, uint64)                      # all unsigned integer types
  sints = (int8, int16, int32, int64)                          # all signed integer types
  ints = uints + sints                                         # all integer types
  all = floats + ints + (bool, weakint) # noqa: A003           # every user-facing type

# allow overriding the default float type via env var, e.g. DEFAULT_FLOAT=float16 for half-precision training
if (env_default_float := getenv("DEFAULT_FLOAT", "")):
  dtypes.default_float = getattr(dtypes, env_default_float.lower())
  assert dtypes.is_float(dtypes.default_float), f"{env_default_float} is not a float dtype"

DTypeLike = str|DType  # accept either a DType object or a string name (e.g. "float32")
def to_dtype(dtype:DTypeLike) -> DType:
  """Convert a DTypeLike (DType or string name) to a DType. String lookup is case-insensitive."""
  return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())

# Type promotion lattice, following JAX's design (minus complex types):
# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
#
# Each entry maps a DType to the list of DTypes it can be promoted *to* (its direct parents in the lattice).
# The lattice is a DAG: bool -> weakint -> {int8, uint8} -> ... -> uint64 -> fp8s -> {float16, bfloat16} -> float32 -> float64.
# least_upper_dtype finds the smallest type that's an ancestor of ALL input types (the "join" in lattice terms).
promo_lattice = { dtypes.bool: [dtypes.weakint], dtypes.weakint: [dtypes.int8, dtypes.uint8],
  dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32], dtypes.int32: [dtypes.int64],
  dtypes.int64: [dtypes.uint64], dtypes.uint8: [dtypes.int16, dtypes.uint16], dtypes.uint16: [dtypes.int32, dtypes.uint32],
  dtypes.uint32: [dtypes.int64, dtypes.uint64], dtypes.uint64: [dtypes.fp8e4m3, dtypes.fp8e5m2, dtypes.fp8e4m3fnuz, dtypes.fp8e5m2fnuz],
  dtypes.fp8e4m3: [dtypes.float16, dtypes.bfloat16], dtypes.fp8e5m2: [dtypes.float16, dtypes.bfloat16],
  dtypes.fp8e4m3fnuz: [dtypes.float16, dtypes.bfloat16], dtypes.fp8e5m2fnuz: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }

@functools.cache
def _get_recursive_parents(dtype:DType) -> set[DType]:
  """Collect all ancestors of `dtype` in the promotion lattice (including itself). float64 is the top/root."""
  return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}

@functools.cache
def least_upper_dtype(*ds:DType) -> DType:
  """Find the least common supertype of all given DTypes using the promotion lattice.

  This is the "join" operation: it intersects the ancestor sets of all input types
  and returns the minimum (lowest-priority) type in that intersection. Images short-circuit
  because they always dominate (they carry shape info that can't be promoted away).

  Args:
    *ds: One or more DTypes to find the common supertype of.

  Returns:
    The smallest DType that all inputs can be safely promoted to.
  """
  return min(set.intersection(*[_get_recursive_parents(d.scalar()) for d in ds])) \
      if not (images:=[d for d in ds if isinstance(d, ImageDType)]) else images[0]

def least_upper_float(dt:DType) -> DType:
  """Promote `dt` to at least the default float type. If already float, return as-is."""
  return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.default_float)

# DTYPES_DICT: maps short names ("float32", "int8", ...) to DType objects. Excludes internal types (void, weakint, _uint128, _uint256).
# INVERSE_DTYPES_DICT: maps C-level names ("float", "signed char", ...) back to short names. Used by __repr__.
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if isinstance(v, DType) and not k.startswith(("default", "void", "weakint", "_"))}
INVERSE_DTYPES_DICT = {**{v.name:k for k,v in DTYPES_DICT.items()}, "void": "void", "weakint":"weakint"}

@functools.cache
def can_lossless_cast(dt0:DType, dt1:DType) -> bool:
  """Check whether casting from dt0 to dt1 preserves all values (no precision or range loss).

  Similar to numpy.can_cast with casting='safe'. The logic is hardcoded per target type
  rather than derived from bit-widths, because float<->int losslessness depends on mantissa
  bits, not just total bits (e.g. float32 can't losslessly hold all int32 values).

  See: https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html

  Args:
    dt0: The source dtype.
    dt1: The target dtype.

  Returns:
    True if every value representable in dt0 is exactly representable in dt1.
  """
  if dt0 == dt1 or dt0 == dtypes.bool: return True
  match dt1:
    case dtypes.weakint: return dt0 in dtypes.ints
    case dtypes.double: return dt0 in (dtypes.float, dtypes.half, dtypes.bfloat16, *dtypes.fp8s,
      dtypes.uint32, dtypes.uint16, dtypes.uint8, dtypes.int32, dtypes.int16, dtypes.int8)
    case dtypes.float: return dt0 in (dtypes.half, dtypes.bfloat16, *dtypes.fp8s, dtypes.uint16, dtypes.uint8, dtypes.int16, dtypes.int8)
    case dtypes.half: return dt0 in (*dtypes.fp8s, dtypes.uint8, dtypes.int8)
    case dtypes.uint64: return dt0 in (dtypes.uint32, dtypes.uint16, dtypes.uint8)
    case dtypes.uint32: return dt0 in (dtypes.uint16, dtypes.uint8)
    case dtypes.uint16: return dt0 in (dtypes.uint8,)
    case dtypes.int64: return dt0 in (dtypes.uint32, dtypes.uint16, dtypes.uint8, dtypes.int32, dtypes.int16, dtypes.int8)
    case dtypes.int32: return dt0 in (dtypes.uint16, dtypes.uint8, dtypes.int16, dtypes.int8)
    case dtypes.int16: return dt0 in (dtypes.uint8, dtypes.int8)
    case _: return False

def sum_acc_dtype(dt:DType):
  """Choose the accumulator dtype for sum reductions to avoid overflow.

  Unsigned ints accumulate in at least uint32, signed ints/bool in at least int32,
  and floats in at least float32 (overridable via SUM_DTYPE env var for mixed-precision training).

  Args:
    dt: The element dtype being summed.

  Returns:
    The dtype to use for the accumulator.
  """
  if dtypes.is_unsigned(dt): return least_upper_dtype(dt, dtypes.uint)
  if dtypes.is_int(dt) or dt == dtypes.bool: return least_upper_dtype(dt, dtypes.int)
  return least_upper_dtype(dt, to_dtype(getenv("SUM_DTYPE", "float32")))

def float_to_fp16(x):
  """Round a float to fp16 precision. Returns +/-inf on overflow (matching hardware behavior)."""
  try: return struct.unpack('e', struct.pack('e', float(x)))[0]
  except OverflowError: return math.copysign(math.inf, x)

def float_to_bf16(x):
  """Round a float to bfloat16 precision using round-to-nearest-even (banker's rounding).

  bfloat16 is just float32 with the bottom 16 mantissa bits truncated (8-bit exponent, 7-bit mantissa).
  The rounding adds 0x7FFF (halfway) plus the LSB of the result for ties-to-even, then masks off the low bits.
  """
  if not math.isfinite(x): return x
  u = struct.unpack('I', struct.pack('f', x))[0]
  # round-to-nearest-even: add half-ulp (0x7FFF) + LSB of truncated result for tie-breaking
  u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
  return struct.unpack('f', struct.pack('I', u))[0]

# fp8-float conversions based on https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp
# Configuration for each fp8 format. All thresholds are stored as raw IEEE-754 double bit patterns
# so the conversion can work entirely in integer arithmetic (avoiding float rounding surprises).
# (bias, sig_bits, mant_mask, min_denorm_half, ovf_threshold, max_norm, min_norm)
_fp8_cfg = {
  dtypes.fp8e4m3: (7, 4, 0x7, 0x3F50000000000000, 0x407D000000000000, 0x7E, 0x3F90000000000000),
  dtypes.fp8e5m2: (15, 3, 0x3, 0x3EE0000000000000, 0x40EE000000000000-1, 0x7B, 0x3F10000000000000),
  dtypes.fp8e4m3fnuz: (8, 4, 0x7, 0x3F40000000000000, 0x406F000000000000-1, 0x7F, 0x3F80000000000000),
  dtypes.fp8e5m2fnuz: (16, 3, 0x3, 0x3ED0000000000000, 0x40EE000000000000-1, 0x7F, 0x3F00000000000000),
}

def float_to_fp8(x: float, dtype: DType) -> int:
  """Convert a Python float to an 8-bit float, returning the raw byte as an int.

  Handles all four fp8 variants (e4m3, e5m2, and their fnuz counterparts). The conversion
  operates on raw IEEE-754 bit patterns to implement round-to-nearest-even without relying
  on float arithmetic. Special values (inf, nan, zero) are handled per-format before the
  general case.

  Args:
    x: The float value to convert.
    dtype: Which fp8 format to target (must be in dtypes.fp8s).

  Returns:
    The encoded fp8 value as an int in [0, 255].
  """
  assert dtype in dtypes.fp8s, "Only for fp8s"
  # FNUZ formats: all non-finite values map to NaN (0x80), and there's only positive zero (0x00)
  if dtype in dtypes.fp8_fnuz and not math.isfinite(x): return 0x80
  if dtype in dtypes.fp8_fnuz and x == 0.0: return 0x00
  # e4m3 don't support inf, return 0x7f(+NaN) and 0xff(-NaN) to match jax
  # NaN is unordered, can't compare with zero, use math.copysign to get sign
  if dtype == dtypes.fp8e4m3 and not math.isfinite(x): return 0x7f if math.copysign(1, x) > 0 else 0xff
  if dtype == dtypes.fp8e5m2 and not math.isfinite(x): return (0 if math.copysign(1, x) > 0 else 0x80) | (0x7c if math.isinf(x) else 0x7f)
  bias, sig_bits, mant_mask, min_denorm_half, ovf_threshold, max_norm, min_norm = _fp8_cfg[dtype]
  xbits, = struct.unpack('Q', struct.pack('d', x))
  half_ulp = 1 << (52 - sig_bits)  # half of the unit-in-last-place for rounding
  sign, exp, mantissa, absx = ((xbits>>63)&1)<<7, ((xbits>>52)&0x7FF)-1023+bias, (xbits>>(53-sig_bits))&mant_mask, xbits&0x7FFFFFFFFFFFFFFF
  if absx <= min_denorm_half: res = 0                          # too small even for denorm -> flush to zero
  elif absx > ovf_threshold: res = max_norm                    # overflow -> clamp to max representable
  elif absx >= min_norm:                                       # normal range: pack exponent + mantissa with rounding
    res, round_bits = (exp << (sig_bits - 1)) | mantissa, xbits & ((half_ulp << 1) - 1)
    if round_bits > half_ulp or (round_bits == half_ulp and mantissa & 1): res += 1  # round-to-nearest-even
  else:                                                        # denormal range: shift mantissa right and round
    shift = 1 - exp
    mantissa |= 1 << (sig_bits - 1)
    res, half = mantissa >> shift, half_ulp << shift
    round_bits = (xbits | (1 << 52)) & ((half << 1) - 1)
    if round_bits > half or (round_bits == half and res & 1): res += 1
  return 0 if dtype in dtypes.fp8_fnuz and res == 0 else int(res | sign)  # fnuz has no negative zero

def fp8_to_float(x: int, dtype: DType) -> float:
  """Decode a raw fp8 byte (int in [0, 255]) back to a Python float.

  Args:
    x: The raw fp8 value as an int.
    dtype: Which fp8 format to decode from (must be in dtypes.fp8s).

  Returns:
    The decoded float value.
  """
  assert dtype in dtypes.fp8s, "Only for fp8s"
  if dtype in dtypes.fp8_fnuz and x == 0x80: return math.nan  # FNUZ: 0x80 is the unique NaN encoding
  if (x & 0x7F) == 0: return -0.0 if x & 0x80 else 0.0       # magnitude is zero -> +0.0 or -0.0
  bias, sig_bits, *_ = _fp8_cfg[dtype]
  mant_bits, exp_bits = sig_bits - 1, 8 - sig_bits
  exp_max, mant_max = (1 << exp_bits) - 1, (1 << mant_bits) - 1
  sign, exp, mantissa = (x >> 7) & 1, (x >> mant_bits) & exp_max, x & mant_max
  # handle special exponent (all-ones) for non-FNUZ formats
  if dtype not in dtypes.fp8_fnuz and exp == exp_max:
    if dtype == dtypes.fp8e5m2: return math.copysign(math.nan if mantissa else math.inf, -1 if sign else 1)
    if mantissa == mant_max: return math.nan  # e4m3: only the max mantissa pattern is NaN (no inf)
  # denormal (exp==0): no implicit leading 1 bit; normal: implicit leading 1 bit
  val = (mantissa / (mant_max + 1)) * 2 ** (1 - bias) if exp == 0 else (1 + mantissa / (mant_max + 1)) * 2 ** (exp - bias)
  return -val if sign else val

def storage_fmt_for_dtype(dtype:DType):
  """Return the struct format char for reading/writing raw bytes of this dtype.

  bfloat16 is stored as uint16 ('H') and fp8s as uint8 ('B') because Python's struct module
  doesn't have native support for these formats.
  """
  return 'H' if dtype == dtypes.bfloat16 else 'B' if dtype in dtypes.fp8s else dtype.fmt

def to_storage_scalar(x, dtype:DType):
  """Convert a Python scalar to its raw storage representation for the given dtype.

  For types without native struct support (bf16, fp8), this returns the raw integer bits.
  For native types (float32, int32, etc.), returns the value unchanged.
  """
  if dtype == dtypes.half: return float_to_fp16(x)
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF  # extract upper 16 bits
  if dtype in dtypes.fp8s: return float_to_fp8(float(x), dtype)
  return x

def from_storage_scalar(x, dtype:DType):
  """Convert a raw storage value back to a Python scalar. Inverse of to_storage_scalar."""
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]  # reconstruct float32 from upper 16 bits
  if dtype in dtypes.fp8s: return fp8_to_float(int(x), dtype)
  return x

# truncate: round-trip a Python value through the dtype's precision, simulating what the hardware would do.
# For float types this means round to the type's precision; for int types this means mask/wrap via ctypes.
truncate: dict[DType, Callable] = {dtypes.bool: bool,
  dtypes.float16: float_to_fp16, dtypes.bfloat16: lambda x: float_to_bf16(float(x)),
  **{fp8: (lambda x, dtype=fp8: fp8_to_float(float_to_fp8(x, dtype), dtype)) for fp8 in dtypes.fp8s},
  # for native C types, use ctypes to get C-level truncation/wrap behavior (e.g. int32 overflow wraps)
  **{getattr(dtypes, n): (lambda x, c=getattr(ctypes, f'c_{n}'): c(x).value)
     for n in ('float', 'double', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')}}

# --- numpy and torch dtype interop ---
# These are used by Tensor.numpy(), Tensor.torch(), and the corresponding from_* constructors.
# Types without direct numpy/torch equivalents (bf16, fp8s) are proxied through float32/uint8.

def _to_np_dtype(dtype:DType) -> type|None:
  """Convert a tinygrad DType to its numpy dtype equivalent. bf16/fp8 are widened to float32."""
  import numpy as np
  if dtype in { dtypes.bfloat16, *dtypes.fp8s }: return np.float32  # no native numpy bf16/fp8 -- upcast
  return np.dtype(dtype.fmt).type if dtype.fmt is not None else None

def _from_np_dtype(npdtype:'np.dtype') -> DType: # type: ignore [name-defined] # noqa: F821
  """Convert a numpy dtype to the corresponding tinygrad DType."""
  import numpy as np
  return DTYPES_DICT[np.dtype(npdtype).name]

@functools.cache
def _to_torch_dtype(dtype:DType) -> 'torch.dtype'|None:  # type: ignore [name-defined] # noqa: F821
  """Convert a tinygrad DType to its torch dtype equivalent. fp8s are stored as uint8."""
  import numpy as np, torch
  if dtype == dtypes.uint64: return torch.uint64
  if dtype == dtypes.bfloat16: return torch.bfloat16
  if dtype in dtypes.fp8s: return torch.uint8  # torch doesn't have fp8 dtypes, store as raw bytes
  # NOTE: torch doesn't expose this mapping with a stable API, so we go through numpy as a bridge
  try: return torch.from_numpy(np.array([], dtype=_to_np_dtype(dtype))).dtype
  except TypeError: return None

@functools.cache
def _from_torch_dtype(torchdtype:'torch.dtype') -> DType: # type: ignore [name-defined] # noqa: F821
  """Convert a torch dtype to the corresponding tinygrad DType. Builds a reverse lookup from _to_torch_dtype."""
  return {v:k for k in DTYPES_DICT.values() if (v:=_to_torch_dtype(k)) is not None}[torchdtype]
