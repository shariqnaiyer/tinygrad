# tinygrad Commenting Standard

> Defines the commenting conventions used across this personal annotated branch.
>
> Last updated: 2025-04-15

---

## Principles

1. Comments explain **WHY**, not **WHAT** — the code already says what it does.
2. Comments that restate code (`# increment counter` on `counter += 1`) are worse than no comment.
3. Be precise — use exact types, parameter names, return shapes.
4. Be concise — every word earns its place.
5. Preserve tinygrad's style: 2-space indent, 150-char lines, compact layout.
6. Never change code behavior. Comments only.

---

## File Headers

Every `.py` file gets a module-level docstring immediately after imports (or before them if no `from __future__` import). Format:

```python
"""Short one-line summary of what this file does.

Longer description if needed — what role this file plays in the architecture,
what key abstractions it defines, and how it relates to other modules.

Key exports: ClassName, function_name, CONSTANT
Dependencies: module_a (for X), module_b (for Y)
"""
```

For files that already have a comment-style header (e.g., `# inspired by...`), keep it and add the docstring below.

---

## Class Documentation

```python
class MyClass:
  """One-line summary of what this class represents.

  Longer description: what it's responsible for, what state it manages,
  and where it fits in the system. Include invariants if relevant.

  Example:
    obj = MyClass(arg1, arg2)
    result = obj.do_thing()
  """
```

---

## Function/Method Documentation

For **public** functions — full docstring:

```python
def public_function(param1: int, param2: str = "default") -> list[int]:
  """One-line summary in imperative mood ("Return the...", "Compute the...").

  Longer explanation if the summary isn't enough. Focus on WHY this exists
  and any non-obvious behavior.

  Args:
    param1: Description of param1. Valid range: [0, 100].
    param2: Description of param2. Defaults to "default".

  Returns:
    Description of what's returned and under what conditions.

  Raises:
    ValueError: When param1 is negative.
    RuntimeError: When device is not available.
  """
```

For **private/internal** functions — shorter format is acceptable:

```python
def _internal_helper(x):
  """Compute the frobnicated value of x for use in the scheduler."""
```

For **very short lambdas or one-liners** — inline comment is fine:

```python
simplify = lambda x: x.ssimplify() if isinstance(x, UOp) else x  # reduce symbolic UOps to concrete ints when possible
```

---

## Inline Comments

Use `#` comments for:

- **Algorithm steps**: `# Step 2: topological sort of kernel dependency graph`
- **Non-obvious conditionals**: `# guard: skip if buffer is already realized`
- **Magic numbers**: `# 256 = max workgroup size for this GPU arch`
- **Workarounds**: `# HACK: Metal doesn't support 64-bit atomics, emulate with CAS loop`
- **Performance choices**: `# using dict.fromkeys() instead of set() to preserve insertion order`
- **Domain context**: `# Winograd transform: F(4x4, 3x3) — see http://arxiv.org/abs/1509.09308`

Place inline comments on the line above the code they describe, not at end-of-line (unless very short).

---

## Tags

| Tag | Meaning |
|-----|---------|
| `# NOTE:` | Important context that isn't obvious from the code |
| `# TODO:` | Something that should be done but isn't yet |
| `# HACK:` | Intentional workaround — explain why |
| `# BUG:` | Known bug — do not fix, just document |
| `# PERF:` | Performance-related choice or concern |
| `# SAFETY:` | Why a cast/assertion/guard is safe |

---

## What NOT to Comment

- Auto-generated files (`runtime/autogen/`) — skip entirely
- Obvious one-liner methods where the name says it all
- Type annotations that are already precise
- Code that will be deleted soon (comment the deletion plan, not the code)
