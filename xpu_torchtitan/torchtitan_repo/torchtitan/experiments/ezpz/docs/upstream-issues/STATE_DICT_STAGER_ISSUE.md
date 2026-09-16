# StateDictStager raises `KeyError: <class 'type'>` on second `stage()` when state dict contains a class object

## Summary

`StateDictStager` (and therefore `DefaultStager`, and therefore
`dcp.async_save(async_checkpointer_type=AsyncCheckpointerType.PROCESS)`)
crashes on the **second** stage() call if the state dict contains any
class object (i.e. a value `x` where `type(x) is type` or `x` is itself
a class).

The first stage() always works; the second always fails after
`StateDictStager.close()` clears `_deepcopy_dispatch`.
`DefaultStager._stage()` calls `close()` after every stage to "break a
reference cycle" (`staging.py:251`), so the bug fires reliably from the
second checkpoint save onward in any production async-with-pinned-mem
loop.

## Repro

Tested on `torch==2.13.0.dev20260429+xpu`. CPU-only, no distributed init
required.

```python
import torch
from torch.distributed.checkpoint._state_dict_stager import StateDictStager


class SomeClass:
    pass


state_dict = {
    "model_weight": torch.zeros(4, 4),
    "some_class_field": SomeClass,  # any class object triggers the bug
}

stager = StateDictStager(pin_memory=False, share_memory=False)

# First stage — succeeds
stager.stage(state_dict)

# DefaultStager._stage calls this after every stage to break a ref
# cycle (see staging.py:251). Mirroring it here.
stager.close()

# Second stage — crashes
stager.stage(state_dict)
```

Output:

```
Traceback (most recent call last):
  File "repro.py", line 18, in <module>
    stager.stage(state_dict)
  File ".../torch/distributed/checkpoint/_state_dict_stager.py", line 191, in stage
    return self.deepcopy_with_tensor_offload(state_dict, None, [], non_blocking)
  File ".../torch/distributed/checkpoint/_state_dict_stager.py", line 374, in deepcopy_with_tensor_offload
    y = self._reconstruct(...)
  File ".../torch/distributed/checkpoint/_state_dict_stager.py", line 425, in <genexpr>
    self.deepcopy_with_tensor_offload(arg, memo, non_blocking=non_blocking)
  File ".../torch/distributed/checkpoint/_state_dict_stager.py", line 320, in deepcopy_with_tensor_offload
    y = self._deepcopy_dispatch[type](x, memo)
        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
KeyError: <class 'type'>
```

## Root cause

`StateDictStager.__init__` builds a per-instance dispatch table including
the entry needed for class objects (`_state_dict_stager.py:117`):

```python
d[type] = _deepcopy_atomic
```

`StateDictStager.close()` clears the entire table to break a closure
reference cycle (`_state_dict_stager.py:275`):

```python
self._deepcopy_dispatch.clear()
```

But `deepcopy_with_tensor_offload` has one branch (line 318-320) that
indexes the dict directly instead of using `.get()`:

```python
if issubclass(cls, type):
    # type copier is also atomic
    y = self._deepcopy_dispatch[type](x, memo)
```

Every other lookup in the function uses `.get(cls)` and falls back to
`__deepcopy__` / `__reduce_ex__` / etc. Only this one branch assumes
the table is intact, and it's the only one that hits class-object
values.

Since `DefaultStager._stage` calls `self._state_dict_stager.close()`
after every stage (`staging.py:251`), the very next stage call on any
state dict containing a class object crashes.

## Suggested fix

The right fix is **either**:

1. **Don't clear `_deepcopy_dispatch` in `close()`** — only the cached
   storages cause memory leaks; the dispatch dict only holds references
   to module-level functions plus the closure `self`, and the closure
   cycle can be broken at GC time. Keep:

   ```python
   def close(self):
       self._cached_storage_mapping.clear()
       # don't clear _deepcopy_dispatch — needed by next stage()
   ```

2. **Rebuild `_deepcopy_dispatch` at the start of each `stage()`** —
   keeps `close()`'s reference-cycle break but makes the stager
   reusable. Move the dispatch construction out of `__init__` into a
   helper called by both `__init__` and `stage()` (cheap — just a few
   dict assignments).

A naive narrow fix won't work. I tried changing line 320 to:

```python
copier = self._deepcopy_dispatch.get(type, _deepcopy_atomic)
y = copier(x, memo)
```

That makes the immediate `KeyError` go away, but the very next call
falls into `__reduce_ex__` (the atomic-types like `int`, `str`, etc.
are also missing from the cleared dispatch) and enters an infinite
recursion via `_reconstruct`:

```
RecursionError: maximum recursion depth exceeded
  ...
  File "_state_dict_stager.py", line 425, in <genexpr>
    self.deepcopy_with_tensor_offload(arg, memo, non_blocking=non_blocking)
  ...
```

Because every atomic-type entry was also cleared, the only real fix is
to keep the dispatch alive across close(), or rebuild it on next stage.

The current `close()` contract is also surprising: it appears to work
for tensor-only state dicts (which never hit the dispatch table at all
— they go through `_offload_tensor` directly) but permanently breaks
the stager for any heterogeneous state dict. Either fix it or document
that StateDictStager is single-shot after close.

## Real-world impact

Surfaced via `torchtitan` (`pytorch/torchtitan`) running async checkpoint
with `async_checkpointer_type=PROCESS` (i.e.
`--checkpoint.async-mode=async_with_pinned_mem` in the trainer CLI).
torchtitan state dicts contain class references via
`Configurable.Config` dataclass fields — common pattern in any
config-system that introspects classes at runtime. First save (e.g.
`enable_first_step_checkpoint=True` at step 1) works; second save (next
checkpoint interval) crashes all ranks with a Fatal Python abort.

The plain `--checkpoint.async-mode=async` path is unaffected because
it uses the in-process default stager (regular `copy.deepcopy`) rather
than the `StateDictStager`-based pinned-memory stager.

## Environment

- `torch==2.13.0.dev20260429+xpu`
- Intel XPU build, but bug is platform-independent (reproduces on CPU-only)
- python 3.12

cc @torch.distributed.checkpoint maintainers
