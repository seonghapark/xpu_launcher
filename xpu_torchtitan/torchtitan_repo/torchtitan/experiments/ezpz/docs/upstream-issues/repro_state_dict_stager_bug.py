"""Minimal repro: DefaultStager raises KeyError on second stage()
when the state dict contains a class object (not an instance).

The bug:
- StateDictStager.__init__ builds _deepcopy_dispatch including
      d[type] = _deepcopy_atomic
- StateDictStager.close() clears the dispatch table after every stage to
  break closure-capture reference cycles
- StateDictStager.deepcopy_with_tensor_offload, when it sees a value x
  whose type is itself a metaclass (i.e. x is itself a class object), does:
      y = self._deepcopy_dispatch[type](x, memo)
  which raises KeyError because the table was cleared by the previous close().

Every other lookup in deepcopy_with_tensor_offload uses .get() and falls
through to a recursive copier; only this one branch indexes the dict
directly, making it the lone failure mode after close().

DefaultStager (the real path used by
torch.distributed.checkpoint.async_save with
AsyncCheckpointerType.PROCESS) calls self._state_dict_stager.close()
inside _stage_state_dict every save, so the bug fires on the SECOND save.

Real-world impact: any state dict containing a class reference (a fairly
common pattern in dataclass-based config systems — e.g. torchtitan stores
class objects via Configurable fields) will succeed on the first
checkpoint and crash on every subsequent one when using
async_with_pinned_mem mode.

Tested:
- torch 2.13.0.dev20260429+xpu (Intel XPU build) — reproduces
- Reproduces on CPU only too — pure-Python issue in the dispatcher;
  no GPU or distributed init required.

Run:
    python repro_state_dict_stager_bug.py
"""

import torch
from torch.distributed.checkpoint._state_dict_stager import StateDictStager


# Any class object as a state-dict value triggers the bug.
# Real torchtitan state dicts contain these via Configurable.Config
# dataclass fields that retain class references at runtime.
class SomeClass:
    pass


state_dict = {
    "model_weight": torch.zeros(4, 4),
    "some_class_field": SomeClass,
}

# Use StateDictStager directly to avoid DefaultStager's accelerator
# requirement — same code path, just sidesteps the
#   AssertionError: Non-blocking copy requires that the current
#   accelerator is available.
# that DefaultStager raises on CPU-only machines. The bug is in
# StateDictStager itself, not the wrapper.
stager = StateDictStager(pin_memory=False, share_memory=False)

print("First stage()...")
result1 = stager.stage(state_dict)
print(f"  OK — got {type(result1).__name__} with {len(result1)} keys")

# This is what DefaultStager._stage() does after every stage call to
# break a closure-capture reference cycle (staging.py:251). It's the
# precondition for the bug — the next stage() then crashes on any
# class-object value.
print("Calling stager.close() (mimics DefaultStager post-stage lifecycle)...")
stager.close()

print("Second stage()...")
try:
    result2 = stager.stage(state_dict)
    print(f"  OK — got {type(result2).__name__} with {len(result2)} keys")
    print()
    print("Did NOT reproduce — close() lifecycle may have changed.")
except KeyError as e:
    print(f"  REPRODUCED: KeyError: {e!r}")
    print()
    print("Root cause: StateDictStager.close() clears _deepcopy_dispatch,")
    print("but deepcopy_with_tensor_offload's `issubclass(cls, type)` branch")
    print("(_state_dict_stager.py:320) indexes the dict directly with [type]")
    print("instead of using .get() with a fallback. Suggested fix:")
    print()
    print("    y = self._deepcopy_dispatch.get(type, copy._deepcopy_atomic)(x, memo)")
    print()
    print("Or — better — rebuild self._deepcopy_dispatch at the start of")
    print("each stage() instead of clearing it in close(). The current")
    print("close() also leaves the stager in an unusable state, which is")
    print("surprising for an object that gets reused across saves.")
    raise
