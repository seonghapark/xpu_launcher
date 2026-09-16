"""Verifies the bug + proposed fix by demonstrating that re-populating
the type entry in _deepcopy_dispatch makes the second stage() succeed.

This is a structural confirmation that .get(type, _deepcopy_atomic)
would work — without monkey-patching the actual function.
"""

import torch
from torch.distributed.checkpoint._state_dict_stager import StateDictStager


class SomeClass:
    pass


state_dict = {
    "model_weight": torch.zeros(4, 4),
    "some_class_field": SomeClass,
}


def _deepcopy_atomic(x, _):
    """The local _deepcopy_atomic that close() removed."""
    return x


# === Without fix: should crash ===
print("=== UNPATCHED: second stage() should crash ===")
stager = StateDictStager(pin_memory=False, share_memory=False)
stager.stage(state_dict)
print("First stage: OK")
stager.close()
try:
    stager.stage(state_dict)
    print("Second stage: OK — bug not present")
except KeyError as e:
    print(f"Second stage: REPRODUCED KeyError: {e!r}")

print()
print("=== PATCHED: re-add d[type] = _deepcopy_atomic to a fresh stager ===")

# A fresh stager + the proposed fix: re-add the type entry after close()
stager2 = StateDictStager(pin_memory=False, share_memory=False)
stager2.stage(state_dict)
print("First stage: OK")
stager2.close()
# This is what the .get(type, _deepcopy_atomic) fallback at line 320
# would do automatically. Demonstrate it works:
stager2._deepcopy_dispatch[type] = _deepcopy_atomic
result2 = stager2.stage(state_dict)
print(f"Second stage: OK — got {type(result2).__name__} with {len(result2)} keys")
print(f"  class object preserved across copy: {result2['some_class_field'] is SomeClass}")
print()
print("FIX VERIFIED — the .get(type, _deepcopy_atomic) fallback at line 320")
print("would prevent the crash without changing close() semantics.")
