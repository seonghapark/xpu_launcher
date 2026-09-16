# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-local SelectiveAC subclass that drops the all-to-all op from the save list.

Background: PR14's ``_normal_equal_a2a_padding`` fast-path in
``token_dispatcher.py:dispatch`` pads tensors before the EP all-to-all and
slices the pad rows off after. When SelectiveAC includes
``all_to_all_single`` in the save list, SAC captures the comm output at the
dispatch boundary (an ``AsyncCollectiveTensor`` wrapping a buffer that the
caching allocator can reuse before backward). On step-1 backward recompute,
the saved wrapper's underlying GPU memory has been freed and reissued for a
new allocation -> GPU page table inconsistency that surfaces as a
``Segmentation fault from GPU at 0x... type: 0 (NotPresent), level: 1 (PDE)``
on the next forward op (typically step-2's ``_local_reorder``).

Dropping the op from the save list forces recompute of the a2a in backward
(extra comm) but avoids the dangling save. Measured on the affected config
(``moe_10b_2b_sdpa_ep`` EP=12 padding=1 AC=selective): 10 clean steps with
loss matching the upstream-dispatcher baseline within 1e-4 nats.

The standard (non-padded) MoE path is fine with the upstream save list, so
keeping the override narrow to ezpz/moe rather than touching upstream.

57th sync: PR #3674 refactored AC into a Configurable policy class
hierarchy. The pre-refactor monkey-patch (intercepting the module-level
``_get_save_ops`` symbol around an ``apply_ac`` call) is replaced by
``MoeSelectiveAC``, a ``SelectiveAC`` subclass that overrides
``get_save_ops``. This is the extension point the new API was designed
for. Reference in config registries with ``MoeSelectiveAC.Config()``.
"""

import torch

from torchtitan.distributed.activation_checkpoint import (
    _get_default_save_ops,
    SelectiveAC,
)


_A2A_OP = torch.ops._c10d_functional.all_to_all_single.default


class MoeSelectiveAC(SelectiveAC):
    """SelectiveAC variant for ezpz/moe: drops ``all_to_all_single`` from saves.

    Forces recompute of the EP all-to-all on backward, avoiding the dangling
    ``AsyncCollectiveTensor`` save problem from PR14's padded a2a fast-path.
    """

    def get_save_ops(self) -> set:
        save_ops = _get_default_save_ops()
        save_ops.discard(_A2A_OP)
        return save_ops
