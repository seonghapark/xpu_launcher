# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workaround for upstream xccl's missing ``supportsSplitting`` override.

``torch/csrc/distributed/c10d/Backend.hpp`` declares
``virtual bool supportsSplitting() const { return false; }``. ``ProcessGroupNCCL``
overrides this to ``return true;`` (see
``torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp``); ``ProcessGroupXCCL`` does
NOT, so it inherits the base ``false``.

Upstream ``torch.distributed.device_mesh.DeviceMesh._init_one_process_group``
(lines 550-562 in torch 2.13) gates the ``split_group`` vs ``new_group`` code
path on:

    bound_device_id is not None  (or torchcomms enabled)
    and torch.accelerator.is_available()
    and (backend is None or default_group._get_backend(accel).name() == backend)

For an xccl-backed default group, all three of these are true on XPU systems
that build with ``device_id=...`` (which xccl's eager init pathway does set),
so we end up in the ``split_group`` branch. ``split_group`` then calls
``parent_backend.supportsSplitting`` (Python: ``supports_splitting``), which
returns ``False`` for xccl, and raises::

    RuntimeError: No backend for the parent process group or its backend does
    not support splitting

This blocks every nested mesh construction on XPU — including the EP-flavored
sparse mesh built unconditionally for ``moe_2b_ep``::

    full_sparse_mesh = unflatten_mesh(
        self._world_mesh,
        ("pp", "dp_replicate", "efsdp", "ep"),
        (self.pp, self.dp_replicate, efsdp, self.ep),
    )

The real fix has to land upstream in two places: ``ProcessGroupXCCL.hpp`` needs
a ``supportsSplitting() override { return true; }``, AND the underlying
``ProcessGroupXCCL::split`` implementation needs to actually work. Until both
land, we need to keep the ``new_group`` fallback alive on XPU even when
``bound_device_id`` is set.

This module installs a wrapper around ``DeviceMesh._init_one_process_group``
that:

  * Detects when the parent backend's effective ``supports_splitting`` is
    ``False`` (the xccl case).
  * In that case, treats ``backend_override`` as the ``new_group`` path would —
    i.e. skips the ``split_group`` branch and falls through to the
    ``new_group`` loop, which is the same code path xccl already takes today
    in single-mesh / non-eager-bound configurations.

We deliberately do NOT touch any upstream files (``Golden Rule #1``). The
wrapper monkey-patches at import time when xccl is detected, is idempotent,
and is a no-op on cuda/cpu builds.

Remove this module once both upstream conditions hold:
  1. ``ProcessGroupXCCL`` declares ``supportsSplitting() override { return
     true; }`` in its header.
  2. ``ProcessGroupXCCL::split`` is implemented and exercised by the existing
     ``test/distributed/test_c10d_xccl.py`` suite.
"""

from __future__ import annotations

import torch

from torchtitan.tools.logging import logger


_PATCHED_ATTR = "_ezpz_xccl_split_group_patched"


def _should_patch() -> bool:
    """Return True iff we're on an XPU build with xccl available.

    Pure no-op on cuda/cpu — the bug only manifests under xccl, so leaving
    the upstream path untouched everywhere else minimises the blast radius.
    """
    return torch.distributed.is_xccl_available() and torch.xpu.is_available()


def maybe_install_xccl_split_group_workaround() -> None:
    """Install the xccl split_group workaround if needed (idempotent).

    Safe to call multiple times; safe to call from non-XPU paths (no-op).
    """
    if not _should_patch():
        return

    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.distributed_c10d import _get_default_group

    if getattr(DeviceMesh, _PATCHED_ATTR, False):
        return

    original_init_one_process_group = DeviceMesh._init_one_process_group

    @staticmethod
    def _patched_init_one_process_group(*args, **kwargs):
        """Force the ``new_group`` path whenever the parent backend is xccl.

        Two failure generations of the same upstream bug:

        * Older builds: ``ProcessGroupXCCL`` inherits
          ``supportsSplitting() == False`` and ``split_group`` raises.
        * 2026.x builds: it reports ``True`` but ``split()`` is still broken —
          some ranks get ``NON_GROUP_MEMBER`` back, which surfaces later as a
          bare ``AssertionError`` in ``DeviceMesh._init_process_groups``
          (mixed None / non-None dim group names).

        So we no longer trust ``supports_splitting`` on xccl at all; set
        ``XPU_XCCL_TRUST_SPLIT=1`` to re-enable the upstream fast path once
        a fixed xccl lands. ``*args`` keeps us compatible with upstream
        signature changes (e.g. the added ``preserve_rank_order``).
        """
        import os

        try:
            default_group = _get_default_group()
        except Exception:
            return original_init_one_process_group(*args, **kwargs)

        accel = torch.accelerator.current_accelerator()
        if accel is None:
            return original_init_one_process_group(*args, **kwargs)

        try:
            parent_backend = default_group._get_backend(accel)
        except Exception:
            return original_init_one_process_group(*args, **kwargs)

        backend_name = ""
        try:
            backend_name = str(parent_backend.name())
        except Exception:
            backend_name = type(parent_backend).__name__

        force_new_group = (
            "xccl" in backend_name.lower()
            and os.environ.get("XPU_XCCL_TRUST_SPLIT", "0") != "1"
        )

        if not force_new_group and getattr(
            parent_backend, "supports_splitting", False
        ):
            # NCCL / trusted backend — take the upstream fast path.
            return original_init_one_process_group(*args, **kwargs)

        # Steer the upstream gate's ``bound_device_id`` clause to ``False`` so
        # every dim takes the ``new_group`` fallback (never xccl split()).
        saved_bound_device_id = getattr(default_group, "bound_device_id", None)
        try:
            default_group.bound_device_id = None  # type: ignore[attr-defined]
            return original_init_one_process_group(*args, **kwargs)
        finally:
            default_group.bound_device_id = saved_bound_device_id  # type: ignore[attr-defined]

    DeviceMesh._init_one_process_group = _patched_init_one_process_group
    setattr(DeviceMesh, _PATCHED_ATTR, True)

    logger.info(
        "Installed xccl split_group workaround on DeviceMesh._init_one_process_group "
        "(upstream ProcessGroupXCCL has no supportsSplitting() override; see "
        "experiments/ezpz/xccl_split_group_workaround.py)."
    )
