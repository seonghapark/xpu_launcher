"""Work around broken process-group splitting in current XCCL builds."""

from __future__ import annotations

import os

import torch

from torchtitan.tools.logging import logger


_PATCHED_ATTR = "_torchtitan_xccl_split_group_patched"


def maybe_install_xccl_split_group_workaround() -> None:
    if (
        not hasattr(torch.distributed, "is_xccl_available")
        or not torch.distributed.is_xccl_available()
        or not hasattr(torch, "xpu")
        or not torch.xpu.is_available()
        or os.environ.get("XPU_XCCL_TRUST_SPLIT", "0") == "1"
    ):
        return

    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.distributed_c10d import _get_default_group

    if getattr(DeviceMesh, _PATCHED_ATTR, False):
        return
    original = DeviceMesh._init_one_process_group

    @staticmethod
    def use_new_group_for_xccl(*args, **kwargs):
        try:
            default_group = _get_default_group()
            accelerator = torch.accelerator.current_accelerator()
            backend_name = str(default_group._get_backend(accelerator).name())
        except Exception:
            return original(*args, **kwargs)
        if "xccl" not in backend_name.lower():
            return original(*args, **kwargs)

        bound_device_id = getattr(default_group, "bound_device_id", None)
        try:
            default_group.bound_device_id = None
            return original(*args, **kwargs)
        finally:
            default_group.bound_device_id = bound_device_id

    DeviceMesh._init_one_process_group = use_new_group_for_xccl
    setattr(DeviceMesh, _PATCHED_ATTR, True)
    logger.info("Installed XCCL DeviceMesh new_group workaround")