"""Diagnostic for the FSDP `Inconsistent compute device and device_id` bug.

Print, per rank:
  - torch.xpu.current_device() (after ezpz.setup_torch)
  - os.environ LOCAL_RANK / ACCELERATE_LOCAL_RANK / RANK
  - accelerate.Accelerator().device (what accelerate believes is "our" device)
  - AutoModelForCausalLM device, parameter-by-parameter
"""

import os

import ezpz
import ezpz.distributed


def main():
    ezpz.distributed.setup_torch()

    import torch
    rank = ezpz.distributed.get_rank()
    local_rank = ezpz.distributed.get_local_rank()

    current_xpu = torch.xpu.current_device() if torch.xpu.is_available() else None
    print(
        f"[diag rank={rank} local_rank={local_rank}] "
        f"after-setup: torch.xpu.current_device()={current_xpu}  "
        f"torch.xpu.device_count()={torch.xpu.device_count()}",
        flush=True,
    )

    relevant_env = {
        k: os.environ.get(k)
        for k in (
            "LOCAL_RANK", "ACCELERATE_LOCAL_RANK", "RANK", "WORLD_SIZE",
            "PMI_LOCAL_RANK", "PMI_RANK", "MPI_LOCALRANKID",
            "ZE_FLAT_DEVICE_HIERARCHY", "ZE_AFFINITY_MASK",
            "ONEAPI_DEVICE_SELECTOR",
        )
    }
    print(f"[diag rank={rank}] env: {relevant_env}", flush=True)

    from accelerate import Accelerator
    acc = Accelerator()
    print(
        f"[diag rank={rank}] accelerator.device={acc.device}  "
        f"local_process_index={acc.local_process_index}  "
        f"process_index={acc.process_index}",
        flush=True,
    )

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    first_param_device = next(model.parameters()).device
    print(
        f"[diag rank={rank}] post-from_pretrained: "
        f"first_param_device={first_param_device}",
        flush=True,
    )

    model = model.to(acc.device)
    moved_device = next(model.parameters()).device
    print(
        f"[diag rank={rank}] post-.to(accelerator.device): "
        f"first_param_device={moved_device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
