"""Tests for the cuda/xpu/rocm accelerator backend modules."""

from __future__ import annotations

import re

import pytest

from cli import accelerators, launch
from cli.accelerators import cuda, rocm, xpu

ALL_BACKENDS = (cuda, xpu, rocm)


def test_public_api_parity() -> None:
    for module in ALL_BACKENDS:
        assert accelerators.missing_api(module) == [], module.__name__


def test_names() -> None:
    assert cuda.name() == "cuda"
    assert xpu.name() == "xpu"
    assert rocm.name() == "rocm"
    assert set(accelerators.BACKENDS) == {"cuda", "xpu", "rocm"}


def test_visible_devices_env_names() -> None:
    assert cuda.visible_devices_env() == "CUDA_VISIBLE_DEVICES"
    assert xpu.visible_devices_env() == "ZE_AFFINITY_MASK"
    assert rocm.visible_devices_env() == "ROCR_VISIBLE_DEVICES"
    assert "HIP_VISIBLE_DEVICES" in rocm.extra_visible_devices_envs()


def test_distributed_backend_names() -> None:
    assert cuda.distributed_backend() == "nccl"
    assert xpu.distributed_backend() == "xccl"
    assert rocm.distributed_backend() == "nccl"
    assert cuda.collective_library() == "nccl"
    assert xpu.collective_library() == "oneccl"
    assert rocm.collective_library() == "rccl"


def test_smi_binaries() -> None:
    assert cuda.smi_binary() == "nvidia-smi"
    assert xpu.smi_binary() == "xpu-smi"
    assert rocm.smi_binary() == "rocm-smi"
    for module in ALL_BACKENDS:
        cmd = module.smi_query_command()
        assert cmd and cmd[0] == module.smi_binary()


def test_set_visible_devices() -> None:
    env: dict[str, str] = {}
    assert cuda.set_visible_devices([0, 1], env=env) == {
        "CUDA_VISIBLE_DEVICES": "0,1"
    }
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    env = {}
    xpu.set_visible_devices(["0.0", "0.1"], env=env)
    assert env["ZE_AFFINITY_MASK"] == "0.0,0.1"

    env = {}
    rocm.set_visible_devices([2, 3], env=env)
    assert env["ROCR_VISIBLE_DEVICES"] == "2,3"
    assert env["HIP_VISIBLE_DEVICES"] == "2,3"


def test_visible_devices_reads_env() -> None:
    assert cuda.visible_devices(env={"CUDA_VISIBLE_DEVICES": "0,1"}) == ["0", "1"]
    assert xpu.visible_devices(env={"ZE_AFFINITY_MASK": "0.0"}) == ["0.0"]
    assert rocm.visible_devices(env={"HIP_VISIBLE_DEVICES": "1"}) == ["1"]
    assert cuda.visible_devices(env={}) is None


def test_crash_patterns_match_backend_signatures() -> None:
    samples = {
        cuda: (
            "ncclInternalError: Internal check failed.",
            "RuntimeError: CUDA error: device-side assert triggered",
            "torch.cuda.OutOfMemoryError: CUDA out of memory",
        ),
        xpu: (
            "UR_RESULT_ERROR_DEVICE_LOST",
            "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
            "oneccl_bindings_for_pytorch failed",
        ),
        rocm: (
            "Memory access fault by GPU node-4",
            "RuntimeError: HIP error: hipErrorOutOfMemory",
            "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
        ),
    }
    for module, lines in samples.items():
        rx = re.compile(module.crash_patterns(), flags=re.IGNORECASE)
        for line in lines:
            assert rx.search(line), f"{module.__name__} should match {line!r}"


def test_detect_accelerator_explicit() -> None:
    assert accelerators.detect_accelerator(explicit="cuda") == "cuda"
    assert accelerators.detect_accelerator(explicit="rocm") == "rocm"
    assert accelerators.detect_accelerator(explicit="none") == "none"
    with pytest.raises(ValueError):
        accelerators.detect_accelerator(explicit="tpu")


def test_detect_accelerator_env_override() -> None:
    assert (
        accelerators.detect_accelerator(env={"XPU_LAUNCH_ACCELERATOR": "rocm"})
        == "rocm"
    )
    assert (
        accelerators.detect_accelerator(env={"XPU_ACCELERATOR": "none"}) == "none"
    )


def test_detect_accelerator_none_when_nothing_available(monkeypatch) -> None:
    for module in ALL_BACKENDS:
        monkeypatch.setattr(module, "is_available", lambda env=None: False)
    assert accelerators.detect_accelerator(env={}) == "none"


def test_doctor_payload_shape() -> None:
    for module in ALL_BACKENDS:
        payload = module.doctor_payload(env={})
        assert payload["name"] == module.name()
        assert isinstance(payload["available"], bool)
        assert isinstance(payload["device_count"], int)
        assert payload["distributed_backend"]


def test_launch_crash_rx_augmented_by_active_accelerator() -> None:
    line = "ncclInternalError: Internal check failed."
    launch.set_active_accelerator(None)
    try:
        assert launch._get_crash_rx("generic").search(line) is None
        launch.set_active_accelerator("cuda")
        rx = launch._get_crash_rx("generic")
        assert rx.search(line)
        # Base patterns must survive augmentation.
        assert rx.search("OutOfMemoryError")
    finally:
        launch.set_active_accelerator(None)


def test_launch_parser_accepts_accelerator_flag() -> None:
    args = launch.parse_args(["--accelerator", "rocm", "--", "echo", "ok"])
    assert args.accelerator == "rocm"
    args = launch.parse_args(["--", "echo", "ok"])
    assert args.accelerator == "auto"
