from __future__ import annotations

from xpu_launch import resource_monitor


def test_cpu_percent() -> None:
    assert resource_monitor._cpu_percent((100, 40), (200, 60)) == 80.0
    assert resource_monitor._cpu_percent((100, 40), (100, 40)) is None


def test_parse_xpu_smi_csv() -> None:
    output = """Timestamp, DeviceId, GPU Utilization (%), GPU Power (W), GPU Energy Consumed (J), GPU Memory Used (MiB)
19:47:29.596, 0, 87.5, 271.10, 95554.67, 2048
"""
    assert resource_monitor._parse_csv_metrics(output) == [
        {
            "device_id": 0,
            "utilization_percent": 87.5,
            "power_watts": 271.1,
            "energy_joules": 95554.67,
            "memory_used_mib": 2048,
        }
    ]


def test_parse_cuda_smi_csv() -> None:
    output = """index,utilization.gpu [%],power.draw [W],memory.used [MiB],temperature.gpu
0,91,305.5,4096,67
"""
    assert resource_monitor._parse_csv_metrics(output) == [
        {
            "device_id": 0,
            "utilization_percent": 91,
            "power_watts": 305.5,
            "memory_used_mib": 4096,
            "temperature_celsius": 67,
        }
    ]


def test_summary_aggregates_power_and_energy() -> None:
    summary = resource_monitor._Summary()
    summary.add(
        {
            "cpu": {"used_percent": 50.0},
            "memory": {"used_bytes": 100, "used_percent": 25.0},
            "accelerators": [
                {"power_watts": 200.0, "utilization_percent": 80.0},
                {"power_watts": 250.0, "utilization_percent": 60.0},
            ],
        },
        elapsed_since_previous=2.0,
    )
    result = summary.as_dict(
        backend="xpu", started_at="start", elapsed_seconds=2.0
    )
    assert result["averages"]["accelerator_power_watts"] == 450.0
    assert result["averages"]["accelerator_utilization_percent"] == 70.0
    assert result["accelerator_energy_joules_estimate"] == 900.0


def test_parse_args_accepts_wrapped_command_without_separator() -> None:
    args = resource_monitor._parse_args(
        ["--output-dir", "/tmp/metrics", "python", "train.py", "--steps", "2"]
    )
    assert args.command == ["python", "train.py", "--steps", "2"]
