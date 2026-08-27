from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import launch  # noqa: E402


def _capture_run(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with patch("sys.stdout", new=out), patch("sys.stderr", new=err):
        rc = launch.run(argv)
    return rc, out.getvalue().strip(), err.getvalue().strip()


def test_dry_run_snapshot_slurm_srun(tmp_path: Path) -> None:
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    with patch("cli.launch.choose_launcher_binary", return_value="srun"):
        rc, stdout, _stderr = _capture_run(
            [
                "--scheduler",
                "slurm",
                "--hostfile",
                str(hostfile),
                "-n",
                "8",
                "-ppn",
                "4",
                "--dry-run",
                "--",
                "python",
                "train.py",
            ]
        )

    assert rc == 0
    assert (
        stdout
        == f"srun -u --verbose -N 2 -n 8 --ntasks-per-node 4 --nodelist={hostfile.resolve()} python train.py"
    )


def test_dry_run_snapshot_pbs_mpiexec(tmp_path: Path) -> None:
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    with patch("cli.launch.choose_launcher_binary", return_value="mpiexec"):
        rc, stdout, _stderr = _capture_run(
            [
                "--scheduler",
                "pbs",
                "--hostfile",
                str(hostfile),
                "-n",
                "8",
                "-ppn",
                "4",
                "--dry-run",
                "--",
                "python",
                "train.py",
            ]
        )

    assert rc == 0
    assert stdout == f"mpiexec -n 8 --ppn 4 --hostfile {hostfile} python train.py"


def test_auto_retry_reinjects_srun_exclude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\nhostC\n", encoding="utf-8")

    commands: list[list[str]] = []

    def _fake_runner(cmd: list[str], idle_timeout_s: int) -> tuple[int, str]:
        commands.append(list(cmd))
        if len(commands) == 1:
            return 1, "RuntimeError: [gloo] Connection closed by peer on hostA"
        return 0, "ok"

    monkeypatch.setattr(launch, "_run_with_watchdog_capture", _fake_runner)
    monkeypatch.setattr(launch.time, "sleep", lambda _x: None)
    monkeypatch.setattr(launch, "choose_launcher_binary", lambda _s: "srun")

    out = io.StringIO()
    err = io.StringIO()
    with patch("sys.stdout", new=out), patch("sys.stderr", new=err):
        rc = launch.run(
            [
                "--scheduler",
                "slurm",
                "--auto-retry",
                "--hostfile",
                str(hostfile),
                "--nhosts",
                "2",
                "--spare-nodes",
                "1",
                "--",
                "python",
                "train.py",
            ]
        )

    assert rc == 0
    assert len(commands) == 2

    first = " ".join(commands[0])
    second = " ".join(commands[1])
    assert "--exclude=" not in first
    assert "--exclude=hostA" in second
    assert "[auto-retry] FAILOVER STOP: success" in err.getvalue()
