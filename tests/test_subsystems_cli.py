from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import main  # noqa: E402


def test_subsystem_commands_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "launch" in result.output
    assert "doctor" in result.output
    assert "submit" in result.output
    assert "dist" in result.output
    assert "integrations" in result.output


def test_doctor_command_outputs_json() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "xpu" in payload
    assert "checks" in payload["xpu"]


def test_submit_generates_script(tmp_path: Path) -> None:
    runner = CliRunner()
    script_path = tmp_path / "job.sh"
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "submit",
            "--hostfile",
            str(hostfile),
            "--nproc",
            "8",
            "--nproc-per-node",
            "4",
            "--command",
            "python train.py",
            "--script-path",
            str(script_path),
            "--no-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["script_path"] == str(script_path)
    assert script_path.is_file()
    text = script_path.read_text(encoding="utf-8")
    assert "python train.py" in text


def test_submit_aurora_directive_mapping(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    script_path = tmp_path / "aurora.sh"
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    monkeypatch.setitem(
        main.commands["submit"].callback.__globals__,
        "get_machine_name",
        lambda: "aurora",
    )

    result = runner.invoke(
        main,
        [
            "submit",
            "--hostfile",
            str(hostfile),
            "--nproc",
            "8",
            "--nproc-per-node",
            "4",
            "--command",
            "python train.py",
            "--script-path",
            str(script_path),
            "--no-run",
        ],
    )
    assert result.exit_code == 0
    text = script_path.read_text(encoding="utf-8")
    assert "#PBS -l filesystems=home:flare" in text
    assert "#PBS -l place=scatter" in text


def test_submit_polaris_directive_mapping(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    script_path = tmp_path / "polaris.sh"
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    monkeypatch.setitem(
        main.commands["submit"].callback.__globals__,
        "get_machine_name",
        lambda: "polaris",
    )

    result = runner.invoke(
        main,
        [
            "submit",
            "--hostfile",
            str(hostfile),
            "--nproc",
            "8",
            "--nproc-per-node",
            "4",
            "--command",
            "python train.py",
            "--script-path",
            str(script_path),
            "--no-run",
        ],
    )
    assert result.exit_code == 0
    text = script_path.read_text(encoding="utf-8")
    assert "#PBS -l filesystems=home:eagle" in text
    assert "#PBS -l place=scatter" in text


def test_dist_validate_outputs_ok_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("hostA\nhostB\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "dist",
            "validate",
            "--hostfile",
            str(hostfile),
            "--nproc",
            "8",
            "--nproc-per-node",
            "4",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "ok" in payload


def test_doctor_reports_mpi_pmi_checks() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    checks = payload["xpu"]["checks"]
    names = {c["name"] for c in checks}
    assert "mpi.launcher.detected" in names
    assert "mpi.launcher.family" in names
    assert "mpi.libs.detected" in names
    assert "pmi.env.detected" in names
    assert "pmi.libs.compatibility" in names

    mpi_check = next(c for c in checks if c["name"] == "mpi.libs.detected")
    assert "launcher_path" in mpi_check
    assert "launcher_realpath" in mpi_check
    assert "launcher_family" in mpi_check
    assert "mpi_sonames" in mpi_check
    assert "pmi_sonames" in mpi_check
    assert isinstance(mpi_check["mpi_sonames"], list)
    assert isinstance(mpi_check["pmi_sonames"], list)
