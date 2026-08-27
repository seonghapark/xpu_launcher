from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import scheduler_topology as st  # noqa: E402


def test_resolve_hostfile_uses_pbs_query_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "pbs.nodefile"
    candidate.write_text("hostA\n", encoding="utf-8")

    monkeypatch.delenv("HOSTFILE", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.delenv("COBALT_NODEFILE", raising=False)
    monkeypatch.setattr(st, "get_pbs_nodefile_of_active_job", lambda: str(candidate))

    resolved = st.resolve_hostfile(None, "pbs", workdir=tmp_path)
    assert resolved == str(candidate)


def test_resolve_hostfile_builds_slurm_hostfile_from_nodelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HOSTFILE", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.delenv("COBALT_NODEFILE", raising=False)
    monkeypatch.setenv("SLURM_NODELIST", "node[001-002]")

    resolved = st.resolve_hostfile(None, "slurm", workdir=tmp_path)
    assert resolved is not None

    out = Path(resolved)
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "node001\nnode002\n"


def test_get_slurm_active_jobid_fallbacks_to_running_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    monkeypatch.setattr(st.socket, "getfqdn", lambda: "node002.cluster")
    monkeypatch.setattr(st, "get_slurm_running_jobids", lambda: ["123", "456"])

    def _fake_nodelist(jobid: str) -> list[str]:
        return ["node001", "node002"] if jobid == "123" else ["node010"]

    monkeypatch.setattr(st, "get_nodelist_from_slurm_jobid", _fake_nodelist)

    assert st.get_slurm_active_jobid() == "123"
