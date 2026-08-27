from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli.failover_models import (  # noqa: E402
    PROVENANCE_BLIND,
    PROVENANCE_SCRAPED,
    NodeAllocation,
    parse_bad_nodes_file,
)


def test_node_allocation_records_provenance(tmp_path: Path) -> None:
    hostfile = tmp_path / "active.hostfile"
    bad_nodes = tmp_path / "bad_nodes.txt"

    alloc = NodeAllocation.from_full_nodelist(
        nodelist=["hostA", "hostB", "hostC"],
        nproc_active_hosts=2,
        hostfile_path=hostfile,
        bad_nodes_path=bad_nodes,
    )

    swaps = alloc.swap_in(["hostA"], attempt=1)
    assert swaps == [("hostA", "hostC")]

    assert alloc.active == ["hostC", "hostB"]
    assert alloc.scraped_bad_hosts() == ["hostA"]

    parsed = parse_bad_nodes_file(bad_nodes)
    assert parsed[0].host == "hostA"
    assert parsed[0].provenance == PROVENANCE_SCRAPED


def test_blind_swap_records_provenance(tmp_path: Path) -> None:
    hostfile = tmp_path / "active.hostfile"
    bad_nodes = tmp_path / "bad_nodes.txt"

    alloc = NodeAllocation.from_full_nodelist(
        nodelist=["hostA", "hostB", "hostC"],
        nproc_active_hosts=2,
        hostfile_path=hostfile,
        bad_nodes_path=bad_nodes,
    )
    bad, spare = alloc.swap_one_blind(attempt=3)
    assert (bad, spare) == ("hostA", "hostC")

    parsed = parse_bad_nodes_file(bad_nodes)
    assert parsed[0].provenance == PROVENANCE_BLIND
