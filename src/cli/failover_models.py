"""Postmortem-oriented failover data model for host swap bookkeeping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

PROVENANCE_SCRAPED = "scraped"
PROVENANCE_BLIND = "blind"


@dataclass(frozen=True)
class BadNodeRecord:
    host: str
    provenance: str
    attempt: Optional[int] = None

    def to_line(self) -> str:
        parts = [self.host, self.provenance]
        if self.attempt is not None:
            parts.append(f"attempt={self.attempt}")
        return "  ".join(parts)


def parse_bad_nodes_file(path: Path) -> list[BadNodeRecord]:
    records: list[BadNodeRecord] = []
    if not path.exists():
        return records

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if not fields:
            continue
        host = fields[0]
        provenance = fields[1] if len(fields) > 1 else ""
        attempt = None
        for tok in fields[2:]:
            if tok.startswith("attempt="):
                try:
                    attempt = int(tok.split("=", 1)[1])
                except ValueError:
                    attempt = None
        records.append(BadNodeRecord(host, provenance, attempt))

    return records


@dataclass
class NodeAllocation:
    active: list[str]
    spare: deque[str]
    hostfile_path: Path
    bad_nodes_path: Path
    bad_nodes: list[BadNodeRecord] = field(default_factory=list)

    @classmethod
    def from_full_nodelist(
        cls,
        nodelist: Sequence[str],
        nproc_active_hosts: int,
        hostfile_path: Path,
        bad_nodes_path: Path,
    ) -> "NodeAllocation":
        if nproc_active_hosts > len(nodelist):
            raise ValueError(
                f"need {nproc_active_hosts} active hosts but only {len(nodelist)} were given"
            )

        alloc = cls(
            active=list(nodelist[:nproc_active_hosts]),
            spare=deque(nodelist[nproc_active_hosts:]),
            hostfile_path=hostfile_path,
            bad_nodes_path=bad_nodes_path,
        )
        alloc._write_active()
        bad_nodes_path.write_text("", encoding="utf-8")
        return alloc

    @staticmethod
    def _node_key(host: str) -> str:
        head = host.split(".", 1)[0]
        return head.split("-hsn", 1)[0]

    def _match_active(self, host: str) -> Optional[str]:
        if host in self.active:
            return host
        key = self._node_key(host)
        for active in self.active:
            if self._node_key(active) == key:
                return active
        return None

    def _write_active(self) -> None:
        self.hostfile_path.write_text(
            "\n".join(self.active) + ("\n" if self.active else ""),
            encoding="utf-8",
        )

    def _append_bad(
        self,
        host: str,
        provenance: str,
        attempt: Optional[int] = None,
    ) -> None:
        rec = BadNodeRecord(host=host, provenance=provenance, attempt=attempt)
        self.bad_nodes.append(rec)
        with self.bad_nodes_path.open("a", encoding="utf-8") as fh:
            fh.write(rec.to_line() + "\n")

    @property
    def has_spares(self) -> bool:
        return len(self.spare) > 0

    def scraped_bad_hosts(self) -> list[str]:
        return [r.host for r in self.bad_nodes if r.provenance == PROVENANCE_SCRAPED]

    def blind_bad_hosts(self) -> list[str]:
        return [r.host for r in self.bad_nodes if r.provenance == PROVENANCE_BLIND]

    def swap_in(
        self,
        bad_hosts: Sequence[str],
        attempt: Optional[int] = None,
    ) -> list[tuple[str, str]]:
        swaps: list[tuple[str, str]] = []
        for bad in bad_hosts:
            active_host = self._match_active(bad)
            if active_host is None:
                continue
            if not self.spare:
                raise RuntimeError(f"out of spare nodes - cannot replace {active_host}")
            spare = self.spare.popleft()
            idx = self.active.index(active_host)
            self.active[idx] = spare
            self._append_bad(active_host, PROVENANCE_SCRAPED, attempt)
            swaps.append((active_host, spare))
        if swaps:
            self._write_active()
        return swaps

    def swap_one_blind(self, attempt: Optional[int] = None) -> tuple[str, str]:
        if not self.spare:
            raise RuntimeError("out of spare nodes - cannot blind-rotate")
        if not self.active:
            raise RuntimeError("active set is empty - nothing to rotate")

        bad = self.active[0]
        spare = self.spare.popleft()
        self.active[0] = spare
        self._append_bad(bad, PROVENANCE_BLIND, attempt)
        self._write_active()
        return bad, spare
