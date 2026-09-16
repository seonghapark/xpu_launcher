#!/usr/bin/env python3
"""scrape_bad_nodes.py — extract bad-node hostnames from a training log.

Reads the log file given on the command line and prints one bad
hostname per line on stdout (deduplicated, in first-seen order).

Detected patterns (each tied to a specific failure mode we've seen
in production training):

  1. "<hostname>: shepherd died from signal 9"
     PALS shepherd kill — node went bad mid-run (Aurora's most
     common failure mode). 8459818 / 8460301 / 8463659 all hit this.
     We do NOT match `rank N died from signal {11,15}` because those
     are almost always cascading deaths from a primary kill on a
     *different* node — including them would falsely tag innocent
     nodes (e.g. 8466848 had rank 1304 die from signal 11 as a
     downstream effect of a `std::bad_alloc` on rank 2413).

  2. "Connection closed by peer [IP]:port" / "Timed out waiting Nms for recv"
     gloo TCP failure — usually one bad node taking down peer
     connections from many ranks. 8470102 / 8470103 / 8479581 hit this.
     We resolve the IP via `getent hosts` to get the hostname.
     Note that gloo errors usually point at a single peer IP across
     many "Connection closed" lines, so the deduplication output is
     typically just one node.

If no specific bad hostname is identifiable (e.g. the
`set_determinism` `std::bad_alloc` init crash), nothing is printed
and the caller (failover_lib.sh) falls back to rotating one spare in
blindly.

Hostname normalization: PBS hostfile entries use the
`.hsn.cm.aurora.alcf.anl.gov` form. We always emit hostnames in that
form regardless of whether the log used `.hostmgmtNNNN.cm.aurora...`
or some other suffix.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Pattern 1: "<host>.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9"
# Excludes "rank N died from signal {11,15}" because those are almost
# always cascading deaths downstream of a primary kill elsewhere.
SIGNAL_RX = re.compile(
    r"^([a-zA-Z0-9.-]+\.hsn\.cm\.aurora\.alcf\.anl\.gov):\s+"
    r"shepherd\s+died\s+from\s+signal\s+9\b",
    re.MULTILINE,
)

# Pattern 2: "RuntimeError: ... Connection closed by peer [IP]:port"
GLOO_PEER_RX = re.compile(
    r"Connection closed by peer\s+\[([0-9.]+)\]:\d+",
    re.MULTILINE,
)

# Aurora compute hostname pattern — matches xNNNNcNsNbNnN.hsn.cm.aurora...
AURORA_HSN_RX = re.compile(r"^x\d+c\d+s\d+b\d+n\d+\.hsn\.cm\.aurora\.alcf\.anl\.gov$")


def normalize_hostname(host: str) -> str | None:
    """Return the canonical `.hsn.cm.aurora.alcf.anl.gov` form, or
    None if `host` doesn't look like a valid Aurora compute hostname.
    """
    # Already in canonical form
    if AURORA_HSN_RX.match(host):
        return host
    # `xNNNNcNsNbNnN.<other-suffix>.cm.aurora.alcf.anl.gov` → strip suffix
    m = re.match(r"^(x\d+c\d+s\d+b\d+n\d+)\.", host)
    if m:
        return f"{m.group(1)}.hsn.cm.aurora.alcf.anl.gov"
    return None


def resolve_ip(ip: str) -> str | None:
    """Reverse-resolve an IP to a hostname via `getent hosts`. Returns
    a canonical Aurora hostname, or None if the lookup fails or
    the result doesn't look like a compute node.
    """
    try:
        out = subprocess.check_output(
            ["getent", "hosts", ip], text=True, timeout=5
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if not out:
        return None
    # `getent hosts <ip>` → `<ip>  <hostname1> <hostname2>...`
    # Walk all returned names, return the first one that normalizes.
    parts = out.split()
    for p in parts[1:]:
        canonical = normalize_hostname(p)
        if canonical:
            return canonical
    return None


def scrape(log_path: Path) -> list[str]:
    """Return deduplicated list of bad hostnames in first-seen order."""
    text = log_path.read_text(errors="replace")
    seen: list[str] = []
    seen_set: set[str] = set()

    for m in SIGNAL_RX.finditer(text):
        host = normalize_hostname(m.group(1))
        if host and host not in seen_set:
            seen.append(host)
            seen_set.add(host)

    for m in GLOO_PEER_RX.finditer(text):
        ip = m.group(1)
        host = resolve_ip(ip)
        if host and host not in seen_set:
            seen.append(host)
            seen_set.add(host)

    return seen


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scrape_bad_nodes.py <log-file>", file=sys.stderr)
        return 2
    log_path = Path(argv[1])
    if not log_path.is_file():
        print(f"log file not found: {log_path}", file=sys.stderr)
        return 1
    for host in scrape(log_path):
        print(host)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
