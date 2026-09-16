#!/usr/bin/env python3
"""Auto-update the ``Notes`` and ``Modified`` columns in
``docs/README.md`` tables.

The top-level ``docs/README.md`` has ~9 tables of the form

    | Page | Notes | Modified |
    |------|-------|---------:|
    | [Production Index](./production/README.md) | ... | 2026-05-23 |

Both columns rot. This script:

1. Parses every table-row that has a markdown link in the first cell.
2. Resolves the link target relative to ``docs/`` (the README's dir).
3. For each resolved file path:
   * Rewrites the ``Modified`` column with the last-commit date
     (``git log -1 --format=%cs``).
   * If the linked file has a recognisable canonical-chain status block
     (``**Latest checkpoint:** ...``, ``**Tokens consumed:** ...``,
     etc.), synthesises a fresh one-liner for ``Notes``.
4. Rows whose linked file is not a canonical chain (no recognisable
   status block) are left alone in the Notes column — that's narrative
   judgment.
5. Rows containing the sentinel ``<!-- noauto -->`` are not touched
   at all. Use this to keep hand-curated descriptions stable.

Usage (from repo root):
    python3 -m torchtitan.experiments.ezpz.utils.refresh_docs_readme_table

By default operates on ``torchtitan/experiments/ezpz/docs/README.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DEFAULT_README = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "README.md"
)

# Matches a table row whose first cell contains a markdown link.
ROW_RE = re.compile(r"^\|\s*\[[^\]]+\]\(([^)]+)\)\s*\|")

# Recognisable status-block fields inside per-trajectory READMEs.
LATEST_CKPT_RE = re.compile(r"^\*\*Latest checkpoint:\*\*\s*(.+)$", re.M)
CUMULATIVE_STEPS_RE = re.compile(r"^\*\*Cumulative(?: \*persisted\*)? steps:\*\*\s*(.+)$", re.M)
TOKENS_RE = re.compile(r"^\*\*Tokens consumed(?: \(persisted\))?:\*\*\s*(.+)$", re.M)
LOSS_RE = re.compile(r"^\*\*Loss:\*\*\s*(.+)$", re.M)

# Pull the trailing-percent/loss-number patterns out of a longer cell
# so the synthesised one-liner stays short.
# Handles both formats:
#   "= **4.047T tokens** (**86.6%** of 4.67T target)"
#   "= **3.07T tokens** (65.7% of 4.67T target)"
TOKEN_TOTAL_RE = re.compile(
    r"\*\*([\d.]+[KMBT])\s*tokens\*\*\s*\(\*{0,2}([\d.]+%)"
)
STEP_NUM_RE = re.compile(r"step-?([\d,]+)", re.I)
LOSS_NUM_RE = re.compile(r"^([\d.]+)")

# Opt-out sentinel
NOAUTO_SENTINEL = "<!-- noauto -->"

# Auto-generated "Recently Updated" region markers in docs/README.md.
RECENT_BEGIN = "<!-- BEGIN recently-updated (auto-generated) -->"
RECENT_END = "<!-- END recently-updated (auto-generated) -->"
RECENT_LIMIT = 25  # rows shown in the always-visible table
RECENT_EXTRA = 25  # additional rows tucked into the <details> block
# H1 title at the top of a markdown doc (first "# ..." line).
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)


def doc_title(path: Path) -> str:
    """First H1 heading in the doc, else its repo-relative stem."""
    try:
        m = H1_RE.search(path.read_text())
    except OSError:
        m = None
    if m:
        # Strip emoji/backtick noise lightly; keep it readable.
        return m.group(1).replace("`", "").strip()
    return path.stem


def _collect_doc_rows(readme_path: Path) -> list[tuple[str, str, str]]:
    """All tracked docs under docs/ as (date, title, rel_link), newest
    first. git commit date covers EVERY doc, not just those in the
    curated tables -- the manifest-independent freshness view.
    """
    repo_root = find_repo_root(readme_path)
    docs_dir = readme_path.parent
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", str(docs_dir.resolve().relative_to(repo_root)) + "/*.md"],
        capture_output=True, text=True, check=True,
    )
    rows: list[tuple[str, str, str]] = []
    for rel in out.stdout.split():
        abs_path = repo_root / rel
        if not abs_path.exists():
            continue
        date = git_last_commit_date(repo_root, abs_path)
        if not date:
            continue
        link = "./" + str(abs_path.resolve().relative_to(docs_dir.resolve()))
        rows.append((date, doc_title(abs_path), link))
    # Newest first; tie-break by path for determinism.
    rows.sort(key=lambda r: (r[0], r[2]), reverse=True)
    return rows


def _md_table(rows: list[tuple[str, str, str]]) -> list[str]:
    """Render (date, title, link) rows as a Modified|Doc markdown table."""
    lines = ["| Modified | Doc |", "|---------:|-----|"]
    for date, title, link in rows:
        lines.append(f"| {date} | [{title}]({link}) |")
    return lines


def build_recently_updated(
    readme_path: Path, *, limit: int = RECENT_LIMIT, extra: int = RECENT_EXTRA
) -> str:
    """Region body: the `limit` most-recent docs as a visible table, then
    the next `extra` tucked into a collapsed <details> block.
    """
    rows = _collect_doc_rows(readme_path)
    body = _md_table(rows[:limit])
    next_rows = rows[limit : limit + extra]
    if next_rows:
        lo, hi = limit + 1, limit + len(next_rows)
        body += [
            "",
            "<details>",
            f"<summary>Next {len(next_rows)} (#{lo}-{hi})</summary>",
            "",
            *_md_table(next_rows),
            "",
            "</details>",
        ]
    return "\n".join(body)


def refresh_recently_updated(readme_path: Path, *, dry_run: bool = False) -> bool:
    """Rewrite the BEGIN/END recently-updated region in place. Returns
    True if the region changed. No-op (with a warning) if the markers
    aren't present.
    """
    text = readme_path.read_text()
    if RECENT_BEGIN not in text or RECENT_END not in text:
        print("  [recently-updated] markers not found; skipping", file=sys.stderr)
        return False
    pre, rest = text.split(RECENT_BEGIN, 1)
    _, post = rest.split(RECENT_END, 1)
    table = build_recently_updated(readme_path)
    new_text = f"{pre}{RECENT_BEGIN}\n{table}\n{RECENT_END}{post}"
    if new_text == text:
        print("  [recently-updated] already current")
        return False
    if not dry_run:
        readme_path.write_text(new_text)
    print(f"  [recently-updated] table refreshed{' (dry-run)' if dry_run else ''}")
    return True

# Default token-budget label when a README isn't in the manifest (every
# current trajectory shares the olmo-mix-1124 4.67T budget).
_DEFAULT_TOKEN_TARGET_LABEL = "4.67T"


def _fmt_token_target(total: int) -> str:
    """Short human label for a token target, e.g. 4_673_780_159_710 -> '4.67T'."""
    if total >= 1e12:
        return f"{total / 1e12:.2f}T"
    return f"{total / 1e9:.0f}B"


def _token_target_label(readme_path: Path) -> str:
    """Resolve the per-page token-budget label from the trajectory
    manifest by matching the linked README path. Falls back to the
    shared olmo-mix default if the page isn't a tracked trajectory.
    """
    try:
        from torchtitan.experiments.ezpz.utils.trajectories import TRAJECTORIES
    except Exception:
        return _DEFAULT_TOKEN_TARGET_LABEL
    rp = str(readme_path.resolve())
    for t in TRAJECTORIES:
        rdm = t.get("readme")
        if rdm and rp.endswith(rdm):
            return _fmt_token_target(t["token_target"])
    return _DEFAULT_TOKEN_TARGET_LABEL


def git_last_commit_date(repo_root: Path, path: Path) -> str | None:
    """Return YYYY-MM-DD of the last commit touching ``path``."""
    rel = path.resolve().relative_to(repo_root)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cs", "--", str(rel)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        return None
    date = out.stdout.strip()
    return date or None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"no .git found walking up from {start}")


def synthesize_notes(target_path: Path) -> str | None:
    """Read the linked file and synthesise a one-line Notes blurb if
    it has the canonical-chain status block. Return None otherwise.
    """
    try:
        text = target_path.read_text()
    except (OSError, UnicodeDecodeError):
        return None

    latest = LATEST_CKPT_RE.search(text)
    if not latest:
        return None  # not a canonical-chain page

    latest_str = latest.group(1).strip()
    tokens_str = TOKENS_RE.search(text)
    loss_str = LOSS_RE.search(text)

    # Pull step number out of the latest-checkpoint sentence.
    step_m = STEP_NUM_RE.search(latest_str)
    step_part = f"step-**{step_m.group(1)}**" if step_m else latest_str.split(" ")[0]

    # Pull "X.XXT tokens (YY%)" tidily. The "of <target>" suffix comes
    # from the trajectory manifest (per-page token_target) rather than a
    # hardcoded 4.67T, so a future non-olmo-mix model labels correctly.
    tokens_part = ""
    if tokens_str:
        ts = tokens_str.group(1).strip()
        m = TOKEN_TOTAL_RE.search(ts)
        if m:
            target_label = _token_target_label(target_path)
            tokens_part = f" ({m.group(1)} tokens, {m.group(2)} of {target_label})"
        else:
            tokens_part = f" ({ts.split('(')[0].strip()})"

    # Loss — first numeric token only.
    loss_part = ""
    if loss_str:
        ls = loss_str.group(1).strip()
        lm = LOSS_NUM_RE.match(ls)
        if lm:
            loss_part = f", loss {lm.group(1)}"

    return f"{step_part}{tokens_part}{loss_part}."


def split_row(line: str) -> list[str] | None:
    """Split a markdown table row into its raw cell contents (between
    the leading and trailing ``|``). Returns None if the line isn't a
    table row.
    """
    if not line.startswith("|") or not line.rstrip().endswith("|"):
        return None
    s = line.strip()
    s = s[1:-1]  # drop leading + trailing |
    # Markdown tables don't support escaped pipes in our usage; safe split.
    return s.split("|")


def join_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def refresh_readme(readme_path: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite rows in ``readme_path``. Returns (rows_with_changes,
    notes_rewritten).
    """
    repo_root = find_repo_root(readme_path)
    readme_dir = readme_path.parent

    lines = readme_path.read_text().splitlines(keepends=False)
    updated_lines: list[str] = []
    n_changed = 0
    n_notes_rewrites = 0

    for line in lines:
        m = ROW_RE.match(line)
        if not m:
            updated_lines.append(line)
            continue

        if NOAUTO_SENTINEL in line:
            updated_lines.append(line)
            continue

        target_str = m.group(1)
        if "://" in target_str or target_str.startswith("mailto:"):
            updated_lines.append(line)
            continue

        cells = split_row(line)
        if cells is None or len(cells) < 3:
            updated_lines.append(line)
            continue

        target_path = (readme_dir / target_str).resolve()
        if not target_path.exists():
            print(f"  [missing] {target_str} — leaving row unchanged",
                  file=sys.stderr)
            updated_lines.append(line)
            continue

        new_date = git_last_commit_date(repo_root, target_path)
        new_notes = synthesize_notes(target_path)

        old_date = cells[-1].strip()
        old_notes = cells[1].strip()

        any_change = False

        if new_date and new_date != old_date:
            cells[-1] = f" {new_date} "
            any_change = True
            print(f"  [date {old_date} -> {new_date}] {target_str}")

        if new_notes and new_notes != old_notes:
            cells[1] = f" {new_notes} "
            any_change = True
            n_notes_rewrites += 1
            print(f"  [notes rewrite] {target_str}")
            print(f"      OLD: {old_notes[:80]}{'...' if len(old_notes) > 80 else ''}")
            print(f"      NEW: {new_notes}")

        if any_change:
            updated_lines.append(join_row(cells))
            n_changed += 1
        else:
            updated_lines.append(line)

    if n_changed and not dry_run:
        readme_path.write_text("\n".join(updated_lines) + "\n")

    # Refresh the auto-generated "Recently Updated" region (reads the file
    # back so it sees the row-date rewrites above).
    refresh_recently_updated(readme_path, dry_run=dry_run)

    suffix = " (dry-run)" if dry_run else ""
    print(
        f"\n=== {n_changed} row(s) updated, {n_notes_rewrites} notes-column "
        f"rewrites{suffix} ===",
    )
    return n_changed, n_notes_rewrites


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
    )
    parser.add_argument(
        "readme",
        nargs="?",
        type=Path,
        default=DEFAULT_README,
        help=f"path to README.md (default: {DEFAULT_README})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show planned changes without rewriting the file",
    )
    args = parser.parse_args()

    refresh_readme(args.readme, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
