#!/usr/bin/env python3
"""Auto-fill the DETERMINISTIC structured fields in production-tracking
trajectory READMEs from on-disk checkpoint state (and optionally W&B).

This is the per-leaf companion to ``refresh_docs_readme_table.py`` (which
owns the top-level ``docs/README.md`` table). It iterates the LIVE
trajectories in ``trajectories.py`` and rewrites ONLY scalar, mechanically
derivable fields in each trajectory's README:

  - ``> Last updated: YYYY-MM-DD``                  -> today (only if the
        marker exists on its own line; never injected into Status prose)
  - ``**Latest checkpoint:** step-N ...``           -> largest VALID step
  - ``**Cumulative steps:** N``                     -> same step number
  - ``**Tokens consumed:** N x GBS x SEQ = X.XXXT (YY.Y%) ...`` -> arithmetic
  - ``**Loss:** L ...``                             -> latest W&B loss
        (only with --wandb; left untouched otherwise)

Scope note: this fills the per-trajectory LEAF READMEs only. Rollup
snapshot cells in docs/production/README.md and docs/production/agpt/
README.md are NOT auto-propagated here (their numbers were hand-set and
carry surrounding narrative); refresh those manually or via the
docs/README.md index table that refresh_docs_readme_table.py owns.

HARD RULES (mirrors the design + CLAUDE.md):
  - NEVER rewrite narrative prose, Status paragraphs, Progress/Logs
    tables, or the parentheticals after a value. Only the scalar token.
  - "VALID" step = a ``step-N`` dir (digits only) that has ``.metadata``
    AND at least one ``*.distcp`` shard. Mid-save/empty dirs are skipped
    (e.g. 20B-512N step-4500 has 0 .distcp -> the latest valid is 4400).
  - Preserve each file's existing punctuation/precision/unit/asterisk
    style on write; normalize only on read. Write only on change.
  - Conflict guard: if the chosen valid step DISAGREES with the step the
    README's prose asserts in a way implying human disagreement, warn and
    SKIP that field rather than overwrite.

Usage:
  python -m torchtitan.experiments.ezpz.utils.fill_trajectory_fields [--dry-run] [--wandb] [--model KEY]
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

from torchtitan.experiments.ezpz.utils.trajectories import (
    REPO_ROOT,
    by_key,
    live_trajectories,
)

# Reuse the leaf-field regexes already proven in refresh_docs_readme_table
# so the two fillers parse identically. We widen a couple to also accept
# the ``*complete*`` / ``*persisted*`` italic variants some pages use.
NOAUTO_SENTINEL = "<!-- noauto -->"

# Tolerant read regexes (match the label, capture the rest of the line).
LATEST_CKPT_RE = re.compile(
    r"^(?P<label>\*\*Latest(?: \*\w+\*)? checkpoint:\*\*\s*)(?P<body>.+)$", re.M
)
CUMULATIVE_STEPS_RE = re.compile(
    # tolerate "Cumulative steps", "Cumulative *persisted* steps" (italic),
    # and "Cumulative persisted steps" (plain) -- different pages vary.
    r"^(?P<label>\*\*Cumulative(?: \*?\w+\*?)? steps:\*\*\s*)(?P<body>.+)$", re.M
)
TOKENS_RE = re.compile(
    r"^(?P<label>\*\*Tokens consumed(?: \(persisted\))?:\*\*\s*)(?P<body>.+)$", re.M
)
LOSS_RE = re.compile(r"^(?P<label>\*\*Loss:\*\*\s*)(?P<body>.+)$", re.M)
LAST_UPDATED_RE = re.compile(
    r"^(?P<pre>>\s*\*{0,2}Last updated:?\*{0,2}\s*)(?P<date>\d{4}-\d{2}-\d{2})(?P<post>.*)$",
    re.M,
)

# step-86,200 or step-30500 (optional comma) — used to read/replace the
# step token inside a longer sentence without touching the rest.
STEP_TOKEN_RE = re.compile(r"step-\*{0,2}([\d,]+)\*{0,2}")
# GBS from the config table row: | GBS | 6,144 (LBS=2) |
GBS_ROW_RE = re.compile(r"^\|\s*GBS\s*\|\s*([\d,]+)", re.M)
# First numeric in a loss body, e.g. "2.656 (last log ...)" -> 2.656
LEADING_NUM_RE = re.compile(r"^([\d.]+)")


def _step_is_valid(step_dir: Path) -> bool:
    """A checkpoint dir is valid iff it has a ``.metadata`` file AND at
    least one ``*.distcp`` shard (i.e. the save finalized; not mid-write).

    Any OSError (perms, race during scan) degrades to "not valid" so one
    unreadable step dir can't abort the whole refresh.
    """
    try:
        if not (step_dir / ".metadata").exists():
            return False
        return any(p.suffix == ".distcp" for p in step_dir.iterdir())
    except OSError:
        return False


def largest_valid_step(ckpt_dir: str | Path) -> int | None:
    """Return the largest step-N (digits only) under ``ckpt_dir`` whose
    save finalized (see ``_step_is_valid``). Skips mid-save/empty dirs.
    Returns None if no valid checkpoint exists.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    best: int | None = None
    for child in d.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"step-(\d+)", child.name)
        if not m:
            continue  # excludes .bak-*, -SCALING-PROBE, etc.
        step = int(m.group(1))
        if best is not None and step <= best:
            continue
        if _step_is_valid(child):
            best = step
    return best


def _stated_step_dir_is_empty(ckpt_dir: str | Path, stated: int) -> bool:
    """True if the README's stated step points at a dir on disk that is
    present but INVALID (mid-save / empty). In that case the README's
    higher number is itself wrong and disk truth should win.
    """
    sd = Path(ckpt_dir) / f"step-{stated}"
    return sd.is_dir() and not _step_is_valid(sd)


def _fmt_int(n: int, *, comma: bool) -> str:
    return f"{n:,}" if comma else str(n)


def format_tokens(
    step: int,
    gbs: int,
    seq_len: int,
    target: int,
    *,
    t_decimals: int = 3,
    b_decimals: int = 1,
    pct_decimals: int = 1,
) -> tuple[str, str]:
    """Return (tokens_str, percent_str) e.g. ("4.339T", "92.9%").

    Unit chosen by magnitude: >=1e12 -> T, else B. Decimal precision for
    each unit + the percent is caller-supplied so the rewrite can match
    the file's EXISTING precision (avoids 3.07T -> 3.070T churn).
    """
    tokens = step * gbs * seq_len
    if tokens >= 1e12:
        tok_str = f"{tokens / 1e12:.{t_decimals}f}T"
    else:
        tok_str = f"{tokens / 1e9:.{b_decimals}f}B"
    pct = f"{100 * tokens / target:.{pct_decimals}f}%"
    return tok_str, pct


def _decimals_in(num_str: str) -> int:
    """Count decimal places in a numeric string like '3.07' -> 2, '453' -> 0."""
    if "." not in num_str:
        return 0
    return len(num_str.split(".", 1)[1])


def _replace_step_token(body: str, step: int) -> str:
    """Replace the first ``step-N`` token in ``body`` with ``step-<step>``,
    PRESERVING the comma style and any surrounding ``**`` already present.
    Everything else in ``body`` (the parenthetical) is untouched.
    """
    m = STEP_TOKEN_RE.search(body)
    if not m:
        return body
    had_comma = "," in m.group(1)
    # Preserve bold-inside-token if the original had it.
    bold = body[m.start():m.end()].count("**") >= 2
    new_num = _fmt_int(step, comma=had_comma)
    new_tok = f"step-**{new_num}**" if bold else f"step-{new_num}"
    return body[: m.start()] + new_tok + body[m.end():]


def _read_step_from_body(body: str) -> int | None:
    m = STEP_TOKEN_RE.search(body)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


class FieldChange:
    __slots__ = ("field", "old", "new")

    def __init__(self, field: str, old: str, new: str):
        self.field = field
        self.old = old
        self.new = new


def fill_one(
    traj: dict, *, today: str, wandb_loss: float | None, dry_run: bool
) -> tuple[list[FieldChange], list[str]]:
    """Fill scalar fields in one trajectory README. Returns (changes,
    warnings). Writes the file unless dry_run.
    """
    changes: list[FieldChange] = []
    warns: list[str] = []
    readme = REPO_ROOT / traj["readme"]
    if not readme.is_file():
        warns.append(f"{traj['key']}: README not found at {traj['readme']}")
        return changes, warns

    text = readme.read_text()
    if NOAUTO_SENTINEL in text.split("\n", 1)[0]:
        warns.append(f"{traj['key']}: <!-- noauto --> at file head; skipped")
        return changes, warns

    step = largest_valid_step(traj["ckpt_dir"])
    if step is None:
        warns.append(f"{traj['key']}: no valid checkpoint on disk; skipped")
        return changes, warns

    # GBS: prefer the file's own config-table value (authoritative for the
    # displayed arithmetic); fall back to the manifest.
    gbs_m = GBS_ROW_RE.search(text)
    gbs = int(gbs_m.group(1).replace(",", "")) if gbs_m else traj["gbs"]
    if gbs != traj["gbs"]:
        warns.append(
            f"{traj['key']}: README GBS {gbs} != manifest GBS {traj['gbs']}; using README value"
        )

    # (Token total + percent are computed inside the Tokens-consumed
    # branch below so they can match the file's existing precision.)

    new_text = text

    # --- Latest checkpoint: replace step token only ---
    m = LATEST_CKPT_RE.search(new_text)
    if m:
        stated = _read_step_from_body(m.group("body"))
        # Conflict guard: if the README asserts a HIGHER step than disk-valid,
        # default to NOT rolling back (the human may know of a ckpt we can't
        # see). EXCEPTION: if that stated step's dir is present-but-invalid
        # (mid-save/empty), the README number is provably wrong -> disk wins.
        if (
            stated is not None
            and stated > step
            and not _stated_step_dir_is_empty(traj["ckpt_dir"], stated)
        ):
            warns.append(
                f"{traj['key']}: README Latest-checkpoint step {stated} > disk-valid {step}; "
                "skipping (human-asserted higher; verify)"
            )
        else:
            if stated is not None and stated > step:
                warns.append(
                    f"{traj['key']}: README Latest-checkpoint step {stated} points at an "
                    f"empty/mid-save dir; correcting to disk-valid {step}"
                )
            new_body = _replace_step_token(m.group("body"), step)
            if new_body != m.group("body"):
                changes.append(FieldChange("Latest checkpoint", m.group("body"), new_body))
                new_text = new_text[: m.start("body")] + new_body + new_text[m.end("body"):]

    # --- Cumulative steps: the bare number ---
    m = CUMULATIVE_STEPS_RE.search(new_text)
    if m:
        body = m.group("body")
        stated = _read_step_from_body("step-" + body) if re.match(r"[\d,]+", body) else None
        new_num = _fmt_int(step, comma=("," in body))
        # Replace only the leading number token, keep any trailing prose.
        nb = re.sub(r"^\*{0,2}[\d,]+\*{0,2}", lambda mm: (
            f"**{new_num}**" if mm.group(0).count("**") >= 2 else new_num
        ), body, count=1)
        guard_skip = (
            stated is not None
            and stated > step
            and not _stated_step_dir_is_empty(traj["ckpt_dir"], stated)
        )
        if guard_skip:
            warns.append(
                f"{traj['key']}: README Cumulative steps {stated} > disk-valid {step}; skipping"
            )
        elif nb != body:
            changes.append(FieldChange("Cumulative steps", body, nb))
            new_text = new_text[: m.start("body")] + nb + new_text[m.end("body"):]

    # --- Tokens consumed: rebuild "N x GBS x SEQ = TOK (PCT of TARGET)"
    # PRESERVING the file's existing decimal precision (no 3.07->3.070
    # churn) and any remainder AFTER the closing paren (a trailing clause
    # like "; persisted" must survive). ---
    m = TOKENS_RE.search(new_text)
    if m:
        body = m.group("body")
        # Match existing token total + percent to clone their precision.
        tok_old = re.search(r"([\d.]+)\s*([TB])\s*tokens", body)
        pct_old = re.search(r"([\d.]+)\s*%", body)
        t_dec = _decimals_in(tok_old.group(1)) if tok_old else 3
        b_dec = _decimals_in(tok_old.group(1)) if (tok_old and tok_old.group(2) == "B") else 1
        pct_dec = _decimals_in(pct_old.group(1)) if pct_old else 1
        tok_str2, pct2 = format_tokens(
            step, gbs, traj["seq_len"], traj["token_target"],
            t_decimals=t_dec, b_decimals=b_dec, pct_decimals=pct_dec,
        )
        step_comma = _fmt_int(step, comma=True)
        gbs_comma = _fmt_int(gbs, comma=True)
        seq_comma = _fmt_int(traj["seq_len"], comma=True)
        tok_bold = bool(re.search(r"\*\*[\d.]+[TB]\s*tokens\*\*", body))
        pct_bold = bool(re.search(r"\(\*\*[\d.]+%", body))
        tok_render = f"**{tok_str2} tokens**" if tok_bold else f"{tok_str2} tokens"
        pct_render = f"**{pct2}**" if pct_bold else pct2
        # Preserve the "of <target>" text inside the paren if present.
        of_text = ""
        of_m = re.search(r"%\*{0,2}\s*(of [^)]+)\)", body)
        if of_m:
            of_text = " " + of_m.group(1).rstrip()
        # Preserve anything AFTER the first ")" that closes the percent
        # parenthetical (a trailing prose clause). Find that close-paren.
        close_idx = body.find(")")
        remainder = body[close_idx + 1:] if close_idx != -1 else ""
        nb = (
            f"{step_comma} x {gbs_comma} x {seq_comma} = "
            f"{tok_render} ({pct_render}{of_text})" + remainder
        )
        if " × " in body:  # honor the file's existing multiply glyph
            nb = nb.replace(" x ", " × ")
        if nb != body:
            changes.append(FieldChange("Tokens consumed", body, nb))
            new_text = new_text[: m.start("body")] + nb + new_text[m.end("body"):]

    # --- Loss: only with --wandb ---
    if wandb_loss is not None:
        m = LOSS_RE.search(new_text)
        if m:
            body = m.group("body")
            lm = LEADING_NUM_RE.match(body)
            if lm:
                new_loss = f"{wandb_loss:.5g}"
                nb = new_loss + body[lm.end():]
                if nb != body:
                    changes.append(FieldChange("Loss", body, nb))
                    new_text = new_text[: m.start("body")] + nb + new_text[m.end("body"):]

    # --- Last updated date ---
    m = LAST_UPDATED_RE.search(new_text)
    if m and m.group("date") != today:
        changes.append(FieldChange("Last updated", m.group("date"), today))
        new_text = (
            new_text[: m.start("date")] + today + new_text[m.end("date"):]
        )
    elif not m:
        warns.append(
            f"{traj['key']}: no standalone '> Last updated:' line "
            "(date likely lives in Status prose); left untouched"
        )

    if changes and not dry_run:
        readme.write_text(new_text)

    return changes, warns


def _wandb_latest_loss(traj: dict) -> float | None:
    """Latest global_avg_loss across the trajectory's W&B runs. Best-effort;
    returns None on any failure (so the disk path never blocks on W&B).
    """
    run_ids = traj.get("wandb_run_ids") or []
    if not run_ids:
        return None
    try:
        import wandb  # noqa: PLC0415

        from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: PLC0415
            PROJECT,
        )

        api = wandb.Api()
        # Walk run-ids newest-first; take the last finite loss we find.
        for rid in reversed(run_ids):
            try:
                run = api.run(f"{PROJECT}/{rid}")
            except Exception:
                continue
            last = None
            for row in run.scan_history(keys=["loss_metrics/global_avg_loss"]):
                v = row.get("loss_metrics/global_avg_loss")
                if v is not None:
                    last = v
            if last is not None:
                return float(last)
    except Exception as e:  # noqa: BLE001
        print(f"  [wandb] loss lookup failed for {traj['key']}: {e}", file=sys.stderr)
    return None


# Rollup pages whose Snapshot tables restate each leaf's step/loss/tokens.
# Each row links to a leaf README; we match the row by that link and
# rewrite ONLY the numeric cells, never the Status/narrative cell.
# (The cross-model agpt/README.md is intentionally absent: it is pure
# narrative Headlines with no snapshot table.)
#
# The top-level docs/production/README.md is ALSO intentionally absent: its
# "Status-at-a-glance" dashboard uses a different column schema
# (Trajectory | State | Persisted step | Loss | % target | Trend) than the
# per-model rollup tables (... | Cumulative steps | Loss | Tokens). The
# rollup propagator assumes the per-model schema, so pointing it at the
# dashboard mangled the Loss/% cells (e.g. wrote "86,200.656" into Loss).
# The dashboard's trend-lights + narrative State are hand-curated; leave it
# to manual upkeep.
ROLLUP_PAGES = [
    "torchtitan/experiments/ezpz/docs/production/agpt/2b/README.md",
    "torchtitan/experiments/ezpz/docs/production/agpt/20b/README.md",
]


def _canon_readme(repo_rel: str) -> str:
    """Normalize a leaf README repo-relative path to its canonical
    docs-relative form, e.g.
    'torchtitan/.../docs/production/agpt/2b/n256/README.md' ->
    'production/agpt/2b/n256/README.md'. Used as the disambiguating key
    so 2B-256N and 20B-256N never collide (both end '.../n256/README.md'
    but differ in the model segment)."""
    marker = "docs/"
    i = repo_rel.find(marker)
    return repo_rel[i + len(marker):] if i != -1 else repo_rel


def _resolve_rollup_link(page_rel: str, link: str) -> str:
    """Resolve a markdown link found inside a rollup page to the same
    canonical docs-relative form as _canon_readme. ``page_rel`` is the
    rollup's repo-relative path; ``link`` is the (possibly relative)
    link target from the table row (e.g. 'n256/README.md',
    'agpt/2b/n256/README.md', or '2b/n256/README.md')."""
    page_docs = _canon_readme(page_rel)           # e.g. production/agpt/2b/README.md
    page_dir = "/".join(page_docs.split("/")[:-1])  # production/agpt/2b
    # Resolve link relative to the page's directory, collapsing any '../'.
    parts = (page_dir.split("/") if page_dir else []) + link.split("/")
    stack: list[str] = []
    for p in parts:
        if p == "..":
            if stack:
                stack.pop()
        elif p in ("", "."):
            continue
        else:
            stack.append(p)
    return "/".join(stack)


def _compute_disk_values(traj: dict) -> dict | None:
    """Authoritative (step, tokens_str, pct, loss_str) for a trajectory,
    from disk + its leaf README. Returns None if no valid ckpt.
    The step/tokens come from disk; loss + precision come from the leaf
    README so the rollup matches the leaf exactly.
    """
    step = largest_valid_step(traj["ckpt_dir"])
    if step is None:
        return None
    leaf = REPO_ROOT / traj["readme"]
    leaf_text = leaf.read_text() if leaf.is_file() else ""
    gbs_m = GBS_ROW_RE.search(leaf_text)
    gbs = int(gbs_m.group(1).replace(",", "")) if gbs_m else traj["gbs"]
    # Clone the leaf Tokens line's precision so rollup matches it.
    t_dec, b_dec, pct_dec = 3, 1, 1
    tm = TOKENS_RE.search(leaf_text)
    if tm:
        tok_old = re.search(r"([\d.]+)\s*([TB])\s*tokens", tm.group("body"))
        pct_old = re.search(r"([\d.]+)\s*%", tm.group("body"))
        if tok_old:
            if tok_old.group(2) == "T":
                t_dec = _decimals_in(tok_old.group(1))
            else:
                b_dec = _decimals_in(tok_old.group(1))
        if pct_old:
            pct_dec = _decimals_in(pct_old.group(1))
    tok_str, pct = format_tokens(
        step, gbs, traj["seq_len"], traj["token_target"],
        t_decimals=t_dec, b_decimals=b_dec, pct_decimals=pct_dec,
    )
    # Loss: read the leaf's **Loss:** field if present (leading number).
    loss_str = None
    lm = LOSS_RE.search(leaf_text)
    if lm:
        lnum = LEADING_NUM_RE.match(lm.group("body").strip())
        if lnum:
            loss_str = lnum.group(1)
    return {"step": step, "tokens": tok_str, "pct": pct, "loss": loss_str}


def propagate_to_rollups(
    trajs: list[dict], *, dry_run: bool
) -> tuple[int, list[str]]:
    """Update rollup Snapshot table rows from each live trajectory's
    disk-authoritative values. Rewrites ONLY the step/loss/tokens numeric
    cells of the row that links to a given leaf README; the Status/
    narrative cell and all other rows are untouched. Returns (n_changed,
    warnings).
    """
    # Build {canonical-docs-rel README: values} -- keyed so 2B-256N and
    # 20B-256N never collide.
    vals: dict[str, dict] = {}
    for t in trajs:
        v = _compute_disk_values(t)
        if v:
            vals[_canon_readme(t["readme"])] = v
    n_changed = 0
    warns: list[str] = []

    for page_rel in ROLLUP_PAGES:
        page = REPO_ROOT / page_rel
        if not page.is_file():
            continue
        lines = page.read_text().splitlines(keepends=False)
        out: list[str] = []
        changed_here = False
        for line in lines:
            if not line.lstrip().startswith("|") or "README.md)" not in line:
                out.append(line)
                continue
            # Resolve THIS row's leaf link to the same canonical key.
            link_m = re.search(r"\]\(([^)]*?n\d+/README\.md)\)", line)
            if not link_m:
                out.append(line)
                continue
            key = _resolve_rollup_link(page_rel, link_m.group(1))
            v = vals.get(key)
            if not v:
                out.append(line)
                continue
            new_line = _rewrite_rollup_row(line, v)
            if new_line != line:
                out.append(new_line)
                changed_here = True
                n_changed += 1
            else:
                out.append(line)
        if changed_here and not dry_run:
            page.write_text("\n".join(out) + "\n")
        if changed_here:
            print(f"  [rollup] {page_rel}: updated snapshot row(s)")
    return n_changed, warns


def _rewrite_rollup_row(line: str, v: dict) -> str:
    """Rewrite the step / loss / tokens(pct) numeric cells of one rollup
    table row, preserving every other cell (esp. the Status narrative)
    and the row's existing bold / comma / '~' / '(persisted)' styling.

    Two row shapes are handled:
      A. model-rollup:  | [name](nN/README.md) ... | <status> | **STEP** | **LOSS** | **TOK (PCT)** |
      B. top-level:     | Model | N | **STEP** (persisted) | **LOSS** | **TOK** (PCT) | [job](..nN/README.md) | <status> |
    We edit cells by matching the numeric *content* with anchored regexes
    rather than by column index, so both shapes work.
    """
    step_comma = f"{v['step']:,}"

    def sub_step(cell: str) -> str:
        # Replace a leading bold-or-plain integer (the cumulative step),
        # keep any '(persisted)' / suffix text.
        return re.sub(
            r"(\*{0,2})[\d,]+(\*{0,2})",
            lambda m: f"{m.group(1)}{step_comma}{m.group(2)}",
            cell, count=1,
        )

    def sub_loss(cell: str) -> str:
        if v["loss"] is None:
            return cell
        return re.sub(
            r"(\*{0,2})[\d.]+(\*{0,2})",
            lambda m: f"{m.group(1)}{v['loss']}{m.group(2)}",
            cell, count=1,
        )

    def sub_tokens(cell: str) -> str:
        # Replace "X.XXT" / "X.XB" and the "(YY.Y%)" while keeping ~, bold,
        # 'of ...' text, and any surrounding words.
        c = re.sub(r"[\d.]+\s*[TB]", v["tokens"], cell, count=1)
        c = re.sub(r"[\d.]+\s*%", v["pct"], c, count=1)
        return c

    cells = line.split("|")
    # Find the cell index that holds the leaf link (shape discriminator).
    link_idx = next(
        (i for i, c in enumerate(cells) if re.search(r"n\d+/README\.md\)", c)),
        None,
    )
    if link_idx is None:
        return line

    if link_idx <= 2:
        # Shape A (model rollup): link in col 1 -> numeric cells are the
        # LAST three pipe-separated cells (steps, loss, tokens).
        # cells[-1] is '' (trailing pipe), so real last is cells[-2].
        if len(cells) >= 5:
            cells[-4] = sub_step(cells[-4])
            cells[-3] = sub_loss(cells[-3])
            cells[-2] = sub_tokens(cells[-2])
    else:
        # Shape B (top-level): link is near the end; numeric cells are
        # cols 3,4,5 (1-indexed within the row) = cells[3],[4],[5].
        if len(cells) >= 6:
            cells[3] = sub_step(cells[3])
            cells[4] = sub_loss(cells[4])
            cells[5] = sub_tokens(cells[5])
    return "|".join(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print diffs, don't write")
    ap.add_argument("--wandb", action="store_true", help="also fill **Loss:** from W&B")
    ap.add_argument("--model", help="limit to one trajectory key (e.g. 2b_v2_256)")
    ap.add_argument(
        "--today",
        help="override today's date (YYYY-MM-DD); for testing. Date.now is "
        "otherwise used.",
    )
    ap.add_argument(
        "--no-rollups", action="store_true",
        help="skip propagating leaf values into rollup Snapshot tables",
    )
    args = ap.parse_args()

    if args.today:
        today = args.today
    else:
        today = datetime.date.today().isoformat()

    if args.model:
        trajs = [by_key(args.model)]
        # by_key returns any class; only proceed if it has a live ckpt dir
        trajs = [t for t in trajs if t.get("ckpt_dir") and Path(t["ckpt_dir"]).is_dir()]
    else:
        trajs = live_trajectories()

    total_changes = 0
    total_warns = 0
    for traj in trajs:
        wandb_loss = _wandb_latest_loss(traj) if args.wandb else None
        changes, warns = fill_one(
            traj, today=today, wandb_loss=wandb_loss, dry_run=args.dry_run
        )
        if changes or warns:
            print(f"=== {traj['key']} ({traj['readme']}) ===")
        for c in changes:
            print(f"  [{c.field}]")
            print(f"      OLD: {c.old[:100]}")
            print(f"      NEW: {c.new[:100]}")
        for w in warns:
            print(f"  ! {w}")
        total_changes += len(changes)
        total_warns += len(warns)

    # Propagate leaf disk values into rollup Snapshot tables (numeric
    # cells only; never the Status narrative). Always over the full live
    # set so a --model run doesn't half-update a rollup row.
    rollup_changed = 0
    if not args.no_rollups:
        rollup_trajs = live_trajectories()
        rollup_changed, rollup_warns = propagate_to_rollups(
            rollup_trajs, dry_run=args.dry_run
        )
        for w in rollup_warns:
            print(f"  ! {w}")

    suffix = " (dry-run)" if args.dry_run else ""
    print(
        f"\n=== {total_changes} leaf field(s), {rollup_changed} rollup row(s) "
        f"changed, {total_warns} warning(s){suffix} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
