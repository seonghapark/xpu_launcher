# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download bigbio/* QA datasets and emit a unified Alpaca-style JSONL for SFT.

The `bigbio` collection provides a unified `*_bigbio_qa` schema across dozens of
biomedical QA datasets:

    {question, context, choices, answer, type}

This script loads a curated set of those datasets, converts each row into

    {"instruction", "input", "output"}

records, and writes them to one JSONL file suitable for the gemma SFT pipeline
(`torchtitan.models.gemma.train --dataset_local_path <path>`).

Usage
-----
Full download + merge (default output = ./sft_bigbio.jsonl):

    python -m torchtitan.models.gemma.build_bigbio_sft \
        --output outputs/sft_bigbio.jsonl

Cap each source at 10K rows and shuffle:

    python -m torchtitan.models.gemma.build_bigbio_sft \
        --output outputs/sft_bigbio_mixed.jsonl \
        --max_per_dataset 10000 --shuffle

Only a subset of sources (comma-separated hf ids):

    python -m torchtitan.models.gemma.build_bigbio_sft \
        --datasets bigbio/med_qa,bigbio/pubmed_qa \
        --output outputs/sft_medqa_pubmed.jsonl

List curated sources and exit (no downloads):

    python -m torchtitan.models.gemma.build_bigbio_sft --list

Notes
-----
* `bigbio` datasets ship loader scripts and therefore require
  `trust_remote_code=True`. The script will fail loudly if HF's environment
  does not allow that -- set ``HF_DATASETS_TRUST_REMOTE_CODE=1`` or accept the
  interactive prompt.
* Some bigbio configs are gated or the exact config-name may drift between
  versions of the ``datasets`` library. Any source that fails to load is
  skipped with a WARNING; the script never aborts mid-run.
* This produces a JSONL, not a HF dataset, so the file is trivially inspectable
  (`head`, `wc -l`) and directly consumable by ``load_sft_records``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Curated source list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BigBioSource:
    hf_id: str
    config: str
    split: str = "train"
    # A short human-readable tag used for logging.
    tag: str = ""


# NOTE: Config names follow the bigbio convention "<name>_bigbio_qa". Some
# datasets have multiple sub-configs (e.g. pubmed_qa's labeled/artificial
# folds); we default to the largest publicly usable one. If a config is not
# available in your installed `datasets` version, the script skips it.
CURATED_SOURCES: list[BigBioSource] = [
    # Classic exam / MCQ QA
    BigBioSource("bigbio/med_qa", "med_qa_en_bigbio_qa", "train", "USMLE MCQ"),
    BigBioSource(
        "bigbio/medmcqa", "medmcqa_bigbio_qa", "train", "Indian med MCQ"
    ),
    BigBioSource(
        "bigbio/head_qa", "head_qa_en_bigbio_qa", "train", "Spanish health exams (EN)"
    ),
    # PubMed / literature
    BigBioSource(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_bigbio_qa",
        "train",
        "PubMedQA (labeled)",
    ),
    BigBioSource(
        "bigbio/bioasq_task_b",
        "bioasq_10b_bigbio_qa",
        "train",
        "BioASQ 10b",
    ),
    # General science / reading comprehension useful for bio
    BigBioSource("bigbio/sciq", "sciq_bigbio_qa", "train", "SciQ"),
    BigBioSource(
        "bigbio/biomrc", "biomrc_large_A_bigbio_qa", "train", "BioMRC large A"
    ),
]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


_MC_LETTERS = "ABCDEFGHIJKLMN"


def _first_str(value: Any) -> str:
    """Return the first non-empty string from `value` (str, list, or None)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for v in value:
            s = _first_str(v)
            if s:
                return s
        return ""
    return str(value).strip()


def format_bigbio_qa(row: dict) -> dict | None:
    """Map a BigBIO QA schema row -> Alpaca-style {instruction, input, output}.

    BigBIO QA rows look like::

        {
            "id": "...",
            "question": "What is ...?",
            "context": "Passage ... " | "",
            "choices": ["A opt", "B opt", ...] | [],
            "answer": ["correct text"],
            "type": "multiple_choice" | "yesno" | "factoid" | "list" | ...
        }

    We return ``None`` for rows that cannot be turned into supervised targets
    (empty question or empty answer).
    """
    question = _first_str(row.get("question"))
    context = _first_str(row.get("context"))
    choices = row.get("choices") or []
    if isinstance(choices, str):
        choices = [choices]

    answer_raw = row.get("answer")
    if isinstance(answer_raw, list):
        # For "list"-type QA (multiple gold answers), join them on newlines so
        # the model sees the full set.
        answer_items = [_first_str(a) for a in answer_raw if _first_str(a)]
        answer = "\n".join(answer_items)
    else:
        answer = _first_str(answer_raw)

    if not question or not answer:
        return None

    # Build the instruction: question, plus multiple-choice options if any.
    parts: list[str] = [question]
    if choices:
        opt_lines = []
        for i, opt in enumerate(choices):
            letter = _MC_LETTERS[i] if i < len(_MC_LETTERS) else str(i + 1)
            opt_lines.append(f"{letter}. {_first_str(opt)}")
        parts.append("Options:\n" + "\n".join(opt_lines))
    instruction = "\n\n".join(parts)

    return {
        "instruction": instruction,
        "input": context,
        "output": answer,
    }


# Registry of format callables, in case future non-QA schemas are added.
Formatter = Callable[[dict], "dict | None"]


# ---------------------------------------------------------------------------
# Downloading + merging
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, flush=True)


def _iter_source(
    source: BigBioSource, formatter: Formatter, max_rows: int | None
) -> Iterable[dict]:
    """Yield formatted records from a single bigbio source.

    Any load / iteration error is caught and logged; the generator then stops
    cleanly so the outer merge continues with the next source.
    """
    try:
        # Imported lazily so `--list` works without `datasets` installed.
        from datasets import load_dataset
    except ImportError as e:
        _log(f"[FATAL] `datasets` is required: {e}")
        raise

    try:
        ds = load_dataset(
            source.hf_id,
            source.config,
            split=source.split,
            trust_remote_code=True,
        )
    except Exception as e:  # noqa: BLE001 - we intentionally swallow per source
        _log(
            f"[WARN] failed to load {source.hf_id} ({source.config}, "
            f"split={source.split}): {type(e).__name__}: {e}"
        )
        return

    n_yielded = 0
    n_skipped = 0
    for row in ds:
        rec = formatter(row)
        if rec is None:
            n_skipped += 1
            continue
        yield rec
        n_yielded += 1
        if max_rows is not None and n_yielded >= max_rows:
            break

    _log(
        f"[OK]   {source.hf_id:<28} config={source.config:<36} "
        f"kept={n_yielded:>6}  skipped={n_skipped}"
    )


def build(
    sources: list[BigBioSource],
    output_path: str,
    *,
    max_per_dataset: int | None,
    shuffle: bool,
    seed: int,
    dry_run: bool,
) -> int:
    """Materialize the merged JSONL. Returns the number of records written."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_records: list[dict] = []
    per_source_counts: dict[str, int] = {}

    for source in sources:
        _log(f"[LOAD] {source.hf_id}  ({source.tag})")
        start = len(all_records)
        for rec in _iter_source(source, format_bigbio_qa, max_per_dataset):
            # Tag origin for downstream analysis; harmless for training.
            rec = dict(rec)
            rec["_source"] = source.hf_id
            all_records.append(rec)
        per_source_counts[source.hf_id] = len(all_records) - start

    _log("")
    _log("[SUMMARY] records per source:")
    for src, n in per_source_counts.items():
        _log(f"  {src:<30} {n:>8}")
    _log(f"  {'TOTAL':<30} {len(all_records):>8}")

    if dry_run:
        _log("[dry-run] not writing output file")
        return len(all_records)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_records)
        _log(f"[shuffle] seed={seed}")

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _log(f"[write] {len(all_records)} records -> {output_path}")
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--output", "-o",
        default="./sft_bigbio.jsonl",
        help="Path to the merged JSONL to write (default: ./sft_bigbio.jsonl).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated subset of hf_ids to include "
        "(default: all curated sources).",
    )
    p.add_argument(
        "--max_per_dataset",
        type=int,
        default=None,
        help="Cap the number of kept rows per source (default: no cap).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the merged records before writing.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)."
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Load and count records but do not write the output file.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the curated source list and exit (no downloads).",
    )
    return p.parse_args(argv)


def _print_curated_list() -> None:
    print("Curated bigbio QA sources:\n")
    print(
        f"  {'hf_id':<28} {'config':<40} {'split':<8}  tag"
    )
    print("  " + "-" * 90)
    for s in CURATED_SOURCES:
        print(f"  {s.hf_id:<28} {s.config:<40} {s.split:<8}  {s.tag}")
    print(
        "\nAll follow the BigBIO unified QA schema (question / context / "
        "choices / answer)."
    )


def _select_sources(spec: str | None) -> list[BigBioSource]:
    if not spec:
        return list(CURATED_SOURCES)
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    selected = [s for s in CURATED_SOURCES if s.hf_id in wanted]
    missing = wanted - {s.hf_id for s in selected}
    if missing:
        raise SystemExit(
            f"Unknown --datasets entries: {sorted(missing)}. "
            f"Run with --list to see valid ids."
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list:
        _print_curated_list()
        return 0

    sources = _select_sources(args.datasets)
    _log(f"[plan] {len(sources)} source(s) -> {args.output}")

    n = build(
        sources,
        args.output,
        max_per_dataset=args.max_per_dataset,
        shuffle=args.shuffle,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
