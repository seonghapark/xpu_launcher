#!/usr/bin/env python3
"""Pre-cache a HuggingFace dataset to a shared HF_HOME on flare.

Solves the "rank 0 prefetch alone burns the rate-limit quota" problem
for datasets with large recursive file manifests (allenai/olmo-mix-1124,
etc.). Run this once from a login node, then submit scripts only need
to export ``HF_HOME=<shared>`` and ``HF_DATASETS_OFFLINE=1``.

Usage (from login node, with proxy on):

    export http_proxy=http://proxy.alcf.anl.gov:3128
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export HF_HOME=/lus/flare/projects/datasets/hf_cache
    python torchtitan/experiments/ezpz/scripts/precache_hf_dataset.py \
        allenai/olmo-mix-1124

The script:
    1. Snapshot-downloads the dataset to ``$HF_HOME/datasets/...``
    2. Resolves the dataset_info.json (metadata) via load_dataset
    3. Verifies it's loadable in HF_DATASETS_OFFLINE=1 mode

Subsequent training jobs add to their submit scripts:

    export HF_HOME=/lus/flare/projects/datasets/hf_cache
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1

— and ``load_dataset(\"allenai/olmo-mix-1124\", streaming=True)``
returns immediately from the local cache with zero Hub API calls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def precache(dataset_path: str, config_name: str | None = None) -> None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("ERROR: HF_HOME is not set. Export it first, e.g.:")
        print("  export HF_HOME=/lus/flare/projects/datasets/hf_cache")
        sys.exit(1)
    print(f"[precache] HF_HOME={hf_home}")
    print(f"[precache] dataset={dataset_path}  config={config_name}")
    print()

    # Stage 1: snapshot_download the full repo (parquet shards +
    # dataset_info.json + README + .gitattributes). This is the bulk
    # of what makes load_dataset(streaming=True) work offline later.
    print("[precache] stage 1: snapshot_download")
    t0 = time.monotonic()
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=dataset_path,
        repo_type="dataset",
    )
    print(f"[precache] snapshot at: {snapshot_path}")
    print(f"[precache] snapshot took {time.monotonic()-t0:.1f}s")
    print()

    # Stage 2: load_dataset once online to populate the datasets-library
    # cache (the JSON manifests under HF_DATASETS_CACHE that load_dataset
    # reads, distinct from the huggingface_hub repo cache).
    print("[precache] stage 2: load_dataset (online, populates datasets cache)")
    t1 = time.monotonic()
    from datasets import load_dataset

    kwargs = {"streaming": True}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(dataset_path, **kwargs)
    print(f"[precache] online load_dataset took {time.monotonic()-t1:.1f}s")
    print()

    # Stage 3: flip to offline mode and re-load to verify it works
    # WITHOUT any Hub calls.
    print("[precache] stage 3: verify offline load")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    t2 = time.monotonic()
    ds_off = load_dataset(dataset_path, **kwargs)
    print(f"[precache] OFFLINE load_dataset took {time.monotonic()-t2:.1f}s")
    print()
    print("[precache] success")
    print()
    print("Add to your submit script:")
    print(f"  export HF_HOME={hf_home}")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="HF dataset path, e.g. allenai/olmo-mix-1124")
    ap.add_argument(
        "--config",
        default=None,
        help="Dataset config / subset name (e.g. 'default', 'en')",
    )
    args = ap.parse_args()
    precache(args.dataset, args.config)


if __name__ == "__main__":
    main()
