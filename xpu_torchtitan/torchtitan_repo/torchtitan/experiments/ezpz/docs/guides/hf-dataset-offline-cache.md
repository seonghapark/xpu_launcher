# Local HF Dataset Cache for Distributed Training

How to use a large HF dataset (`allenai/olmo-mix-1124`, `HuggingFaceFW/fineweb-edu`,
etc.) at 8+ nodes without `429 Too Many Requests` errors. Validated end-to-end on
8N Sunspot (96 ranks).

## What goes wrong with streaming at scale

`load_dataset(repo, streaming=True)` makes two classes of HF Hub calls during
training:

1. **Startup metadata** — `/api/datasets/<repo>/revision/<sha>` + paginated
   `/api/datasets/<repo>/tree?recursive=true` to enumerate parquet/json shards.
   For `allenai/olmo-mix-1124` (28,880 files): ~29 paginated tree calls × 96 ranks
   = ~2,800 calls in seconds — well above HF's **1000 req / 5 min per-IP** quota.
2. **Per-shard `hf://` opens** at iteration time. Even when each rank should
   read disjoint shards (via `split_dataset_by_node`), the first batch can have
   96 concurrent opens of the same shard before sharding kicks in, which the
   data CDN treats as a hot spot.

Repro from an 8N run on Sunspot:

```
huggingface_hub.errors.HfHubHTTPError: 429 Too Many Requests
Url: https://huggingface.co/api/datasets/allenai/olmo-mix-1124/tree/.../data?recursive=true
```

Or, more confusingly, the data CDN failure surfaces as:

```
FileNotFoundError: gzip://...::hf://datasets/allenai/olmo-mix-1124@<sha>/data/algebraic-stack/train/algebraic-stack-train-0000.json.gz
```

(HF's fsspec resolver wraps both 429 and "not found" as `FileNotFoundError`.)

For **small** HF datasets (`eliplutchok/fineweb-small-sample`, `Salesforce/wikitext`)
neither class of call exceeds the quota, so streaming "just works."

## The fix: snapshot a sub-config + register as local dataset

The rank-0 prefetch wrapper in `ezpz/datasets.py` is sufficient for small
datasets but not for olmo-mix-scale ones. The reliable approach is to
pre-download the slice you actually need, then point the dataloader at the
local directory.

### Step 1 — pre-download from a login node (one-time per sub-config)

```bash
ssh sunspot.alcf.anl.gov
cd ~/path/to/torchtitan
source .venv/bin/activate

# Proxy required on login nodes for outbound HTTPS
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Shared cache on flare so all compute nodes can read it
export HF_HOME=/lus/flare/projects/datasets/hf_cache

# Use allow_patterns to pull only the sub-config you need
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='allenai/olmo-mix-1124',
    repo_type='dataset',
    allow_patterns=['data/wiki/**', 'README.md', '*.json', '.gitattributes'],
)
"
```

For convenience there's also `torchtitan/experiments/ezpz/scripts/precache_hf_dataset.py`
which does the full-snapshot version of this with verification — but for
large datasets the `snapshot_download(..., allow_patterns=[...])` form above
is what you actually want.

### Step 2 — register the local snapshot in `ezpz/datasets.py`

Add a line near the other `register_local_dataset(...)` calls at the bottom of
the file:

```python
register_local_dataset(
    "olmo_mix_wiki_local",  # whatever name you want to pass to --dataloader.dataset
    "/lus/flare/projects/datasets/hf_cache/hub/datasets--allenai--olmo-mix-1124/snapshots/<sha>/data/wiki/",
    format="json",  # olmo-mix shards are .json.gz; default is "parquet"
)
```

The snapshot path is what `snapshot_download` printed. The format must match
the on-disk file type (`parquet`, `json`, `arrow`, `csv`, or `text`).

### Step 3 — use it in training

```bash
python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_2b \
    --dataloader.dataset olmo_mix_wiki_local
```

Zero Hub API calls. Validated on 8N (96 ranks): 5 steps clean, MFU 20%.

## olmo-mix-1124 sub-config inventory

Sizes measured 2026-06-11 against the canonical
`99ee6aaace88779d1ef099d36251b91101c1679b` revision:

| Component | Files | Size  | Fits on flare (~3 TB free)? |
|---|--:|--:|---|
| `wiki`            |     2 | **6.5 GB**   | ✅ (validated) |
| `algebraic-stack` |    16 | 11 GB        | ✅ |
| `open-web-math`   |    13 | 13 GB        | ✅ |
| `arxiv`           |    20 | 22 GB        | ✅ |
| `pes2o`           |    26 | 106 GB       | ✅ |
| `starcoder`       |   863 | 103 GB       | ✅ |
| `dclm`            | 27938 | 7,221 GB     | ❌ (don't try) |
| **(full)**        | 28880 | **7,482 GB** | ❌ |

The full mix won't fit on flare — `dclm` is 96% of the bytes. Pre-cache
whichever components you actually need.

## Other large datasets

| Dataset                       | Files | Approach |
|---|--:|---|
| `HuggingFaceFW/fineweb-edu`   | 3,038 | use `--config sample-10BT` or `sample-100BT` (≤100 GB), or the existing `fineweb_edu_local` registration |
| `HuggingFaceFW/fineweb`       |  many | 93 TB total — don't try; use a sub-config |
| `Salesforce/wikitext`         |    16 | 14 MB, full snapshot fine |

## Notes

- **Don't set `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1` globally** — they
  block legitimate `hf://` reads as well as metadata calls. The `register_local_dataset`
  path skips Hub entirely without needing those env vars.
- Compute nodes mount `/lus/flare` the same as login nodes, so a single shared
  cache works for any job on Sunspot/Aurora.
- The HF cache stores LFS blobs by SHA in `<HF_HOME>/hub/datasets--<repo>/blobs/`;
  the snapshot directory contains symlinks into that. Don't move or duplicate
  the snapshot dir — symlinks will break.
- Refreshing: re-run `snapshot_download` with the same `allow_patterns` to pick
  up new shards added upstream.
