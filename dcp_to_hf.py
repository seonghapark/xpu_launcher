#!/usr/bin/env python3
"""Convert a TorchTitan DCP checkpoint into HuggingFace safetensors.

This replaced an earlier hand-rolled converter (``convert_model_format.py``,
since deleted) that never called ``torch.distributed.checkpoint``, so it could
not read a real sharded DCP checkpoint at all; it also stripped the leading
``model.`` prefix that every HF key needs, and renamed keys only by removing
prefixes, so it could never produce the ``model.layers.N.self_attn.q_proj.
weight`` names ``from_hf_map`` expects on the way back in. Do not reintroduce a
key-rewriting converter.

This converter instead reuses TorchTitan's *own* ``state_dict_adapter.to_hf()``,
reached through the ordinary ``ConfigManager`` path. That matters for more than
key names: ``to_hf`` decides whether to apply the HF<->Meta rotary permutation
to Q/K based on the model's RoPE flavor. Re-deriving that here would mean
duplicating the rule, and a copy that drifts produces a checkpoint that loads
cleanly and trains on silently corrupted attention weights.

Because the adapter is built from the same ``--module/--config`` the training
run uses, the conversion is correct by construction for whichever flavor is
being trained.

Usage (torchtitan args after ``--`` are the SAME ones training will use):

    python dcp_to_hf.py \\
        --dcp-path  /path/to/checkpoint/step-1000 \\
        --output    /path/to/out-safetensors \\
        --assets-from /path/to/base-model-safetensors \\
        -- --module agpt --config agpt_2b_yarn --training.seq_len 16384
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Put the torchtitan REPO ahead of the installed wheel, exactly as
# titan_train.py does. Both are importable here, but only the repo carries the
# agpt model package -- without this, "--module agpt" fails with "Cannot import
# module 'agpt'" and lists only the wheel's built-in models. Using the same
# TORCHTITAN_ROOT override keeps the converter and the trainer pointed at one
# checkout, which also means the adapter is the one training will use.
_TORCHTITAN_ROOT = Path(
    os.environ.get(
        "TORCHTITAN_ROOT",
        Path(__file__).resolve().parent / "xpu_torchtitan" / "torchtitan_repo",
    )
)
if str(_TORCHTITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHTITAN_ROOT))

# Files copied verbatim from the reference HF assets directory. The weights are
# regenerated, but tokenizer and architecture metadata are unchanged by
# training, so carrying them over keeps the output directory directly usable as
# --hf_assets_path.
_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

# Default safetensors shard ceiling. Matches the HF hub convention; keeps any
# single file comfortably under filesystem and mmap limits on Lustre.
_DEFAULT_SHARD_BYTES = 5 * 1000 * 1000 * 1000


def _log(msg: str) -> None:
    print(f"[dcp_to_hf] {msg}", flush=True)


def _build_adapter(titan_args: list[str], hf_assets_path: str):
    """Build the state dict adapter for the config training will use.

    ``titan_args`` is forwarded verbatim to ConfigManager so the adapter is
    constructed from exactly the config the training run builds. ``sys.argv`` is
    rewritten to match only because ConfigManager's tyro layer inspects it for
    usage/error reporting; nothing in the config path derives a value from it.
    """
    from torchtitan.config.manager import ConfigManager

    args = list(titan_args)
    if not any(a == "--hf_assets_path" or a.startswith("--hf_assets_path=") for a in args):
        args += ["--hf_assets_path", hf_assets_path]

    saved_argv = sys.argv
    sys.argv = [saved_argv[0], *args]
    try:
        config = ConfigManager().parse_args(args)
    finally:
        sys.argv = saved_argv

    # NOTE: the trainer calls model_config.update_from_config(config=config)
    # before building its adapter; this does not. That is deliberate. What
    # update_from_config changes on the model config is the two length-derived
    # RoPE values -- max_seq_len (via dataclasses.replace, which preserves the
    # dataclass type) and the YaRN rope_factor. Neither reaches to_hf: the
    # adapter consults the RoPE *class* to decide on the Q/K permutation, not
    # its field values, and otherwise reads only n_heads, n_kv_heads, dim,
    # head_dim and enable_weight_tying. Key naming and the permutation are
    # therefore identical either way, while calling it here would surface
    # unrelated parallelism validation (TP head divisibility, EP dispatcher
    # checks) that has no meaning for a single-rank offline conversion.
    model_spec = config.model_spec
    if model_spec.state_dict_adapter is None:
        raise SystemExit(
            "error: this model spec has no state_dict_adapter, so there is no "
            "defined HF representation to convert into."
        )
    adapter = model_spec.state_dict_adapter(model_spec.model, config.hf_assets_path)
    _log(f"adapter: {type(adapter).__name__}")
    rope = model_spec.model.layers[0].attention.rope
    _log(f"rope: {type(rope).__qualname__}")
    return adapter


def _load_dcp(dcp_path: Path) -> dict[str, Any]:
    """Read a DCP checkpoint into a plain, unsharded state dict."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: dict[str, Any] = {}
    _load_state_dict(
        sd,
        storage_reader=dcp.FileSystemReader(str(dcp_path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd


def _model_keys_only(sd: dict[str, Any], adapter) -> dict[str, Any]:
    """Drop everything that is not a model weight.

    A full (resumable) TorchTitan checkpoint also carries optimizer, dataloader
    and train_state entries. ``to_hf`` indexes its map directly for non-layer
    keys, so those extras would raise KeyError rather than be ignored.
    """
    import re

    import torch

    native_keys = {v for v in adapter.from_hf_map.values() if v is not None}
    kept: dict[str, Any] = {}
    skipped: set[str] = set()
    for key, value in sd.items():
        abstract = re.sub(r"(\d+)", "{}", key, count=1)
        if (key in native_keys or abstract in native_keys) and torch.is_tensor(value):
            kept[key] = value
        else:
            skipped.add(abstract)
    if skipped:
        _log(f"ignoring {len(skipped)} non-model key pattern(s): {sorted(skipped)[:8]}")
    return kept


def _shard(hf_sd: dict[str, Any], max_bytes: int) -> list[dict[str, Any]]:
    """Greedily pack tensors into shards of at most ``max_bytes``.

    A tensor larger than the limit still gets its own shard rather than being
    split; safetensors cannot split a single tensor across files.
    """
    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0
    for key in sorted(hf_sd):
        value = hf_sd[key]
        nbytes = value.numel() * value.element_size()
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = value
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def _write_safetensors(hf_sd: dict[str, Any], out_dir: Path, max_bytes: int) -> None:
    from safetensors.torch import save_file

    # safetensors rejects shared storage and non-contiguous tensors; a clone is
    # cheap next to the I/O and sidesteps both. Weight tying in particular makes
    # lm_head and embed_tokens alias the same storage.
    hf_sd = {k: v.detach().contiguous().clone() for k, v in hf_sd.items()}

    shards = _shard(hf_sd, max_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(shards) == 1:
        # Single-file layout, matching the base agpt checkpoints (no index).
        path = out_dir / "model.safetensors"
        save_file(shards[0], str(path), metadata={"format": "pt"})
        _log(f"wrote {path.name} ({len(shards[0])} tensors)")
        return

    total = len(shards)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(out_dir / name), metadata={"format": "pt"})
        for key, value in shard.items():
            weight_map[key] = name
            total_bytes += value.numel() * value.element_size()
        _log(f"wrote {name} ({len(shard)} tensors)")

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    _log("wrote model.safetensors.index.json")


def _copy_assets(assets_from: Path, out_dir: Path) -> None:
    copied = []
    for name in _ASSET_FILES:
        src = assets_from / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    _log(f"copied assets: {', '.join(copied) if copied else '(none)'}")
    if not (out_dir / "config.json").is_file():
        _log(
            "WARNING: no config.json in output. Training reads architecture and "
            "tokenizer metadata from --hf_assets_path, so point --assets-from at "
            "the base model directory."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a TorchTitan DCP checkpoint to HF safetensors.",
        epilog="Arguments after -- are passed to TorchTitan's ConfigManager.",
    )
    parser.add_argument("--dcp-path", required=True, help="DCP checkpoint directory")
    parser.add_argument("--output", required=True, help="Output HF directory")
    parser.add_argument(
        "--assets-from",
        help="Directory to copy config.json/tokenizer from (usually the base model)",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=_DEFAULT_SHARD_BYTES,
        help=f"Max bytes per safetensors shard (default {_DEFAULT_SHARD_BYTES})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty output directory",
    )
    argv = sys.argv[1:]
    titan_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, titan_args = argv[:split], argv[split + 1 :]
    args = parser.parse_args(argv)

    dcp_path = Path(args.dcp_path).resolve()
    out_dir = Path(args.output).resolve()

    if not dcp_path.is_dir():
        raise SystemExit(f"error: DCP path not found: {dcp_path}")
    if not (dcp_path / ".metadata").is_file():
        raise SystemExit(
            f"error: {dcp_path} has no .metadata file, so it is not a DCP "
            f"checkpoint directory. Pass the step-NNNN directory itself."
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"error: output directory is not empty: {out_dir}\n"
            f"       pass --force to overwrite."
        )

    assets_from = Path(args.assets_from).resolve() if args.assets_from else None
    if assets_from and not assets_from.is_dir():
        raise SystemExit(f"error: --assets-from not found: {assets_from}")

    adapter = _build_adapter(titan_args, str(assets_from or out_dir))

    _log(f"loading DCP checkpoint: {dcp_path}")
    raw = _load_dcp(dcp_path)
    _log(f"loaded {len(raw)} top-level entries")

    native = _model_keys_only(raw, adapter)
    if not native:
        raise SystemExit(
            "error: no model weights found in the checkpoint. Its keys do not "
            "match the adapter's native names -- check that --module/--config "
            "describe the model this checkpoint was saved from."
        )
    _log(f"{len(native)} model tensors")

    hf_sd = adapter.to_hf(native)
    _log(f"converted to {len(hf_sd)} HF tensors")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_safetensors(hf_sd, out_dir, args.max_shard_bytes)
    if assets_from:
        _copy_assets(assets_from, out_dir)

    _log(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
