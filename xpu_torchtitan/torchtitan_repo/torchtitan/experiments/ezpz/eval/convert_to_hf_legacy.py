# Variant of convert_to_hf.py that handles legacy DCP checkpoints
# from before the qkv_linear submodule was introduced.
#
# Legacy DCP key shape (pre-2026-04-30, agpt 2B 256N gap-fill ckpts in
# /flare/.../projects/saforem2/torchtitan/outputs/checkpoints/):
#     layers.N.attention.{wq,wk,wv}.weight
#
# Current DCP key shape (v2 stack):
#     layers.N.attention.qkv_linear.{wq,wk,wv}.weight
#
# Shapes are identical — only the FQN prefix changed when the
# attention module factored {wq,wk,wv} into a `qkv_linear` submodule.
# This script renames the keys at load time via a LoadPlanner subclass,
# then converts to HF safetensors. Used for one-off gap-fill of evals
# on legacy ckpts; not for production resume.

import argparse
import importlib
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


# Pre-qkv_linear field names.
# Pattern (per layer): NEW -> OLD
LEGACY_RENAMES = {
    "attention.qkv_linear.wq.weight": "attention.wq.weight",
    "attention.qkv_linear.wk.weight": "attention.wk.weight",
    "attention.qkv_linear.wv.weight": "attention.wv.weight",
}


def _new_to_old(new_key: str) -> str | None:
    """If new_key looks like layers.N.<suffix> where suffix has a legacy
    rename, return the legacy key. Else None."""
    for new_suffix, old_suffix in LEGACY_RENAMES.items():
        if new_key.endswith("." + new_suffix):
            prefix = new_key[: -len(new_suffix)]
            return prefix + old_suffix
    return None


class LegacyKeyRenamePlanner(DefaultLoadPlanner):
    """LoadPlanner that maps current FQNs to legacy FQNs so a state_dict
    with the new key layout can be populated from a checkpoint written
    with the old layout. Keys not in the rename map pass through.
    """

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        renamed_sd = {}
        # remember new->old map so we can stash loaded tensors back under the
        # new key after load
        self._reverse_map: dict[str, str] = {}
        for new_key, tensor in state_dict.items():
            old_key = _new_to_old(new_key)
            if old_key is not None and metadata is not None \
                    and old_key in metadata.state_dict_metadata:
                renamed_sd[old_key] = tensor
                self._reverse_map[old_key] = new_key
            else:
                renamed_sd[new_key] = tensor
        super().set_up_planner(renamed_sd, metadata, is_coordinator)


@torch.inference_mode()
def convert_legacy_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    if "." in model_name:
        model_module = importlib.import_module(f"torchtitan.{model_name}")
    else:
        model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert sd_adapter is not None

    state_dict = model._get_state_dict()

    planner = LegacyKeyRenamePlanner(allow_partial_load=True)
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
        planner=planner,
    )

    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LEGACY DCP weights to HF format.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--hf_assets_path", type=Path, default="./assets/hf/gemma-7b")
    parser.add_argument("--model_name", type=str, default="experiments.ezpz.agpt")
    parser.add_argument("--model_flavor", type=str, default="2b")
    parser.add_argument(
        "--export_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    args = parser.parse_args()

    convert_legacy_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
