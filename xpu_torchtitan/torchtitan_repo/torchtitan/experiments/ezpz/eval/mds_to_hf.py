"""Convert Megatron-DeepSpeed (MDS) AuroraGPT-2B checkpoints to HuggingFace.

Adapted from argonne-lcf/Megatron-DeepSpeed/mds_to_hf.py to:
  - build LlamaConfig from scratch (no need to download Llama-2-7b)
  - copy tokenizer files into the output dir alongside the weights
  - save weights as `model-00001-of-00001.safetensors` to match our other
    converted checkpoints (so lm-eval picks them up)

Usage:
    python3 mds_to_hf.py \
        --mds_checkpoint <ckpt_dir>/global_step{N}/mp_rank_00_model_states.pt \
        --output_dir <out_dir>
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig


def _make_dummy(module: str, name: str):
    def _setstate(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            if isinstance(state[0], dict):
                self.__dict__.update(state[0])
            if isinstance(state[1], dict):
                self.__dict__.update(state[1])
        elif isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["__state__"] = state

    return type(name, (), {
        "__module__": module,
        "__init__": lambda self, *a, **kw: None,
        "__setstate__": _setstate,
    })


class _MegatronCompatUnpickler(pickle.Unpickler):
    # MDS checkpoints reference classes from the `megatron` package (in
    # optimizer/state objects). We only read `mds['args'].__dict__` and the
    # weight tensors under `mds['module']`, so we never invoke those classes —
    # the unpickler just needs to not crash on the references.
    def find_class(self, module: str, name: str):
        if module == "megatron" or module.startswith("megatron."):
            return _make_dummy(module, name)
        return super().find_class(module, name)


class _PickleShim:
    Unpickler = _MegatronCompatUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _MegatronCompatUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return _MegatronCompatUnpickler(io.BytesIO(data), **kwargs).load()


REPO_ROOT = Path(__file__).resolve().parents[4]
TOKENIZER_DIR = REPO_ROOT / "assets/hf/gemma-7b"


def build_llama_config(mds_args: dict) -> LlamaConfig:
    """Build a LlamaConfig matching the MDS training arguments."""
    return LlamaConfig(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        hidden_size=mds_args["hidden_size"],
        intermediate_size=mds_args["ffn_hidden_size"],
        num_hidden_layers=mds_args["num_layers"],
        num_attention_heads=mds_args["num_attention_heads"],
        num_key_value_heads=mds_args["num_key_value_heads"],
        max_position_embeddings=mds_args["max_position_embeddings"],
        rope_theta=mds_args["rope_theta"],
        rms_norm_eps=mds_args["layernorm_epsilon"],
        vocab_size=mds_args["padded_vocab_size"],
        hidden_act="silu" if mds_args.get("swiglu") else "gelu",
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=mds_args.get("bos_token_id", 2),
        eos_token_id=mds_args.get("eos_token_id", 1),
        tie_word_embeddings=False,
    )


def convert(mds_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MDS checkpoint from {mds_path}")
    # MDS checkpoints reference `megatron.*` classes (in optimizer state and the
    # `args` namespace). We don't need megatron itself — the shim returns dummy
    # classes that just hold whatever __dict__ was pickled, which is enough for
    # `mds['args'].__dict__` and the weight tensors under `mds['module']`.
    mds = torch.load(
        mds_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_PickleShim,
    )
    args_dict = mds["args"].__dict__

    cfg = build_llama_config(args_dict)
    cfg.save_pretrained(out_dir)

    enc = mds["module"]["language_model"]["encoder"]
    state_dict: dict[str, torch.Tensor] = {}

    dim = args_dict["kv_channels"]
    hidden_size = args_dict["hidden_size"]
    n_kv = args_dict["num_key_value_heads"]
    n_heads = args_dict["num_attention_heads"]
    kv_groups = n_heads // n_kv

    for layer_i in range(cfg.num_hidden_layers):
        # Split the fused QKV projection into separate Q, K, V tensors.
        fused_qkv = enc[f"layers.{layer_i}.self_attention.query_key_value.weight"]
        fused = fused_qkv.view(n_kv, (kv_groups + 2) * dim, hidden_size)

        q = fused[:, : kv_groups * dim, :].contiguous().view(-1, hidden_size)
        k = fused[:, kv_groups * dim : (kv_groups + 1) * dim, :].contiguous().view(-1, hidden_size)
        v = fused[:, (kv_groups + 1) * dim :, :].contiguous().view(-1, hidden_size)

        prefix = f"model.layers.{layer_i}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = q
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = k
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = v
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = enc[
            f"layers.{layer_i}.self_attention.dense.weight"
        ]

        # MLP: split the fused gate-up projection.
        fused_mlp = enc[f"layers.{layer_i}.mlp.dense_h_to_4h.weight"]
        gate, up = torch.chunk(fused_mlp, 2, dim=0)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate
        state_dict[f"{prefix}.mlp.up_proj.weight"] = up
        state_dict[f"{prefix}.mlp.down_proj.weight"] = enc[
            f"layers.{layer_i}.mlp.dense_4h_to_h.weight"
        ]

        # Norms.
        state_dict[f"{prefix}.input_layernorm.weight"] = enc[
            f"layers.{layer_i}.input_layernorm.weight"
        ]
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = enc[
            f"layers.{layer_i}.post_attention_layernorm.weight"
        ]

    # Embed, final norm, lm_head.
    state_dict["model.embed_tokens.weight"] = mds["module"]["language_model"]["embedding"][
        "word_embeddings"
    ]["weight"]
    state_dict["model.norm.weight"] = enc["final_layernorm.weight"]
    state_dict["lm_head.weight"] = mds["module"]["language_model"]["output_layer"]["weight"]

    # Cast to bf16 to match the dtype the model was trained in.
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

    # Save as a single safetensors shard. Write the matching index file so
    # transformers' `from_pretrained` finds it (it checks for the index before
    # the sharded file pattern).
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_path = out_dir / shard_name
    save_file(state_dict, safetensors_path)
    print(f"Saved weights: {safetensors_path}")

    index = {
        "metadata": {
            "total_size": sum(t.numel() * t.element_size() for t in state_dict.values()),
        },
        "weight_map": {k: shard_name for k in state_dict},
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Copy tokenizer files (gemma-7b — same as torchtitan-ezpz uses).
    for f in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = TOKENIZER_DIR / f
        if src.exists():
            shutil.copy(src, out_dir / f)
    print(f"Copied tokenizer from {TOKENIZER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mds_checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.mds_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
