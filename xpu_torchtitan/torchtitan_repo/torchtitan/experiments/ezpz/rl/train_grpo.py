# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# GRPO training with pluggable tasks using TRL + ezpz.
#
# Every TRL ``GRPOConfig`` field (and every HF ``TrainingArguments``
# field it inherits) is exposed as a CLI flag via ``HfArgumentParser``.
# Run with ``--help`` to see the full list — the curated defaults below
# only override values where the ezpz/XPU defaults differ from upstream
# TRL.
#
# Usage:
#   # Plain DDP (every rank holds the full model)
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --task multiply --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
#       --max_steps 20 --per_device_train_batch_size 1
#
#   # Real FSDP (model + grads + optimizer state sharded across all ranks)
#   # Mirrors ezpz.examples.hf.py's explicit-plugin pattern by bootstrapping
#   # the FSDP_/ACCELERATE_USE_FSDP env vars before GRPOTrainer.__init__.
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --task multiply --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
#       --per_device_train_batch_size 2 --fsdp full_shard \
#       --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
#       --gradient_checkpointing --bf16 --beta 0.0
#
#   # vLLM rollouts
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --task sum_digits --beta 0.04 --num_generations 8 \
#       --use_vllm --vllm_gpu_memory_utilization 0.5
#
# Note: HfArgumentParser uses ``snake_case`` flags (e.g.
# ``--per_device_train_batch_size``), not the hyphenated form.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import ezpz
import ezpz.distributed

from torchtitan.experiments.ezpz.rl.tasks import TASK_REGISTRY, get_task

log = logging.getLogger(__name__)

DEFAULT_MODEL = "argonne_private/AuroraGPT-7B"
FALLBACK_MODEL = "Qwen/Qwen3-0.6B"


def _task_help() -> str:
    """Render a one-line help string listing every registered task.

    Importing tasks/__init__.py populates TASK_REGISTRY via per-module
    register_task() side effects, so by the time this dataclass is
    constructed the registry is fully populated. The choices+help
    are kept in sync automatically — adding a new task module is the
    only step needed to expose it in --help.
    """
    if not TASK_REGISTRY:
        return "Task name (registry is empty — no tasks imported?)."
    rows = "; ".join(f"{n}: {t.description}" for n, t in sorted(TASK_REGISTRY.items()))
    return f"Task name from torchtitan.experiments.ezpz.rl.tasks registry. Choices: {rows}"


@dataclass
class EzpzGRPOArgs:
    """ezpz-side CLI args that are not part of GRPOConfig."""

    task: str = field(
        default="sum_digits",
        metadata={
            "help": _task_help(),
            "choices": sorted(TASK_REGISTRY) or None,
        },
    )
    model_name_or_path: str = field(
        default="",
        metadata={
            "help": (
                f"HuggingFace model name or local path. If empty, resolves "
                f"{DEFAULT_MODEL!r} with fallback to {FALLBACK_MODEL!r}."
            )
        },
    )
    num_samples: int = field(
        default=0,
        metadata={
            "help": (
                "Number of training prompts to materialize. 0 (default) "
                "materializes a large pool (~100k) so a typical run "
                "never reuses the same prompt. Set to a positive "
                "integer to materialize that exact count. (TRL's "
                "GRPOTrainer doesn't yet accept true IterableDataset "
                "streams — see trl#3213.)"
            )
        },
    )
    no_save: bool = field(
        default=False,
        metadata={"help": "Skip the final trainer.save_model() call."},
    )
    # --- FSDP knobs (ezpz-managed; mirrors ezpz.examples.hf.py) ---------------
    # HF Trainer / GRPOTrainer construct their internal accelerate.Accelerator
    # from a tangle of env vars. Under ``ezpz launch`` (mpiexec) none of these
    # env vars are set by default, so passing TRL's --fsdp full_shard is a
    # silent no-op — every rank ends up holding the full model. These knobs
    # populate the FSDP_/ACCELERATE_USE_FSDP env vars before GRPOTrainer init
    # so the internal Accelerator builds a real FullyShardedDataParallelPlugin
    # the same way ezpz.examples.hf.py does.
    #
    # NOTE: We deliberately do NOT add a top-level ``--fsdp`` flag here
    # because TrainingArguments already owns that name; instead we read the
    # value of ``config.fsdp`` after parsing. The companion knobs below ARE
    # ezpz-owned (TrainingArguments doesn't have them as scalar fields).
    fsdp_transformer_layer_cls_to_wrap: str = field(
        default="LlamaDecoderLayer",
        metadata={
            "help": (
                "Transformer block class name to auto-wrap for FSDP. Defaults "
                "to LlamaDecoderLayer (correct for the AuroraGPT-* family). "
                "Set to e.g. Qwen3DecoderLayer for Qwen-family models. Has no "
                "effect unless --fsdp is set."
            )
        },
    )
    fsdp_cpu_ram_efficient_loading: bool = field(
        default=False,
        metadata={
            "help": (
                "Load model on rank 0 only and broadcast shards. Slower init "
                "but avoids OOM during model loading on small-memory tiles. "
                "Has no effect unless --fsdp is set."
            )
        },
    )


def _ezpz_grpo_config_cls():
    """Build EzpzGRPOConfig lazily so importing this module doesn't drag
    in TRL/transformers (which pull in torch). Tasks/tests that just want
    the dataclass shape can stay light."""

    from trl import GRPOConfig

    @dataclass
    class EzpzGRPOConfig(GRPOConfig):
        # --- output / cadence -------------------------------------------------
        # output_dir is required by HF TrainingArguments. We give it a sentinel
        # so the user can leave it unset and we'll fill it in from --task below.
        output_dir: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "Output directory. Defaults to outputs/rl/grpo-{task} if unset."
                )
            },
        )
        max_steps: int = 50
        logging_steps: float = 1
        # Disable mid-training checkpoints to avoid safetensors E2BIG errors
        # on filesystems with path-length limits.
        save_strategy: str = "no"

        # --- precision / memory ------------------------------------------------
        bf16: bool = True
        gradient_accumulation_steps: int = 1
        gradient_checkpointing: bool = True
        # Empties XPU cache every N optimizer steps to fight allocator
        # fragmentation. XPU's allocator is more fragmentation-prone than CUDA.
        torch_empty_cache_steps: Optional[int] = 1

        # --- GRPO objective ----------------------------------------------------
        # beta=0.0 disables KL-against-reference. With beta=0 TRL doesn't
        # allocate or load the frozen reference model at all (saves ~bf16
        # model size per rank, e.g. ~4 GB for a 2B llama). Override to e.g.
        # 0.04 if you want the KL term back.
        beta: float = 0.0

        # --- generation --------------------------------------------------------
        num_generations: int = 4
        max_completion_length: int = 64
        temperature: float = 0.7

        # --- observability -----------------------------------------------------
        # Stream sample prompts + generated completions to W&B (as a table)
        # every logging_steps. For RL experiments where you want to see what
        # the model is actually generating, this is the single most useful
        # signal. Trivially cheap (only logged on rank 0).
        log_completions: bool = True
        # Display-only — how many rows to render in the Rich completions
        # table on rank 0 each logging_steps. wandb gets all completions
        # regardless. 2 is enough to see what the model is doing without
        # a wall of text per step.
        num_completions_to_print: int = 2

        # --- vLLM (off by default — XPU vLLM is fragile) -----------------------
        use_vllm: bool = False

        def __post_init__(self):
            # When --fsdp is on and gradient_checkpointing would otherwise
            # be True, migrate to fsdp_config["activation_checkpointing"]
            # BEFORE TrainingArguments.__post_init__ runs — that's where
            # the "redundant AllGather" warning lives (training_args.py:2732),
            # and once super() emits it, it's emitted on every rank for
            # the rest of the run.
            #
            # We do this in __post_init__ rather than in main() so that
            # transformers' own check passes (gradient_checkpointing=False
            # by the time it looks).
            if self.fsdp and self.gradient_checkpointing:
                fsdp_cfg = self.fsdp_config
                if fsdp_cfg is None:
                    fsdp_cfg = {}
                elif isinstance(fsdp_cfg, str):
                    # HF parses --fsdp_config <path> later; we have a path
                    # string here. Load it now so we can merge our flag in.
                    import json
                    with open(fsdp_cfg, encoding="utf-8") as f:
                        fsdp_cfg = json.load(f)
                else:
                    fsdp_cfg = dict(fsdp_cfg)
                fsdp_cfg.setdefault("activation_checkpointing", True)
                self.fsdp_config = fsdp_cfg
                self.gradient_checkpointing = False

            # When FSDP is on, force model_init_kwargs["device_map"]=None
            # so TRL doesn't override it to "auto". TRL's
            # create_model_from_path (trl/trainer/utils.py:1022) defaults
            # device_map to "auto" which under ZE_FLAT_DEVICE_HIERARCHY=FLAT
            # picks the last visible tile (xpu:11) on every process — every
            # rank ends up with its model on xpu:11 regardless of local_rank,
            # and FSDP's _get_compute_device check fires:
            #   ValueError: Inconsistent compute device and `device_id` on
            #   rank N: xpu:11 vs xpu:<local_rank>
            # Setting device_map=None makes from_pretrained leave the model
            # on CPU; HF Trainer then moves it to args.device (the per-rank
            # accelerator.device = xpu:<local_rank>), and FSDP wraps cleanly.
            if self.fsdp:
                mik = self.model_init_kwargs
                if mik is None:
                    mik = {}
                elif isinstance(mik, str):
                    import json
                    mik = json.loads(mik)
                else:
                    mik = dict(mik)
                # Use setdefault so an explicit --model_init_kwargs from CLI
                # still wins. None tells from_pretrained "don't dispatch".
                if "device_map" not in mik:
                    mik["device_map"] = None
                self.model_init_kwargs = mik
            super().__post_init__()

    return EzpzGRPOConfig


def _resolve_model(name: str) -> str:
    """Try to load tokenizer for *name*; fall back if it doesn't exist.

    Only rank 0 should call this. Other ranks should receive the
    resolved name via broadcast (see _prefetch_and_broadcast_model).
    """
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(name)
        return name
    except Exception:
        log.warning(f"Model {name!r} not available, falling back to {FALLBACK_MODEL!r}")
        return FALLBACK_MODEL


# Mapping of HF model_type → the canonical decoder-block class name
# FSDP should auto-wrap. Used when the user hasn't explicitly set
# --fsdp_transformer_layer_cls_to_wrap. Add new families here as the
# zoo grows. Falls back to the user-provided value (default
# LlamaDecoderLayer) if model_type isn't in the map.
_DEFAULT_WRAP_CLS_BY_MODEL_TYPE = {
    "llama": "LlamaDecoderLayer",
    "llama4": "Llama4DecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "mistral": "MistralDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "gemma2": "Gemma2DecoderLayer",
    "phi": "PhiDecoderLayer",
    "phi3": "Phi3DecoderLayer",
    "gpt_neox": "GPTNeoXLayer",
    "gpt2": "GPT2Block",
    "deepseek_v3": "DeepseekV3DecoderLayer",
    "olmo": "OlmoDecoderLayer",
    "olmo2": "Olmo2DecoderLayer",
}


def _autodetect_wrap_cls(model_name: str, user_provided: str) -> str:
    """Pick the right FSDP auto-wrap class for *model_name*.

    If the user explicitly passed --fsdp_transformer_layer_cls_to_wrap
    with a non-default value, respect it. Otherwise look up the
    model's HF model_type in _DEFAULT_WRAP_CLS_BY_MODEL_TYPE.

    Falls back to the user-provided value (default LlamaDecoderLayer)
    if model_type is unknown — the FSDP plugin's lookup will then
    raise a clear ValueError naming the missing class, which is the
    same behavior as before this helper existed.
    """
    # If the user opted in to a non-default class, don't override
    if user_provided != "LlamaDecoderLayer":
        return user_provided

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        model_type = getattr(cfg, "model_type", None)
        if model_type and model_type in _DEFAULT_WRAP_CLS_BY_MODEL_TYPE:
            cls = _DEFAULT_WRAP_CLS_BY_MODEL_TYPE[model_type]
            if cls != user_provided:
                log.info(
                    f"[FSDP] auto-detected wrap class {cls!r} for "
                    f"model_type={model_type!r} (override with "
                    f"--fsdp_transformer_layer_cls_to_wrap)"
                )
            return cls
    except Exception as e:
        log.warning(
            f"[FSDP] model_type autodetect failed ({e!r}); falling back "
            f"to user-provided wrap class {user_provided!r}"
        )
    return user_provided


def _prefetch_and_broadcast_model(model_name: str, rank: int) -> str:
    """Resolve + prefetch model files on rank 0, then barrier so all
    ranks load from a populated HF cache.

    HF Hub rate-limits per-IP HEAD requests; 48 ranks doing
    ``AutoConfig.from_pretrained`` / ``AutoTokenizer.from_pretrained``
    / ``AutoModel.from_pretrained`` concurrently against the same
    repo hits 429s and stalls every retry by 200+s. Pre-warming the
    cache from rank 0 collapses that to a single sequence of
    requests, then all ranks read from disk after the barrier.

    Returns the resolved model name (broadcast from rank 0 so all
    ranks see the same fallback decision).
    """
    import torch.distributed as dist
    from transformers import AutoConfig, AutoTokenizer

    if rank == 0:
        # First: resolve fallback (tokenizer HEAD; populates cache too)
        resolved = _resolve_model(model_name) if model_name else _resolve_model(DEFAULT_MODEL)
        log.info(f"[prefetch] rank 0 resolved model={resolved!r}; warming HF cache...")
        # Skip the Hub snapshot_download when `resolved` is a local
        # path — worker ranks read straight from disk and we'd just
        # log a confusing 404 ("Repository Not Found for url:
        # https://huggingface.co/api/models/<local-name>") that makes
        # it look like the local checkpoint wasn't found.
        is_local = os.path.isdir(resolved) or os.path.isfile(
            os.path.join(resolved, "config.json")
        )
        if is_local:
            log.info(f"[prefetch] {resolved!r} is a local path; skipping Hub snapshot_download")
        else:
            # Force the config + tokenizer + weights HEAD/GET to populate
            # the on-disk cache. Weights pulled here so worker ranks just
            # mmap them later instead of each issuing their own HEAD.
            try:
                AutoConfig.from_pretrained(resolved)
                AutoTokenizer.from_pretrained(resolved)
                # snapshot_download pulls weight shards into the cache
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=resolved, allow_patterns=[
                    "*.json", "*.txt", "*.model", "tokenizer*",
                    "*.safetensors", "*.bin",
                ])
                log.info(f"[prefetch] rank 0 cache warm for {resolved!r}")
            except Exception as e:
                # Non-fatal: per-rank loads will still hit network.
                log.warning(f"[prefetch] rank 0 cache warm failed: {e}; continuing")
    else:
        resolved = ""

    # Broadcast the resolved name from rank 0 so worker ranks pick
    # up the same fallback choice. object_list broadcast handles
    # arbitrary Python types.
    if dist.is_initialized():
        names = [resolved]
        dist.broadcast_object_list(names, src=0)
        resolved = names[0]
        dist.barrier()  # don't let workers hit the cache until rank 0 finishes writing

    return resolved


_FSDP_STRATEGY_MAP = {
    "full_shard": "FULL_SHARD",
    "shard_grad_op": "SHARD_GRAD_OP",
    "no_shard": "NO_SHARD",
    "hybrid_shard": "HYBRID_SHARD",
    "hybrid_shard_zero2": "HYBRID_SHARD_ZERO2",
}


def _bootstrap_fsdp_env(
    fsdp: str,
    *,
    transformer_layer_cls_to_wrap: str,
    cpu_ram_efficient_loading: bool,
    bf16: bool,
) -> None:
    """Populate the FSDP_/ACCELERATE_USE_FSDP env vars that HF Trainer's
    internal Accelerator reads at __init__ time.

    Mirrors ezpz.examples.hf.py's explicit-FSDP-plugin pattern but adapted
    to GRPOTrainer (which builds its own Accelerator internally rather
    than letting us pass one). We set env vars BEFORE GRPOTrainer.__init__
    runs; Accelerator picks them up via
    accelerate.utils.dataclasses.FullyShardedDataParallelPlugin.__post_init__.

    The ``fsdp`` argument is the value of TrainingArguments.fsdp after
    parsing — TRL/HF accept either a string ('full_shard', 'shard_grad_op',
    ...) or a list of FSDPOption enums. We coerce both shapes here.
    """
    # TrainingArguments.fsdp accepts str | list[FSDPOption] | None
    if not fsdp:
        return  # plain DDP — leave env alone

    def _opt_name(x) -> str:
        # FSDPOption is a str enum; its .value is e.g. 'full_shard'.
        # Plain strings come through unchanged.
        return getattr(x, "value", str(x)).lower()

    if isinstance(fsdp, (list, tuple)):
        # CLI like --fsdp "full_shard auto_wrap" parses into a list of
        # FSDPOption enums; we only care about the sharding-strategy slot.
        tokens = [_opt_name(x) for x in fsdp]
    else:
        tokens = _opt_name(fsdp).split()

    strategy_key = next(
        (t for t in tokens if t in _FSDP_STRATEGY_MAP), None
    )
    if strategy_key is None:
        raise ValueError(
            f"--fsdp must include one of {sorted(_FSDP_STRATEGY_MAP)}; "
            f"got {fsdp!r}"
        )

    # Required by Accelerator() to actually build an FSDP plugin
    os.environ.setdefault("ACCELERATE_USE_FSDP", "true")
    # Sharding strategy (FSDP1 vocabulary; FSDP2 reshard_after_forward is bool)
    os.environ.setdefault(
        "FSDP_SHARDING_STRATEGY", _FSDP_STRATEGY_MAP[strategy_key]
    )
    # Auto-wrap policy — without this the whole model wraps as a single FSDP
    # unit, defeating the point.
    os.environ.setdefault("FSDP_AUTO_WRAP_POLICY", "TRANSFORMER_BASED_WRAP")
    os.environ.setdefault(
        "FSDP_TRANSFORMER_CLS_TO_WRAP", transformer_layer_cls_to_wrap
    )
    # Match ezpz.examples.hf.py's plugin defaults
    os.environ.setdefault("FSDP_BACKWARD_PREFETCH", "BACKWARD_PRE")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")
    # Sharded state dict avoids the rank-0-gathers-everything OOM at save time
    os.environ.setdefault("FSDP_STATE_DICT_TYPE", "SHARDED_STATE_DICT")
    # Mixed precision under bf16
    if bf16:
        os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
    # Optional cpu_ram_efficient_loading (load on rank 0 + broadcast)
    if cpu_ram_efficient_loading:
        os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
        # Required companion: sync module states after broadcast
        os.environ.setdefault("FSDP_SYNC_MODULE_STATES", "true")

    log.info(
        "[FSDP] enabled via env: strategy=%s wrap_cls=%s cpu_eff_load=%s",
        _FSDP_STRATEGY_MAP[strategy_key],
        transformer_layer_cls_to_wrap,
        cpu_ram_efficient_loading,
    )


def _build_wandb_config(
    ezpz_args, config, model_name: str, device_type: str, rank: int
) -> dict:
    """Build the wandb init-config block from every dataclass field.

    The old hand-curated dict only logged ~11 of GRPOConfig's 60+ fields
    (and none of TrainingArguments' 100+ inherited fields). Anything not
    in that hand-curated list silently disappeared from wandb hyperparams,
    making sweeps hard to interpret.

    Now: dump every field of ezpz_args + config to wandb config, with
    light filtering for fields that don't serialize cleanly (callables,
    huge strings, internal HF state).
    """
    import dataclasses

    EZPZ_PREFIX = "ezpz/"
    TRAIN_PREFIX = "train/"
    # Fields that aren't useful or don't serialize well
    SKIP = {
        "hub_token", "push_to_hub_token",  # secrets
        "logging_dir", "output_dir", "run_name",  # path noise
        "label_names",  # internal HF
    }

    def _safe(v):
        # Coerce non-JSON-serializable values into reprs wandb can handle
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _safe(val) for k, val in v.items()}
        return repr(v)

    out: dict = {}

    # ezpz-side args
    for f in dataclasses.fields(ezpz_args):
        if f.name in SKIP:
            continue
        out[f"{EZPZ_PREFIX}{f.name}"] = _safe(getattr(ezpz_args, f.name))

    # GRPOConfig + TrainingArguments inherited fields
    for f in dataclasses.fields(config):
        if f.name in SKIP or f.name.startswith("_"):
            continue
        val = getattr(config, f.name, None)
        out[f"{TRAIN_PREFIX}{f.name}"] = _safe(val)

    # Resolved runtime extras
    out["runtime/model_name"] = model_name
    out["runtime/device_type"] = device_type
    out["runtime/rank"] = rank
    out["runtime/world_size"] = int(os.environ.get("WORLD_SIZE", "1"))

    return out


# Jinja chat templates. Use single-token turn boundaries that exist in
# the model's vocab — otherwise the model has never seen them and
# treats them as ordinary text, producing prompt-echoing completions
# like "<|user|> What is 1+2? <|user|> 1+2 = 3 What is ...".
#
# The assistant content is wrapped in {% generation %}…{% endgeneration %}
# so TRL's SFTTrainer with assistant_only_loss=True can mask out the
# user turn and compute loss only on the assistant span. These markers
# are renderer-neutral (jinja just ignores them at render time) so the
# same template works for both inference / GRPO and SFT.
_CHAT_TEMPLATE_GEMMA = (
    # Gemma format: single-token <start_of_turn>/<end_of_turn> (ids
    # 106/107 in Gemma 2/3 tokenizer family — AuroraGPT-2B uses this).
    # 'user' and 'model' are the canonical role names.
    #
    # The `.lstrip('\n')` calls on each message's content keep the
    # template "prefix-preserving" under sentencepiece tokenization
    # — otherwise an assistant message starting with `\n` produces
    # a `\n\n` (id 109) token boundary when the user prompt's
    # trailing `\n` (id 108) merges with the assistant's leading `\n`,
    # while prompt-alone rendering keeps the `\n` (id 108) terminal.
    # That mismatch trips TRL's SFTTrainer warning at sft_trainer:1480
    # ("Mismatch between tokenized prompt and the start of tokenized
    # prompt+completion") and breaks the assistant-only-loss mask.
    # Stripping leading newlines is a no-op on the rendered content
    # for clean data and only affects the merge-boundary case.
    "{% for message in messages %}"
    "{% if message['role'] == 'system' or message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'].lstrip('\n') }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n"
    "{% generation %}{{ message['content'].lstrip('\n') }}<end_of_turn>\n{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
_CHAT_TEMPLATE_CHATML = (
    # ChatML format: <|im_start|>/<|im_end|> — used by Qwen, OpenAI's
    # public chatml spec, etc. Tokenizers in this family encode the
    # markers as single tokens.
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n"
    "{% generation %}{{ message['content'] }}<|im_end|>\n{% endgeneration %}"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_CHAT_TEMPLATE_PLAIN = (
    # Plain-text fallback. No special tokens. Uses ALL-CAPS role names
    # + double-newline boundaries which most pretrain corpora contain.
    # Works on any tokenizer, but the model has no native turn boundary
    # signal — generation will tend to ramble past the answer.
    "{% for message in messages %}"
    "{% if message['role'] == 'system' or message['role'] == 'user' %}"
    "USER: {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "ASSISTANT: "
    "{% generation %}{{ message['content'] }}\n\n{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
)


def _pick_chat_template(tokenizer) -> tuple[str, str]:
    """Pick a chat template the tokenizer's vocab actually understands.

    Returns (kind, template). Kind is a short label used in the log
    message so it's clear which format was chosen.

    Strategy: probe the tokenizer for known single-token turn markers
    (Gemma's <start_of_turn>, ChatML's <|im_start|>) and pick the
    matching template. If neither is single-token, fall back to a
    plain-text USER/ASSISTANT format which works anywhere but is
    less effective at constraining the model.
    """
    def _is_single_token(s: str) -> bool:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return len(ids) == 1
        except Exception:
            return False

    if _is_single_token("<start_of_turn>") and _is_single_token("<end_of_turn>"):
        return "gemma", _CHAT_TEMPLATE_GEMMA
    if _is_single_token("<|im_start|>") and _is_single_token("<|im_end|>"):
        return "chatml", _CHAT_TEMPLATE_CHATML
    return "plaintext", _CHAT_TEMPLATE_PLAIN


def main() -> None:
    # Apply XPU compatibility patches BEFORE importing trl (some are
    # active at import time of TRL submodules, e.g. the cuda alias
    # patch needs to be in place before GRPOTrainer constructs
    # VLLMGeneration which hardcodes torch.cuda.current_device()).
    from torchtitan.experiments.ezpz.rl.xpu_overrides import (
        apply_all_xpu_patches,
    )

    apply_all_xpu_patches()

    from trl import GRPOTrainer
    from transformers import AutoTokenizer, HfArgumentParser

    # Silence noisy HTTP request logs from huggingface_hub / httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    EzpzGRPOConfig = _ezpz_grpo_config_cls()
    parser = HfArgumentParser((EzpzGRPOArgs, EzpzGRPOConfig))
    ezpz_args, config = parser.parse_args_into_dataclasses()

    rank = ezpz.distributed.get_rank()
    device_type = ezpz.distributed.get_torch_device_type()

    # Resolve + pre-warm HF cache on rank 0, then barrier so worker
    # ranks read from cache instead of hammering HF Hub with 48
    # concurrent HEAD requests (which trips per-IP rate limits, see
    # HTTP 429 retries with 200s+ backoffs).
    model_name = _prefetch_and_broadcast_model(ezpz_args.model_name_or_path, rank)

    # Auto-detect the right FSDP wrap class for this model family if
    # the user didn't override it. Runs AFTER prefetch so the config
    # is in the local HF cache (no per-rank network hits).
    wrap_cls = _autodetect_wrap_cls(
        model_name, ezpz_args.fsdp_transformer_layer_cls_to_wrap
    )
    ezpz_args.fsdp_transformer_layer_cls_to_wrap = wrap_cls

    # Bootstrap FSDP env vars BEFORE GRPOTrainer.__init__ — its internal
    # accelerate.Accelerator only reads them once at construction time.
    # Mirrors the explicit-FSDP-plugin pattern in ezpz.examples.hf.py.
    _bootstrap_fsdp_env(
        config.fsdp,
        transformer_layer_cls_to_wrap=wrap_cls,
        cpu_ram_efficient_loading=ezpz_args.fsdp_cpu_ram_efficient_loading,
        bf16=config.bf16,
    )

    # (FSDP + gradient_checkpointing migration happens in
    # EzpzGRPOConfig.__post_init__ — see _ezpz_grpo_config_cls.)

    # Fill in output_dir sentinel from task
    if config.output_dir is None:
        config.output_dir = f"outputs/rl/grpo-{ezpz_args.task}"

    # report_to to wandb only on rank 0 (avoids 48 ranks all writing).
    # If the user passed --report_to explicitly we respect it on rank 0
    # but force "none" on every other rank.
    # Rank 0 defaults to wandb so the run is always observable; worker
    # ranks always get silenced so we don't double-log. Pass
    # --report_to none explicitly to opt out. TRL's own default is
    # "none", which silently means runs without --report_to wandb
    # don't show up in the wandb project at all.
    if rank == 0:
        not_explicitly_set = (
            config.report_to == "none"
            or config.report_to == ["none"]
            or not config.report_to
        )
        if not_explicitly_set:
            config.report_to = ["wandb"]
    else:
        config.report_to = []

    # save_steps must be > max_steps so the in-train save never fires.
    # (Belt-and-suspenders alongside save_strategy="no".)
    if config.save_steps and config.save_steps <= config.max_steps:
        config.save_steps = config.max_steps + 1

    # Resolve task from registry
    task = get_task(ezpz_args.task)

    log.info(
        f"[rank {rank}] GRPO config: model={model_name} task={ezpz_args.task} "
        f"max_steps={config.max_steps} lr={config.learning_rate} "
        f"bsz={config.per_device_train_batch_size} "
        f"num_gens={config.num_generations} beta={config.beta} "
        f"grad_ckpt={config.gradient_checkpointing} use_vllm={config.use_vllm} "
        f"fsdp={config.fsdp or 'off'} device={device_type}"
    )

    # W&B tracking via ezpz (rank 0 only). Initializing BEFORE GRPOTrainer
    # constructs its internal Accelerator gives HF Trainer's WandbCallback
    # a live wandb run to attach to, so every self.log() inside training
    # (GRPO metrics, rewards, completions, sampling stats) makes it to wandb.
    if rank == 0:
        ezpz.distributed.setup_wandb(
            project_name="torchtitan.ezpz.rl",
            config=_build_wandb_config(
                ezpz_args, config, model_name, device_type, rank,
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # GRPOTrainer._tokenize_prompts (trl/trainer/grpo_trainer.py:1312)
    # calls tokenizer.apply_chat_template, which crashes on base /
    # pretraining-only tokenizers that have no chat template. Inject a
    # template the tokenizer can actually use as turn boundaries — see
    # _pick_chat_template for the per-vocab logic. Existing chat
    # templates (Instruct variants, Qwen, etc.) are untouched.
    if tokenizer.chat_template is None:
        kind, tokenizer.chat_template = _pick_chat_template(tokenizer)
        log.info(
            f"[rank {rank}] tokenizer has no chat_template; injected "
            f"{kind!r} fallback (override by setting tokenizer.chat_template "
            f"explicitly before launching, or pass --chat_template_kwargs)"
        )

    dataset = task.build_dataset(num_samples=ezpz_args.num_samples)
    log.info(f"[rank {rank}] Built dataset: {len(dataset)} samples")

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=task.reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info(f"[rank {rank}] Starting GRPO training...")
    trainer.train()
    log.info(f"[rank {rank}] Training complete.")

    if rank == 0 and not ezpz_args.no_save:
        save_path = os.path.join(config.output_dir, "final")
        try:
            trainer.save_model(save_path)
            log.info(f"Model saved to {save_path}")
        except Exception as e:
            log.warning(f"Failed to save model to {save_path}: {e}")


if __name__ == "__main__":
    ezpz.distributed.setup_torch()
    main()
