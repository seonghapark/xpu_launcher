import ezpz
import ezpz.distributed
import json
import os
from dataclasses import is_dataclass
from typing import Any, Literal

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.experiments.ezpz.validator import EzpzValidator
from torchtitan.config import CommConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.config.configs import CompileConfig
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_debugmodel()


def agpt_2b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b()


def agpt_2b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_2b()
    cfg.dataloader.dataset_path = None
    return cfg


def _set_rope_backend(
    cfg: FaultTolerantTrainer.Config,
    backend: Literal["complex", "cos_sin"],
) -> FaultTolerantTrainer.Config:
    """Switch every RoPE callsite in the model spec to ``backend``.

    PR #3458 (RoPE refactor) split ``RoPE.Config`` into
    ``ComplexRoPE.Config`` / ``CosSinRoPE.Config`` and dropped the
    ``backend`` string field — backend is now encoded in the type.
    The top-level ``Model.Config.rope`` field is also gone; each
    layer's ``Attention.Config`` owns its own rope. So flipping the
    backend means rebuilding each layer's ``attention.rope`` as a
    fresh instance of the target subclass, copying over all other
    fields (dim / max_seq_len / theta / scaling / yarn params).
    """
    from dataclasses import fields

    from torchtitan.models.common import ComplexRoPE, CosSinRoPE

    target_cls = ComplexRoPE.Config if backend == "complex" else CosSinRoPE.Config
    model = cfg.model_spec.model
    for layer in model.layers:
        old_rope = layer.attention.rope
        if old_rope is None:
            continue
        kwargs = {f.name: getattr(old_rope, f.name) for f in fields(old_rope)}
        layer.attention.rope = target_cls(**kwargs)
    return cfg


def agpt_2b_real() -> FaultTolerantTrainer.Config:
    """agpt_2b with real-valued (cos_sin) RoPE instead of complex.

    The default `RoPE.Config(backend="complex")` uses torch.complex64
    ops that torch.compile inductor refuses to lower:

        UserWarning: Torchinductor does not support code generation
        for complex operators. Performance may be worse than eager.

    The `cos_sin` backend uses real-valued sin/cos rotations that
    inductor can compile, so this flavor exists to A/B test whether
    eliminating the eager fallback inside the compiled graph
    improves XPU throughput.
    """
    return _set_rope_backend(ezpz_agpt_2b(), "cos_sin")


def agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b_flex_attn()


def agpt_7b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_7b()


def agpt_7b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_7b()
    cfg.dataloader.dataset_path = None
    return cfg


def ezpz_agpt_8b() -> FaultTolerantTrainer.Config:
    return _base_config("8B")


def agpt_8b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_8b()


def agpt(
    flavor: str,
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full"] = "full",
    seq_len: int = 8192,
    # IMPORTANT: bfloat16 master weights silently freeze RMSNorm.weight.
    # Norm weights init to 1.0 (bf16 ulp = 7.8e-3); per-step updates
    # ~1.6e-5 round to zero forever, so the model's normalization layers
    # never train. FSDP MixedPrecisionPolicy keeps the bf16 cast for
    # forward/backward; reduce stays fp32 — the fp32 master copy is
    # what enables sub-ulp accumulation.
    # See docs/guides/known-bugs/training-dtype-bf16-norm-freeze.md.
    dtype: Literal["bfloat16", "float32"] = "float32",
    compile: bool = True,
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default",
    tensor_parallel_degree: int = 1,
    checkpoint_interval: int = 50,
    hf_assets_path: str = "./assets/hf/gemma-7b",
    dataset_path: str | None = None,
) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    cfg.hf_assets_path = hf_assets_path
    cfg.debug.print_config = True
    cfg.training.local_batch_size = local_batch_size
    # 57th sync: PR #3674 replaced the `mode` string with a policy class
    # hierarchy. `None` disables AC (was mode="none"); FullAC.Config()
    # is the agpt default (was mode="full").
    cfg.activation_checkpoint = (
        None if activation_checkpoint_mode == "none" else FullAC.Config()
    )
    cfg.training.seq_len = seq_len
    cfg.training.dtype = dtype
    cfg.dataloader.dataset = "blendcorpus"
    if dataset_path is None:
        dataset_path = f"torchtitan/experiments/ezpz/data-lists/{ezpz.distributed.get_machine().lower()}/books.txt"
    cfg.dataloader.dataset_path = dataset_path
    # Validator reads from the same blendcorpus corpus, but it pulls from
    # the validation split (see BlendCorpusDataLoader.Config.serve_validation).
    # Also default the validator's data_cache_path to the trainer's so it does
    # not keep the bare ".cache/blendcorpus" default and cold-build the
    # validation index at full scale on the first validate() call -- the race
    # that crashed job 12469584 ("mmap length > file size" at TP>1, mistaken
    # for a validator collective deadlock; see docs/guides/known-bugs/).
    # NOTE: this only aligns the in-config DEFAULT. In production the submit
    # scripts pass the warm path explicitly via
    # --validator.dataloader.data-cache-path (applied by tyro AFTER this
    # builder), which is the operative fix; this copy is defense-in-depth for
    # interactive / non-script callers. Either way the validation-split index
    # must be prewarmed (prewarm_blendcorpus_cache.sh builds it).
    if isinstance(cfg.validator.dataloader, BlendCorpusDataLoader.Config):
        cfg.validator.dataloader.dataset_path = dataset_path
        cfg.validator.dataloader.data_cache_path = cfg.dataloader.data_cache_path
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    if compile:
        cfg.compile = CompileConfig(enable=True)
    cfg.parallelism.fsdp_reshard_after_forward = fsdp_reshard_after_forward
    cfg.parallelism.tensor_parallel_degree = tensor_parallel_degree
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = checkpoint_interval
    return cfg


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/hf/gemma-7b",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        loss=CrossEntropyLoss.Config(),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
        # Walltime-aware checkpointing: guarantee a save before the PBS walltime
        # runs out (otherwise a short job can save nothing -- see the train loop
        # in trainer.py). The failover submit scripts export
        # $WALLTIME_DEADLINE_EPOCH (absolute job_start+walltime timestamp,
        # survives failover retries -- PREFERRED) and $WALLTIME_SECONDS (relative
        # fallback). 0/unset disables it (interactive runs, non-PBS). Overridable
        # on the CLI via --walltime-deadline-epoch / --walltime-seconds.
        walltime_deadline_epoch=int(
            os.environ.get("WALLTIME_DEADLINE_EPOCH", "0") or "0"
        ),
        walltime_seconds=int(os.environ.get("WALLTIME_SECONDS", "0") or "0"),
        # Validator runs on the blendcorpus validation split (5% of the
        # corpus by default — see BlendCorpusDataLoader.Config.split). The
        # validator builds its own dataloader from this Config every time
        # validate() is called, so serve_validation=True ensures it gets
        # held-out samples instead of the train split.
        # Default enable=False keeps prior behavior; flip per config or via
        # --validator.enable on the CLI.
        # Uses EzpzValidator (subclass of Validator) which fixes loss
        # reporting on TP > 1 — see torchtitan/experiments/ezpz/validator.py.
        validator=EzpzValidator.Config(
            enable=False,
            freq=200,
            steps=10,
            dataloader=BlendCorpusDataLoader.Config(
                dataset="blendcorpus",
                serve_validation=True,
                infinite=False,
            ),
        ),
    )


def ezpz_agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return agpt("debugmodel", local_batch_size=2)


def ezpz_agpt_2b() -> FaultTolerantTrainer.Config:
    return agpt("2b", activation_checkpoint_mode="none")


def agpt_2b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_2b with ChunkedLossWrapper to keep peak memory low.

    With vocab=256128 the unchunked logits are ~16 GB at LBS=2 / seq=8192,
    which OOMs on Aurora's 64 GB tiles. ChunkedLossWrapper(num_chunks=8) caps
    the peak slice at ~2 GB. Set lm_head module reference at trainer init
    via the set_lm_head plumbing in `experiments/ezpz/trainer.py`.
    """
    cfg = ezpz_agpt_2b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


def ezpz_agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("2b_flex_attn", local_batch_size=2)


def ezpz_agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def ezpz_agpt_7b() -> FaultTolerantTrainer.Config:
    return agpt("7b", local_batch_size=2, seq_len=4096, hf_assets_path="./assets/hf/llama-2-7b-hf")


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def ezpz_agpt_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("debugmodel")


def ezpz_agpt_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("2b")


def ezpz_agpt_7b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("7b")


def ezpz_agpt_8b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("8B")


def ezpz_agpt_blendcorpus_debugmodel() -> FaultTolerantTrainer.Config:
    cfg = _base_config("debugmodel")
    cfg.dataloader.dataset = "blendcorpus"
    if isinstance(cfg.tokenizer, EZPZTokenizer.Config):
        cfg.tokenizer.backend = "sptoken"
    return cfg


def ezpz_agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def agpt_20b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_20b with ChunkedLossWrapper. See agpt_2b_chunkedce for rationale."""
    cfg = ezpz_agpt_20b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


def agpt_20b_real() -> FaultTolerantTrainer.Config:
    """agpt_20b with real-valued (cos_sin) RoPE. See agpt_2b_real."""
    return _set_rope_backend(ezpz_agpt_20b(), "cos_sin")


def ezpz_agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def ezpz_agpt_50b_wide() -> FaultTolerantTrainer.Config:
    return agpt("50B_wide", tensor_parallel_degree=2)


def agpt_50b_wide() -> FaultTolerantTrainer.Config:
    return agpt("50B_wide", tensor_parallel_degree=2)


def ezpz_agpt_70b_wide() -> FaultTolerantTrainer.Config:
    return agpt("70B_wide", tensor_parallel_degree=2)


def agpt_70b_wide() -> FaultTolerantTrainer.Config:
    return agpt("70B_wide", tensor_parallel_degree=2)


def ezpz_agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def agpt_80b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_80b with ChunkedLossWrapper. See agpt_2b_chunkedce for rationale."""
    cfg = ezpz_agpt_80b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


def agpt_80b_real() -> FaultTolerantTrainer.Config:
    """agpt_80b with real-valued (cos_sin) RoPE. See agpt_2b_real."""
    return _set_rope_backend(ezpz_agpt_80b(), "cos_sin")


def ezpz_agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B")


def ezpz_agpt_80b_wide_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_wide")


def ezpz_agpt_80b_deep_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_deep")


# Competition speedrun configs — makes them discoverable via --config
from torchtitan.experiments.ezpz.competition.configs import (  # noqa: E402, F401
    speedrun_2b_adamw,
    speedrun_2b_adamw_cosine,
    speedrun_2b_adamw_fast_warmup,
    speedrun_2b_adamw_high_lr,
    speedrun_2b_adamw_qknorm,
    speedrun_2b_adamw_short_decay,
    speedrun_2b_mano,
    speedrun_2b_mano_1e3,
    speedrun_2b_mano_cosine,
    speedrun_2b_mano_high_lr,
    speedrun_2b_mano_qknorm,
    speedrun_2b_muon,
    speedrun_2b_muon_aggressive,
    speedrun_2b_muon_cosine,
    speedrun_2b_muon_fast_warmup,
    speedrun_2b_muon_qknorm,
    speedrun_2b_muon_short_decay,
    speedrun_2b_sophiag,
    speedrun_2b_spam,
    speedrun_2b_torchmuon,
    speedrun_2b_torchmuon_cosine,
)
from torchtitan.experiments.ezpz.competition.configs import (  # noqa: E402, F401
    full_2b_adamw,
    full_2b_adamw_qknorm,
    full_2b_mano,
    full_2b_mano_qknorm,
    full_2b_muon,
    r4_adamw,
    r4_adamw_higher_lr,
    r4_adamw_qknorm,
    r4_adamw_qknorm_softcap,
    r4_adamw_softcap,
    r4_mano,
    r4_mano_higher_lr,
    r4_mano_qknorm,
    r5_adamw_constant_lr,
    r5_mano_constant_lr,
    r5_mano_lr1e3,
    r5_mano_lr2e3,
    r5_mano_lr3e4,
    r5_mano_lr6e4,
    r5_schedulefree,
    smoke_2b_50steps,
    smoke_2b_async_ckpt,
    smoke_2b_async_ckpt_pinned,
    speedrun_2b_kitchen_sink,
    speedrun_2b_mano_kitchen_sink,
    speedrun_2b_relu2,
    speedrun_2b_softcap,
)
