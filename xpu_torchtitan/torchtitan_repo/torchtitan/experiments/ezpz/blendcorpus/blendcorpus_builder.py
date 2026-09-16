from dataclasses import dataclass, field
import importlib
import os
from types import SimpleNamespace
from typing import Any

import ezpz
import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import logger


def _import_blendcorpus_modules():
    try:
        bc_mpu = importlib.import_module("blendcorpus.parallel_state")
        bc_config_mod = importlib.import_module("blendcorpus.data.config")
        bc_sampler_mod = importlib.import_module("blendcorpus.data.data_samplers")
        bc_dataset_mod = importlib.import_module("blendcorpus.data.gpt_dataset")
    except ImportError as exc:
        # exc.name is the module Python actually failed on. Distinguish
        # "blendcorpus itself missing" from "blendcorpus IS installed but
        # one of its transitive deps (e.g. deepspeed) isn't" — the two
        # cases need different fixes and the same error message for both
        # is actively misleading.
        missing = getattr(exc, "name", None) or "<unknown>"
        top = missing.split(".", 1)[0]
        if top == "blendcorpus":
            detail = (
                f"BlendCorpus dataset was requested but the `blendcorpus` "
                f"package is not installed (failed to import `{missing}`). "
                "Install it (e.g. `pip install blendcorpus`) or set "
                "`--dataloader.dataset` to a non-blendcorpus dataset."
            )
        else:
            detail = (
                f"BlendCorpus dataset was requested. `blendcorpus` itself "
                f"is installed, but importing it pulled in a missing "
                f"transitive dependency: `{missing}` (top-level module: "
                f"`{top}`). Install the missing dependency in the active "
                f"env (e.g. `pip install {top}`), or set "
                f"`--dataloader.dataset` to a non-blendcorpus dataset to "
                f"sidestep the import entirely. The original ImportError "
                f"is chained below."
            )
        raise ImportError(detail) from exc

    bc_get_config = getattr(bc_config_mod, "get_config")
    bc_set_config = getattr(bc_config_mod, "set_config")
    build_pretraining_data_loader = getattr(
        bc_sampler_mod, "build_pretraining_data_loader"
    )
    build_gpt_datasets = getattr(bc_dataset_mod, "build_gpt_datasets")

    return (
        bc_mpu,
        bc_get_config,
        bc_set_config,
        build_gpt_datasets,
        build_pretraining_data_loader,
    )


class BlendCorpusDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        num_workers: int = 0
        persistent_workers: bool = False
        pin_memory: bool = field(
            default_factory=lambda: ezpz.get_torch_device_type() == "cuda"
        )
        prefetch_factor: int | None = None
        infinite: bool = True

        split: str = "95,5,0"
        dataloader_type: str = "single"
        shuffle: bool = True
        shuffle_sample_in_corpus: bool = True
        blend_sample_in_corpus: bool = False
        append_eod: bool = True
        provide_attention_mask: bool = False
        eod_token_id: int | None = None
        data_cache_path: str = ".cache/blendcorpus"

        train_iters: int | None = None
        # When True, this dataloader serves the validation split of the
        # blendcorpus corpus instead of the train split. Used by the
        # Validator to compute held-out NLL on a slice of the same corpus
        # the model is training on. eval_iters controls how many distinct
        # validation samples blendcorpus pre-builds; larger is wasteful but
        # safe — the actual number of validation steps is set on
        # Validator.Config.steps.
        serve_validation: bool = False
        eval_iters: int = 100

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        self._mode = "hf"
        self._delegate: BaseDataLoader | None = None

        if config.dataset != "blendcorpus":
            hf_cfg = HuggingFaceTextDataLoader.Config(
                dataset=config.dataset,
                dataset_path=config.dataset_path,
                num_workers=config.num_workers,
                persistent_workers=config.persistent_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                infinite=config.infinite,
            )
            self._delegate = hf_cfg.build(
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                tokenizer=tokenizer,
                seq_len=seq_len,
                local_batch_size=local_batch_size,
            )
            return

        self._mode = "blendcorpus"
        (
            bc_mpu,
            bc_get_config,
            bc_set_config,
            build_gpt_datasets,
            build_pretraining_data_loader,
        ) = _import_blendcorpus_modules()

        parallel_dims = kwargs.get("parallel_dims")
        tp_degree = getattr(parallel_dims, "tp", 1)
        pp_degree = getattr(parallel_dims, "pp", 1)
        cp_degree = getattr(parallel_dims, "cp", 1)

        requested_global_batch_size = kwargs.get("global_batch_size")
        if not requested_global_batch_size or requested_global_batch_size <= 0:
            requested_global_batch_size = local_batch_size * dp_world_size

        train_iters = config.train_iters
        if train_iters is None:
            train_iters = kwargs.get("training_steps")
        if train_iters is None:
            train_iters = 1
            logger.warning(
                "BlendCorpus train_iters was not provided; defaulting to 1. "
                "Set --training.steps or --dataloader.train-iters explicitly."
            )
        else:
            train_iters = int(train_iters)
            if train_iters <= 0:
                logger.warning(
                    "BlendCorpus got non-positive train_iters=%s; defaulting to 1 "
                    "to avoid oversized index allocation.",
                    train_iters,
                )
                train_iters = 1

        bc_cfg = SimpleNamespace(
            data_file_list=config.dataset_path,
            seq_length=seq_len,
            train_iters=train_iters,
            # eval_iters > 0 triggers validation index file construction
            # in blendcorpus. Production runs have always used 0 here, so
            # the valid index files do not exist on disk for our corpora.
            # Only request eval_iters > 0 when this loader is actually
            # being asked to serve the validation split.
            eval_iters=int(config.eval_iters) if config.serve_validation else 0,
            seed=42,
            data_impl="mmap",
            micro_batch_size=int(local_batch_size),
            global_batch_size=int(requested_global_batch_size),
            tensor_model_parallel_size=int(tp_degree),
            pipeline_model_parallel_size=int(pp_degree),
            sequence_parallel_size=int(cp_degree),
            num_workers=int(config.num_workers),
            pin_memory=bool(config.pin_memory),
            split=config.split,
            dataloader_type=config.dataloader_type,
            shuffle=bool(config.shuffle),
            shuffle_sample_in_corpus=bool(config.shuffle_sample_in_corpus),
            blend_sample_in_corpus=bool(config.blend_sample_in_corpus),
            append_eod=bool(config.append_eod),
            provide_attention_mask=bool(config.provide_attention_mask),
            eod_token_id=(
                int(config.eod_token_id)
                if config.eod_token_id is not None
                else getattr(tokenizer, "eos_id", None)
            ),
            data_cache_path=os.path.abspath(config.data_cache_path),
        )
        os.makedirs(bc_cfg.data_cache_path, exist_ok=True)

        # blendcorpus's initialize_model_parallel asserts that no parallel
        # groups already exist, so calling it twice (e.g. once for the
        # train loader, again for a validation loader) fatal-errors. Skip
        # re-init if the trainer already set it up.
        if not bc_mpu.model_parallel_is_initialized():
            bc_mpu.initialize_model_parallel(
                tensor_model_parallel_size=bc_cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size=bc_cfg.pipeline_model_parallel_size,
                sequence_parallel_size=bc_cfg.sequence_parallel_size,
            )

        # On XCCL (XPU) with torch <2.13, barrier() hangs because the
        # C++ XCCL backend ignores opts.device and defaults all ranks
        # to device 0.  Work around by replacing barrier() with a
        # CPU-side gloo barrier for the duration of dataset building.
        # On torch >=2.13 this is fixed upstream — skip the monkey-patch
        # so we don't silently route every other dist.barrier() call
        # (FSDP, DCP, validator, etc.) through CPU/gloo for the rest of
        # the session.
        import torch
        import torch.distributed as dist

        _torch_version_tuple = tuple(
            int(p) for p in torch.__version__.split("+")[0].split(".")[:2]
        )
        _xccl_needs_barrier_fix = (
            _torch_version_tuple < (2, 13)
            and hasattr(torch, "xpu")
            and torch.xpu.is_available()
            and getattr(
                dist.distributed_c10d._get_default_group(),
                "bound_device_id",
                None,
            )
            is None
        )
        if _xccl_needs_barrier_fix:
            logger.info(
                "XCCL barrier workaround: using gloo (CPU) barriers "
                "for blendcorpus dataset building (torch %s)",
                torch.__version__,
            )
            _prev_gloo_log = os.environ.get("GLOO_LOG_LEVEL")
            os.environ["GLOO_LOG_LEVEL"] = "WARN"
            _gloo_world = dist.new_group(backend="gloo")
            if _prev_gloo_log is None:
                os.environ.pop("GLOO_LOG_LEVEL", None)
            else:
                os.environ["GLOO_LOG_LEVEL"] = _prev_gloo_log
            _orig_barrier = dist.barrier

            def _gloo_barrier(group=None, async_op=False, device_ids=None):
                # Always barrier on the gloo world group (CPU-side).
                return _orig_barrier(group=_gloo_world, async_op=async_op)

            dist.barrier = _gloo_barrier  # type: ignore[assignment]

        bc_set_config(bc_cfg)
        self._bc_cfg = bc_get_config()

        # All ranks call build_gpt_datasets together — the library has
        # internal barriers that require all ranks to participate.
        # Pre-cache step in benchmark_80b.sh ensures index files exist
        # on disk before this point.
        rank = int(os.environ.get("RANK", 0))
        logger.info(
            "Rank %d: building blendcorpus datasets (data=%s, cache=%s, "
            "serve_validation=%s)...",
            rank,
            bc_cfg.data_file_list,
            bc_cfg.data_cache_path,
            config.serve_validation,
        )
        train_ds, valid_ds, _ = build_gpt_datasets(self._bc_cfg)

        # Keep gloo barrier active for the entire session — the XCCL
        # barrier bug affects all collectives, not just dataset building.
        # The gloo barrier is CPU-side and works reliably on all backends.

        # Pick which split this loader serves. The validator constructs a
        # second BlendCorpusDataLoader.Config(serve_validation=True) so
        # both train and valid splits stay live in the same process.
        if config.serve_validation:
            if valid_ds is None:
                raise RuntimeError(
                    "BlendCorpusDataLoader.Config(serve_validation=True) but "
                    f"blendcorpus returned valid_ds=None — check that split "
                    f"({bc_cfg.split!r}) gives the validation slice non-zero "
                    f"weight and eval_iters ({bc_cfg.eval_iters}) > 0."
                )
            served_ds = valid_ds
        else:
            served_ds = train_ds

        logger.info("Rank %d: blendcorpus datasets ready.", rank)
        # ``_served_ds`` is whichever split this loader serves (train OR
        # valid, decided by ``config.serve_validation``). It feeds both
        # ``self._loader`` below and the resampler invoked from
        # ``set_consumed_by_global_step`` / ``load_state_dict``.
        self._served_ds = served_ds
        self._build_pretraining_data_loader = build_pretraining_data_loader
        self._loader = build_pretraining_data_loader(served_ds, 0, self._bc_cfg)
        self._consumed_samples = 0

        try:
            self._len = len(self._loader)
        except TypeError:
            self._len = int(1e12)

        logger.info("Using BlendCorpus dataloader backend")

    def __len__(self):
        if self._delegate is not None:
            if hasattr(self._delegate, "__len__"):
                return len(self._delegate)  # type: ignore[arg-type]
            return int(1e12)
        return self._len

    def __iter__(self):
        if self._delegate is not None:
            yield from iter(self._delegate)
            return

        for batch in self._loader:
            tokens = batch["text"].long()
            input_ids = tokens[:, :-1].contiguous()
            labels = tokens[:, 1:].contiguous()
            yield {"input": input_ids}, labels

    def set_consumed_by_global_step(self, global_step: int, global_batch_size: int):
        if self._delegate is not None:
            return

        consumed = int(global_step) * int(global_batch_size)
        self._consumed_samples = consumed
        self._loader = self._build_pretraining_data_loader(
            self._served_ds, consumed, self._bc_cfg
        )

    def state_dict(self) -> dict[str, Any]:
        if self._delegate is not None:
            return self._delegate.state_dict()
        return {"consumed_samples": int(self._consumed_samples)}

    def load_state_dict(self, state_dict: dict[str, Any]):
        if self._delegate is not None:
            self._delegate.load_state_dict(state_dict)
            return

        consumed = int(state_dict.get("consumed_samples", 0))
        if consumed != self._consumed_samples:
            self._consumed_samples = consumed
            self._loader = self._build_pretraining_data_loader(
                self._served_ds, consumed, self._bc_cfg
            )
