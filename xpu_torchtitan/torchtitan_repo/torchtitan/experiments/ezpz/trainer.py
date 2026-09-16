# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import ezpz

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.ezpz.lr_finder import LRFinderConfig
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import (
    TorchFTManager as FTManager,
    maybe_semi_sync_training,
)
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptimizersContainer as FTOptimizersContainer,
)
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer


def _set_pg_timeouts_xpu_aware(
    timeout: timedelta,
    parallel_dims: ParallelDims,
) -> None:
    """Apply ``timeout`` to every PG in the mesh, with explicit XPU support.

    Upstream ``dist_utils.set_pg_timeouts`` delegates to
    ``torch.distributed.distributed_c10d._set_pg_timeout``, whose device
    dispatch only knows about cpu (gloo) and cuda (nccl / gloo /
    torchcomms). On XPU the loop adds no backends, emits the warning
    ``"Set timeout is now only supported for either nccl or gloo."``,
    and never calls ``_set_default_timeout``. Result: ``train_timeout_seconds``
    silently no-ops and a hung collective burns the full PBS walltime
    instead of aborting (see
    ``docs/upstream-issues/train_timeout_xpu_silent_noop.md``).

    Workaround: set the timeout on every mesh PG ourselves (with the
    safety barrier the upstream helper uses), then additionally call
    ``ProcessGroupXCCL.set_timeout`` on any xccl-backed groups.

    61st sync: we no longer delegate to ``dist_utils.set_pg_timeouts``.
    The spmd_types series switched it from
    ``distributed_c10d._set_pg_timeout`` to ``torch.distributed.set_timeout``,
    which does NOT exist in our pinned torch 2.13 (AttributeError ->
    every collective unguarded). So we inline the pre-sync behavior
    against ``_set_pg_timeout`` (present in torch 2.13's
    ``distributed_c10d``), keeping this shim independent of upstream's
    timeout-API churn. Remove once PyTorch's timeout dispatch learns
    about XPU and our torch exposes the matching API.
    """
    from torch.distributed import distributed_c10d as c10d

    device_module = dist_utils.device_module
    # Flush in-flight work under the old timeout before lowering it
    # (mirrors upstream set_pg_timeouts' safety barrier).
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    timeout_groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    for group in timeout_groups:
        c10d._set_pg_timeout(timeout, group)

    xpu_device = torch.device("xpu")
    if not (
        torch.distributed.is_xccl_available() and torch.xpu.is_available()
    ):
        return

    from torch._C._distributed_c10d import ProcessGroupXCCL

    groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    patched = 0
    for group in groups:
        if group is None:
            group = torch.distributed.distributed_c10d._get_default_group()
        if xpu_device not in group._device_types:
            continue
        backend = group._get_backend(xpu_device)
        if isinstance(backend, ProcessGroupXCCL):
            backend.set_timeout(timeout)
            patched += 1
    if patched:
        logger.info(
            f"Applied train timeout {timeout} to {patched} xccl ProcessGroup(s) "
            "(upstream _set_pg_timeout has no xpu branch)."
        )


class FaultTolerantTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
        lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)

        # Batch-size ramp (analogous to LR warmup, but for global batch
        # size). Ramps the effective gradient-accumulation count -- and
        # thus effective GBS = LBS * dp_degree * GAS -- linearly from
        # `batch_ramp_start_gas` up to the full `gradient_accumulation_steps`
        # over the first `batch_ramp_steps` optimizer steps, then holds at
        # full GBS. LBS and DP degree are unchanged, so this needs no
        # dataloader/parallelism changes; loss stays correct because the
        # ezpz train_step normalizes by global_valid_tokens (token count),
        # not by microbatch count.
        #
        # Motivation: 80B NaNs at GBS>=372 (grad_norm NaNs at step 2) but
        # is stable at GBS=168. A ramp lets training stabilize at small GBS
        # before reaching the target GBS. `batch_ramp_steps=0` disables it
        # (default), preserving exact current behavior.
        batch_ramp_steps: int = 0
        """Number of steps to linearly ramp GAS from batch_ramp_start_gas to
        the full gradient_accumulation_steps. 0 disables the ramp."""
        batch_ramp_start_gas: int = 1
        """GAS to start the ramp from (effective GBS at step 0 =
        LBS * dp_degree * batch_ramp_start_gas). Must be >= 1 and <= the
        full gradient_accumulation_steps."""

        # Walltime-aware checkpointing. A short job can otherwise run its whole
        # PBS window and save NOTHING (e.g. 20B 512N at ~48s/step never reaches
        # the next interval=100 boundary inside a 2h window), wasting all the
        # compute. The train loop watches the clock and, once it is within
        # walltime_checkpoint_margin_seconds of the deadline, forces a final
        # checkpoint and stops cleanly -- guaranteeing a save before walltime
        # regardless of step rate, model size, or startup cost.
        #
        # TWO ways to set the deadline (deadline_epoch wins if both set):
        #
        # * walltime_deadline_epoch (PREFERRED) -- an ABSOLUTE Unix timestamp
        #   (job_start + PBS_walltime), exported once as $WALLTIME_DEADLINE_EPOCH
        #   by the failover submit scripts. Because it is absolute, it SURVIVES
        #   failover relaunches: a mid-job bad-node swap restarts the trainer but
        #   the deadline is unchanged, so the backstop still fires before the real
        #   PBS walltime. This fixes the 2026-06-27 failure where the relative
        #   budget reset on retry and the job was SIGTERM'd mid-save (see
        #   memory project_walltime_ckpt_resets_on_failover_retry).
        #
        # * walltime_seconds (FALLBACK) -- a RELATIVE budget in seconds, measured
        #   from loop entry. Resets on each relaunch, so it is unreliable across
        #   failover retries; kept only for backward compatibility / non-PBS use.
        #
        # Both 0/unset disables the feature (exact prior behavior).
        walltime_deadline_epoch: int = 0
        """Absolute Unix timestamp (job_start + walltime) past which a final
        checkpoint is forced. From $WALLTIME_DEADLINE_EPOCH. Survives failover
        retries. Takes precedence over walltime_seconds. 0 disables."""
        walltime_seconds: int = 0
        """RELATIVE walltime budget in seconds from loop entry (fallback when
        walltime_deadline_epoch is unset). Resets on failover relaunch -- prefer
        walltime_deadline_epoch. 0 disables."""
        walltime_checkpoint_margin_seconds: int = 600
        """Force a final checkpoint + stop once within this many seconds of the
        deadline. Must cover one checkpoint save + async flush at the target
        scale (20B/512N+ may want ~900; 600 is safe for <=2B/256N)."""

    ft_manager: FTManager

    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert config.model_spec is not None, (
            "model_spec must be set before creating Trainer"
        )
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes (FT override handles ft_manager creation)
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # FT addition: adjust dp info via ft_manager
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            training_steps=config.training.steps,
            global_batch_size=config.training.global_batch_size,
            parallel_dims=parallel_dims,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
        )
        self.model_config = model_config

        # logger.info(
        #     f"Building {model_spec.name} {model_spec.flavor} "
        #     f"with {json.dumps(model_config.to_dict(), indent=2, ensure_ascii=False)}"
        # )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # if ezpz.dist
        # if ezpz.distributed.asni
        if ezpz.distributed.verify_wandb():
            import wandb
            if wandb.run is not None:
                wandb.run.watch(model, log="all")

        # Quantization is now applied to the config at model_registry time
        # (#3127). The runtime model_converters layer is gone.

        # Verify all submodules satisfy the Module protocol
        model.verify_module_protocol()

        # Check if any quantization converter is on the model_config
        from torchtitan.components.quantization.utils import has_quantization as _has_quantization
        has_quantization = _has_quantization(model_config)

        # metrics logging (FT addition: ft_enable, ft_replica_id)
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
            has_quantization=has_quantization,
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        heading = 80 * "="
        logger.info(
            "\n".join([
                "\n",
                f"{heading}",
                f"{color.blue}Model: {model_spec.name} {model_spec.flavor} ",
                f"{color.red}config: {model_param_count:,} total parameters{color.reset}",
                f"{heading}",
                "\n",
            ])
        )
        # logger.info(
        #     "\n" + 80 * "="
        #     f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
        #     f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        #
        # )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        # Loss is now built from the JobConfig.loss field (upstream #2937 /
        # ChunkedLossWrapper). The FT integration no longer wraps the loss
        # function — FTOptimizersContainer below still passes ft_manager
        # for gradient sync.
        self.loss_fn = config.loss.build(compile_config=config.compile)

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # Batch-size ramp config validation (see Config docstrings).
        self.batch_ramp_steps = config.batch_ramp_steps
        self.batch_ramp_start_gas = config.batch_ramp_start_gas
        if self.batch_ramp_steps < 0:
            raise ValueError(
                f"batch_ramp_steps must be >= 0, got {self.batch_ramp_steps}"
            )
        if self.batch_ramp_steps > 0:
            if not 1 <= self.batch_ramp_start_gas <= self.gradient_accumulation_steps:
                raise ValueError(
                    "batch_ramp_start_gas must be in "
                    f"[1, {self.gradient_accumulation_steps}], got "
                    f"{self.batch_ramp_start_gas}"
                )
            logger.info(
                "Batch-size ramp ENABLED: GAS %d -> %d over %d steps "
                "(effective GBS %d -> %d)",
                self.batch_ramp_start_gas,
                self.gradient_accumulation_steps,
                self.batch_ramp_steps,
                self.batch_ramp_start_gas
                * config.training.local_batch_size
                * batch_degree,
                global_batch_size,
            )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            from torchtitan.components.metrics import ensure_pp_loss_visible

            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    cast(BaseModel, m).init_states(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model).init_states(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Replayed from upstream torchtitan/trainer.py (lines 391-411). Required
        # whenever loss=ChunkedLossWrapper.Config(...) — the loss object computes
        # logits from hidden states in chunks, so it needs a handle to lm_head
        # and signals the model to skip its own lm_head pass via _skip_lm_head.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(lm_head)
                    self.model_parts[-1]._skip_lm_head = True
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)
                self.model_parts[0]._skip_lm_head = True

        # FT addition: set all reduce hook
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        # FT addition: pass ft_manager for FTOptimizersContainer
        if isinstance(config.optimizer, FTOptimizersContainer.Config):
            self.optimizers = config.optimizer.build(
                model_parts=self.model_parts, ft_manager=self.ft_manager
            )
        else:
            self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # The post-optimizer model_converters hook is gone in #3127 —
        # quantization is applied to the config and runs as part of
        # forward, not via a runtime post-step hook.
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # Build checkpoint manager.
        # When fault tolerance is enabled and config.checkpoint uses
        # FTCheckpointManager.Config, ft_manager is passed through.
        # Otherwise the base CheckpointManager is used without it.
        ckpt_kwargs: dict = dict(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )
        # FTCheckpointManager accepts ft_manager; base CheckpointManager does not
        from torchtitan.experiments.torchft.checkpoint import (
            TorchFTCheckpointManager as FTCheckpointManager,
        )

        if isinstance(config.checkpoint, FTCheckpointManager.Config):
            ckpt_kwargs["ft_manager"] = self.ft_manager
        self.checkpointer = config.checkpoint.build(**ckpt_kwargs)

        # 57th sync: PR #3694 deleted the --disable_loss_parallel flag.
        # TP-on now always implies LP-on; the context no longer takes
        # enable_loss_parallel (loss-parallel autograd moved into
        # cross_entropy_loss).
        # 61st sync: the spmd_types series renamed get_train_context ->
        # get_spmd_context and added the spmd_typechecking kwarg. Mirror
        # upstream Trainer; we run spmd_backend=default so typechecking is
        # off (the kwarg is inert unless backend == "spmd_types").
        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                job_config=config,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
            ranks=global_ranks,
        )

        # On XPU, ProcessGroupXCCL inherits Backend::supportsSplitting() ==
        # false, but DeviceMesh._init_one_process_group still routes nested
        # mesh PG creation through ``split_group`` whenever
        # ``bound_device_id`` is set on the default group. That always blows
        # up with "No backend for the parent process group or its backend
        # does not support splitting", which kills every nested mesh
        # construction the EP sparse mesh requires. Steer the gate to the
        # ``new_group`` fallback for xccl until upstream lands the
        # supportsSplitting override + split implementation.
        from torchtitan.experiments.ezpz.xccl_split_group_workaround import (
            maybe_install_xccl_split_group_workaround,
        )

        maybe_install_xccl_split_group_workaround()

        # FT addition: build FTManager
        self.ft_manager = config.fault_tolerance.build()

        world_size = int(os.environ["WORLD_SIZE"])

        return ParallelDims.from_config(config.parallelism, world_size)

    def _effective_gas(self) -> int:
        """Gradient-accumulation steps for the current step under the ramp.

        Linearly interpolates GAS from ``batch_ramp_start_gas`` (at step 0)
        to the full ``gradient_accumulation_steps`` (at ``batch_ramp_steps``),
        holding at full after. Returns the full GAS when the ramp is
        disabled (``batch_ramp_steps == 0``). ``self.step`` is 0-indexed at
        the point train_step runs.
        """
        if self.batch_ramp_steps <= 0:
            return self.gradient_accumulation_steps
        if self.step >= self.batch_ramp_steps:
            return self.gradient_accumulation_steps
        start = self.batch_ramp_start_gas
        full = self.gradient_accumulation_steps
        # Linear interpolation; round to nearest int, clamp to [start, full].
        frac = self.step / self.batch_ramp_steps
        gas = round(start + (full - start) * frac)
        return max(start, min(full, gas))

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Effective grad-accum count for this step (batch-size ramp).
        # Equals self.gradient_accumulation_steps unless the ramp is on.
        gas = self._effective_gas()

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(gas):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            # Upstream PR #3586 (2026-06-09) retyped global_valid_tokens
            # as `float | None` and switched the no-DP branch to
            # `float(local_valid_tokens.item())`. Mirror that here so the
            # annotation contract holds. DP branch keeps returning a
            # tensor from dist_sum — upstream itself does the same; the
            # consumer (BaseLoss.__call__) accepts either at runtime.
            global_valid_tokens = float(local_valid_tokens.item())

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return float(loss.detach().item())

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            # FT addition: use ft_manager.loss_sync_pg for extra process group
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

        if isinstance(global_avg_loss, torch.Tensor):
            return float(global_avg_loss.item())
        return float(global_avg_loss)

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        # FT addition: per-replica profiling leaf folder
        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            config.profiler.build(
                global_step=self.step,
                base_folder=config.dump_folder,
                leaf_folder=leaf_folder,
            ) as profiler,
            # FT addition: maybe_semi_sync_training context manager
            maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    len(self.model_config.layers)
                    if hasattr(self.model_config, "layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    config.model_spec.fragment_fn
                    if hasattr(config.model_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            # Walltime-aware checkpointing: resolve an ABSOLUTE wall-clock
            # deadline (time.time() epoch) so the backstop survives failover
            # relaunches. Prefer walltime_deadline_epoch (set once per job);
            # fall back to the relative walltime_seconds measured from now.
            wall_margin = config.walltime_checkpoint_margin_seconds
            if config.walltime_deadline_epoch > 0:
                wall_deadline = float(config.walltime_deadline_epoch)
                logger.info(
                    f"walltime-ckpt: absolute deadline {wall_deadline:.0f} "
                    f"(~{(wall_deadline - time.time()) / 60:.0f} min from now), "
                    f"margin {wall_margin}s"
                )
            elif config.walltime_seconds > 0:
                wall_deadline = time.time() + config.walltime_seconds
                logger.info(
                    f"walltime-ckpt: relative budget {config.walltime_seconds}s "
                    f"from loop entry (NOTE: resets on failover retry -- prefer "
                    f"walltime_deadline_epoch), margin {wall_margin}s"
                )
            else:
                wall_deadline = None

            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                saved_this_step = self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # Walltime guard: once within margin of the (absolute) deadline,
                # force a final checkpoint now and stop cleanly. Reuses the
                # interval-bypassing last_step path, then flushes any async save
                # so it actually lands on disk before the process exits. Without
                # this, a short job can run its whole window and save nothing.
                if wall_deadline is not None:
                    remaining = wall_deadline - time.time()
                    if remaining <= wall_margin:
                        logger.info(
                            f"walltime deadline near at step {self.step} "
                            f"({remaining:.0f}s left <= margin {wall_margin}s); "
                            "forcing final checkpoint and stopping"
                        )
                        if not saved_this_step:
                            self.checkpointer.save(self.step, last_step=True)
                        # Block until any async save is fully on disk before we
                        # break to teardown (close() does NOT wait for pending
                        # saves).
                        self.checkpointer.maybe_wait_for_saving()
                        break

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    _set_pg_timeouts_xpu_aware(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")
