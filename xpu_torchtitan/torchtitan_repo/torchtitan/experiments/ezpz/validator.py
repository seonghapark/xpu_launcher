# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Subclass of upstream Validator with two ezpz-side adjustments.

1. Caches the validation dataloader on the instance instead of rebuilding
   it on every `validate()` call. Upstream rebuilds per call, which is
   cheap for the default `c4_validation` HF stream but re-runs
   `build_gpt_datasets()` for our blendcorpus-backed loader.
2. Captures `job_config` in `__init__` and forwards `training_steps` +
   `global_batch_size` to the validation dataloader, so the blendcorpus
   loader gets the same sample budget the trainer used.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.tools import utils


class EzpzValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        pass

    def __init__(self, *args, **kwargs):
        # Capture job_config before delegating so we can pass training_steps
        # and global_batch_size through to the validator dataloader.
        # Upstream Validator.__init__ swallows job_config via **kwargs and
        # never stores it; without these the BlendCorpus build path falls
        # back to train_iters=1 / global_batch_size=local*dp (warning
        # logged on every validate() call) and the bc_set_config global
        # state ends up with the wrong sample budget for the rest of the
        # process — a footgun for any later resume-from-checkpoint
        # rebuild of the train dataloader.
        self._job_config = kwargs.get("job_config")
        super().__init__(*args, **kwargs)

    def _get_validation_dataloader(self):
        """Cache the validation dataloader on the instance.

        Upstream `Validator.validate()` rebuilds it on every call. For the
        default `c4_validation` HF stream that's cheap, but for our
        blendcorpus-backed loader it re-runs `build_gpt_datasets()` every
        validation pass (cached index file load + dataset wrapper
        construction + a no-op `bc_mpu` re-init guarded by our patch),
        which adds visible noise to the log and ~few hundred ms of
        wasted work per call.

        `BlendCorpusDataLoader.__iter__` yields a fresh iterator each
        call by re-instantiating the underlying torch DataLoader, so it
        is safe to cache the wrapper and re-iterate it per validation.
        """
        if getattr(self, "_cached_dataloader", None) is None:
            # Pass training_steps + global_batch_size through so the
            # blendcorpus loader gets the same sample budget the trainer
            # used. Without these, the train_iters defaults to 1 and the
            # global_batch_size collapses to local_batch_size * dp_world,
            # producing a misleading "Global batch size: <small>" log line
            # and (more dangerously) overwriting the bc_set_config global
            # state with the wrong values.
            extra: dict = {}
            if self._job_config is not None:
                training_cfg = getattr(self._job_config, "training", None)
                if training_cfg is not None:
                    if getattr(training_cfg, "steps", None):
                        extra["training_steps"] = training_cfg.steps
                    gbs = getattr(training_cfg, "global_batch_size", None)
                    if gbs and gbs > 0:
                        extra["global_batch_size"] = gbs
            self._cached_dataloader = self.dl_config.build(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                local_batch_size=self.local_batch_size,
                parallel_dims=self.parallel_dims,
                **extra,
            )
        return self._cached_dataloader

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        validation_dataloader = self._get_validation_dataloader()

        for input_dict, labels in validation_dataloader:
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            self.metrics_processor.ntokens_since_last_log += labels.numel()
            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            # post_dataloading_process returns a 3-tuple (inputs, labels,
            # extra_kwargs); an older upstream signature also returned a
            # separate `extra_inputs`, which has since been folded into
            # extra_kwargs. Match the current upstream contract (see
            # torchtitan/components/validate.py).
            inputs, labels, extra_kwargs = self.post_dataloading_process(
                input_dict, labels, model_parts
            )

            local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=device_type)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()

            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh, None
                )
            else:
                # Upstream PR #3586 (2026-06-09) retyped global_valid_tokens
                # as `float | None`; mirror that in the no-DP branch. See
                # the matching note in ezpz/trainer.py.
                global_valid_tokens = float(local_valid_tokens.item())

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None
                with self.validation_context():
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )

                if self.pp_has_last_stage:
                    assert losses is not None
                    loss_sum = torch.sum(torch.stack(losses)).to(device_type)
                else:
                    loss_sum = torch.tensor([-1.0], device=device_type)
            else:
                with self.validation_context():
                    assert len(model_parts) == 1
                    predictions = model_parts[0](inputs, **extra_kwargs)
                    # loss_fn (BaseLoss.__call__) returns (loss, metrics_dict)
                    # -- upstream validate.py unpacks the same way. Without the
                    # unpack, loss_sum is a tuple and loss_sum.detach() below
                    # raises AttributeError on the first validation batch.
                    loss_sum, _ = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss_sum.detach() / global_valid_tokens)
            num_steps += 1

        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_sum(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = float(loss.item())

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        for model in model_parts:
            model.train()
