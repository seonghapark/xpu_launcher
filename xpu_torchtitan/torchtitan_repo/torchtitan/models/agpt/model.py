# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""agpt model class.

agpt reuses Llama3's architecture verbatim but needs a sharding hook that
knows about the optional QK-Norm sub-module. We subclass Llama3Model to
override the sharding-config setter; everything else (forward, init,
weight-tying, etc.) is inherited.
"""

from dataclasses import dataclass

from torchtitan.models.llama3.model import Llama3Model


class AgptModel(Llama3Model):
    """Llama3 with agpt's QK-Norm-aware sharding setter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            # Run llama3's validation + rope sync first. It calls
            # set_llama3_sharding_config at the end, which we then
            # idempotently overwrite to also fill in QK-Norm sharding.
            #
            # Explicit super(AgptModel.Config, self) is required because
            # bare super() in a slots=True nested-class dataclass can't
            # resolve the enclosing class name correctly.
            super(AgptModel.Config, self).update_from_config(
                config=config, **kwargs
            )

            # Derive the YaRN context-extension ratio from the one authoritative
            # length: config.training.seq_len. This hook is the correct place
            # because it runs AFTER tyro has applied --training.seq_len and
            # BEFORE the model (and its immutable RoPE cache) is built, so the
            # value here is always the one the user actually asked for.
            #
            # Decoder.Config.update_from_config, called via super() above, has
            # just set every layer's rope.max_seq_len to training.seq_len. The
            # flavor's own max_seq_len (262144) and the hard-coded rope_factor
            # (32.0) are therefore both stale by this point: keeping factor=32
            # while training at 16K would apply the full 256K interpolation,
            # and an mscale of 1 + 0.1*ln(32) = 1.347, for an extension that is
            # never exercised -- needlessly perturbing attention magnitudes away
            # from the loaded checkpoint.
            from torchtitan.trainer import Trainer

            if isinstance(config, Trainer.Config):
                self._sync_yarn_rope_factor(config.training.seq_len)

            from torchtitan.models.agpt.sharding import (
                set_agpt_sharding_config,
            )

            parallelism = config.parallelism
            set_agpt_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def _sync_yarn_rope_factor(self, seq_len: int) -> None:
            """Recompute each YaRN layer's rope_factor from ``seq_len``.

            ``seq_len`` is ``config.training.seq_len`` -- the only length input
            this takes. It is passed in explicitly rather than read back off
            ``rope.max_seq_len`` (which the base class has just set to the same
            value) so the single source of truth is visible at the callsite.

            ``rope_factor`` is the context-extension ratio target/original, so
            it must track the length actually being trained rather than stay
            pinned to whatever the flavor declared.

            ``original_seq_len`` is the model's pretraining length -- a property
            of the checkpoint, not of this run -- so it stays as declared. It is
            the denominator of the ratio, never an alternative target.

            Training at or below the pretraining length gives factor 1.0, which
            both backends treat as "no YaRN scaling" and which makes mscale 1.0,
            leaving the loaded weights' attention magnitudes untouched.
            """
            for layer in self.layers:
                attention = getattr(layer, "attention", None)
                rope = getattr(attention, "rope", None) if attention else None
                if rope is None or rope.scaling != "yarn":
                    continue
                rope.rope_factor = max(1.0, seq_len / rope.original_seq_len)
