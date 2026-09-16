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

            from torchtitan.experiments.ezpz.agpt.sharding import (
                set_agpt_sharding_config,
            )

            parallelism = config.parallelism
            set_agpt_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )
