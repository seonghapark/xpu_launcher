# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import ezpz

import torch


logger = logging.getLogger()


def _detect_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())

    for key in (
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "MV2_COMM_WORLD_RANK",
    ):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue

    try:
        return int(ezpz.get_rank())
    except Exception:
        return 0


def reset_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


def init_logger() -> None:
    reset_logger(logger)

    rank = _detect_rank()
    level = logging.INFO if rank == 0 else logging.CRITICAL

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)

    if rank == 0:
        logging.disable(logging.NOTSET)
    else:
        # Keep CRITICAL logs from non-zero ranks and suppress everything else.
        logging.disable(logging.CRITICAL - 1)
        # Also gag the Python `warnings` module on non-zero ranks. The
        # `logging.disable` above only covers the `logging` package;
        # things like torch.autocast's UserWarning ("XPU autocast only
        # supports bf16/fp16") and the inductor "complex operators"
        # warning go through `warnings.warn`, which isn't routed
        # through our logger. Without this they fan out N_rank-fold.
        import warnings

        warnings.filterwarnings("ignore")

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
