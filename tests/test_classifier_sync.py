from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import launch  # noqa: E402


def test_shell_success_with_inner_nonzero_is_failure() -> None:
    log_text = "Execution finished with 1.\n"
    result = launch._classify_attempt(
        0,
        log_text,
        [],
        profile="generic",
        prior_attempt_had_progress=None,
        has_spares=True,
        consecutive_unattributed=0,
    )
    assert result.reason is launch.TerminationReason.BAD_NODE_BLIND


def test_walltime_without_crash_is_walltime() -> None:
    log_text = "rank 7 died from signal 15\njob wall clock limit\n"
    result = launch._classify_attempt(
        143,
        log_text,
        [],
        profile="generic",
        prior_attempt_had_progress=None,
        has_spares=True,
        consecutive_unattributed=0,
    )
    assert result.reason is launch.TerminationReason.WALLTIME


def test_walltime_with_crash_signature_is_failover() -> None:
    log_text = "srun: error: nid001321: tasks 4-7: Killed\n"
    result = launch._classify_attempt(
        143,
        log_text,
        ["nid001321"],
        profile="slurm",
        prior_attempt_had_progress=None,
        has_spares=True,
        consecutive_unattributed=0,
    )
    assert result.reason is launch.TerminationReason.BAD_NODE_KNOWN


def test_retryable_unattributed_under_budget() -> None:
    log_text = "OSError: [Errno 28] No space left on device\n"
    result = launch._classify_attempt(
        1,
        log_text,
        ["hostA"],
        profile="aurora",
        prior_attempt_had_progress=None,
        has_spares=True,
        consecutive_unattributed=0,
    )
    assert result.reason is launch.TerminationReason.RETRYABLE_UNATTRIBUTED


def test_stuck_pre_training_two_attempts_without_progress() -> None:
    log_text = "RuntimeError: misc failure without progress marker\n"
    result = launch._classify_attempt(
        1,
        log_text,
        [],
        profile="generic",
        prior_attempt_had_progress=False,
        has_spares=True,
        consecutive_unattributed=3,
    )
    assert result.reason is launch.TerminationReason.STUCK_PRE_TRAINING
