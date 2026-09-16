# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest port of ``verify_pr1_dispatch.py``.

Exercises the PR 1 contract for ``_set_pg_timeout`` on an xccl
ProcessGroup:

- **Unpatched torch**: ``_set_pg_timeout`` on an xccl PG emits the
  ``"Set timeout is now only supported for either nccl or gloo."``
  warning and never reaches ``ProcessGroupXCCL.set_timeout``.
- **Patched torch (PR 1 applied)**: the warning is suppressed and the
  call routes through ``ProcessGroupXCCL.set_timeout``, which in turn
  mutates ``backend.options._timeout``.

The body of this test is structured to port cleanly into upstream
``test/distributed/test_c10d_xccl.py`` — only the harness around it
(``unittest.TestCase`` + ``MultiProcessTestCase``) needs to change.
"""

from __future__ import annotations

import os
import socket
import unittest
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist


TARGET_WARNING_SUBSTR = "Set timeout is now only supported"
INITIAL_TIMEOUT = timedelta(minutes=2)
NEW_TIMEOUT = timedelta(seconds=37)


def _xpu_xccl_available() -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    if not hasattr(dist, "is_xccl_available"):
        return False
    try:
        return dist.is_xccl_available()
    except Exception:
        return False


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


@unittest.skipUnless(
    _xpu_xccl_available(),
    "requires Intel XPU + xccl distributed backend",
)
class TestSetPgTimeoutXccl(unittest.TestCase):
    """PR 1 dispatch contract for ``_set_pg_timeout`` on xccl PGs."""

    def setUp(self) -> None:
        # Single-rank xccl group is sufficient: the dispatch path being
        # exercised is purely Python and per-rank.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        torch.xpu.set_device(0)
        dist.init_process_group(
            backend="xccl",
            init_method="env://",
            timeout=INITIAL_TIMEOUT,
        )

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _invoke_with_spy(self):
        """Invoke ``_set_pg_timeout`` and return observable side effects.

        Returns
        -------
        tuple of (warning_fired, dispatch_reached_backend,
                  pre_options_timeout, post_options_timeout, backend)
        """
        from torch.distributed.distributed_c10d import _set_pg_timeout

        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device("xpu"))

        calls: list[timedelta] = []
        original = type(backend).set_timeout

        def _spy(self_, timeout):  # type: ignore[no-untyped-def]
            calls.append(timeout)
            return original(self_, timeout)

        type(backend).set_timeout = _spy
        try:
            pre = getattr(getattr(backend, "options", None), "_timeout", None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _set_pg_timeout(NEW_TIMEOUT, pg)
            post = getattr(getattr(backend, "options", None), "_timeout", None)
        finally:
            type(backend).set_timeout = original

        warning_fired = any(
            TARGET_WARNING_SUBSTR in str(w.message) for w in caught
        )
        dispatch_reached = any(t == NEW_TIMEOUT for t in calls)
        return warning_fired, dispatch_reached, pre, post, backend

    def test_dispatch_reaches_xccl_backend(self) -> None:
        """PR 1's positive contract: when patched, dispatch reaches xccl."""
        warning_fired, dispatch_reached, pre, post, backend = (
            self._invoke_with_spy()
        )
        # Either pass is observable — let the asserts say which patch
        # state we're actually in, with a helpful failure message.
        if warning_fired:
            self.skipTest(
                "torch is unpatched (warning fired): "
                f"backend={type(backend).__name__} "
                f"pre={pre} post={post} dispatch_reached={dispatch_reached}. "
                "Apply PR 1 diff to torch/distributed/distributed_c10d.py "
                "to exercise this test."
            )
        self.assertTrue(
            dispatch_reached,
            f"PR 1 applied (no warning) but ProcessGroupXCCL.set_timeout "
            f"was never called. backend={type(backend).__name__}",
        )
        self.assertEqual(
            post,
            NEW_TIMEOUT,
            f"backend.options._timeout did not update: pre={pre} post={post}",
        )

    def test_unpatched_contract(self) -> None:
        """PR 1's negative contract: without it, dispatch never reaches xccl.

        This is the regression guard — if upstream ever changes
        ``_set_pg_timeout`` to silently no-op without the warning, we
        want to catch that too.
        """
        warning_fired, dispatch_reached, _pre, _post, backend = (
            self._invoke_with_spy()
        )
        if not warning_fired:
            self.skipTest(
                "torch already includes PR 1 (no warning fired); "
                "this negative test is N/A."
            )
        self.assertFalse(
            dispatch_reached,
            f"Warning fired but ProcessGroupXCCL.set_timeout was still "
            f"called — unexpected. backend={type(backend).__name__}",
        )


if __name__ == "__main__":
    unittest.main()
