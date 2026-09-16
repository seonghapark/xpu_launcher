# PR 1 (xccl Python dispatch fix) — empirical verification

## TL;DR

The proposed diff in
[`PLAN_xccl_timeout_upstream_pr.md`](PLAN_xccl_timeout_upstream_pr.md)
PR 1 works as intended on Intel XPU + xccl. Verified empirically on
**Sunspot**, 2026-05-21, **torch 2.13.0.dev20260519+xpu** from the
repo `.venv/`, 12 ranks on a single node, alloc 12467214 (own job).

| Pass         | warning fired | dispatch reached `ProcessGroupXCCL.set_timeout` | `options._timeout` after | Verdict |
|--------------|:-------------:|:------------------------------------------------:|:-------------------------:|:-------:|
| **unpatched** | ✅ yes (`Set timeout is now only supported for either nccl or gloo.`) | ❌ never called           | unchanged (`0:02:00`)     | UNPATCHED, contract holds |
| **patched**   | ❌ no          | ✅ called with `timedelta(seconds=37)`            | updated to `0:00:37`      | PATCHED, contract holds   |

Both passes were exit-0 across all 12 ranks. Logs in
`logs/pr1-verify/`.

## Method

1. **Verifier**:
   [`verify_pr1_dispatch.py`](verify_pr1_dispatch.py). Calls
   `torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=37), pg)`
   on the default xccl PG and reports four signals:
   - `pre_options_timeout` / `post_options_timeout` (read off
     `backend.options._timeout`)
   - whether the `"Set timeout is now only supported..."` warning fires
   - whether `ProcessGroupXCCL.set_timeout` was actually called
     (detected via a monkey-patched class-method spy)
   - the verdict (UNPATCHED vs PATCHED) and pass/fail vs that contract.

2. **PR 1 diff** applied to the in-repo `.venv` copy of
   `torch/distributed/distributed_c10d.py` (backup taken first):

   ```diff
   @@ around the existing cuda branch in _set_pg_timeout @@
        if torch.device("cuda") in devices:
            ...
   +    # PR 1 (xccl Python dispatch fix): route xpu PGs to ProcessGroupXCCL
   +    # via the public Backend API method (set_timeout), since XCCL does not
   +    # expose _set_default_timeout the way NCCL does.
   +    xccl_backends = set()
   +    if torch.device("xpu") in devices and is_xccl_available():
   +        backend = group._get_backend(torch.device("xpu"))
   +        if isinstance(backend, ProcessGroupXCCL):
   +            xccl_backends.add(backend)
   -    if len(backends) == 0:
   +    if len(backends) == 0 and len(xccl_backends) == 0:
            warnings.warn(
                "Set timeout is now only supported for either nccl or gloo.", stacklevel=2
            )
        for backend in backends:
            backend._set_default_timeout(timeout)
   +    for backend in xccl_backends:
   +        backend.set_timeout(timeout)
   ```

   File restored to its original state via Edit afterwards (verified
   `diff -q` clean).

3. **Launcher**:
   ```bash
   source $HOME/.ezpz/utils.sh && ezpz_setup_job && ezpz_setup_xpu \
     && source .venv/bin/activate \
     && NRANKS=2 ezpz launch --filter=2 python3 \
       torchtitan/experiments/ezpz/docs/upstream-issues/verify_pr1_dispatch.py
   ```
   (`--filter=2` keeps Sunspot from launching all 12 tiles per node
   when we only want 2 — verifier still runs the dispatch on every
   rank that launches.)

## Bonus finding: xccl C++ DOES update `options._timeout`

The previous understanding in
[`PLAN_xccl_timeout_upstream_pr.md`](PLAN_xccl_timeout_upstream_pr.md)
PR 2 was that "xccl stores the value but does not enforce it." The
"stores the value" half is now **observable from Python**: every rank
in the patched run shows `options._timeout` flip from `0:02:00` (the
ezpz `setup_torch` default) to `0:00:37` after the dispatch. So at
minimum the value reaches xccl's `Options` storage.

That does **not** mean enforcement works — the standing repro
(`repro_xccl_timeout_abort.py`) is the canonical demonstration that
the deadline does not actually fire an abort. PR 2's scope is
unchanged: the watchdog / `ccl_abort()` chain still needs to be wired
in C++.

## Implications for PR 1 submission

- Diff is **safe** (no behavior change for non-xpu users; xccl users
  get the warning suppressed and the value plumbed through).
- Diff is **necessary even if PR 2 never lands** — without it, the
  `ezpz` xpu-aware workaround in `22847fcb3`
  (`_set_pg_timeouts_xpu_aware`) is the only way to push the value
  into the backend, and that workaround lives in every torchtitan
  branch we have to keep rebased.
- Test scaffold for the upstream PR: port
  [`verify_pr1_dispatch.py`](verify_pr1_dispatch.py) into
  `test/distributed/test_c10d_xccl.py`. The two assertions are
  exactly the two signals the spy collects: warning not raised, and
  `set_timeout` reached on the xccl backend.

## Artifacts

- Verifier: [`verify_pr1_dispatch.py`](verify_pr1_dispatch.py)
- Logs:
  - `logs/pr1-verify/baseline-unpatched-20260521-112520.log` (initial
    baseline with `pre/post == None` because pyi-only `timeout`
    property doesn't exist — kept for the bare warning evidence)
  - `logs/pr1-verify/patched-20260521-112811.log` (decisive
    patched-state evidence: warning suppressed, dispatch reached
    backend, `options._timeout` flipped)
  - `logs/pr1-verify/baseline-v2-20260521-112922.log` (re-baseline
    post-restore with the instrumented verifier — confirms
    unpatched contract too)
- Job alloc: 12467214 (1N Sunspot workq, released after run).
