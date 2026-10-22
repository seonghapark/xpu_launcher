"""Per-rank Inductor cache isolation, plus the (unchanged) os._exit teardown shim.

WHY THIS FILE EXISTS
    Job 8929205 (rank39) and job 8929490 (rank7) both died in the very first
    Inductor compile with

        InductorError: ValueError: @jit functions should be defined in a Python file

    taking all 48 ranks down with them (PALS: "rank 0 died from signal 15").
    Two of five 48-rank attempts have been lost this way -- ~40%.

    The leading hypothesis is a read-after-write race in PyCodeCache: all 48
    ranks share ONE TORCHINDUCTOR_CACHE_DIR on Lustre and race to populate the
    same cache keys, so a rank can exec() a generated module whose bytes are
    not yet visible to it, leaving inspect.getsource() unable to recover the
    source for @triton.jit. The offending .py was complete and readable after
    the fact, which is what a visibility race looks like.

    This shim gives every rank its OWN cache directory, removing the 48-way
    race. It is a MITIGATION OF AN UNPROVEN HYPOTHESIS, not a fix: if the real
    cause is a torch/triton source-resolution bug, isolation will not help and
    the trials will show that.

WHY sitecustomize AND NOT A WRAPPER SCRIPT
    Job 8924922 tried to influence ranks by pointing TRAIN_PYTHON_BIN at a bash
    shim. Every rank then died in Triton compile with the SAME @jit error,
    because Triton resolves a @jit function's source through the interpreter
    and the shim displaced it. A PYTHONPATH sitecustomize displaces nothing --
    it runs inside the real interpreter, before user code.

WHY IT IS SAFE TO SET THE DIR HERE
    torch._inductor.runtime.cache_dir_utils.cache_dir() reads
    TORCHINDUCTOR_CACHE_DIR lazily, at first use, and memoizes it into the
    environment (line 15-21 of that file in frameworks/2026.1.0). sitecustomize
    runs before `import torch`, so rewriting the variable now is seen by every
    consumer. TRITON_CACHE_DIR is deliberately left UNSET: triton_cache_dir()
    derives from cache_dir(), so Triton follows automatically. Setting it
    explicitly would be a second place to get wrong.

COST, WHICH IS THE POINT OF MEASURING
    48 ranks compiling into 48 directories means 48x the compile work and 48x
    the cache bytes. That is the trade this experiment has to price, not
    assume. The job measures both.

GATES
    AGPT_INDUCTOR_ISOLATE=1  -> per-rank cache dir. Unset/0 -> untouched
                                (the shared-directory control).
    AGPT_OSEXIT_EXPERIMENT=1 -> the teardown shim, loaded from osexit_exp/
                                so that file stays the single source of truth.
    AGPT_CORE_LIMIT_BYTES=N  -> cap this rank's core-dump size to N bytes.
                                Unset -> untouched.
"""
import os
import sys


# --- core-dump size cap -----------------------------------------------------
# MUST be done in-process, not with `ulimit -c` in the job script: PALS does
# not propagate rlimits to ranks by default (hence its --rlimits flag), so a
# shell-level limit stays in the job shell while ranks dump at `unlimited`.
#
# Why cap at all: with the os._exit shim OFF, every rank reaches the
# sycl.descriptor_unreg SIGSEGV and dumps core. Job 8924552 wrote 47 cores
# averaging 11.3 GB (529.8 GB total), so a 48-rank trial would write ~542 GB.
# That is not only a quota problem -- the write lands INSIDE the teardown
# window, inflating the shutdown_s measurement that the experiment exists to
# take. A small cap keeps the dump as a countable EVENT without letting its
# cost contaminate the metric.
_CORE_LIMIT = os.environ.get("AGPT_CORE_LIMIT_BYTES")
if _CORE_LIMIT and _CORE_LIMIT.strip().isdigit():
    try:
        import resource

        _n = int(_CORE_LIMIT)
        _soft, _hard = resource.getrlimit(resource.RLIMIT_CORE)
        # Never raise the hard limit; only lower the soft one.
        if _hard != resource.RLIM_INFINITY:
            _n = min(_n, _hard)
        resource.setrlimit(resource.RLIMIT_CORE, (_n, _hard))
        sys.stderr.write(
            "INDUCTOR_ISO rank=%s core limit -> %d bytes\n"
            % (os.environ.get("PALS_RANKID", "?"), _n)
        )
        sys.stderr.flush()
    except Exception as _exc:  # never let this stop the run
        try:
            sys.stderr.write("INDUCTOR_ISO core limit FAILED: %r\n" % (_exc,))
            sys.stderr.flush()
        except Exception:
            pass


def _rank():
    for var in ("PALS_RANKID", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(var)
        if val is not None and val.strip().isdigit():
            return val.strip()
    return None


def _say(msg):
    try:
        sys.stderr.write("INDUCTOR_ISO rank=%s %s\n" % (_rank() or "?", msg))
        sys.stderr.flush()
    except Exception:
        pass


# --- per-rank Inductor cache ------------------------------------------------
if os.environ.get("AGPT_INDUCTOR_ISOLATE") == "1":
    _base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    _r = _rank()
    if not _base:
        # No shared dir to derive from. Do NOT invent one -- torch's default is
        # per-node /tmp, which is already unshared, and silently relocating it
        # would make the measurement meaningless.
        _say("TORCHINDUCTOR_CACHE_DIR unset; leaving torch's default alone")
    elif _r is None:
        # Without a rank we cannot isolate. Falling back to the shared dir is
        # the honest move: it keeps the run correct and makes the control/
        # treatment distinction visible in the log rather than silently absent.
        _say("NO RANK ENV (PALS_RANKID/PMI_RANK/RANK missing); staying shared at %s" % _base)
    else:
        _iso = os.path.join(_base, "rank_%s" % _r)
        try:
            os.makedirs(_iso, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = _iso
            _say("isolated cache -> %s" % _iso)
        except Exception as exc:
            # A failed mkdir must not kill the rank; the shared dir still works,
            # it is just the configuration we were trying to avoid.
            _say("mkdir failed (%r); staying shared at %s" % (exc, _base))
else:
    _say("isolation OFF (control: shared %s)"
         % os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<torch default>"))


# --- reuse the existing teardown shim, unmodified ---------------------------
# Only one sitecustomize can win on PYTHONPATH, so load the osexit module by
# path rather than copying its logic here. If it ever changes, this picks the
# change up; there is no second copy to drift.
_OSEXIT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "osexit_exp", "sitecustomize.py")
_OSEXIT = os.path.normpath(_OSEXIT)
if os.path.isfile(_OSEXIT):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_agpt_osexit_shim", _OSEXIT)
        _mod = _ilu.module_from_spec(_spec)
        # Keep a reference: the module registers atexit handlers that close over
        # its globals, and a garbage-collected module clears those to None.
        sys.modules["_agpt_osexit_shim"] = _mod
        _spec.loader.exec_module(_mod)
    except Exception as exc:
        _say("could not load osexit shim from %s: %r" % (_OSEXIT, exc))
else:
    _say("osexit shim not found at %s" % _OSEXIT)
