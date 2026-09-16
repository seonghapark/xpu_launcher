# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import os
from collections import Counter
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.ops.scatter_add import deterministic_scatter_add


@torch.library.custom_op(
    "torchtitan_ezpz::deterministic_scatter_add_1d", mutates_args=()
)
def deterministic_scatter_add_1d(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    index_2d = index.reshape(-1, 1).expand(-1, src.shape[-1])
    return deterministic_scatter_add(out, index_2d, src)


def deterministic_scatter_add_(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return out.scatter_add_(dim=0, index=index, src=src)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


@deterministic_scatter_add_1d.register_fake
def _(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _backward_scatter_add_1d(ctx, grad_output: torch.Tensor):
    (index,) = ctx.saved_tensors
    index_2d = index.reshape(-1, 1).expand(-1, grad_output.shape[-1])
    grad_src = torch.gather(grad_output, dim=0, index=index_2d)
    return grad_output, None, grad_src


def _setup_scatter_add_1d_context(ctx, inputs, output) -> None:
    _out, index, _src = inputs
    ctx.save_for_backward(index)


deterministic_scatter_add_1d.register_autograd(
    _backward_scatter_add_1d,
    setup_context=_setup_scatter_add_1d_context,
)


_MOE_FASTPATH_COUNTERS: Counter[str] = Counter()
_MOE_FASTPATH_ATEXIT_REGISTERED = False


def _moe_fastpath_debug_enabled() -> bool:
    return os.environ.get("TT_MOE_DEBUG_FASTPATHS", "").lower() in {
        "1",
        "true",
        "yes",
        "rank0",
        "all",
    }


def _rank_for_fastpath_debug() -> int:
    for name in ("RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return 0


def _print_moe_fastpath_counters() -> None:
    if not _moe_fastpath_debug_enabled():
        return
    mode = os.environ.get("TT_MOE_DEBUG_FASTPATHS", "").lower()
    rank = _rank_for_fastpath_debug()
    if mode != "all" and rank != 0:
        return
    if not _MOE_FASTPATH_COUNTERS:
        print(f"[rank{rank}] TT_MOE_FASTPATH_COUNTERS empty", flush=True)
        return
    counts = " ".join(
        f"{name}={count}" for name, count in sorted(_MOE_FASTPATH_COUNTERS.items())
    )
    print(f"[rank{rank}] TT_MOE_FASTPATH_COUNTERS {counts}", flush=True)


def _record_moe_fastpath(name: str, count: int = 1) -> None:
    global _MOE_FASTPATH_ATEXIT_REGISTERED
    if not _moe_fastpath_debug_enabled() or torch.compiler.is_compiling():
        return
    if not _MOE_FASTPATH_ATEXIT_REGISTERED:
        atexit.register(_print_moe_fastpath_counters)
        _MOE_FASTPATH_ATEXIT_REGISTERED = True
    _MOE_FASTPATH_COUNTERS[name] += count


def _record_moe_fastpath_max(name: str, value: int) -> None:
    if not _moe_fastpath_debug_enabled() or torch.compiler.is_compiling():
        return
    if name not in _MOE_FASTPATH_COUNTERS or value > _MOE_FASTPATH_COUNTERS[name]:
        _record_moe_fastpath(name, value - _MOE_FASTPATH_COUNTERS.get(name, 0))


def _record_moe_fastpath_min(name: str, value: int) -> None:
    if not _moe_fastpath_debug_enabled() or torch.compiler.is_compiling():
        return
    if name not in _MOE_FASTPATH_COUNTERS or value < _MOE_FASTPATH_COUNTERS[name]:
        _MOE_FASTPATH_COUNTERS[name] = value


def _record_moe_fastpath_stat(name: str, value: int) -> None:
    _record_moe_fastpath(f"{name}_count")
    _record_moe_fastpath(f"{name}_sum", value)
    _record_moe_fastpath_max(f"{name}_max", value)
    _record_moe_fastpath_min(f"{name}_min", value)


def _normal_equal_a2a_padding_policy() -> str:
    value = os.environ.get("TT_MOE_NORMAL_EQUAL_A2A_PADDING", "").lower()
    if value in {"1", "true", "yes", "force"}:
        return "force"
    if value == "adaptive":
        return "adaptive"
    return "off"


def _normal_equal_a2a_padding_enabled() -> bool:
    return _normal_equal_a2a_padding_policy() in {"force", "adaptive"}


def _normal_equal_a2a_padding_threshold() -> float:
    value = os.environ.get("TT_MOE_NORMAL_EQUAL_A2A_PADDING_THRESHOLD", "0.10")
    try:
        threshold = float(value)
    except ValueError:
        threshold = 0.10
    return max(threshold, 0.0)


def _normal_equal_a2a_padding_telemetry_enabled() -> bool:
    return os.environ.get("TT_MOE_NORMAL_EQUAL_A2A_PADDING_TELEMETRY", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _equal_a2a_padding_overhead_ratio(
    *,
    input_splits: list[int],
    output_splits: list[int],
    equal_split_size: int,
) -> tuple[int, float]:
    real_tokens = sum(input_splits) + sum(output_splits)
    padded_tokens = sum(equal_split_size - split for split in input_splits) + sum(
        equal_split_size - split for split in output_splits
    )
    if real_tokens <= 0:
        return padded_tokens, float("inf")
    return padded_tokens, padded_tokens / real_tokens


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _scatter_add_1d_forward_or_autograd(
    out: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    protected_input: torch.Tensor,
) -> torch.Tensor:
    if torch.is_grad_enabled() or _shares_storage(out, protected_input):
        _record_moe_fastpath("scatter_add_1d_autograd_or_alias")
        return deterministic_scatter_add_1d(out, index, src)
    _record_moe_fastpath("scatter_add_inplace_no_grad")
    scatter_index = index.reshape(-1, 1).expand(-1, src.shape[-1])
    return deterministic_scatter_add_(out, scatter_index, src)


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]
    equal_a2a_split_size: int | None = None
    normal_equal_a2a_padding: bool = False


class LocalTokenDispatcher(Configurable):
    """Token dispatcher for EP=1. Handles local token reordering only.

    Also serves as the base class for EP dispatchers (AllToAllTokenDispatcher,
    DeepEPTokenDispatcher, HybridEPTokenDispatcher) which override
    dispatch() and combine().

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int
        score_before_experts: bool = True

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_before_experts = config.score_before_experts

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_mesh, tp_mesh

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via argsort and optionally applies
        routing scores (when ``score_before_experts`` is True).

        Args:
            x_TD: ``(T, D)`` input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices

        Returns:
            routed_input_ND: ``(N, D)`` where N = T*K. Tokens in expert-sorted
                order, score-weighted if ``score_before_experts``.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
        """
        # Reorder the token indices to match the order of the experts where N = T*K
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        # Apply scores before expert computation if configured
        if self.score_before_experts:
            _record_moe_fastpath("score_before_experts")
            routed_input_ND = (
                routed_input_ND.to(torch.float32)
                * topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(x_TD.dtype)

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices per token
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert

        Returns:
            routed_input_RD: ``[R = sum(num_local_tokens_per_expert_E), input_dim(D)]``.
                Tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        # R = N (no EP all-to-all)
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int | None = None,
        local_seq_len_after_padding: int | None = None,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Unused for local dispatch; kept
                for a shared dispatcher combine signature.
            local_seq_len_after_padding: Unused for local dispatch; kept for
                a shared dispatcher combine signature.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        del num_local_tokens_after_padding, local_seq_len_after_padding
        out_TD = torch.zeros_like(x_TD)

        if not self.score_before_experts:
            if routed_output_RD.dtype == torch.bfloat16:
                _record_moe_fastpath("score_after_experts_bf16")
            else:
                _record_moe_fastpath("score_after_experts")
            routed_output_RD = routed_output_RD * metadata.topk_scores_experts_sorted_N.to(
                routed_output_RD.dtype
            ).reshape(-1, 1)

        out_TD = _scatter_add_1d_forward_or_autograd(
            out_TD,
            metadata.token_indices_experts_sorted_N,
            routed_output_RD,
            x_TD,
        )
        return out_TD

    def _sp_global_token_indices(
        self,
        local_indices: torch.Tensor,
        local_seq_len: int,
    ) -> torch.Tensor:
        """Map SP-local token indices to full-sequence global indices.

        Replays upstream PR #3604 (commit ``5ba439938``): with sequence
        parallel and ``B > 1``, the previous ``local_idx + local_T * sp_rank``
        offset placed tokens into wrong global positions because each batch
        has its own sequence shard. Convert to batch-aware indexing instead.

        For ``sp_size == 1`` this is a no-op; the EP=1 ``LocalTokenDispatcher``
        path never sets SP attrs, but inheriting subclasses do.
        """
        sp_size = getattr(self, "sp_size", 1)
        if sp_size == 1:
            return local_indices

        local_pos = local_indices % local_seq_len
        batch_idx = local_indices // local_seq_len
        global_seq_len = local_seq_len * sp_size
        global_indices = batch_idx * global_seq_len + local_pos
        return torch.add(  # pyrefly: ignore [no-matching-overload]
            global_indices, self.sp_rank * local_seq_len
        )


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ``ep_mesh`` and the ``sp_size`` / ``sp_rank`` SP coordinates are wired
    by the owning ``GroupedExperts.parallelize`` override via
    ``wire_meshes``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.force_load_balance = config.force_load_balance
        # DeviceMesh (not ProcessGroup) so that CooR precompile can use
        # torch.ops._dtensor.mesh_get_process_group to keep the FX graph
        # rank-agnostic. None when EP=1 so dispatch falls back to the
        # LocalTokenDispatcher path.
        self.ep_mesh: DeviceMesh | None = None
        # Sequence-parallel split coordinates derived from tp_mesh.
        # ``sp_rank`` uses ``DeviceMesh._sym_get_coordinate`` so it is a
        # ``SymInt`` under CooR precompile, keeping the FX graph
        # rank-agnostic. Defaults are the TP=1 values.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0
        # Defaults for the normal-equal-A2A-padding policy and its
        # telemetry flag. `wire_meshes` overwrites these with values
        # reduced across the EP mesh, so every rank agrees. Without
        # the reduction, per-rank env-var-read drift inside dispatch()
        # could cause ranks to take different branches and the
        # all_to_all_single to hang. Keep local-read defaults for the
        # EP=1 case (when wire_meshes is called with ep_mesh=None).
        self._normal_equal_a2a_policy: str = _normal_equal_a2a_padding_policy()
        self._normal_equal_a2a_telemetry: bool = (
            _normal_equal_a2a_padding_telemetry_enabled()
        )

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh and SP coordinates used by dispatch / combine.

        Both arguments may be ``None`` when the corresponding parallelism
        dimension is disabled; ``dispatch`` / ``combine`` handle the
        disabled cases internally.
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

        # Sample the TT_MOE_NORMAL_EQUAL_A2A_PADDING env var once per
        # dispatcher construction and reduce across the EP mesh so every
        # rank agrees on the policy. Without this, env propagation skew
        # across hosts (per-host .bashrc differences, partial PBS
        # broadcast of `-v` exports) can leave rank 0 seeing "force"
        # while rank 7 sees "" — the two ranks then take different
        # branches inside dispatch() and the all_to_all_single hangs
        # with mismatched argument shapes.
        #
        # Reduction is "max" over an int encoding (off=0, adaptive=1,
        # force=2) so the strongest policy wins. This biases toward
        # turning on the optimization when any rank requested it,
        # which keeps behavior consistent if a user mistakenly only
        # exports the env var on rank 0's host.
        self._normal_equal_a2a_policy = self._resolve_normal_equal_a2a_policy(ep_mesh)
        self._normal_equal_a2a_telemetry = (
            _normal_equal_a2a_padding_telemetry_enabled()
        )

    @staticmethod
    def _resolve_normal_equal_a2a_policy(ep_mesh: DeviceMesh | None) -> str:
        """Reduce the per-rank env-var-derived policy across ep_mesh.

        Returns one of "off", "adaptive", "force". When ep_mesh is None
        (EP=1) the local read is authoritative.
        """
        local = _normal_equal_a2a_padding_policy()
        if ep_mesh is None or ep_mesh.size() == 1:
            return local
        # Encode policy as int for the collective; max-reduce; decode.
        # off < adaptive < force so max gives the strongest policy.
        encoding = {"off": 0, "adaptive": 1, "force": 2}
        decoding = {v: k for k, v in encoding.items()}
        local_code = encoding.get(local, 0)
        # Tensor must live on the mesh's device — XCCL/NCCL backends
        # have no CPU op handler, so a CPU tensor would raise
        # "No backend type associated with device type cpu". Use
        # ep_mesh.device_type to stay portable across xpu/cuda/etc.
        local_tensor = torch.tensor(
            [local_code], dtype=torch.int32, device=ep_mesh.device_type
        )
        torch.distributed.all_reduce(
            local_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=ep_mesh.get_group(),
        )
        return decoding[int(local_tensor.item())]

    @staticmethod
    def _pad_to_equal_splits(
        x: torch.Tensor,
        splits: list[int],
        equal_split_size: int,
    ) -> torch.Tensor:
        if len(splits) == 0:
            return x
        chunks = x.split(splits, dim=0)
        padded_chunks = []
        for chunk, split in zip(chunks, splits):
            pad = equal_split_size - split
            if pad > 0:
                chunk = torch.cat(
                    [chunk, chunk.new_zeros((pad, *chunk.shape[1:]))], dim=0
                )
            padded_chunks.append(chunk)
        return torch.cat(padded_chunks, dim=0)

    @staticmethod
    def _compact_equal_splits(
        x: torch.Tensor,
        splits: list[int],
        equal_split_size: int,
    ) -> torch.Tensor:
        if len(splits) == 0:
            return x
        chunks = x.split([equal_split_size] * len(splits), dim=0)
        return torch.cat(
            [chunk[:split] for chunk, split in zip(chunks, splits)],
            dim=0,
        )

    @staticmethod
    def _can_use_equal_a2a_splits(splits: list[int]) -> bool:
        return len(splits) > 0 and min(splits) != max(splits)

    def _can_use_equal_a2a_splits_global(self, splits: list[int]) -> bool:
        """All-reduce the local can-use-equal-padding decision across ep_mesh.

        Without this, ranks compute the local boolean from their own
        ``splits`` and can disagree: rank A's splits are uniform
        (min==max -> False, no padding) while rank B's splits are
        uneven (min!=max -> True, pad and pass None/None to
        all_to_all_single). The two ranks then call the collective
        with different argument shapes and hang.

        Reduce with op=MAX (True > False) so any rank's decision to
        pad forces all peers to also pad.
        """
        local = self._can_use_equal_a2a_splits(splits)
        if self.ep_mesh is None or self.ep_mesh.size() == 1:
            return local
        # Same XPU/CUDA caveat as _resolve_normal_equal_a2a_policy:
        # XCCL/NCCL has no CPU op handler.
        local_tensor = torch.tensor(
            [1 if local else 0],
            dtype=torch.int32,
            device=self.ep_mesh.device_type,
        )
        torch.distributed.all_reduce(
            local_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=self.ep_mesh.get_group(),
        )
        return bool(local_tensor.item())

    def _global_equal_a2a_split_size(
        self,
        local_splits: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        local_max = max(local_splits) if local_splits else 0
        local_max_tensor = torch.tensor([local_max], device=device, dtype=dtype)
        global_max = all_to_all_single(
            local_max_tensor.expand(self.ep_mesh.size()).contiguous(),
            None,
            None,
            group=self.ep_mesh,
        )
        global_max = torch.ops._c10d_functional.wait_tensor(global_max)
        return int(global_max.max().item())

    def _normal_equal_a2a_adaptive_allowed(
        self,
        *,
        input_splits: list[int],
        output_splits: list[int],
        equal_split_size: int,
        device: torch.device,
    ) -> bool:
        padded_tokens, local_ratio = _equal_a2a_padding_overhead_ratio(
            input_splits=input_splits,
            output_splits=output_splits,
            equal_split_size=equal_split_size,
        )
        local_ratio_bp = int(local_ratio * 10000)
        local_ratio_tensor = torch.tensor([local_ratio_bp], device=device)
        worst_ratio = all_to_all_single(
            local_ratio_tensor.expand(self.ep_mesh.size()).contiguous(),
            None,
            None,
            group=self.ep_mesh,
        )
        worst_ratio = torch.ops._c10d_functional.wait_tensor(worst_ratio)
        worst_ratio_bp = int(worst_ratio.max().item())
        _record_moe_fastpath_stat(
            "normal_equal_a2a_padding_candidate_padded_tokens", padded_tokens
        )
        _record_moe_fastpath_stat(
            "normal_equal_a2a_padding_local_overhead_bp", local_ratio_bp
        )
        _record_moe_fastpath_stat(
            "normal_equal_a2a_padding_global_worst_overhead_bp", worst_ratio_bp
        )
        if worst_ratio_bp <= int(_normal_equal_a2a_padding_threshold() * 10000):
            _record_moe_fastpath("normal_equal_a2a_padding_adaptive_dispatch")
            return True
        _record_moe_fastpath("normal_equal_a2a_padding_adaptive_skip")
        return False

    def _record_normal_equal_a2a_padding_stats_for_splits(
        self,
        *,
        input_splits: list[int],
        output_splits: list[int],
        equal_split_size: int,
        device: torch.device,
        reduce_global: bool,
    ) -> None:
        padded_tokens, local_ratio = _equal_a2a_padding_overhead_ratio(
            input_splits=input_splits,
            output_splits=output_splits,
            equal_split_size=equal_split_size,
        )
        _record_moe_fastpath_stat(
            "normal_equal_a2a_padding_candidate_padded_tokens", padded_tokens
        )
        _record_moe_fastpath_stat(
            "normal_equal_a2a_padding_local_overhead_bp", int(local_ratio * 10000)
        )
        if reduce_global:
            local_ratio_tensor = torch.tensor(
                [int(local_ratio * 10000)], device=device
            )
            worst_ratio = all_to_all_single(
                local_ratio_tensor.expand(self.ep_mesh.size()).contiguous(),
                None,
                None,
                group=self.ep_mesh,
            )
            worst_ratio = torch.ops._c10d_functional.wait_tensor(worst_ratio)
            _record_moe_fastpath_stat(
                "normal_equal_a2a_padding_global_worst_overhead_bp",
                int(worst_ratio.max().item()),
            )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ep_mesh is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        With SP, x_TD/topk_scores_TK/topk_expert_ids_TK are already
        the local SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        # EP=1: fall back to local dispatch (no all-to-all needed)
        if self.ep_mesh is None:
            return super().dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
            )

        ep_size = self.ep_mesh.size()
        # _local_reorder returns (N, D) where N = T*K.
        # EP all-to-all below produces (R, D) where R != N.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_global_tokens_per_local_expert_E = all_to_all_single(
                num_local_tokens_per_expert_E,
                None,
                None,
                group=self.ep_mesh,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_global_tokens_per_local_expert_E = (
                torch.ops._c10d_functional.wait_tensor(
                    num_global_tokens_per_local_expert_E
                )
            )
            input_splits = (
                num_local_tokens_per_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_global_tokens_per_local_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        equal_a2a_split_size = None
        normal_equal_a2a_padding = False
        dispatch_input_splits = input_splits_list
        dispatch_output_splits = output_splits_list
        # Branch decisions for the equal-A2A-padding fast paths must be
        # IDENTICAL across all EP-mesh ranks, otherwise the
        # all_to_all_single below gets different argument shapes from
        # different ranks and hangs. Two sources of per-rank divergence
        # to globally reduce:
        #   1. `_can_use_equal_a2a_splits` is a local min!=max check.
        #      `_can_use_equal_a2a_splits_global` reduces with op=MAX.
        #   2. The TT_MOE_NORMAL_EQUAL_A2A_PADDING env var is read at
        #      `wire_meshes` time and reduced into
        #      `self._normal_equal_a2a_policy` to be robust against
        #      env-propagation skew across hosts.
        if (
            self.force_load_balance
            and self.sp_size == 1
            and self._can_use_equal_a2a_splits_global(input_splits_list)
        ):
            equal_a2a_split_size = self._global_equal_a2a_split_size(
                input_splits_list,
                num_local_tokens_per_expert_E.device,
                num_local_tokens_per_expert_E.dtype,
            )
            _record_moe_fastpath("equal_a2a_padding_dispatch")
            dispatch_input_splits = None
            dispatch_output_splits = None
        elif (
            self._normal_equal_a2a_policy in {"force", "adaptive"}
            and self.sp_size == 1
            and len(input_splits_list) > 0
        ):
            normal_equal_policy = self._normal_equal_a2a_policy
            equal_a2a_split_size = self._global_equal_a2a_split_size(
                input_splits_list,
                num_local_tokens_per_expert_E.device,
                num_local_tokens_per_expert_E.dtype,
            )
            adaptive_allowed = True
            if normal_equal_policy == "adaptive":
                adaptive_allowed = self._normal_equal_a2a_adaptive_allowed(
                    input_splits=input_splits_list,
                    output_splits=output_splits_list,
                    equal_split_size=equal_a2a_split_size,
                    device=num_local_tokens_per_expert_E.device,
                )
            else:
                self._record_normal_equal_a2a_padding_stats_for_splits(
                    input_splits=input_splits_list,
                    output_splits=output_splits_list,
                    equal_split_size=equal_a2a_split_size,
                    device=num_local_tokens_per_expert_E.device,
                    reduce_global=self._normal_equal_a2a_telemetry,
                )
            if adaptive_allowed:
                normal_equal_a2a_padding = True
                _record_moe_fastpath("normal_equal_a2a_padding_dispatch")
                _record_moe_fastpath(
                    "normal_equal_a2a_dispatch_padded_tokens",
                    sum(equal_a2a_split_size - split for split in input_splits_list),
                )
                dispatch_input_splits = None
                dispatch_output_splits = None
            else:
                equal_a2a_split_size = None

        if equal_a2a_split_size is not None:
            routed_input_ND = self._pad_to_equal_splits(
                routed_input_ND,
                input_splits_list,
                equal_a2a_split_size,
            )

        # All-to-all dispatch tokens to EP ranks
        routed_input_RD = all_to_all_single_autograd(
            routed_input_ND,
            dispatch_output_splits,
            dispatch_input_splits,
            self.ep_mesh,
        )
        if equal_a2a_split_size is not None:
            routed_input_RD = self._compact_equal_splits(
                routed_input_RD,
                output_splits_list,
                equal_a2a_split_size,
            )

        # Reorder from rank-major to expert-major via _permute.
        #
        # num_global_tokens_per_local_expert_E layout after all-to-all
        # (e = local experts, EP = EP ranks):
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # _permute reshuffles to:
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        # TODO: Consider using num_global_tokens_per_local_expert_e as the
        # expert_bias_e update buffer, then all-gather on EP ranks. This
        # is blocked by clarification on HybridEP token dropping.
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = self._permute(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            equal_a2a_split_size=equal_a2a_split_size,
            normal_equal_a2a_padding=normal_equal_a2a_padding,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)

        Collapses token count matrix ``t_mat`` from ``(EP, e)`` to
        ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing across ranks.
        """
        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device

        # (EP, e) matrix of token counts per (rank, local_expert)
        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # Transpose to expert-major (e, EP) and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # For each output position, find its input position:
        #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        # seg_ids.shape[0] == segment_lens.sum(), but reuses the unbacked symint
        # already created by repeat_interleave above.
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int | None = None,
        local_seq_len_after_padding: int | None = None,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        When sp_size > 1, dispatch uses local token indices.
        Combine offsets them to global positions so scatter_add
        into full x_TD is correct.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Local token count to use for the
                combined SP view after logical padding. MoE padding passes this
                count without materializing pad rows. Defaults to
                ``x_TD.shape[0]`` if not provided (back-compat for callers
                that pre-date the upstream PR #3604 signature).
            local_seq_len_after_padding: Per-batch local sequence length after
                logical padding, used to map local token indices to global SP
                positions. Required when ``sp_size > 1``.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.ep_mesh is None:
            return super().combine(
                routed_output_RD,
                metadata,
                x_TD,
                num_local_tokens_after_padding=num_local_tokens_after_padding,
                local_seq_len_after_padding=local_seq_len_after_padding,
            )

        # Reverse expert-major reordering
        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )

        combine_input_splits = metadata.output_splits
        combine_output_splits = metadata.input_splits
        equal_a2a_split_size = metadata.equal_a2a_split_size
        if equal_a2a_split_size is not None:
            if metadata.normal_equal_a2a_padding:
                _record_moe_fastpath("normal_equal_a2a_padding_combine")
                _record_moe_fastpath(
                    "normal_equal_a2a_combine_padded_tokens",
                    sum(
                        equal_a2a_split_size - split for split in metadata.output_splits
                    ),
                )
            else:
                _record_moe_fastpath("equal_a2a_padding_combine")
            routed_output_RD = self._pad_to_equal_splits(
                routed_output_RD,
                metadata.output_splits,
                equal_a2a_split_size,
            )
            combine_input_splits = None
            combine_output_splits = None

        # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
        # on the NCCL stream and won't block until the tensor is accessed.
        routed_output_RD = all_to_all_single_autograd(
            routed_output_RD,
            combine_output_splits,
            combine_input_splits,
            self.ep_mesh,
        )
        if equal_a2a_split_size is not None:
            routed_output_RD = self._compact_equal_splits(
                routed_output_RD,
                metadata.input_splits,
                equal_a2a_split_size,
            )

        # With SP, x_TD is the local shard. Create full-size buffer for
        # scatter_add so routed results from all SP ranks can be placed
        # at global positions. Use ``num_local_tokens_after_padding`` when
        # the caller provides it (post-PR #3604 upstream callsite); fall
        # back to ``x_TD.shape[0]`` to preserve the prior behavior when
        # callers haven't been updated yet.
        local_T = (
            num_local_tokens_after_padding
            if num_local_tokens_after_padding is not None
            else x_TD.shape[0]
        )
        out_TD = torch.zeros(
            local_T * self.sp_size,
            x_TD.shape[-1],
            device=x_TD.device,
            dtype=x_TD.dtype,
        )

        if not self.score_before_experts:
            if routed_output_RD.dtype == torch.bfloat16:
                _record_moe_fastpath("score_after_experts_bf16")
            else:
                _record_moe_fastpath("score_after_experts")
            routed_output_RD = routed_output_RD * metadata.topk_scores_experts_sorted_N.to(
                routed_output_RD.dtype
            ).reshape(-1, 1)

        # With SP, token indices are 0-based within the local shard.
        # Map them to global positions in the full-size scatter buffer
        # via the batch-aware helper (replays upstream PR #3604).
        if self.sp_size > 1:
            assert local_seq_len_after_padding is not None, (
                "AllToAllTokenDispatcher.combine requires "
                "local_seq_len_after_padding when sp_size > 1"
            )
            token_indices_experts_sorted_N = self._sp_global_token_indices(
                metadata.token_indices_experts_sorted_N,
                local_seq_len_after_padding,
            )
        else:
            token_indices_experts_sorted_N = metadata.token_indices_experts_sorted_N

        assert isinstance(token_indices_experts_sorted_N, torch.Tensor)
        out_TD = _scatter_add_1d_forward_or_autograd(
            out_TD,
            token_indices_experts_sorted_N,
            routed_output_RD,
            x_TD,
        )
        return out_TD


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Requires EP to be enabled (ep_mesh must be set). Raises ValueError
    if ep_mesh is None, since quantized grouped GEMMs need padded token
    groups which are only produced by the EP permute_and_pad path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(
        self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    ):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        # FP8/MXFP8 require groups to be permuted to expert major order AND
        # padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert
        # gets locally is a multiple of `self.pad_multiple`.
        # Note that this will create side effects when wrapping the for-loop
        # implementation of GroupedExperts, as it does not need padding.
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
            _group_offsets,
        ) = permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD[:-1]


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP and HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using DeepEP for efficient token dispatch/combine.

    Uses DeepEP library kernels (H100/NVLink Switch) instead of standard
    all-to-all collectives. Combine is asynchronous — callers must call
    sync_combine() before using the result.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import deepep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by DeepEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. DeepEP returns the number
        # of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "MoE.parallelize() must call token_dispatcher.wire_meshes()."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.deepep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int | None = None,
        local_seq_len_after_padding: int | None = None,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        ``local_seq_len_after_padding`` is required when ``sp_size > 1``
        so the SP expansion uses batch-aware global indices (upstream
        PR #3604).
        """
        del num_local_tokens_after_padding  # unused; combined_TD.shape[0] carries it
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        combined_TD = combine_tokens(routed_output_RD, metadata.state)

        if self.sp_size > 1:
            assert local_seq_len_after_padding is not None, (
                "DeepEPTokenDispatcher.combine requires "
                "local_seq_len_after_padding when sp_size > 1"
            )
            sync_combine()
            out_TD = torch.zeros(
                combined_TD.shape[0] * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            local_indices = torch.arange(
                combined_TD.shape[0], device=combined_TD.device
            )
            global_indices = self._sp_global_token_indices(
                local_indices,
                local_seq_len_after_padding,
            )
            out_TD[global_indices] = combined_TD
            return out_TD

        return combined_TD


class HybridEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using HybridEP for efficient token dispatch/combine.

    Uses HybridEP library kernels (GB200/NVLink72) instead of standard
    all-to-all collectives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        """Config for HybridEP token dispatcher.

        Args:
            non_blocking_capacity_factor: Enable non-blocking HybridEP dispatch
                with a given capacity factor.

                Setting this to a float in (0, 1] enables CPU-free non-blocking
                dispatch and controls num_permuted_tokens — the fused-permute
                output capacity, estimated as:
                num_tokens × ep_size × min(num_local_experts, top_k) × cf,
                aligned for MXFP8.  Tokens whose permuted offset exceeds this
                limit are silently dropped (overflow_flag is set on GPU).

                - None = blocking mode (default).  HybridEP calls
                  cudaStreamSynchronize after dispatch, copies
                  tokens_per_expert to pinned CPU memory, and computes the
                  exact num_permuted_tokens on the host.  No token dropping.
                - 1.0 = non-blocking, worst-case sizing: every token can reach
                  every local expert, no drops, highest memory.
                - < 1.0 = non-blocking, reduced memory; controls the
                  fused-permute output tensor size (num_permuted_tokens).
                  Safe in practice when forced load balancing (e.g. aux-loss /
                  round-robin) keeps distribution roughly uniform.

                Note: this factor has no lasting effect on the all-to-all
                communication buffer.  HybridEP's dispatch_with_permute
                internally passes the actual num_tokens to
                update_template_config, which auto-grows the buffer to the
                full token count on the first dispatch regardless of this
                setting.
        """

        non_blocking_capacity_factor: float | None = None
        pad_multiple: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by HybridEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. HybridEP returns the
        # number of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "MoE.parallelize() must call token_dispatcher.wire_meshes()."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.hybridep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
            non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
            pad_multiple=self.pad_multiple,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int | None = None,
        local_seq_len_after_padding: int | None = None,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP.

        ``local_seq_len_after_padding`` is required when ``sp_size > 1``
        so the SP expansion uses batch-aware global indices (upstream
        PR #3604).
        """
        del num_local_tokens_after_padding  # unused; combined_TD.shape[0] carries it
        from torchtitan.distributed.deepep import hybridep

        combined_TD = hybridep.combine_tokens(
            routed_output_RD,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            assert local_seq_len_after_padding is not None, (
                "HybridEPTokenDispatcher.combine requires "
                "local_seq_len_after_padding when sp_size > 1"
            )
            out_TD = torch.zeros(
                combined_TD.shape[0] * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            local_indices = torch.arange(
                combined_TD.shape[0], device=combined_TD.device
            )
            global_indices = self._sp_global_token_indices(
                local_indices,
                local_seq_len_after_padding,
            )
            out_TD[global_indices] = combined_TD
            return out_TD

        return combined_TD
