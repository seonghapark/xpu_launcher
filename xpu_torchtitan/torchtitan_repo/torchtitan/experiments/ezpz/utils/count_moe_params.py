"""Count total and active parameters for each MOE config.

Usage:
    python3 -m torchtitan.experiments.ezpz.utils.count_moe_params
"""
from torchtitan.experiments.ezpz.moe import moe_configs


def count_params(name: str) -> tuple[int, int]:
    cfg = moe_configs[name]()
    total = 0
    active = 0
    dim = cfg.dim

    # Embedding + output + norm
    shared = cfg.vocab_size * dim + dim * cfg.vocab_size + dim
    total += shared
    active += shared

    for layer_cfg in cfg.layers:
        # Attention (always active)
        attn = layer_cfg.attention
        kv_lora = attn.kv_lora_rank
        q_lora = attn.q_lora_rank
        n_heads = attn.n_heads
        qk_nope = attn.qk_nope_head_dim
        qk_rope = attn.qk_rope_head_dim
        v_head = attn.v_head_dim
        head_dim = qk_nope + qk_rope

        if q_lora > 0:
            attn_p = (dim * q_lora) + (q_lora * n_heads * head_dim)
        else:
            attn_p = dim * n_heads * head_dim
        attn_p += dim * (kv_lora + qk_rope)
        attn_p += kv_lora * n_heads * (qk_nope + v_head)
        attn_p += n_heads * v_head * dim
        attn_p += kv_lora
        if q_lora > 0:
            attn_p += q_lora
        total += attn_p
        active += attn_p

        # Norms
        total += 2 * dim
        active += 2 * dim

        # Dense FFN
        ff = getattr(layer_cfg, "feed_forward", None)
        if ff is not None:
            w1 = getattr(ff, "w1", None)
            if w1 is not None:
                h = w1.out_features
                ff_p = 3 * dim * h
                total += ff_p
                active += ff_p

        # MoE
        moe = getattr(layer_cfg, "moe", None)
        if moe is not None:
            num_e = moe.num_experts
            top_k = moe.router.top_k
            e_dim = moe.experts.hidden_dim

            # Router gate
            total += dim * num_e
            active += dim * num_e

            # Routed experts (3 matrices per expert: w1, w2, w3)
            per_expert = 3 * dim * e_dim
            total += num_e * per_expert
            active += top_k * per_expert

            # Shared experts
            se = moe.shared_experts
            if se is not None:
                s_hidden = se.w1.out_features
                n_shared = s_hidden // e_dim if e_dim > 0 else 1
                se_p = se.w1.in_features * se.w1.out_features  # w1
                se_p += se.w2.in_features * se.w2.out_features  # w2
                se_p += se.w3.in_features * se.w3.out_features  # w3
                total += se_p
                active += se_p

    return total, active


if __name__ == "__main__":
    header = (
        f"{'Config':15s}  {'Total':>10s}  {'Active':>10s}  "
        f"{'Ratio':>6s}  {'Layers':>6s}  {'Experts':>7s}  "
        f"{'TopK':>4s}  {'Heads':>5s}  {'Dim':>5s}"
    )
    print(header)
    print("-" * len(header))
    for name in ["debugmodel", "500M", "2B", "4B", "7B", "10B_2B"]:
        cfg = moe_configs[name]()
        t, a = count_params(name)
        num_e = 0
        top_k = 0
        for lc in cfg.layers:
            moe = getattr(lc, "moe", None)
            if moe is not None:
                num_e = moe.num_experts
                top_k = moe.router.top_k
                break
        n_heads = cfg.layers[0].attention.n_heads
        print(
            f"{name:15s}  {t/1e9:9.2f}B  {a/1e9:9.2f}B  "
            f"{a/t:5.1%}  {len(cfg.layers):6d}  {num_e:7d}  "
            f"{top_k:4d}  {n_heads:5d}  {cfg.dim:5d}"
        )
