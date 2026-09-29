



# ---- AdamW param groups：只对权重矩阵做 weight decay（bias/Norm/1D 参数不衰减） ----
def get_adamw_param_groups(model, weight_decay: float):
    # 这些名字命中就强制 no_decay
    NO_DECAY_NAME_SUBSTRS = (
        "struct_bias_module",   # TreeStructBias 整体
        "logit_scale",          # 0D 温度标量
        ".cls",                 # cls token
        "special.weight",       # special token embedding
        "rule_id.weight",       # rule id embedding（若启用）
    )

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        n_low = n.lower()
        if not p.requires_grad:
            continue

        
        if p.ndim <= 1 or n.endswith(".bias") or ("norm" in n_low) or ("bn" in n_low):
            # 经验规则：1D 参数（bias、LayerNorm/BatchNorm 的 γ/β 等）不做衰减；
            no_decay.append(p)
        elif any(substr in n_low for substr in NO_DECAY_NAME_SUBSTRS):
            no_decay.append(p)
        else:
            # 其他矩阵权重做 decay
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]