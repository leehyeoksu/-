import torch


def register_hooks(model):
    """
    각 레이어 attention에 pre_hook 등록
    QK attention weight 계산 후 캐싱
    Returns:
        qk_maps: {layer_idx: tensor(B, H, N, N)}
        hooks: hook handle 리스트 (제거용)
    """
    qk_maps = {}
    hooks = []

    num_heads = model.backbone.config.num_attention_heads
    head_dim  = model.backbone.config.hidden_size // num_heads

    for i, layer in enumerate(model.backbone.layer):
        def make_hook(idx, attn_module):
            def hook(module, args, kwargs):
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hidden_states is None:
                    return

                B, N, _ = hidden_states.shape

                with torch.no_grad():
                    Q = attn_module.q_proj(hidden_states)
                    K = attn_module.k_proj(hidden_states)

                    Q = Q.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, Dh)
                    K = K.view(B, N, num_heads, head_dim).transpose(1, 2)

                    scale = head_dim ** -0.5
                    attn  = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
                    qk_maps[idx] = attn.detach().cpu()  # (B, H, N, N)

            return hook

        h = layer.attention.register_forward_pre_hook(
            make_hook(i, layer.attention),
            with_kwargs=True,
        )
        hooks.append(h)

    return qk_maps, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def collect_gates(model):
    """
    forward 후 각 레이어 GatedOutputProjection.last_gate 수집
    Returns:
        gate_maps: {layer_idx: tensor(B, N, H, Dh)}
    """
    gate_maps = {}
    for i, layer in enumerate(model.backbone.layer):
        o_proj = layer.attention.o_proj
        if hasattr(o_proj, "last_gate") and o_proj.last_gate is not None:
            gate_maps[i] = o_proj.last_gate.detach().cpu()
    return gate_maps
