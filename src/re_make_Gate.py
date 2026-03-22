import torch
import torch.nn as nn

# =========================================================
# 1️⃣ G1 Gate (논문 그대로: SDPA output gating)
# =========================================================
class GatedOutputProjection(nn.Module):
    def __init__(
        self,
        original_o_proj,
        d_model,
        num_heads,
        gate_type="elementwise",   # 논문 best: elementwise
        keep_cls_ungated=True,
    ):
        super().__init__()

        self.original_o_proj = original_o_proj
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.gate_type = gate_type
        self.keep_cls_ungated = keep_cls_ungated

        # -------------------------
        # Gate projection
        # -------------------------
        if gate_type == "elementwise":
            self.gate_proj = nn.Linear(d_model, d_model)
        elif gate_type == "headwise":
            self.gate_proj = nn.Linear(d_model, num_heads)
        else:
            raise ValueError

        # 🔥 논문 스타일 초기화 (bias=0 → sparsity 유도)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.cached_hidden_states = None
        self.last_gate = None

    def set_hidden_states(self, hidden_states):
        self.cached_hidden_states = hidden_states

    def forward(self, x):
        if self.cached_hidden_states is None:
            return self.original_o_proj(x)

        hs = self.cached_hidden_states

        # -------------------------
        # Gating (논문 Eq.5)
        # -------------------------
        g = torch.sigmoid(self.gate_proj(hs))

        if self.gate_type == "headwise":
            B, N, H = g.shape
            g = g.unsqueeze(-1).expand(B, N, H, self.head_dim)
            g = g.reshape(B, N, self.d_model)

        # CLS 보호 (선택)
        if self.keep_cls_ungated and g.size(1) > 0:
            g = g.clone()
            g[:, 0, :] = 1.0

        # 저장 (loss용)
        self.last_gate = g

        # 🔥 핵심: multiplicative gating
        x = g * x

        # 로그 (디버깅)
        if self.training and torch.rand(1).item() < 0.01:
            print(f"[Gate] mean={g.mean().item():.4f}, min={g.min().item():.4f}, max={g.max().item():.4f}")

        out = self.original_o_proj(x)

        self.cached_hidden_states = None
        return out


# =========================================================
# 2️⃣ hook
# =========================================================
def _make_cache_hook(wrapper):
    def hook(module, args, kwargs):
        hidden_states = None
        if len(args) > 0:
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif "x" in kwargs:
            hidden_states = kwargs["x"]

        if hidden_states is not None:
            wrapper.set_hidden_states(hidden_states)

    return hook


# =========================================================
# 3️⃣ inject (🔥 전체 layer 적용 = 논문 그대로)
# =========================================================
def inject_gating(
    model,
    gate_type="elementwise",   # 논문 best
    keep_cls_ungated=True,
):
    d_model = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    hooks = []

    # 🔥 전체 layer 적용 (논문 그대로)
    for layer in model.layer:
        attn = layer.attention

        wrapper = GatedOutputProjection(
            original_o_proj=attn.o_proj,
            d_model=d_model,
            num_heads=num_heads,
            gate_type=gate_type,
            keep_cls_ungated=keep_cls_ungated,
        )

        handle = attn.register_forward_pre_hook(
            _make_cache_hook(wrapper),
            with_kwargs=True
        )
        hooks.append(handle)

        attn.o_proj = wrapper

    return model, hooks


# =========================================================
# 4️⃣ sparsity loss (🔥 논문 핵심)
# =========================================================
def compute_gate_loss(model):
    loss = 0.0
    count = 0

    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            if attn.o_proj.last_gate is not None:
                loss += attn.o_proj.last_gate.mean()
                count += 1

    if count == 0:
        return 0.0

    return loss / count


# =========================================================
# 5️⃣ gate만 학습 (선택)
# =========================================================
def freeze_except_gate(model):
    for p in model.parameters():
        p.requires_grad = False

    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            for p in attn.o_proj.gate_proj.parameters():
                p.requires_grad = True