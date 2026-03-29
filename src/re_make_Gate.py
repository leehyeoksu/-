import torch
import torch.nn as nn


# =========================================================
# 1️⃣ G1 Gate (논문 그대로: SDPA output gating)
#    - elementwise: head-specific (n × q × dk) ← 논문 best
#    - headwise:    head-specific (n × q)
#    gate 위치: concat(AV) = o_proj 입력 x 에 곱하기 = 논문 Eq.5 Y
# =========================================================
class GatedOutputProjection(nn.Module):
    def __init__(
        self,
        original_o_proj,
        d_model,
        num_heads,
        gate_type="elementwise",   # 논문 best: elementwise (head-specific)
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
            # 🔥 head-specific: d_model → num_heads * head_dim (= d_model)
            # 논문 Table1 row5: score shape = n × q × dk
            # Linear(d_model, d_model) 이지만 해석은 (H, Dh) head별로 독립
            # → 학습 시 각 헤드가 자기 파라미터 블록을 가짐 (weight 구조상 head-specific)
            # 실제로 head-specific이 되려면 출력을 (B, N, H, Dh)로 reshape해서 gate 적용
            self.gate_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=True)

        elif gate_type == "headwise":
            # 논문 Table1 row10: score shape = n × q (헤드별 스칼라)
            self.gate_proj = nn.Linear(d_model, num_heads, bias=True)

        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        # 🔥 zero-init (논문 스타일: 초기 gate≈0.5, sparsity 유도)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.cached_hidden_states = None
        self.last_gate = None

    def set_hidden_states(self, hidden_states):
        self.cached_hidden_states = hidden_states

    def forward(self, x):
        # x = concat(AV) = SDPA output, shape: (B, N, d_model) ← 논문의 Y
        if self.cached_hidden_states is None:
            return self.original_o_proj(x)

        hs = self.cached_hidden_states   # pre-norm hidden states (논문 각주1)
        B, N, _ = hs.shape

        # -------------------------
        # Gate 계산 (논문 Eq.5)
        # g = σ(X · Wθ)
        # -------------------------
        if self.gate_type == "elementwise":
            # (B, N, H*Dh) → (B, N, H, Dh) → head-specific elementwise gate
            g = torch.sigmoid(self.gate_proj(hs))               # (B, N, d_model)
            g = g.view(B, N, self.num_heads, self.head_dim)     # (B, N, H, Dh)

            # CLS 보호
            if self.keep_cls_ungated and N > 0:
                g = g.clone()
                g[:, 0, :, :] = 1.0

            self.last_gate = g

            # x를 (B, N, H, Dh)로 reshape해서 head-specific gate 적용
            x = x.view(B, N, self.num_heads, self.head_dim)     # (B, N, H, Dh)
            x = g * x                                            # 🔥 head-specific elementwise
            x = x.reshape(B, N, self.d_model)                   # (B, N, d_model)

        elif self.gate_type == "headwise":
            # (B, N, H) → 헤드별 스칼라를 head_dim에 broadcast
            g = torch.sigmoid(self.gate_proj(hs))               # (B, N, H)
            g = g.view(B, N, self.num_heads, 1)                 # (B, N, H, 1)

            # CLS 보호
            if self.keep_cls_ungated and N > 0:
                g = g.clone()
                g[:, 0, :, :] = 1.0

            self.last_gate = g

            x = x.view(B, N, self.num_heads, self.head_dim)     # (B, N, H, Dh)
            x = g * x                                            # 🔥 헤드별 broadcast
            x = x.reshape(B, N, self.d_model)                   # (B, N, d_model)

        # 디버깅 로그
        if self.training and torch.rand(1).item() < 0.01:
            print(f"[Gate/{self.gate_type}] mean={g.mean():.4f}  min={g.min():.4f}  max={g.max():.4f}")

        out = self.original_o_proj(x)
        self.cached_hidden_states = None
        return out


# =========================================================
# 2️⃣ hook (pre-norm hidden states 캐싱)
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
# 3️⃣ inject_gating (전체 layer 적용 = 논문 그대로)
# =========================================================
def inject_gating(
    model,
    gate_type="elementwise",
    keep_cls_ungated=True,
    target_layers=None,   # None=전체, int=마지막N개, list=[0,1,2]=지정
):
    d_model   = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    total     = len(model.layer)

    # ── 적용할 레이어 인덱스 결정 ──────────────────────────
    if target_layers is None:
        indices = list(range(total))          # 전체
    elif isinstance(target_layers, int):
        indices = list(range(total - target_layers, total))  # 마지막 N개
    elif isinstance(target_layers, list):
        indices = target_layers               # 직접 지정
    else:
        raise ValueError("target_layers: None | int | list[int]")

    hooks = []

    for i, layer in enumerate(model.layer):
        if i not in indices:
            continue

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
            with_kwargs=True,
        )
        hooks.append(handle)
        attn.o_proj = wrapper

    print(f"[inject_gating] 총 {total}개 중 {len(indices)}개 레이어에 G1 gate 적용")
    print(f"  → 적용 인덱스: {indices}  (gate_type={gate_type})")
    return model, hooks

# =========================================================
# 4️⃣ sparsity loss
# =========================================================
def compute_gate_loss(model):
    loss  = 0.0
    count = 0

    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            if attn.o_proj.last_gate is not None:
                loss  += attn.o_proj.last_gate.mean()
                count += 1

    return loss / count if count > 0 else 0.0


# =========================================================
# 5️⃣ gate만 학습
# =========================================================
def freeze_except_gate(model):
    for p in model.parameters():
        p.requires_grad = False

    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            for p in attn.o_proj.gate_proj.parameters():
                p.requires_grad = True

    print("[freeze_except_gate] gate_proj 파라미터만 학습 가능")