import torch
import torch.nn as nn

# =========================================================
# 1️⃣ Gate Wrapper (G1: concat 뒤, o_proj 전)
# =========================================================
class GatedOutputProjection(nn.Module):
    def __init__(
        self,
        original_o_proj,
        d_model,
        num_heads,
        gate_type="elementwise",   # "elementwise" or "headwise"
        keep_cls_ungated=True,
        init_as_identity=True,
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
            raise ValueError("gate_type must be elementwise or headwise")

        # -------------------------
        # 초기 안정화 (중요)
        # -------------------------
        if init_as_identity:
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 4.0)

        self.cached_hidden_states = None

    def set_hidden_states(self, hidden_states):
        self.cached_hidden_states = hidden_states

    def forward(self, x):
        # x: (B, N, D)

        if self.cached_hidden_states is None:
            return self.original_o_proj(x)

        hs = self.cached_hidden_states

        # -------------------------
        # gate 계산
        # -------------------------
        g = torch.sigmoid(self.gate_proj(hs))

        if self.gate_type == "headwise":
            B, N, H = g.shape
            g = g.unsqueeze(-1).expand(B, N, H, self.head_dim)
            g = g.reshape(B, N, self.d_model)

        # 🚨 [수정됨] CLS 보호 (In-place 에러 방지를 위해 clone 추가)
        if self.keep_cls_ungated and g.size(1) > 0:
            g = g.clone()
            g[:, 0, :] = 1.0

        # -------------------------
        # 핵심: gating
        # -------------------------
        x = g * x

        out = self.original_o_proj(x)

        self.cached_hidden_states = None
        return out


# =========================================================
# 2️⃣ hook (attention input 캐싱)
# =========================================================
def _make_cache_hook(wrapper):
    # 🚨 [수정됨] args뿐만 아니라 kwargs(키워드 인자)도 받을 수 있게 수정
    def hook(module, args, kwargs):
        hidden_states = None
        if len(args) > 0:
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif "x" in kwargs: # timm 모델 등에서 x를 쓸 경우 대비
            hidden_states = kwargs["x"]
            
        if hidden_states is not None:
            wrapper.set_hidden_states(hidden_states)
    return hook


# =========================================================
# 3️⃣ 기존 gate 제거 (중복 방지 및 Hook 제거)
# =========================================================
# 🚨 [수정됨] 등록했던 Hook 메모리까지 깔끔하게 지워주도록 수정
def remove_gating(model, hooks=None):
    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            attn.o_proj = attn.o_proj.original_o_proj
            
    if hooks is not None:
        for handle in hooks:
            handle.remove()
        hooks.clear()


# =========================================================
# 4️⃣ 메인 함수 (이거만 쓰면 됨)
# =========================================================
def inject_gating(
    model,
    gate_type="elementwise",
    keep_cls_ungated=True,
    init_as_identity=True,
):
    """
    DINOv3 attention에 G1 gate 삽입
    """
    # 기존거 제거 (초기화)
    d_model = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    hooks = []

    for layer in model.layer:
        attn = layer.attention

        wrapper = GatedOutputProjection(
            original_o_proj=attn.o_proj,
            d_model=d_model,
            num_heads=num_heads,
            gate_type=gate_type,
            keep_cls_ungated=keep_cls_ungated,
            init_as_identity=init_as_identity,
        )

        # 🚨 [수정됨] kwargs 처리를 위해 with_kwargs=True 옵션 추가
        handle = attn.register_forward_pre_hook(_make_cache_hook(wrapper), with_kwargs=True)
        hooks.append(handle)

        # 교체
        attn.o_proj = wrapper

    return model, hooks


# =========================================================
# 5️⃣ gate만 학습 (옵션)
# =========================================================
def freeze_except_gate(model):
    for p in model.parameters():
        p.requires_grad = False

    for layer in model.layer:
        attn = layer.attention
        if isinstance(attn.o_proj, GatedOutputProjection):
            for p in attn.o_proj.gate_proj.parameters():
                p.requires_grad = True