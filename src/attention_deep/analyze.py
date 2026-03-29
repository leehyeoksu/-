import numpy as np
from scipy import stats


def analyze(qk_maps: dict, gate_maps: dict, num_heads: int = 6):
    """
    통계 분석 4종
      ① Spearman  : gate 평균 vs QK sink score
      ② Wilcoxon  : QK sink vs Gate×QK sink (gate 억제 효과)
      ③ Kruskal-Wallis : 레이어별 sink score 차이
      ④ Friedman  : head별 gate 분포 차이

    Args:
        qk_maps  : {layer_idx: tensor(B, H, N, N)}
        gate_maps: {layer_idx: tensor(B, N, H, Dh)}
        num_heads: 헤드 수 (기본 6)
    """
    sink_scores_qk    = []   # 레이어별 QK sink score 평균
    sink_scores_gated = []   # 레이어별 Gate×QK sink score 평균
    gate_means        = []   # 레이어별 gate 평균

    layer_sink_qk     = []   # Kruskal-Wallis용: [[head0..head5], ...]
    head_gates        = {h: [] for h in range(num_heads)}  # Friedman용

    for layer_idx in sorted(qk_maps.keys()):
        if layer_idx not in gate_maps:
            continue

        attn = qk_maps[layer_idx][0]    # (H, N, N)
        gate = gate_maps[layer_idx][0]  # (N, H, Dh)

        layer_qk    = []
        layer_gated = []
        layer_gate  = []

        for h in range(attn.shape[0]):
            # sink score: 다른 패치들이 CLS 토큰을 얼마나 보는가
            patch_to_cls = attn[h, 1:, 0].numpy()       # (N-1,)
            sink_qk      = patch_to_cls.mean()

            gate_h      = gate[1:, h, :].mean(axis=-1).numpy()  # (N-1,)
            gate_norm   = gate_h / (gate_h.max() + 1e-8)
            gated       = patch_to_cls * gate_norm
            sink_gated  = gated.mean()

            layer_qk.append(sink_qk)
            layer_gated.append(sink_gated)
            layer_gate.append(gate_h.mean())
            head_gates[h].append(gate_h.mean())

        sink_scores_qk.append(np.mean(layer_qk))
        sink_scores_gated.append(np.mean(layer_gated))
        gate_means.append(np.mean(layer_gate))
        layer_sink_qk.append(layer_qk)

    print("\n" + "=" * 55)
    print("  통계 분석 결과")
    print("=" * 55)

    # ① Spearman
    r, p = stats.spearmanr(gate_means, sink_scores_qk)
    sig  = "✅ 유의미" if p < 0.05 else "❌ 유의미하지않음"
    print(f"\n① Spearman  (gate평균 vs QK sink score)")
    print(f"   r={r:.4f},  p={p:.4f}  {sig}")
    if r < 0:
        print("   → 음의 상관: gate가 높을수록 sink 감소 (억제 효과)")
    else:
        print("   → 양의 상관 or 무관")

    # ② Wilcoxon
    try:
        stat, p = stats.wilcoxon(sink_scores_qk, sink_scores_gated)
        sig = "✅ 유의미한 억제" if p < 0.05 else "❌ 유의미하지않음"
        print(f"\n② Wilcoxon  (QK sink vs Gate×QK sink)")
        print(f"   stat={stat:.4f},  p={p:.4f}  {sig}")
    except ValueError as e:
        print(f"\n② Wilcoxon  → 계산 불가: {e}")

    # ③ Kruskal-Wallis
    stat, p = stats.kruskal(*layer_sink_qk)
    sig = "✅ 레이어간 차이 있음" if p < 0.05 else "❌ 차이 없음"
    print(f"\n③ Kruskal-Wallis  (레이어별 sink score 차이)")
    print(f"   stat={stat:.4f},  p={p:.4f}  {sig}")

    # ④ Friedman
    friedman_data = [head_gates[h] for h in range(num_heads)]
    try:
        stat, p = stats.friedmanchisquare(*friedman_data)
        sig = "✅ head간 차이 있음" if p < 0.05 else "❌ 차이 없음"
        print(f"\n④ Friedman  (head별 gate 분포 차이)")
        print(f"   stat={stat:.4f},  p={p:.4f}  {sig}")
    except ValueError as e:
        print(f"\n④ Friedman  → 계산 불가: {e}")

    print("\n" + "=" * 55)

    return {
        "gate_means"        : gate_means,
        "sink_scores_qk"    : sink_scores_qk,
        "sink_scores_gated" : sink_scores_gated,
    }
