import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ──────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────
def _to_2d(arr, num_patches):
    try:
        return arr.reshape(num_patches, num_patches)
    except ValueError:
        n = int(np.sqrt(len(arr)))
        return arr[:n * n].reshape(n, n)


def _resize_map(arr_2d, target_size):
    mn, mx = arr_2d.min(), arr_2d.max()
    if mx - mn < 1e-8:
        normed = np.zeros_like(arr_2d, dtype=np.float32)
    else:
        normed = (arr_2d - mn) / (mx - mn)
    pil = Image.fromarray((normed * 255).astype(np.uint8))
    pil = pil.resize(target_size, Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    arr = img.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────
# 메인 시각화
# ──────────────────────────────────────────────────────────
def visualize(
    qk_maps: dict,
    gate_maps: dict,
    original_img,
    save_dir: str      = "outputs",
    patch_size: int    = 14,
    img_size: int      = 224,
    top_percent: float = 10.0,
):
    """
    레이어당 1장 (3행 × 6열)
      Row 0 : QK Attention  — Ungated  (CLS→patch, 상위 top%)
      Row 1 : Gate Heatmap  — 각 head 별 gate 값 (RdBu_r)
      Row 2 : Gate × QK     — Gated    (상위 top%)

    타이틀 구조
      큰 제목  : Layer N
      열 제목  : Head 0 ~ Head 5
      행 레이블: 왼쪽 세로
    """
    os.makedirs(save_dir, exist_ok=True)

    num_patches = img_size // patch_size
    percentile  = 100 - top_percent

    orig      = _to_pil(original_img).resize((img_size, img_size))
    num_heads = next(iter(qk_maps.values())).shape[1]

    ROW_LABELS = [
        "QK Attention\n(Ungated)",
        "Gate Heatmap\n(Head Gating)",
        "Gate × QK\n(Gated)",
    ]

    BG         = "#12121f"
    TEXT_COLOR = "white"
    LABEL_COLOR = "#c8c8e8"

    for layer_idx in sorted(qk_maps.keys()):
        if layer_idx not in gate_maps:
            print(f"[skip] Layer {layer_idx}: gate 없음")
            continue

        attn = qk_maps[layer_idx][0]    # (H, N, N)
        gate = gate_maps[layer_idx][0]  # (N, H, Dh)

        # gate 실제 범위 (패치 토큰만, head 평균)
        gate_patch = gate[1:, :, :].mean(axis=-1).numpy()  # (N-1, H)
        gate_vmin  = float(gate_patch.min())
        gate_vmax  = float(gate_patch.max())

        fig = plt.figure(figsize=(3.8 * num_heads, 12))
        fig.patch.set_facecolor(BG)

        # col 0 = 행 레이블, col 1~ = 데이터
        gs = gridspec.GridSpec(
            3, num_heads + 1,
            figure=fig,
            width_ratios=[0.22] + [1] * num_heads,
            hspace=0.38, wspace=0.06,
            left=0.01, right=0.97,
            top=0.91, bottom=0.02,
        )

        fig.suptitle(
            f"Layer {layer_idx}",
            fontsize=20, fontweight="bold",
            color=TEXT_COLOR, y=0.97,
        )

        for row in range(3):
            # ── 행 레이블
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.set_facecolor(BG)
            ax_label.axis("off")
            ax_label.text(
                0.95, 0.5, ROW_LABELS[row],
                va="center", ha="right",
                fontsize=10, color=LABEL_COLOR,
                fontweight="bold",
                transform=ax_label.transAxes,
            )

            for h in range(num_heads):
                ax = fig.add_subplot(gs[row, h + 1])
                ax.set_facecolor(BG)
                ax.axis("off")

                # 열 제목 (row 0만)
                if row == 0:
                    ax.set_title(
                        f"Head {h}",
                        fontsize=12, color=TEXT_COLOR,
                        fontweight="bold", pad=5,
                    )

                # ── Row 0: QK Ungated ────────────────────────
                if row == 0:
                    raw  = attn[h, 0, 1:].numpy()
                    thr  = np.percentile(raw, percentile)
                    data = np.where(raw >= thr, raw, 0.0)
                    up   = _resize_map(_to_2d(data, num_patches), orig.size)

                    ax.imshow(orig, interpolation="bilinear")
                    ax.imshow(up, cmap="jet", alpha=0.65,
                              vmin=0, vmax=1, interpolation="bilinear")

                # ── Row 1: Gate Heatmap ──────────────────────
                elif row == 1:
                    gate_h  = gate[1:, h, :].mean(axis=-1).numpy()  # (N-1,)
                    gate_2d = _to_2d(gate_h, num_patches)

                    # vmin/vmax를 레이어 전체 범위로 통일
                    span = gate_vmax - gate_vmin + 1e-8
                    normed = ((gate_2d - gate_vmin) / span * 255).astype(np.uint8)
                    gate_up = np.array(
                        Image.fromarray(normed).resize(orig.size, Image.BILINEAR),
                        dtype=np.float32,
                    ) / 255.0

                    im = ax.imshow(
                        gate_up,
                        cmap="RdBu_r",   # 빨강=높음(gate 열림), 파랑=낮음(gate 닫힘)
                        vmin=0, vmax=1,
                        interpolation="bilinear",
                    )
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(colors=LABEL_COLOR, labelsize=7)
                    mid = (gate_vmin + gate_vmax) / 2
                    cbar.set_ticks([0, 0.5, 1])
                    cbar.set_ticklabels(
                        [f"{gate_vmin:.3f}", f"{mid:.3f}", f"{gate_vmax:.3f}"]
                    )

                # ── Row 2: Gate × QK Gated ───────────────────
                else:
                    raw_attn = attn[h, 0, 1:].numpy()
                    gate_h   = gate[1:, h, :].mean(axis=-1).numpy()
                    gate_n   = gate_h / (gate_h.max() + 1e-8)
                    raw      = raw_attn * gate_n
                    thr      = np.percentile(raw, percentile)
                    data     = np.where(raw >= thr, raw, 0.0)
                    up       = _resize_map(_to_2d(data, num_patches), orig.size)

                    ax.imshow(orig, interpolation="bilinear")
                    ax.imshow(up, cmap="jet", alpha=0.65,
                              vmin=0, vmax=1, interpolation="bilinear")

        save_path = os.path.join(save_dir, f"layer_{layer_idx:02d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"저장: {save_path}")
