"""
실행 진입점

사용법:
    # 파일 경로로
    run(model, image_path="test.jpg")

    # 허깅페이스 PIL 이미지로
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet")
    pil_img = ds['valid'][0]['image']
    run(model, pil_img=pil_img)
"""
from .preprocess import preprocess, preprocess_pil
from .hooks      import register_hooks, remove_hooks, collect_gates
from .visualize  import visualize
from .analyze    import analyze
import torch

def run(
    model,
    image_path: str  = None,
    pil_img          = None,
    save_dir: str    = "outputs",
    device: str      = None,
    patch_size: int  = 14,
    img_size: int    = 224,
    top_percent: float = 10.0,
):
    """
    Args:
        model      : 로드된 dinosplus_classfier
        image_path : 파일 경로 (image_path 또는 pil_img 중 하나)
        pil_img    : PIL Image (허깅페이스 데이터셋에서 꺼낸 이미지)
        save_dir   : 결과 저장 폴더
        device     : 'cuda' / 'cpu' (None이면 자동)
        patch_size : 14
        img_size   : 224
        top_percent: attention threshold (상위 몇 %)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # ── 전처리
    if pil_img is not None:
        img_tensor, original_img = preprocess_pil(pil_img, img_size)
    elif image_path is not None:
        img_tensor, original_img = preprocess(image_path, img_size)
    else:
        raise ValueError("image_path 또는 pil_img 중 하나는 필요합니다.")

    img_tensor = img_tensor.to(device)

    # ── Hook 등록
    qk_maps, hooks = register_hooks(model)

    # ── Forward
    with torch.no_grad():
        _ = model(img_tensor)

    # ── Hook 제거
    remove_hooks(hooks)

    # ── Gate 수집
    gate_maps = collect_gates(model)

    print(f"QK maps  : {len(qk_maps)}개 레이어")
    print(f"Gate maps: {len(gate_maps)}개 레이어")

    # ── 시각화
    visualize(
        qk_maps, gate_maps, original_img,
        save_dir    = save_dir,
        patch_size  = patch_size,
        img_size    = img_size,
        top_percent = top_percent,
    )

    # ── 통계
    results = analyze(qk_maps, gate_maps)

    return qk_maps, gate_maps, results


# ── 직접 실행할 때 ──────────────────────────────────────────
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    import mlflow
    from src.re_make_Gate import inject_gating, GatedOutputProjection

    class dinosplus_classfier(nn.Module):
        def __init__(self, model, num):
            super().__init__()
            self.backbone = model
            clsdim = self.backbone.config.hidden_size
            self.fc = nn.Sequential(
                nn.Linear(clsdim, 512),
                nn.ReLU(),
                nn.Linear(512, num),
            )
        def forward(self, x):
            output = self.backbone(pixel_values=x)
            cls    = output.last_hidden_state[:, 0]
            return self.fc(cls)

    # ── 모델 로드
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    run_id = "여기에_run_id"
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "best dino v3 splus gated last ONLY.pth", ".")

    backbone = AutoModel.from_pretrained("facebook/dinov2-small")
    backbone, hooks = inject_gating(backbone, gate_type="elementwise", keep_cls_ungated=True, target_layers=None)
    model = dinosplus_classfier(backbone, num=200)
    model.load_state_dict(torch.load(local_path))

    # ── 실행 (파일 경로)
    run(model, image_path="test.jpg", save_dir="outputs")

    # ── 실행 (허깅페이스 PIL)
    # from datasets import load_dataset
    # ds = load_dataset("zh-plus/tiny-imagenet")
    # run(model, pil_img=ds['valid'][0]['image'], save_dir="outputs")
