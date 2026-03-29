from torchvision import transforms
from PIL import Image


def preprocess(image_path: str, img_size: int = 224):
    """
    학습때랑 동일한 전처리 (Resize + ToTensor만)
    Returns:
        img_tensor: (1, 3, H, W)
        original_img: PIL Image
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0), img


def preprocess_pil(pil_img: Image.Image, img_size: int = 224):
    """
    허깅페이스 데이터셋에서 꺼낸 PIL 이미지 바로 전처리
    Returns:
        img_tensor: (1, 3, H, W)
        original_img: PIL Image (원본)
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = pil_img.convert("RGB")
    return tf(img).unsqueeze(0), img
