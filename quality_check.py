from PIL import Image
import numpy as np

def check_white_blob(image_path: str, threshold: float = 0.15) -> bool:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=float)
    white_mask = (arr[:,:,0] > 240) & (arr[:,:,1] > 240) & (arr[:,:,2] > 240)
    white_ratio = white_mask.sum() / white_mask.size
    if white_ratio > threshold:
        print(f"  ⚠️ 白塗り検出: {white_ratio:.1%} ({image_path})")
        return False
    return True

def check_image_valid(image_path: str) -> bool:
    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=float)
        if arr.mean() / 255.0 < 0.02:
            print(f"  ⚠️ 黒画像検出: {image_path}")
            return False
        if not check_white_blob(image_path):
            return False
        return True
    except Exception as e:
        print(f"  ⚠️ 画像チェックエラー: {e}")
        return False
