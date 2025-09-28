import os
from typing import Iterable, List, Tuple
from PIL import Image

PATCH_SIZE = 224

def _compute_box(cx: float, cy: float, patch_size: int, w: int, h: int) -> Tuple[int,int,int,int]:
    half = patch_size // 2
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + patch_size
    bottom = top + patch_size
    # clamp
    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right > w:
        shift = right - w
        left -= shift
        right = w
    if bottom > h:
        shift = bottom - h
        top -= shift
        bottom = h
    left = max(0, left); top = max(0, top)
    return left, top, right, bottom

def extract_patches(large_image_path: str,
                    coords: Iterable[Tuple[str,str,float,float]],
                    output_dir: str) -> List[str]:
    """Extract 224x224 RGB patches centered on (x_pixel,y_pixel).

    coords: iterable of (barcode, wsi_id, x, y)
    Returns list of written patch file paths.
    """
    if not os.path.exists(large_image_path):
        raise FileNotFoundError(f"Large image not found: {large_image_path}")
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(large_image_path).convert('RGB')
    W,H = img.size
    written = []
    for barcode, wsi_id, x, y in coords:
        patch_name = f"{barcode}_{wsi_id}.png"
        out_path = os.path.join(output_dir, patch_name)
        if os.path.exists(out_path):
            written.append(out_path)
            continue
        box = _compute_box(x, y, PATCH_SIZE, W, H)
        patch = img.crop(box)
        if patch.size != (PATCH_SIZE, PATCH_SIZE):
            # pad if needed (rare edge case)
            pad_img = Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (0,0,0))
            pad_img.paste(patch, (0,0))
            patch = pad_img
        patch.save(out_path)
        written.append(out_path)
    return written
