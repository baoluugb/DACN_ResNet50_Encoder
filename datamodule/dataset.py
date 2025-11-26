import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .image_aug import ImageAugmentation


def pad_to_square(img: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Pad image to square, with optional downsizing if too large

    If image max dimension > max_size, resize to max_size while preserving aspect ratio.
    Then pad to square. This prevents extreme memory usage from very large images.

    Args:
        img: Input image
        max_size: Maximum allowed dimension (default 512)
    """
    h, w = img.shape[:2]

    # If image is too large, resize it first (preserving aspect ratio)
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Now pad to square
    max_side = max(h, w)

    # Calculate padding to center the image in a square
    pad_h = (max_side - h) // 2
    pad_w = (max_side - w) // 2

    # Pad to square (black background to match dataset images)
    img_padded = cv2.copyMakeBorder(
        img,
        pad_h, max_side - h - pad_h,
        pad_w, max_side - w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0  # Black padding (matches dataset background)
    )

    return img_padded


def split_caption(line: str) -> Tuple[str, str]:
    if "\t" in line:
        k, v = line.split("\t", 1)
        return k.strip(), v.strip()
    if "	" in line:
        k, v = line.split("	", 1)
        return k.strip(), v.strip()
    parts = line.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


class HMEDataset(Dataset):
    def __init__(self, source_path: Path, vocab_obj: str, use_augmentation: bool = False, max_image_size: int = 512):
        self.source_path = Path(source_path)
        self.max_image_size = max_image_size  # Maximum allowed image dimension
        self.use_augmentation = use_augmentation

        self.token_to_id: Dict[str, int] = vocab_obj["token_to_id"]
        self.special = vocab_obj["special"]
        self.pad_id = self.token_to_id[self.special["pad"]]
        self.bos_id = self.token_to_id[self.special["bos"]]
        self.eos_id = self.token_to_id[self.special["eos"]]
        self.unk_id = self.token_to_id[self.special["unk"]]

        self.items: List[Dict[str, Any]] = []
        self._images: Dict[str, np.ndarray] = {}

        # Initialize augmentation
        if self.use_augmentation:
            self.augmentation = ImageAugmentation(
                rotation_range=5.0,
                scale_range=(0.9, 1.1),
                shear_range=0.1,
                blur_prob=0.3,
                noise_prob=0.3,
                noise_std=0.05,
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                elastic_prob=0.2,
                elastic_alpha=20.0,
                elastic_sigma=5.0,
            )
            print("[INFO] Data augmentation enabled")
        else:
            self.augmentation = None

        self.load_data(self.source_path)

    def load_data(self, dir_path: Path):
        pkl_path = dir_path / "images.pkl"
        cap_path = dir_path / "caption.txt"
        if not pkl_path.exists() or not cap_path.exists():
            raise FileNotFoundError(f"Expected {pkl_path} and {cap_path}")

        # Load images
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self._images = data
        else:
            self._images = {k: v for k, v in data}

        # Load captions
        for line in Path(cap_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            key, latex = split_caption(line)

            tokens = latex.strip().split()  # Split by whitespace
            if not tokens:  # Empty after splitting, skip
                continue

            self.items.append({"key": key, "latex": latex, "tokens": tokens})

        # Filter out items without corresponding images
        self.items = [it for it in self.items if it["key"] in self._images]

        if not self.items:
            raise ValueError(f"No valid items found in {dir_path}")

        print(f"[INFO] Loaded {len(self.items)} samples from {dir_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        key = rec["key"]
        tokens = rec["tokens"]
        latex = rec["latex"]
        img = self._images[key].copy()

        # Pad to square with max size limit (resizes if > max_size)
        img = pad_to_square(img, max_size=self.max_image_size)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img.astype("float32") / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = img.transpose(2, 0, 1)  # Tạo ra (3, H, W)

        # Tokenize LaTeX: <bos> + tokens + <eos>
        latex_ids = [self.bos_id] + \
            [self.token_to_id.get(t, self.unk_id)
             for t in tokens] + [self.eos_id]

        result = {
            "image": torch.from_numpy(img),  # Bây giờ shape là (3, 224, 224)
            "target_ids": torch.tensor(latex_ids, dtype=torch.long),
            "length": len(latex_ids),
            "path": str(key),
            "tokens": tokens
        }

        return result


def collate_fn(batch: List[Dict[str, Any]], latex_pad_id: int) -> Dict[str, Any]:
    """Collate function for batching samples with variable-size images

    Args:
        batch: List of samples from dataset
        latex_pad_id: Padding ID for LaTeX sequences

    Returns:
        Dictionary with batched tensors
    """
    B = len(batch)

    # Pad images to same size within batch
    # Find max dimensions in this batch
    max_h = max(x["image"].shape[1] for x in batch)
    max_w = max(x["image"].shape[2] for x in batch)

    # Create padded image tensor (B, 3, max_h, max_w)
    images = torch.zeros(B, 3, max_h, max_w, dtype=torch.float32)

    for i, x in enumerate(batch):
        img = x["image"]
        C, H, W = img.shape
        # Place image at top-left corner, rest is zero-padded (black)
        images[i, :, :H, :W] = img

    # Pad LaTeX sequences
    target_lens = [x["target_ids"].shape[0] for x in batch]
    max_L = max(target_lens)
    latex_tgt = torch.full((B, max_L), latex_pad_id, dtype=torch.long)

    paths, tokens_list = [], []
    for i, x in enumerate(batch):
        L = x["target_ids"].shape[0]
        latex_tgt[i, :L] = x["target_ids"]
        paths.append(x["path"])
        tokens_list.append(x["tokens"])

    return {
        "images": images,
        "target_ids": latex_tgt,
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
        "paths": paths,
        "tokens": tokens_list,
    }
