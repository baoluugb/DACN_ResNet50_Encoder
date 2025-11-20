"""Inference script for HMER with Auxiliary Tasks

Generate LaTeX predictions for test images.
"""

from utils import build_vocab_from_dict, tokens_to_string
from models import HMERWithAuxiliary
import sys
from pathlib import Path
import argparse
import torch
from PIL import Image
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_image(image_path: str, target_size: int = 224) -> torch.Tensor:
    """Load and preprocess image

    Args:
        image_path: Path to image file
        target_size: Target size (224)

    Returns:
        Image tensor (1, 3, 224, 224)
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize with padding (same as training)
    h, w = img.shape[:2]
    max_side = max(h, w)
    scale = target_size / max_side
    new_h, new_w = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)

    # Normalize
    img_normalized = img_rgb.astype('float32') / 255.0

    # Transpose to (C, H, W)
    img_chw = img_normalized.transpose(2, 0, 1)

    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

    return img_tensor


def main():
    parser = argparse.ArgumentParser(description='HMER Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--dict', type=str, default='data/CROHME/dictionary.txt',
                        help='Path to dictionary file')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size for decoding (1=greedy)')
    parser.add_argument('--max-len', type=int, default=200,
                        help='Maximum generation length')
    parser.add_argument('--length-penalty', type=float, default=1.0,
                        help='Length penalty for beam search')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"Loading vocabulary from {args.dict}")
    vocab = build_vocab_from_dict(args.dict)
    id_to_token = vocab["id_to_token"]
    special = vocab["special"]

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        model_config = {
            "d_model": cfg.get('d_model', 512),
            "num_heads": cfg.get('num_heads', 8),
            "num_layers": cfg.get('num_layers', 6),
            "ffn_dim": cfg.get('ffn_dim', 2048),
            "dropout": cfg.get('dropout', 0.1),
            "max_len": cfg.get('max_len', 512),
            "num_soft_prompts": cfg.get('num_soft_prompts', 0),
            "aux_type_loss_weight": cfg.get('aux_type_loss_weight', 0.1),
            "aux_depth_loss_weight": cfg.get('aux_depth_loss_weight', 0.1),
            "aux_rel_loss_weight": cfg.get('aux_rel_loss_weight', 0.1),
            "coverage_loss_weight": cfg.get('coverage_loss_weight', 0.01),
            "enable_type_head": cfg.get('enable_type_head', True),
            "enable_depth_head": cfg.get('enable_depth_head', True),
            "enable_rel_head": cfg.get('enable_rel_head', True)
        }
    else:
        # Default config
        model_config = {
            "d_model": 512,
            "num_heads": 8,
            "num_layers": 6,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "max_len": 512,
            "num_soft_prompts": 8,
            "aux_type_loss_weight": 0.1,
            "aux_depth_loss_weight": 0.1,
            "aux_rel_loss_weight": 0.1,
            "coverage_loss_weight": 0.01,
            "enable_type_head": True,
            "enable_depth_head": True,
            "enable_rel_head": True
        }

    # Create model
    print("Building model...")
    model = HMERWithAuxiliary(model_config, vocab)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Load image
    print(f"Loading image from {args.image}")
    image = load_image(args.image).to(device)

    # Generate prediction
    print(f"Generating prediction (beam_size={args.beam_size})...")
    with torch.no_grad():
        predictions = model.generate(
            image,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty
        )

    # Convert to string
    pred_ids = predictions[0]
    pred_str = tokens_to_string(pred_ids, id_to_token, special)

    # Print result
    print("\n" + "="*80)
    print("PREDICTION:")
    print("="*80)
    print(pred_str)
    print("="*80)
    print(f"\nToken IDs: {pred_ids}")
    print(f"Length: {len(pred_ids)} tokens")

    return pred_str


if __name__ == '__main__':
    main()
