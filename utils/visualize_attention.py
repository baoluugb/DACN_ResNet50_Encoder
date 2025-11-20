"""Attention Map Visualization

Loads a random sample from the training dataset and visualizes:
- Original image
- Cross-attention heatmap from the last decoder layer

Usage examples:
    python utils/visualize_attention.py --dataset CROHME --checkpoint checkpoints/CROHME_final.pt --out attention_vis.png
    python utils/visualize_attention.py --dataset HME100k --checkpoint checkpoints/HME100K_best.pt --out attention_vis.png --layer -1 --head 0
"""
from datamodule.dataset import HMEDataset
from models.model import HMERWithAuxiliary
from utils import build_vocab_from_dict
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        # try common keys
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def get_random_sample_from_dataset(dataset_name: str, dict_path: Path):
    """Load a random sample from the training dataset

    Returns:
        sample: Dict with 'image' (padded square tensor, NOT resized) and 'target_ids'
        dataset: The dataset object
    """
    if dataset_name == 'CROHME':
        data_dir = Path('data/CROHME')
        dict_file = data_dir / 'dictionary.txt'
    elif dataset_name == 'HME100k':
        data_dir = Path('data/HME100k')
        dict_file = data_dir / 'dictionary.txt'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Use provided dict_path if available, otherwise use default
    if dict_path is None:
        dict_path = dict_file

    # Load dataset (training split)
    dataset = HMEDataset(
        source_path=data_dir / 'train',
        vocab_obj=build_vocab_from_dict(dict_file),
        use_augmentation=False,
        max_image_size=512  # Match training config
    )
    # Pick random sample
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    # Note: sample['image'] is padded to square (NOT resized - preserves original quality)
    # This is the exact image that the encoder sees

    return sample, dataset


def plot_image_and_attention(image: torch.Tensor, attn: np.ndarray, out_path: Path,
                             hw: tuple, predicted_tokens: str = None, target_tokens: str = None):
    """Plot padded image with overlaid attention heatmap and separate heatmap

    Args:
        image: Padded square image tensor (C, H, W) - NOT resized, only padded
        attn: Cross-attention map (T, S) where S = H_p * W_p (encoder spatial size)
        out_path: Output file path
        hw: Tuple of (H_p, W_p) encoder spatial dimensions (e.g., (14, 14) = 196 patches)
    """
    from scipy.ndimage import zoom

    fig = plt.figure(figsize=(18, 6))

    # Prepare padded image (only padded to square, NOT resized - preserves quality)
    img_np = image.cpu().numpy()
    if img_np.shape[0] == 1:  # Grayscale
        img_display = img_np[0]
        is_grayscale = True
    elif img_np.shape[0] == 3:  # RGB
        img_display = np.transpose(img_np, (1, 2, 0))
        # Denormalize
        img_display = (img_display - img_display.min()) / \
            (img_display.max() - img_display.min() + 1e-8)
        is_grayscale = False

    # Image is square (max_side × max_side) due to padding
    H_img, W_img = img_display.shape[:2]

    # Average attention over decoder time steps to get (S,) spatial attention
    attn_spatial = attn.mean(axis=0)  # (S,) - average over time

    # Reshape to 2D grid (H_p, W_p) - assuming encoder outputs sqrt(S) x sqrt(S) grid
    H_p, W_p = hw
    if attn_spatial.shape[0] == H_p * W_p:
        attn_2d = attn_spatial.reshape(H_p, W_p)
    else:
        # Fallback: try to make it square
        S = attn_spatial.shape[0]
        side = int(np.sqrt(S))
        attn_2d = attn_spatial[:side*side].reshape(side, side)
        H_p, W_p = side, side

    # Resize attention map to match image size
    zoom_factors = (H_img / H_p, W_img / W_p)
    attn_resized = zoom(attn_2d, zoom_factors, order=1)

    # Normalize attention for better visualization
    attn_resized = (attn_resized - attn_resized.min()) / \
        (attn_resized.max() - attn_resized.min() + 1e-8)

    # Create subplots: original, overlay, heatmap
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot 1: Padded image (original quality preserved, only white padding added)
    if is_grayscale:
        ax1.imshow(img_display, cmap='gray')
    else:
        ax1.imshow(img_display)
    ax1.set_title('Padded Image (No Resize)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Plot 2: Overlay attention on image
    if is_grayscale:
        ax2.imshow(img_display, cmap='gray', alpha=1.0)
    else:
        ax2.imshow(img_display, alpha=1.0)

    im_overlay = ax2.imshow(attn_resized, cmap='jet',
                            alpha=0.6, interpolation='bilinear')
    ax2.set_title('Attention Overlay', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Plot 3: Pure attention heatmap (per time step)
    im_heatmap = ax3.imshow(attn, aspect='auto',
                            cmap='hot', interpolation='nearest')
    ax3.set_title('Cross-Attention (T × S)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Encoder Positions (S)', fontsize=11)
    ax3.set_ylabel('Decoder Time Steps (T)', fontsize=11)
    plt.colorbar(im_heatmap, ax=ax3, fraction=0.046, pad=0.04)

    # Add token info if available
    if target_tokens or predicted_tokens:
        info_text = ""
        if target_tokens:
            info_text += f"Target: {target_tokens}\n"
        if predicted_tokens:
            info_text += f"Predicted: {predicted_tokens}"
        plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention maps from HMER model')
    parser.add_argument('--dataset', type=str, default='CROHME', choices=['CROHME', 'HME100k'],
                        help='Dataset to sample from')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dict', type=str, default=None,
                        help='Path to dictionary.txt (auto-detected if not provided)')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--out', type=str, default='utils/attention_visualization.png',
                        help='Output image path for visualization')
    parser.add_argument('--layer', type=str, default='last',
                        help='Layer index or "last"')
    parser.add_argument('--head', type=str, default='mean',
                        help='Head index (int) or "mean"')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load config and vocabulary
    cfg = load_config(Path(args.config))

    # Get random sample from dataset
    print(f"Loading random sample from {args.dataset} training dataset...")
    sample, dataset = get_random_sample_from_dataset(
        args.dataset, Path(args.dict) if args.dict else None)
    vocab = build_vocab_from_dict(Path('data/CROHME/dictionary.txt'))
    # vocab = dataset.vocab
    image = sample['image']  # (C, H, W)
    target_ids = sample['target_ids']  # (T,)

    print(f"  - Image shape: {image.shape}")
    print(f"  - Target length: {len(target_ids)}")

    # Build model config (matching new DenseNet encoder architecture)
    model_cfg = {
        'd_model': cfg.get('d_model', 512),
        'num_heads': cfg.get('num_heads', 8),
        'num_layers': cfg.get('num_layers', 6),
        'ffn_dim': cfg.get('ffn_dim', 2048),
        'dropout': cfg.get('dropout', 0.1),
        'max_len': cfg.get('max_len', 512),
        'num_soft_prompts': cfg.get('num_soft_prompts', 0),
        'enable_type_head': cfg.get('enable_type_head', True),
        'enable_depth_head': cfg.get('enable_depth_head', True),
        'enable_rel_head': cfg.get('enable_rel_head', True),
        # Encoder config (new DenseNet parameters)
        'encoder_growth_rate': cfg.get('encoder_growth_rate', 32),
        'encoder_block_config': cfg.get('encoder_block_config', [6, 12, 24, 16]),
        'encoder_num_init_features': cfg.get('encoder_num_init_features', 64),
        'encoder_pos_max_h': cfg.get('encoder_pos_max_h', 64),
        'encoder_pos_max_w': cfg.get('encoder_pos_max_w', 256),
    }

    print("Initializing model...")
    model = HMERWithAuxiliary(model_cfg, vocab).to(device)
    model.eval()

    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, Path(args.checkpoint), device)

    # Prepare batch
    img_batch = image.unsqueeze(0).to(device)  # (1, C, H, W)
    tgt_batch = target_ids.unsqueeze(0).to(device)  # (1, T)

    # Get encoder memory
    print("Running encoder...")
    with torch.no_grad():
        memory, hw = model.image_encoder(img_batch)

    # Prepare decoder input (teacher forcing with ground truth)
    pad_id = model.pad_id
    decoder_input = tgt_batch[:, :-1]  # Exclude last token for teacher forcing

    # Run decoder with return_attn=True
    print("Running decoder with attention extraction...")
    with torch.no_grad():
        out = model.decoder(tgt_ids=decoder_input,
                            memory=memory, pad_id=pad_id, return_attn=True)

    attn_maps = out.get('attn_maps', None)
    if attn_maps is None or len(attn_maps) == 0:
        raise RuntimeError(
            'No attention maps returned. Ensure decoder supports return_attn=True')

    # Select layer
    if args.layer == 'last':
        layer_idx = -1
    else:
        layer_idx = int(args.layer)

    layer_attn = attn_maps[layer_idx]  # (B, H, T, S)
    layer_attn = layer_attn[0].cpu().numpy()  # (H, T, S)

    # Select head
    if args.head == 'mean':
        attn = layer_attn.mean(axis=0)  # (T, S)
        head_str = 'Mean'
    else:
        head_idx = int(args.head)
        attn = layer_attn[head_idx]  # (T, S)
        head_str = f'Head {head_idx}'

    # Get tokens for display
    from utils import tokens_to_string
    target_str = tokens_to_string(
        target_ids.cpu(), vocab['id_to_token'], vocab['special'])

    # Get prediction
    logits = out['logits']
    pred_ids = logits.argmax(dim=-1)[0].cpu()
    pred_str = tokens_to_string(
        pred_ids, vocab['id_to_token'], vocab['special'])

    print(f"\nTarget:    {target_str}")
    print(f"Predicted: {pred_str}")
    print(
        f"\nAttention shape: {attn.shape} (T={attn.shape[0]}, S={attn.shape[1]})")
    print(f"Layer: {layer_idx}, Head: {head_str}")

    # Plot and save
    out_path = Path(args.out)
    plot_image_and_attention(image, attn, out_path, hw,
                             predicted_tokens=pred_str,
                             target_tokens=target_str)


if __name__ == '__main__':
    main()
