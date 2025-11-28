"""Visualize Model Testing Results for Error Analysis

This script loads a trained HMER model and generates comprehensive visualizations
of test predictions for error analysis, including:
- Input images
- Encoder feature heatmaps
- Predicted LaTeX with token-type analysis
"""

from models.token_categories import build_token_categories_from_dict, create_type_mapping_from_categories
from utils import build_vocab_from_dict
from datamodule.dataset import HMEDataset
from models import HMERWithAuxiliary
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
# Add project root to path BEFORE importing local modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Now import everything else

# Import local modules


# Type class names for visualization
TYPE_CLASS_NAMES = {
    0: 'IGNORE',
    1: 'Digit',
    2: 'Letter',
    3: 'Greek',
    4: 'Operator',
    5: 'Function',
    6: 'Delimiter',
    7: 'Relation',
    8: 'Command/Special'
}

# Color mapping for types
TYPE_COLORS = {
    0: '#CCCCCC',  # Gray
    1: '#FF6B6B',  # Red
    2: '#4ECDC4',  # Cyan
    3: '#FFE66D',  # Yellow
    4: '#95E1D3',  # Light green
    5: '#F38181',  # Pink
    6: '#AA96DA',  # Purple
    7: '#FCBAD3',  # Light pink
    8: '#A8E6CF',  # Mint green
}


def load_model(checkpoint_path: str, vocab_obj: Dict, device: str = 'cuda') -> HMERWithAuxiliary:
    """Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_obj: Vocabulary object
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config if not in checkpoint
        config = {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'ffn_dim': 2048,
            'dropout': 0.1,
            'max_len': 512,
            'num_soft_prompts': 0,
            'aux_type_loss_weight': 0.1,
            'aux_depth_loss_weight': 0.1,
            'aux_rel_loss_weight': 0.1,
            'coverage_loss_weight': 0.01,
            'use_aux_type': True,
            'use_aux_depth': False,
            'use_aux_rel': False,
        }

    # Initialize model
    model = HMERWithAuxiliary(config, vocab_obj)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"[INFO] Loaded model from {checkpoint_path}")
    return model


def tokens_to_string(token_ids: List[int], id_to_token: Dict[int, str]) -> str:
    """Convert token IDs to LaTeX string

    Args:
        token_ids: List of token IDs
        id_to_token: Mapping from ID to token string

    Returns:
        LaTeX string
    """
    tokens = [id_to_token.get(tid, '<unk>') for tid in token_ids]
    return ' '.join(tokens)


def get_encoder_heatmap(model: HMERWithAuxiliary, image: torch.Tensor) -> np.ndarray:
    """Extract and visualize encoder feature map

    Args:
        model: HMER model
        image: Input image tensor (1, 3, H, W)

    Returns:
        Heatmap as numpy array (H, W)
    """
    with torch.no_grad():
        # Get encoder output
        memory, encoder_output = model.image_encoder(
            image)  # memory: (1, num_patches, d_model)

        # Average across feature dimension
        if memory.dim() == 3:  # (B, N, D)
            feature_map = memory[0].mean(dim=-1)  # (N,)

            # Reshape to 2D grid (assuming square patches)
            num_patches = feature_map.size(0)
            grid_size = int(np.sqrt(num_patches))

            if grid_size * grid_size == num_patches:
                heatmap = feature_map.view(grid_size, grid_size).cpu().numpy()
            else:
                # Fallback: create a simple heatmap
                heatmap = np.ones((14, 14)) * feature_map.mean().item()
        else:
            # Fallback
            heatmap = np.ones((14, 14)) * 0.5

    return heatmap


def predict_with_types(model: HMERWithAuxiliary, image: torch.Tensor,
                       id_to_type: Dict[int, int]) -> Tuple[List[int], List[int]]:
    """Generate prediction with type predictions

    Args:
        model: HMER model
        image: Input image tensor (1, 3, H, W)
        id_to_type: Mapping from token ID to type class

    Returns:
        Tuple of (predicted_token_ids, predicted_type_ids)
    """
    with torch.no_grad():
        # Generate sequence
        predicted_ids = model.generate(image, max_len=200, beam_size=1)[0]

        # Get type predictions if available
        if not predicted_ids:
            return [], []

        # Run forward pass to get type logits
        device = image.device
        bos_id = model.bos_id

        # Build input sequence with BOS
        input_ids = torch.tensor(
            [[bos_id] + predicted_ids], dtype=torch.long, device=device)

        # Forward pass
        outputs = model.forward(image, input_ids)

        # Extract type predictions
        predicted_types = []
        if 'type_logits' in outputs and outputs['type_logits'] is not None:
            type_logits = outputs['type_logits']  # (B, T, num_types)
            type_preds = type_logits.argmax(dim=-1)[0].cpu().tolist()  # (T,)
            predicted_types = type_preds[1:]  # Remove BOS position
        else:
            # Fallback: use ground truth type mapping
            predicted_types = [id_to_type.get(tid, 0) for tid in predicted_ids]

    return predicted_ids, predicted_types


def visualize_sample(image_np: np.ndarray, heatmap: np.ndarray,
                     predicted_latex: str, predicted_tokens: List[str],
                     predicted_types: List[int], ground_truth_latex: str,
                     output_path: str):
    """Create visualization figure for one sample

    Args:
        image_np: Original image as numpy array
        heatmap: Encoder feature heatmap
        predicted_latex: Predicted LaTeX string
        predicted_tokens: List of predicted tokens
        predicted_types: List of type predictions
        ground_truth_latex: Ground truth LaTeX
        output_path: Path to save figure
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # 1. Input Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_np, cmap='gray')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Encoder Heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(heatmap, cmap='jet', interpolation='bilinear')
    ax2.set_title('Encoder Feature Heatmap\n(Attention Focus)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Prediction Analysis
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    # Title
    ax3.text(0.5, 0.98, 'Prediction Analysis',
             ha='center', va='top', fontsize=14, fontweight='bold',
             transform=ax3.transAxes)

    # Predicted LaTeX
    ax3.text(0.05, 0.88, 'Predicted LaTeX:',
             ha='left', va='top', fontsize=11, fontweight='bold',
             transform=ax3.transAxes)

    # Wrap long LaTeX strings
    if len(predicted_latex) > 80:
        latex_wrapped = '\n'.join([predicted_latex[i:i+80]
                                  for i in range(0, len(predicted_latex), 80)])
    else:
        latex_wrapped = predicted_latex

    ax3.text(0.05, 0.82, latex_wrapped,
             ha='left', va='top', fontsize=9, family='monospace',
             transform=ax3.transAxes, wrap=True)

    # Ground Truth LaTeX
    y_pos = 0.82 - 0.05 * (1 + len(predicted_latex) // 80)
    ax3.text(0.05, y_pos, 'Ground Truth:',
             ha='left', va='top', fontsize=11, fontweight='bold',
             transform=ax3.transAxes)

    if len(ground_truth_latex) > 80:
        gt_wrapped = '\n'.join([ground_truth_latex[i:i+80]
                               for i in range(0, len(ground_truth_latex), 80)])
    else:
        gt_wrapped = ground_truth_latex

    y_pos -= 0.06
    ax3.text(0.05, y_pos, gt_wrapped,
             ha='left', va='top', fontsize=9, family='monospace',
             transform=ax3.transAxes, wrap=True, color='green')

    # Token-Type Table
    y_pos -= 0.06 * (1 + len(ground_truth_latex) // 80) + 0.05
    ax3.text(0.05, y_pos, 'Token-Type Table:',
             ha='left', va='top', fontsize=11, fontweight='bold',
             transform=ax3.transAxes)

    # Create table data
    table_data = []
    for i, (token, type_id) in enumerate(zip(predicted_tokens, predicted_types)):
        type_name = TYPE_CLASS_NAMES.get(type_id, 'Unknown')
        table_data.append([token, type_name])
        if i >= 19:  # Limit to 20 tokens for display
            table_data.append(['...', '...'])
            break

    if table_data:
        y_pos -= 0.06
        # Create simple table
        table = ax3.table(cellText=table_data,
                          colLabels=['Token', 'Type'],
                          cellLoc='left',
                          colWidths=[0.3, 0.4],
                          loc='upper left',
                          bbox=[0.05, y_pos - 0.4, 0.9, 0.35])

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Color code by type
        for i, (token, type_id) in enumerate(zip(predicted_tokens[:20], predicted_types[:20])):
            color = TYPE_COLORS.get(type_id, '#FFFFFF')
            table[(i+1, 1)].set_facecolor(color)

    # Add legend for type colors at bottom
    legend_elements = [mpatches.Patch(facecolor=TYPE_COLORS[i], label=TYPE_CLASS_NAMES[i])
                       for i in sorted(TYPE_CLASS_NAMES.keys()) if i > 0]
    ax3.legend(handles=legend_elements, loc='lower left', fontsize=7, ncol=2,
               framealpha=0.9)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HMER Test Results for Error Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Path to test data folder')
    parser.add_argument('--dict', type=str, default='data/CROHME/dictionary.txt',
                        help='Path to dictionary file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = 'cpu'

    print(f"[INFO] Using device: {device}")

    # Load vocabulary
    print(f"[INFO] Loading vocabulary from {args.dict}")
    vocab_obj = build_vocab_from_dict(args.dict)
    id_to_token = vocab_obj['id_to_token']
    token_to_id = vocab_obj['token_to_id']

    # Build token categories for type prediction
    print("[INFO] Building token categories...")
    categories = build_token_categories_from_dict(args.dict)
    id_to_type = create_type_mapping_from_categories(categories, token_to_id)

    # Load model
    print(f"[INFO] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, vocab_obj, device)

    # Load test dataset
    print(f"[INFO] Loading test dataset from {args.test_path}")
    test_dataset = HMEDataset(
        source_path=Path(args.test_path),
        vocab_obj=vocab_obj,
        use_augmentation=False,
        max_image_size=512
    )

    print(f"[INFO] Test dataset size: {len(test_dataset)}")

    # Select random samples
    num_samples = min(args.num_samples, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)

    print(f"[INFO] Processing {num_samples} random samples...")

    # Process each sample
    for i, idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{num_samples}] Processing sample {idx}...")

        # Get sample from dataset
        sample = test_dataset[idx]
        image_tensor = sample['image'].unsqueeze(0).to(device)  # (1, 3, H, W)
        ground_truth_latex = sample['latex']

        # Get original image for visualization (convert from tensor)
        image_np = sample['image'].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        image_np = (image_np * 255).astype(np.uint8)
        # Convert to grayscale for display
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Get encoder heatmap
        print(f"  - Extracting encoder heatmap...")
        heatmap = get_encoder_heatmap(model, image_tensor)

        # Generate prediction with types
        print(f"  - Generating prediction...")
        predicted_ids, predicted_types = predict_with_types(
            model, image_tensor, id_to_type)

        # Convert to LaTeX string
        predicted_latex = tokens_to_string(predicted_ids, id_to_token)
        predicted_tokens = [id_to_token.get(
            tid, '<unk>') for tid in predicted_ids]

        print(f"  - Predicted: {predicted_latex[:100]}...")
        print(f"  - Ground Truth: {ground_truth_latex[:100]}...")

        # Create visualization
        output_path = output_dir / f"sample_{idx:04d}.png"
        print(f"  - Creating visualization...")
        visualize_sample(
            image_np=image_np,
            heatmap=heatmap,
            predicted_latex=predicted_latex,
            predicted_tokens=predicted_tokens,
            predicted_types=predicted_types,
            ground_truth_latex=ground_truth_latex,
            output_path=str(output_path)
        )

    print(f"\n[SUCCESS] All visualizations saved to {output_dir}/")
    print(f"[INFO] Generated {num_samples} visualization(s)")


if __name__ == '__main__':
    main()
