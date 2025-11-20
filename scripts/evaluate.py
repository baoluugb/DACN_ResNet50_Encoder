"""Evaluation script for HMER with Auxiliary Tasks

Evaluate model on test set and compute metrics.
"""

# Add project root to path BEFORE any local imports
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import json
import argparse
from datamodule import HMEDataset, collate_fn
from models import HMERWithAuxiliary
from utils import build_vocab_from_dict, tokens_to_string
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Now import local modules

# Standard library imports


def compute_exact_match(pred_seq, target_seq):
    """Compute exact match between sequences"""
    return pred_seq == target_seq


def compute_edit_distance(pred_seq, target_seq):
    """Compute Levenshtein edit distance"""
    m, n = len(pred_seq), len(target_seq)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_seq[i-1] == target_seq[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )

    return dp[m][n]


def evaluate(
    model,
    dataloader,
    device,
    id_to_token,
    special,
    max_len=200,
    beam_size=1,
    save_results=None
):
    """Evaluate model on dataset

    Args:
        model: HMERWithAuxiliary model
        dataloader: DataLoader for test set
        device: torch device
        id_to_token: Token ID to string mapping
        special: Special tokens dict
        max_len: Maximum generation length
        beam_size: Beam size for decoding
        save_results: Path to save detailed results (optional)

    Returns:
        Dictionary with metrics
    """
    model.eval()

    total_samples = 0
    exact_matches = 0
    exprate_1 = 0  # Edit distance <= 1
    exprate_2 = 0  # Edit distance <= 2
    exprate_3 = 0  # Edit distance <= 3
    total_edit_distance = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["images"].to(device)
            target_ids = batch["target_ids"]

            # Generate predictions
            predictions = model.generate(
                images,
                max_len=max_len,
                beam_size=beam_size
            )

            # Process each sample in batch
            for i in range(len(predictions)):
                pred_seq = predictions[i]
                target_seq = target_ids[i].tolist()

                # Remove BOS and truncate at EOS in target
                if model.bos_id in target_seq:
                    target_seq.remove(model.bos_id)
                if model.eos_id in target_seq:
                    eos_idx = target_seq.index(model.eos_id)
                    target_seq = target_seq[:eos_idx]
                # Remove PAD
                target_seq = [t for t in target_seq if t != model.pad_id]

                # Compute metrics
                edit_dist = compute_edit_distance(pred_seq, target_seq)

                # Accumulate
                total_samples += 1
                total_edit_distance += edit_dist

                # ExpRate metrics
                if edit_dist == 0:
                    exact_matches += 1
                    exprate_1 += 1
                    exprate_2 += 1
                    exprate_3 += 1
                elif edit_dist <= 1:
                    exprate_1 += 1
                    exprate_2 += 1
                    exprate_3 += 1
                elif edit_dist <= 2:
                    exprate_2 += 1
                    exprate_3 += 1
                elif edit_dist <= 3:
                    exprate_3 += 1

    # Compute metrics
    metrics = {
        "total_samples": total_samples,
        "exact_match": exact_matches,
        "ExpRate": exact_matches / total_samples if total_samples > 0 else 0,
        "ExpRate_≤1": exprate_1 / total_samples if total_samples > 0 else 0,
        "ExpRate_≤2": exprate_2 / total_samples if total_samples > 0 else 0,
        "ExpRate_≤3": exprate_3 / total_samples if total_samples > 0 else 0,
        "average_edit_distance": total_edit_distance / total_samples if total_samples > 0 else 0
    }

    # Save only metrics (not individual predictions)
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {save_results}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='HMER Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--dict', type=str, required=True,
                        help='Path to dictionary file')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size for decoding')
    parser.add_argument('--max-len', type=int, default=200,
                        help='Maximum generation length')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save detailed results (JSON)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"Loading vocabulary from {args.dict}")
    vocab = build_vocab_from_dict(args.dict)
    id_to_token = vocab["id_to_token"]
    special = vocab["special"]
    pad_id = vocab["token_to_id"][special["pad"]]

    # Load dataset
    print(f"Loading test dataset from {args.test_path}")
    test_dataset = HMEDataset(
        source_path=args.test_path,
        vocab_obj=vocab,
        use_augmentation=False,
        max_image_size=512  # Match training config
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=4
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config
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
            "enable_rel_head": cfg.get('enable_rel_head', True),
            # Encoder config (ResNet backbone parameters)
            "encoder_backbone": cfg.get('encoder_backbone', 'resnet50'),
            "encoder_pos_max_h": cfg.get('encoder_pos_max_h', 64),
            "encoder_pos_max_w": cfg.get('encoder_pos_max_w', 256),
        }
    else:
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
            "enable_rel_head": True,
            # Encoder defaults for ResNet
            "encoder_backbone": "resnet50",
            "encoder_pos_max_h": 64,
            "encoder_pos_max_w": 256,
        }

    # Create and load model
    print("Building model...")
    model = HMERWithAuxiliary(model_config, vocab)

    # Load weights with strict=False to handle encoder architecture changes
    # (Old checkpoints may have incompatible encoder architectures, new model uses ResNet)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False)

    if missing_keys:
        print(
            f"Warning: Missing keys (ResNet encoder initialization): {len(missing_keys)} keys")
        print(
            "  This may occur when loading checkpoints with different encoder architectures")
    if unexpected_keys:
        print(
            f"Warning: Unexpected keys (old encoder incompatible): {len(unexpected_keys)} keys")

    model = model.to(device)

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Evaluate
    print(f"\nEvaluating with beam_size={args.beam_size}...")
    metrics = evaluate(
        model,
        test_loader,
        device,
        id_to_token,
        special,
        max_len=args.max_len,
        beam_size=args.beam_size,
        save_results=args.save_results
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples:         {metrics['total_samples']}")
    print(f"Exact matches:         {metrics['exact_match']}")
    print(f"ExpRate (exact):       {metrics['ExpRate']*100:.2f}%")
    print(f"ExpRate ≤1:            {metrics['ExpRate_≤1']*100:.2f}%")
    print(f"ExpRate ≤2:            {metrics['ExpRate_≤2']*100:.2f}%")
    print(f"ExpRate ≤3:            {metrics['ExpRate_≤3']*100:.2f}%")
    print(f"Avg edit distance:     {metrics['average_edit_distance']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
