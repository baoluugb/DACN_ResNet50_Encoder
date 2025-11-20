"""Training script for HMER with Auxiliary Tasks

Single-stage encoder-decoder training with:
- Main token prediction
- Auxiliary syntax type classification
- Auxiliary structural depth prediction
- Auxiliary relation type classification
- Coverage regularization
"""

from utils import build_vocab_from_dict, print_trainable_parameters, tokens_to_string
from models.auxiliary_targets import build_type_targets, build_depth_targets, build_relation_targets
from models.model import HMERWithAuxiliary
from datamodule.dataset import HMEDataset, collate_fn
import sys
from pathlib import Path
import pickle
import logging
import yaml
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


warnings.filterwarnings("ignore")


def setup_logging(log_file='train.log'):
    """Setup logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(path='config.yaml'):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def compute_accuracy(logits, targets, pad_id):
    """Compute token-level accuracy ignoring padding"""
    pred_ids = logits.argmax(dim=-1)  # (B, T)
    mask = (targets != pad_id)
    correct = ((pred_ids == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    cfg,
    logger,
    id_to_token
):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_main_loss = 0
    total_type_loss = 0
    total_depth_loss = 0
    total_rel_loss = 0
    total_coverage_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        images = batch["images"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Build auxiliary targets
        aux_targets = {
            "type_targets": build_type_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device),
            "depth_targets": build_depth_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device),
            "rel_targets": build_relation_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device)
        }

        # Prepare batch dict
        batch_dict = {
            "images": images,
            "target_ids": target_ids
        }

        # Forward pass with mixed precision
        optimizer.zero_grad()

        use_amp = cfg.get('use_amp', False)

        if use_amp:

            with autocast(device_type='cuda'):
                loss, loss_dict = model.compute_loss(
                    batch_dict,
                    aux_targets,
                    label_smoothing=cfg.get('label_smoothing', 0.1)
                )

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if cfg.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model.compute_loss(
                batch_dict,
                aux_targets,
                label_smoothing=cfg.get('label_smoothing', 0.1)
            )

            loss.backward()

            # Gradient clipping
            if cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['grad_clip'])

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
        total_main_loss += loss_dict.get('main', 0)
        total_type_loss += loss_dict.get('type', 0)
        total_depth_loss += loss_dict.get('depth', 0)
        total_rel_loss += loss_dict.get('rel', 0)
        total_coverage_loss += loss_dict.get('coverage', 0)

        # Compute accuracy (on main task)
        with torch.no_grad():
            outputs = model(images, target_ids)
            logits = outputs["logits"]
            main_targets = target_ids[:, 1:]
            acc = compute_accuracy(logits, main_targets, model.pad_id)
            total_accuracy += acc

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'main': loss_dict.get('main', 0),
            'acc': acc,
            'lr': optimizer.param_groups[0]['lr']
        })

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_main_loss = total_main_loss / num_batches
    # avg_type_loss = total_type_loss / num_batches
    # avg_depth_loss = total_depth_loss / num_batches
    # avg_rel_loss = total_rel_loss / num_batches
    # avg_coverage_loss = total_coverage_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    logger.info(
        f"Epoch {epoch} - Train Loss: {avg_loss:.4f} "
        f"(main: {avg_main_loss:.4f}) - Acc: {avg_accuracy:.4f}"
    )

    return avg_loss, avg_accuracy


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    epoch,
    cfg,
    logger,
    id_to_token
):
    """Validation"""
    model.eval()

    total_loss = 0
    total_main_loss = 0
    total_accuracy = 0
    num_batches = 0

    # Sample predictions for logging
    sample_predictions = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        images = batch["images"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Build auxiliary targets
        aux_targets = {
            "type_targets": build_type_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device),
            "depth_targets": build_depth_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device),
            "rel_targets": build_relation_targets(
                target_ids,
                id_to_token,
                model.pad_id,
                model.bos_id,
                model.eos_id
            ).to(device)
        }

        batch_dict = {
            "images": images,
            "target_ids": target_ids
        }

        # Compute loss
        loss, loss_dict = model.compute_loss(
            batch_dict,
            aux_targets,
            label_smoothing=0.0  # No smoothing for validation
        )

        total_loss += loss.item()
        total_main_loss += loss_dict.get('main', 0)

        # Compute accuracy
        outputs = model(images, target_ids)
        logits = outputs["logits"]
        main_targets = target_ids[:, 1:]
        acc = compute_accuracy(logits, main_targets, model.pad_id)
        total_accuracy += acc

        num_batches += 1

        # Sample predictions for first batch
        if batch_idx == 0 and len(sample_predictions) < 3:
            # Generate predictions
            pred_ids = model.generate(images[:3], max_len=200, beam_size=1)

            for i in range(min(3, len(pred_ids))):
                target = target_ids[i].tolist()
                target_str = tokens_to_string(
                    target, id_to_token, model.special)
                pred_str = tokens_to_string(
                    pred_ids[i], id_to_token, model.special)

                sample_predictions.append({
                    'target': target_str,
                    'predicted': pred_str
                })

    avg_loss = total_loss / num_batches
    avg_main_loss = total_main_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    logger.info(
        f"Epoch {epoch} - Val Loss: {avg_loss:.4f} "
        f"(main: {avg_main_loss:.4f}) - Acc: {avg_accuracy:.4f}"
    )

    # Log sample predictions
    # if sample_predictions:
    #     logger.info("Sample predictions:")
    #     for i, pred in enumerate(sample_predictions):
    #         logger.info(f"  [{i+1}] Target: {pred['target']}")
    #         logger.info(f"      Pred:   {pred['predicted']}")

    return avg_loss, avg_accuracy


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Train HMER with Auxiliary Tasks')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Setup
    logger = setup_logging('train_aux.log')
    cfg = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {cfg.get('dataset', 'CROHME')}")

    # Paths
    dataset_name = cfg.get('dataset', 'CROHME')
    data_root = Path("data") / dataset_name
    train_path = data_root / "train"
    dict_path = data_root / "dictionary.txt"

    # Build vocabulary
    logger.info("Building vocabulary...")
    latex_vocab = build_vocab_from_dict(dict_path)

    token_to_id = latex_vocab["token_to_id"]
    id_to_token = latex_vocab["id_to_token"]
    special = latex_vocab["special"]

    pad_id = token_to_id[special["pad"]]
    bos_id = token_to_id[special["bos"]]
    eos_id = token_to_id[special["eos"]]

    logger.info(f"Vocabulary size: {len(token_to_id)}")
    logger.info(
        f"Special tokens - PAD: {pad_id}, BOS: {bos_id}, EOS: {eos_id}")

    # Load dataset
    logger.info("Loading dataset...")
    max_image_size = cfg.get('max_image_size', 512)
    logger.info(f"Max image size: {max_image_size}")

    full_dataset = HMEDataset(
        source_path=train_path,
        vocab_obj=latex_vocab,
        use_augmentation=cfg.get('use_aug', False),
        max_image_size=max_image_size
    )

    # Split train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * cfg.get('val_split', 0.1))
    train_size = dataset_size - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get('seed', 42))
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get('batch', 16),
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg.get('num_workers', 4),
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get('batch', 16),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg.get('num_workers', 4)
    )

    # Create model
    logger.info("Building model...")

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
        "encoder_backbone": cfg.get("encoder_backbone", "resnet34"),
        "encoder_pos_max_h": cfg.get("encoder_pos_max_h", 64),
        "encoder_pos_max_w": cfg.get("encoder_pos_max_w", 256)
    }

    model = HMERWithAuxiliary(model_config, latex_vocab).to(device)

    # Print parameter info
    print_trainable_parameters(model)
    if cfg.get('use_amp', False):
        print("[INFO] AMP: True")

    # Optimizer
    lr = cfg.get('lr', 5e-4)
    weight_decay = cfg.get('weight_decay', 0.01)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Learning rate scheduler
    scheduler = None
    if cfg.get('use_scheduler', True):
        scheduler_type = cfg.get('scheduler_type', 'cosine')
        warmup_steps = cfg.get('warmup_steps', 1000)

        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            total_steps = len(train_loader) * cfg.get('epochs', 20)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=cfg.get('min_lr', 1e-6)
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            )

            logger.info(
                f"Using CosineAnnealing scheduler with {warmup_steps} warmup steps")

    # Gradient scaler for mixed precision
    scaler = GradScaler() if cfg.get('use_amp', False) else None

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))

        # Also load scheduler state if it exists
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state if using mixed precision
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")

    # Training loop
    epochs = cfg.get('epochs', 20)
    early_stopping = cfg.get('early_stopping', True)
    patience = cfg.get('early_stopping_patience', 10)

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info(f"Starting training from epoch {start_epoch} to {epochs}...")
    logger.info(
        f"Early stopping: {'Enabled' if early_stopping else 'Disabled'}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, cfg, logger, id_to_token
        )

        val_loss, val_acc = validate(
            model, val_loader, device, epoch, cfg, logger, id_to_token
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': cfg
            }

            # Save scheduler state if exists
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            # Save scaler state if using mixed precision
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint, checkpoint_dir / f'{dataset_name}_best.pt')

            logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if early_stopping:
                logger.info(
                    f"No improvement for {patience_counter}/{patience} epochs")

        # Early stopping (only if enabled)
        if early_stopping and patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

        # Save periodic checkpoint
        # if epoch % 5 == 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'train_loss': train_loss,
        #         'config': cfg
        #     }

        #     if scheduler is not None:
        #         checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        #     if scaler is not None:
        #         checkpoint['scaler_state_dict'] = scaler.state_dict()

        #     torch.save(checkpoint, checkpoint_dir / f'{dataset_name}_epoch_{epoch}.pt')

    # Save final model
    final_checkpoint = {
        'epoch': epoch,  # Use actual last epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'config': cfg
    }

    if scheduler is not None:
        final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        final_checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(final_checkpoint, checkpoint_dir / f'{dataset_name}_final.pt')

    # from huggingface_hub import HfApi
    # api = HfApi(token="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")
    # api.upload_file(
    #     path_or_fileobj=checkpoint_dir / f'{dataset_name}_best.pt',
    #     path_in_repo=f"new/{dataset_name}_best.pt",
    #     repo_id="toanhac/test",
    #     repo_type="model"
    # )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
