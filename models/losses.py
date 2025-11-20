"""Loss functions for HMER training with auxiliary tasks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def compute_main_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """Compute cross-entropy loss for main token prediction
    
    Args:
        logits: (B, T, vocab_size) predicted logits
        targets: (B, T) target token IDs
        pad_id: ID of padding token to ignore
        label_smoothing: label smoothing factor (0.0 = no smoothing)
        
    Returns:
        Scalar loss value
    """
    B, T, V = logits.shape
    
    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, V)  # (B*T, V)
    targets_flat = targets.reshape(-1)    # (B*T,)
    
    # Compute cross-entropy with label smoothing
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    
    return loss


def compute_aux_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0
) -> torch.Tensor:
    """Compute cross-entropy for auxiliary classification tasks
    
    Args:
        logits: (B, T, num_classes) predicted logits
        targets: (B, T) target class IDs
        ignore_index: class ID to ignore (typically 0 for IGNORE label)
        
    Returns:
        Scalar loss value
    """
    B, T, C = logits.shape
    
    # Flatten
    logits_flat = logits.reshape(-1, C)  # (B*T, C)
    targets_flat = targets.reshape(-1)    # (B*T,)
    
    # Cross-entropy
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


def compute_coverage_loss(
    coverage: torch.Tensor,
    coverage_weight_mask: Optional[torch.Tensor] = None,
    min_coverage: float = 0.1
) -> torch.Tensor:
    """Compute coverage regularization loss
    
    Penalizes encoder positions that are never attended to,
    encouraging the decoder to cover all visual features.
    
    Args:
        coverage: (B, N_enc) or (B, T_dec, N_enc) attention statistics
            Typically sum or mean of cross-attention weights over time
        coverage_weight_mask: (B, N_enc) optional mask for valid positions
        min_coverage: minimum attention each position should receive
        
    Returns:
        Scalar loss value
    """
    # If coverage is 3D, sum over decoder time dimension
    if coverage.dim() == 3:
        coverage = coverage.sum(dim=1)  # (B, N_enc)
    
    # Compute under-coverage penalty
    # ReLU(min_coverage - actual_coverage)
    under_coverage = F.relu(min_coverage - coverage)
    
    # Apply mask if provided
    if coverage_weight_mask is not None:
        under_coverage = under_coverage * coverage_weight_mask
        # Normalize by number of valid positions
        loss = under_coverage.sum() / coverage_weight_mask.sum().clamp(min=1.0)
    else:
        loss = under_coverage.mean()
    
    return loss


def combine_losses(
    main_loss: torch.Tensor,
    aux_losses_dict: Dict[str, torch.Tensor],
    weights_dict: Dict[str, float]
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Combine main and auxiliary losses with weights
    
    Args:
        main_loss: main token prediction loss
        aux_losses_dict: dict of auxiliary losses
            e.g., {"type": type_loss, "depth": depth_loss, "rel": rel_loss, "coverage": cov_loss}
        weights_dict: dict of loss weights
            e.g., {"type": 0.1, "depth": 0.1, "rel": 0.1, "coverage": 0.01}
            
    Returns:
        total_loss: combined weighted loss
        loss_components: dict of individual loss values (detached) for logging
    """
    total_loss = main_loss
    
    # Store individual components for logging
    loss_components = {
        "main": main_loss.item()
    }
    
    # Add auxiliary losses
    for key, aux_loss in aux_losses_dict.items():
        if aux_loss is not None and key in weights_dict:
            weight = weights_dict[key]
            weighted_loss = weight * aux_loss
            total_loss = total_loss + weighted_loss
            
            # Log both weighted and unweighted
            loss_components[key] = aux_loss.item()
            loss_components[f"{key}_weighted"] = weighted_loss.item()
    
    return total_loss, loss_components


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing
    
    Alternative implementation if not using F.cross_entropy's label_smoothing parameter
    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) or (B, T, C)
            targets: (N,) or (B, T)
        """
        if logits.dim() > 2:
            # Flatten if needed
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (logits.size(-1) - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            # Mask out ignore_index
            mask = (targets == self.ignore_index).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)
        
        # KL divergence
        loss = -(true_dist * log_probs).sum(dim=-1)
        
        # Average over non-ignored positions
        mask = (targets != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        
        return loss
