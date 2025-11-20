"""Simple LaTeX Decoder with Auxiliary Task Heads

Standard transformer decoder architecture for all tasks:
- Shared transformer decoder layers
- Multiple prediction heads from the same features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer Decoder Layer: Self-Attention → Cross-Attention → FFN"""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model) input features
            memory: (B, N, d_model) encoder features
            self_attn_mask: (T, T) causal mask
            self_key_padding_mask: (B, T) padding mask
            memory_key_padding_mask: (B, N) encoder padding mask
            need_attn_weights: whether to return cross-attention weights
        """
        # Self-attention with residual and causal mask
        residual = x
        x_norm = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=False
        )
        x = residual + self.self_attn_dropout(x_attn)
        
        # Cross-attention with residual
        residual = x
        x_norm = self.cross_attn_norm(x)
        x_attn, attn_weights = self.cross_attn(
            x_norm, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_attn_weights,
            average_attn_weights=False
        )
        x = residual + self.cross_attn_dropout(x_attn)
        
        # Feed-forward with residual
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        
        return x, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class LatexDecoderWithAux(nn.Module):
    """Simple LaTeX decoder with auxiliary task heads
    
    Architecture:
    - Shared transformer decoder layers (all tasks use same layers)
    - Multiple prediction heads from the same features:
      * Main: LaTeX token prediction
      * Auxiliary: Syntax type, structural depth, relation type
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        vocab_size: int,
        max_len: int,
        num_soft_prompts: int = 0,
        num_type_classes: int = 9,
        num_depth_classes: int = 11,
        num_rel_classes: int = 7,
        enable_type_head: bool = True,
        enable_depth_head: bool = True,
        enable_rel_head: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_soft_prompts = num_soft_prompts
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Optional soft prompts
        if num_soft_prompts > 0:
            self.soft_prompts = nn.Parameter(torch.randn(1, num_soft_prompts, d_model))
            nn.init.normal_(self.soft_prompts, std=0.02)
        else:
            self.soft_prompts = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Shared transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Main prediction head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Auxiliary heads
        self.enable_type_head = enable_type_head
        self.enable_depth_head = enable_depth_head
        self.enable_rel_head = enable_rel_head
        
        if enable_type_head:
            self.type_head = nn.Linear(d_model, num_type_classes)
        
        if enable_depth_head:
            self.depth_head = nn.Linear(d_model, num_depth_classes)
        
        if enable_rel_head:
            self.rel_head = nn.Linear(d_model, num_rel_classes)
        
        # Initialize weights
        self._init_weights()
        
        self.pad_id = None
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        if self.enable_type_head:
            nn.init.xavier_uniform_(self.type_head.weight)
            nn.init.zeros_(self.type_head.bias)
        
        if self.enable_depth_head:
            nn.init.xavier_uniform_(self.depth_head.weight)
            nn.init.zeros_(self.depth_head.bias)
        
        if self.enable_rel_head:
            nn.init.xavier_uniform_(self.rel_head.weight)
            nn.init.zeros_(self.rel_head.bias)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        pad_id: int,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tgt_ids: (B, T) target token IDs
            memory: (B, N, d_model) encoder features
            pad_id: padding token ID
            memory_key_padding_mask: (B, N) encoder padding mask
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device
        
        # Token embeddings + positional encoding
        x = self.token_embed(tgt_ids)
        x = self.pos_embed(x)
        x = self.dropout(x)
        
        # Prepend soft prompts if enabled
        if self.soft_prompts is not None:
            prompts = self.soft_prompts.expand(B, -1, -1)
            x = torch.cat([prompts, x], dim=1)
            T_total = T + self.num_soft_prompts
        else:
            T_total = T
        
        # Create causal mask
        causal_mask = self._create_causal_mask(T_total, device)
        
        # Create key padding mask
        if self.soft_prompts is not None:
            prompt_mask = torch.zeros(B, self.num_soft_prompts, dtype=torch.bool, device=device)
            tgt_padding_mask = (tgt_ids == pad_id)
            key_padding_mask = torch.cat([prompt_mask, tgt_padding_mask], dim=1)
        else:
            key_padding_mask = (tgt_ids == pad_id)
        
        # Pass through transformer layers
        coverage_maps = []
        for layer in self.layers:
            x, attn_weights = layer(
                x,
                memory,
                self_attn_mask=causal_mask,
                self_key_padding_mask=key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                need_attn_weights=True
            )
            if attn_weights is not None:
                coverage_maps.append(attn_weights)
        
        # Remove soft prompts from output
        if self.soft_prompts is not None:
            x = x[:, self.num_soft_prompts:, :]
        
        # Main token prediction
        logits = self.lm_head(x)
        
        # Auxiliary predictions from same features
        type_logits = self.type_head(x) if self.enable_type_head else None
        depth_logits = self.depth_head(x) if self.enable_depth_head else None
        rel_logits = self.rel_head(x) if self.enable_rel_head else None
        
        # Coverage for loss
        coverage = None
        if coverage_maps:
            avg_attn = torch.stack(coverage_maps, dim=0).mean(dim=0)  # (B, H, T, N)
            avg_attn = avg_attn.mean(dim=1)  # (B, T, N)
            cumulative_coverage = torch.cumsum(avg_attn, dim=1)
            coverage = cumulative_coverage[:, -1, :]  # (B, N)
        
        out = {
            "logits": logits,
            "type_logits": type_logits,
            "depth_logits": depth_logits,
            "rel_logits": rel_logits,
            "coverage": coverage
        }

        if return_attn:
            # Return the list of attention maps (per-layer). Each element has shape (B, H, T, S)
            out["attn_maps"] = coverage_maps

        return out
