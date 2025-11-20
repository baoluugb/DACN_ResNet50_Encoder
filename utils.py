"""Utility functions for vocabulary building and model analysis"""

from pathlib import Path
from typing import Dict, Any


def build_vocab_from_dict(dict_path: Path) -> Dict[str, Any]:
    """Build LaTeX vocabulary from dictionary file
    
    Args:
        dict_path: Path to dictionary.txt file
        
    Returns:
        Dictionary containing:
            - token_to_id: mapping from token string to id
            - id_to_token: mapping from id to token string
            - special: dict with special tokens (pad, bos, eos, unk)
    """
    dict_path = Path(dict_path)
    
    # Define special tokens
    special = {
        "pad": "<pad>",
        "bos": "<s>",
        "eos": "</s>",
        "unk": "<unk>"
    }
    
    # Start with special tokens
    token_to_id = {
        special["pad"]: 0,
        special["bos"]: 1,
        special["eos"]: 2,
        special["unk"]: 3
    }
    
    # Read dictionary and add tokens
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token and token not in token_to_id:
                token_to_id[token] = len(token_to_id)
    
    # Create reverse mapping
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "special": special
    }


def print_trainable_parameters(model):
    """Print trainable parameter statistics for the model (compact format)
    
    Args:
        model: PyTorch model
    """
    trainable_params = 0
    all_params = 0
    
    # Collect stats by module
    module_stats = {}
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        
        # Extract module name (first part before .)
        module_name = name.split('.')[0] if '.' in name else name
        
        if module_name not in module_stats:
            module_stats[module_name] = {
                'trainable': 0,
                'frozen': 0,
                'total': 0
            }
        
        module_stats[module_name]['total'] += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            module_stats[module_name]['trainable'] += param.numel()
        else:
            module_stats[module_name]['frozen'] += param.numel()
    
    # Print compact summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    print(f"{'Module':<25} {'Trainable':>15} {'Frozen':>15} {'Total':>12}")
    print("-"*70)
    
    for module_name in sorted(module_stats.keys()):
        stats = module_stats[module_name]
        trainable = stats['trainable']
        frozen = stats['frozen']
        total = stats['total']
        
        # Format with M/K suffix
        t_str = f"{trainable/1e6:.2f}M" if trainable > 1e6 else f"{trainable/1e3:.1f}K"
        f_str = f"{frozen/1e6:.2f}M" if frozen > 1e6 else f"{frozen/1e3:.1f}K"
        tot_str = f"{total/1e6:.2f}M" if total > 1e6 else f"{total/1e3:.1f}K"
        
        # Color indicator
        indicator = "✓" if trainable > 0 else "✗"
        
        print(f"{indicator} {module_name:<23} {t_str:>14} {f_str:>14} {tot_str:>11}")
    
    print("-"*70)
    
    # Format totals
    t_total = f"{trainable_params/1e6:.2f}M" if trainable_params > 1e6 else f"{trainable_params/1e3:.1f}K"
    a_total = f"{all_params/1e6:.2f}M" if all_params > 1e6 else f"{all_params/1e3:.1f}K"
    
    print(f"{'TOTAL':<25} {t_total:>14} {'':<15} {a_total:>11}")
    print(f"{'Trainable Ratio':<25} {100 * trainable_params / all_params:>13.2f}%")
    print("="*70 + "\n")


def tokens_to_string(token_ids, id_to_token, special_tokens):
    """Convert token IDs to readable string
    
    Args:
        token_ids: list or tensor of token IDs
        id_to_token: mapping from id to token string
        special_tokens: dict with special token strings
        
    Returns:
        String representation of tokens
    """
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    tokens = []
    for tid in token_ids:
        token = id_to_token.get(tid, special_tokens["unk"])
        # Stop at EOS
        if token == special_tokens["eos"]:
            break
        # Skip BOS and PAD
        if token not in [special_tokens["bos"], special_tokens["pad"]]:
            tokens.append(token)
    
    return " ".join(tokens)
