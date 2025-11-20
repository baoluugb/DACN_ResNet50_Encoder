"""Auxiliary target generation for HMER training

Functions to build auxiliary labels directly from LaTeX token sequences:
- Syntax type classification (digits, operators, commands, etc.) - DICTIONARY-BASED
- Structural depth estimation (nesting level)
- Local relation type (superscript, subscript, fraction, etc.)
"""

import torch
from typing import Dict, Optional


# Global cache for token category mapping
_TOKEN_TYPE_CACHE = None


def _get_token_type_mapping(id_to_token: Dict[int, str], token_to_id: Dict[str, int]) -> Dict[int, int]:
    """Get or create token ID to type class mapping based on actual vocabulary
    
    Returns:
        Dictionary mapping token_id -> type_class
            0: IGNORE (reserved for special tokens PAD/BOS/EOS)
            1: digit (0-9)
            2: letter (a-z, A-Z)
            3: greek (\\alpha, \\beta, etc.)
            4: operator (+, -, \\frac, \\sqrt, \\int, etc.)
            5: function (\\sin, \\cos, \\log, etc.)
            6: delimiter ((, ), [, ], {, })
            7: relation (<, >, \\leq, \\geq, etc.)
            8: command/special (other LaTeX commands and symbols)
    """
    global _TOKEN_TYPE_CACHE
    
    if _TOKEN_TYPE_CACHE is not None:
        return _TOKEN_TYPE_CACHE
    
    # Define categories based on token characteristics
    id_to_type = {}
    
    # Greek letters
    greek_letters = {
        '\\alpha', '\\beta', '\\gamma', '\\delta', '\\Delta', '\\epsilon', 
        '\\zeta', '\\eta', '\\theta', '\\Theta', '\\iota', '\\kappa', 
        '\\lambda', '\\Lambda', '\\mu', '\\nu', '\\xi', '\\Xi',
        '\\pi', '\\Pi', '\\rho', '\\sigma', '\\Sigma', '\\tau',
        '\\upsilon', '\\Upsilon', '\\phi', '\\Phi', '\\chi', '\\psi', 
        '\\Psi', '\\omega', '\\Omega'
    }
    
    # Functions
    functions = {
        '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
        '\\log', '\\ln', '\\exp', '\\lim', '\\limits',
        '\\arcsin', '\\arccos', '\\arctan',
        '\\sinh', '\\cosh', '\\tanh'
    }
    
    # Operators (including structural commands)
    operators = {
        '+', '-', '*', '/', '=', '^', '_',
        '\\times', '\\cdot', '\\div', '\\pm', '\\mp',
        '\\frac', '\\sqrt', '\\sum', '\\int', '\\prod',
        '\\partial', '\\nabla'
    }
    
    # Relations
    relations = {
        '<', '>', '\\leq', '\\geq', '\\neq', '\\equiv',
        '\\approx', '\\sim', '\\simeq', '\\cong',
        '\\in', '\\notin', '\\subset', '\\supset',
        '\\subseteq', '\\supseteq', '\\rightarrow', '\\leftarrow',
        '\\Rightarrow', '\\Leftarrow', '\\leftrightarrow'
    }
    
    # Delimiters
    delimiters = {
        '(', ')', '[', ']', '{', '}', '|',
        '\\{', '\\}', '\\lbrack', '\\rbrack',
        '\\lparen', '\\rparen', '\\langle', '\\rangle'
    }
    
    # Categorize each token in vocabulary
    for token, tid in token_to_id.items():
        # Skip special tokens (will be set to 0 during target building)
        if token in ['<pad>', '<s>', '</s>', '<unk>']:
            continue
            
        if token.isdigit() or (len(token) == 1 and token in '0123456789'):
            id_to_type[tid] = 1  # digit
        elif len(token) == 1 and token.isalpha():
            id_to_type[tid] = 2  # letter
        elif token in greek_letters:
            id_to_type[tid] = 3  # greek
        elif token in operators:
            id_to_type[tid] = 4  # operator
        elif token in functions:
            id_to_type[tid] = 5  # function
        elif token in delimiters:
            id_to_type[tid] = 6  # delimiter
        elif token in relations:
            id_to_type[tid] = 7  # relation
        else:
            # Other commands and special symbols
            id_to_type[tid] = 8  # command/special
    
    # Cache for future use
    _TOKEN_TYPE_CACHE = id_to_type
    
    return id_to_type


def build_type_targets(
    token_ids: torch.Tensor,
    id_to_token: Dict[int, str],
    pad_id: int,
    bos_id: int,
    eos_id: int,
    token_to_id: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """Build syntax type labels for each token position using vocabulary-based categorization
    
    Args:
        token_ids: (B, T) LaTeX token IDs
        id_to_token: mapping from id to token string
        pad_id, bos_id, eos_id: special token IDs
        token_to_id: (optional) reverse mapping for building type cache
        
    Returns:
        (B, T) long tensor with type labels:
            0: IGNORE (for PAD/BOS/EOS)
            1: Digit (0-9)
            2: Letter (a-z, A-Z)
            3: Greek (\\alpha, \\beta, etc.)
            4: Operator (+, -, \\frac, \\sqrt, \\int, etc.)
            5: Function (\\sin, \\cos, \\log, etc.)
            6: Delimiter ((, ), [, ], {, })
            7: Relation (<, >, \\leq, \\geq, etc.)
            8: Command/Special (other symbols)
    """
    B, T = token_ids.shape
    type_targets = torch.zeros(B, T, dtype=torch.long)
    
    # Build type mapping if needed
    if token_to_id is None:
        token_to_id = {v: k for k, v in id_to_token.items()}
    
    id_to_type = _get_token_type_mapping(id_to_token, token_to_id)
    
    for b in range(B):
        for t in range(T):
            tid = token_ids[b, t].item()
            
            # Ignore special tokens
            if tid in [pad_id, bos_id, eos_id]:
                type_targets[b, t] = 0
                continue
            
            # Use precomputed type mapping
            type_targets[b, t] = id_to_type.get(tid, 8)  # Default to 8 (command/special)
    
    return type_targets


def build_depth_targets(
    token_ids: torch.Tensor,
    id_to_token: Dict[int, str],
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_depth: int = 10
) -> torch.Tensor:
    """Build structural depth labels using stack-based parsing
    
    Depth increases when entering nested structures like superscripts,
    subscripts, fractions, etc.
    
    Args:
        token_ids: (B, T) LaTeX token IDs
        id_to_token: mapping from id to token string
        pad_id, bos_id, eos_id: special token IDs
        max_depth: maximum depth to track (clip at this value)
        
    Returns:
        (B, T) long tensor with depth labels [0, max_depth]
            0 for PAD/BOS/EOS and top-level tokens
    """
    B, T = token_ids.shape
    depth_targets = torch.zeros(B, T, dtype=torch.long)
    
    # Tokens that increase depth when followed by {
    depth_increase_tokens = {"^", "_", "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\prod"}
    
    for b in range(B):
        depth = 0
        depth_stack = []  # Track depth changes at each brace
        
        for t in range(T):
            tid = token_ids[b, t].item()
            
            # Special tokens get depth 0
            if tid in [pad_id, bos_id, eos_id]:
                depth_targets[b, t] = 0
                continue
            
            token = id_to_token.get(tid, "")
            
            # Assign current depth (clip to max_depth)
            depth_targets[b, t] = min(depth, max_depth)
            
            # Update depth based on token
            if token == "{":
                # Check previous token to see if this opens a nested structure
                if t > 0:
                    prev_tid = token_ids[b, t-1].item()
                    prev_token = id_to_token.get(prev_tid, "")
                    if prev_token in depth_increase_tokens:
                        depth += 1
                        depth_stack.append(True)  # Mark this brace as depth-increasing
                    else:
                        depth_stack.append(False)  # Regular grouping brace
                else:
                    depth_stack.append(False)
            elif token == "}":
                # Close brace - decrease depth if it was a depth-increasing brace
                if depth_stack:
                    was_depth_increase = depth_stack.pop()
                    if was_depth_increase and depth > 0:
                        depth -= 1
    
    return depth_targets


def build_relation_targets(
    token_ids: torch.Tensor,
    id_to_token: Dict[int, str],
    pad_id: int,
    bos_id: int,
    eos_id: int
) -> torch.Tensor:
    """Build local relation type labels based on syntactic context
    
    Args:
        token_ids: (B, T) LaTeX token IDs
        id_to_token: mapping from id to token string
        pad_id, bos_id, eos_id: special token IDs
        
    Returns:
        (B, T) long tensor with relation labels:
            0: IGNORE (PAD/BOS/EOS)
            1: NORMAL (baseline position)
            2: SUPERSCRIPT
            3: SUBSCRIPT
            4: NUMERATOR (fraction)
            5: DENOMINATOR (fraction)
            6: ARGUMENT (of function/command)
    """
    B, T = token_ids.shape
    rel_targets = torch.ones(B, T, dtype=torch.long)  # Default: NORMAL
    
    for b in range(B):
        context_stack = []  # Stack to track nested contexts
        
        for t in range(T):
            tid = token_ids[b, t].item()
            
            # Special tokens get label 0 (IGNORE)
            if tid in [pad_id, bos_id, eos_id]:
                rel_targets[b, t] = 0
                continue
            
            token = id_to_token.get(tid, "")
            
            # Determine current context
            if context_stack:
                current_context = context_stack[-1]
                rel_targets[b, t] = current_context
            else:
                rel_targets[b, t] = 1  # NORMAL
            
            # Update context based on token
            if token == "^":
                # Next token (or group) is superscript
                # Look ahead to see if next is {
                if t + 1 < T:
                    next_tid = token_ids[b, t+1].item()
                    next_token = id_to_token.get(next_tid, "")
                    if next_token == "{":
                        context_stack.append(2)  # SUPERSCRIPT context
                    else:
                        # Single token superscript - mark next position
                        if t + 1 < T:
                            rel_targets[b, t+1] = 2
            
            elif token == "_":
                # Next token (or group) is subscript
                if t + 1 < T:
                    next_tid = token_ids[b, t+1].item()
                    next_token = id_to_token.get(next_tid, "")
                    if next_token == "{":
                        context_stack.append(3)  # SUBSCRIPT context
                    else:
                        # Single token subscript
                        if t + 1 < T:
                            rel_targets[b, t+1] = 3
            
            elif token == "\\frac":
                # Next two groups are numerator and denominator
                # This is simplified - assumes \\frac { num } { denom }
                if t + 1 < T:
                    next_tid = token_ids[b, t+1].item()
                    next_token = id_to_token.get(next_tid, "")
                    if next_token == "{":
                        context_stack.append(4)  # NUMERATOR context
            
            elif token == "{":
                # Opening brace - context already pushed by previous token
                pass
            
            elif token == "}":
                # Closing brace - pop context
                if context_stack:
                    popped = context_stack.pop()
                    # If we just closed numerator, next group is denominator
                    if popped == 4 and t + 1 < T:  # Just closed numerator
                        next_tid = token_ids[b, t+1].item()
                        next_token = id_to_token.get(next_tid, "")
                        if next_token == "{":
                            context_stack.append(5)  # DENOMINATOR context
            
            # Commands with arguments
            elif token.startswith("\\") and token not in ["\\frac", "\\sqrt"]:
                # Generic command - next group is ARGUMENT
                if t + 1 < T:
                    next_tid = token_ids[b, t+1].item()
                    next_token = id_to_token.get(next_tid, "")
                    if next_token == "{":
                        context_stack.append(6)  # ARGUMENT context
    
    return rel_targets
