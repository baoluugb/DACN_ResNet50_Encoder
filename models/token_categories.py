"""Build token categories from dictionary

This module analyzes the dictionary.txt and creates category mappings
for auxiliary task supervision.
"""

from pathlib import Path
from typing import Dict, Set


def build_token_categories_from_dict(dict_path: str) -> Dict[str, Set[str]]:
    """Analyze dictionary and categorize tokens
    
    Args:
        dict_path: Path to dictionary.txt
        
    Returns:
        Dictionary with categories:
            'digit': set of digit tokens
            'letter': set of letter tokens  
            'greek': set of Greek letters
            'operator': set of operators
            'function': set of functions (sin, cos, log, etc.)
            'delimiter': set of delimiters/brackets
            'relation': set of relations (<, >, \\leq, etc.)
            'command': set of LaTeX commands
            'special': set of special tokens
    """
    dict_path = Path(dict_path)
    
    # Read all tokens
    tokens = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    
    # Initialize categories
    categories = {
        'digit': set(),
        'letter': set(),
        'greek': set(),
        'operator': set(),
        'function': set(),
        'delimiter': set(),
        'relation': set(),
        'command': set(),
        'special': set()
    }
    
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
    
    # Operators
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
    
    # Special symbols
    special_symbols = {
        '!', ',', '.', ':', ';', '?',
        '\\ldots', '\\cdots', '\\vdots', '\\ddots',
        '\\infty', '\\exists', '\\forall', '\\prime',
        '\\emptyset', '\\varnothing'
    }
    
    # Categorize each token from dictionary
    for token in tokens:
        if token.isdigit() or (len(token) == 1 and token in '0123456789'):
            categories['digit'].add(token)
        elif len(token) == 1 and token.isalpha():
            categories['letter'].add(token)
        elif token in greek_letters:
            categories['greek'].add(token)
        elif token in functions:
            categories['function'].add(token)
        elif token in operators:
            categories['operator'].add(token)
        elif token in relations:
            categories['relation'].add(token)
        elif token in delimiters:
            categories['delimiter'].add(token)
        elif token in special_symbols:
            categories['special'].add(token)
        elif token.startswith('\\'):
            # Other LaTeX commands
            categories['command'].add(token)
        else:
            # Fallback to special
            categories['special'].add(token)
    
    return categories


def create_type_mapping_from_categories(
    categories: Dict[str, Set[str]],
    token_to_id: Dict[str, int]
) -> Dict[int, int]:
    """Create mapping from token ID to type class
    
    Args:
        categories: Token categories from build_token_categories_from_dict
        token_to_id: Vocabulary mapping
        
    Returns:
        Dictionary mapping token_id -> type_class
            0: IGNORE (reserved for special tokens)
            1: digit
            2: letter  
            3: greek
            4: operator
            5: function
            6: delimiter
            7: relation
            8: command/special
    """
    id_to_type = {}
    
    for token, tid in token_to_id.items():
        if token in categories['digit']:
            id_to_type[tid] = 1
        elif token in categories['letter']:
            id_to_type[tid] = 2
        elif token in categories['greek']:
            id_to_type[tid] = 3
        elif token in categories['operator']:
            id_to_type[tid] = 4
        elif token in categories['function']:
            id_to_type[tid] = 5
        elif token in categories['delimiter']:
            id_to_type[tid] = 6
        elif token in categories['relation']:
            id_to_type[tid] = 7
        else:
            # Command or special
            id_to_type[tid] = 8
    
    return id_to_type


if __name__ == '__main__':
    # Test with CROHME dictionary
    categories = build_token_categories_from_dict('data/CROHME/dictionary.txt')
    
    print("Token Categories from Dictionary:")
    print("="*60)
    for cat_name, tokens in categories.items():
        print(f"\n{cat_name.upper()}: {len(tokens)} tokens")
        print(f"  {sorted(tokens)[:10]}")  # Show first 10
