"""HMER Models Package

Contains all model components for Handwritten Mathematical Expression Recognition.
"""

from .model import HMERWithAuxiliary
from .encoder import Encoder
from .latex_decoder import LatexDecoderWithAux
from .auxiliary_targets import build_type_targets, build_depth_targets, build_relation_targets
from .losses import compute_main_loss, compute_aux_ce_loss, compute_coverage_loss, combine_losses

__all__ = [
    'HMERWithAuxiliary',
    'Encoder',
    'LatexDecoderWithAux',
    'build_type_targets',
    'build_depth_targets',
    'build_relation_targets',
    'compute_main_loss',
    'compute_aux_ce_loss',
    'compute_coverage_loss',
    'combine_losses'
]
