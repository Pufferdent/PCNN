"""PC-NN-v4 configuration.

v4 reframes the problem: instead of predicting per-step placements (v3), the
model scores a *finished solve* — the broken piece board (cleared lines kept
in place) plus the order every cell was filled in — and outputs a single
probability: "would the player have picked this solve, played in this order?".
"""
import os

PIECE_TYPES = ['T', 'I', 'L', 'J', 'S', 'Z', 'O']
MIRROR_MAP = {'L': 'J', 'J': 'L', 'S': 'Z', 'Z': 'S',
              'T': 'T', 'I': 'I', 'O': 'O'}

# Color vocabulary: index 0 = empty ('N'), then the seven pieces.
COLOR_CHARS = ['N'] + PIECE_TYPES
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_CHARS)}
NUM_COLORS = len(COLOR_CHARS)          # 8

BOARD_W = 10
BOARD_H = 4
BOARD_CELLS = BOARD_W * BOARD_H        # 40

MAX_ORDINAL = 10                       # pieces in a 4-line PC
ORDINAL_VOCAB = MAX_ORDINAL + 1        # index 0 = empty cell

MODEL_CONFIG = {
    'color_embed_dim': 8,
    'ordinal_embed_dim': 8,
    'conv_channels': [64, 128, 128],
    'conv_kernel': 3,
    'head_hidden': [256, 64],
    'dropout': 0.1,
}

# V*-relative sample weighting (see weighting.py). d = V*(solve) - V*(chosen).
WEIGHT_CONFIG = {
    'v_window': 1.0,       # consider solves with d > -v_window (smooth gate)
    'd_significant': 0.5,  # d at which a skipped alternative reaches max boost
    'max_boost': 10.0,     # negative-weight multiplier at d >= d_significant
    'pos_ratio': 1.0,      # total positive weight = ratio * total negative weight
    'pos_floor': 1.0,      # minimum weight per positive sample
    'order_weight_mode': 'split',  # 'split': divide a solve's weight across its
                                   # orders; 'replicate': every order keeps it
}

DATASET_CONFIG = {
    'mirror_prob': 0.5,
    'val_frac': 0.1,
}

TRAIN_CONFIG = {
    'epochs': 40,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'seed': 7,
}

BUILD_CONFIG = {
    'max_solutions': None,   # cap sfinder solutions per situation (logged)
    'max_orders': 64,        # cap playable orders per solve (logged)
    'sfinder_timeout': 900,
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V3_DEFAULT = os.path.normpath(os.path.join(_ROOT, '..', 'PC-NN-v3', 'pc-nn'))

PATHS = {
    'root': _ROOT,
    'data_dir': os.path.join(_ROOT, 'data'),
    'dataset': os.path.join(_ROOT, 'data', 'dataset.jsonl'),
    'dataset_meta': os.path.join(_ROOT, 'data', 'dataset_meta.json'),
    'sfinder_cache_dir': os.path.join(_ROOT, 'data', 'sfinder_cache'),
    'sfinder_output_dir': os.path.join(_ROOT, 'output', 'sfinder'),
    'checkpoint_dir': os.path.join(_ROOT, 'checkpoints'),
    'v3_pcnn_dir': os.environ.get('PCNN_V3_PATH', _V3_DEFAULT),
}
PATHS['v3_parsed_weighted'] = os.path.join(
    PATHS['v3_pcnn_dir'], 'data', 'parsed_weighted')
