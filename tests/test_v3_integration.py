"""Sanity checks against real v3 data and the v3/SDK toolchain.

These need PC-NN-v3 (and the Tetris SDK it bootstraps) on disk; they do NOT
invoke sfinder. Skipped when the corpus is missing.
"""
import json
import os

import pytest

from pcnn4.config import PATHS, PIECE_TYPES
from pcnn4.encoding import EncodingError, replay_original_placements

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PATHS['v3_parsed_weighted']),
    reason='PC-NN-v3 parsed_weighted corpus not found')


def _first_full_sequence():
    files = sorted(os.listdir(PATHS['v3_parsed_weighted']))
    for fn in files:
        with open(os.path.join(PATHS['v3_parsed_weighted'], fn)) as f:
            data = json.load(f)
        for seq in data.get('sequences', []):
            steps = seq.get('steps', [])
            if len(steps) == 10 and all(
                    st.get('placed_piece') in PIECE_TYPES for st in steps):
                return seq
    return None


def test_replay_placements_on_real_sequence():
    seq = _first_full_sequence()
    assert seq is not None, 'no full 4L PC in the corpus?'
    placements = replay_original_placements(seq['steps'])
    assert len(placements) == 10
    cells = [c for _, cs in placements for c in cs]
    assert len(cells) == 40
    assert len(set(cells)) == 40          # a perfect tiling, no overlap
    ys = {y for y, _ in cells}
    assert ys == {0, 1, 2, 3}             # exactly the 4-row window


def test_stream_reconstruction_and_glue_roundtrip():
    from pcnn4 import v3bridge  # noqa: F401
    from build_weights import infer_round, reconstruct_stream

    seq = _first_full_sequence()
    variants = reconstruct_stream(seq)
    assert variants, 'stream reconstruction failed on real data'
    pc, stream = infer_round(variants, 1, seq.get('pc_number', 1))
    assert pc is not None
    assert len(stream) == 11
    # the played pieces must be obtainable from the stream (save = 1 piece)
    from pc_value import end_of_pc_save
    placed = [st['placed_piece'] for st in seq['steps']]
    assert end_of_pc_save(stream, placed) is not None
