import pytest

from pcnn4.config import BOARD_CELLS
from pcnn4.encoding import (EncodingError, cell_index, grids_from_order,
                            grids_to_tensors, mirror_sample,
                            replay_original_placements)


def test_cell_index_orientation():
    # y=0 is the bottom row, string row 0 is the top
    assert cell_index(0, 0) == 30
    assert cell_index(3, 0) == 0
    assert cell_index(3, 9) == 9
    with pytest.raises(EncodingError):
        cell_index(4, 0)


def _column_i_placements():
    """Ten vertical I pieces, one per column: a full synthetic 4L PC."""
    return [('I', frozenset((y, x) for y in range(4))) for x in range(10)]


def test_grids_full_pc():
    placements = _column_i_placements()
    color, ordinal = grids_from_order(placements, list(range(10)))
    assert color == 'I' * 40
    # each column carries its placement ordinal (1-based), in every row
    for r in range(4):
        assert ordinal[r * 10:(r + 1) * 10] == list(range(1, 11))


def test_grids_partial_and_overlap():
    placements = _column_i_placements()
    color, ordinal = grids_from_order(placements, [3, 0])
    assert color.count('I') == 8 and color.count('N') == 32
    assert ordinal[0] == 2 and ordinal[3] == 1 and ordinal[5] == 0
    with pytest.raises(EncodingError):
        grids_from_order(placements, [0, 0])  # same piece twice overlaps


def _delta(cells):
    """cells: [(y, x)] with y=0 bottom -> 40-float delta list."""
    d = [0.0] * BOARD_CELLS
    for y, x in cells:
        d[(3 - y) * 10 + x] = 1.0
    return d


def test_replay_placements_no_clear():
    steps = [
        {'placed_piece': 'I', 'delta': _delta([(0, x) for x in range(4)])},
        {'placed_piece': 'O',
         'delta': _delta([(0, 4), (0, 5), (1, 4), (1, 5)])},
    ]
    placements = replay_original_placements(steps)
    assert placements[0] == ('I', frozenset((0, x) for x in range(4)))
    assert placements[1] == ('O', frozenset([(0, 4), (0, 5), (1, 4), (1, 5)]))


def test_replay_placements_across_clear():
    # fill the bottom row with I,I + an O reaching into row 1, then place a
    # piece after the clear: its current-space cells must map up one row
    steps = [
        {'placed_piece': 'I', 'delta': _delta([(0, x) for x in range(4)])},
        {'placed_piece': 'I', 'delta': _delta([(0, x) for x in range(4, 8)])},
        {'placed_piece': 'O',
         'delta': _delta([(0, 8), (0, 9), (1, 8), (1, 9)])},
        # bottom row (orig y=0) is now cleared; current y=0 is orig y=1
        {'placed_piece': 'O',
         'delta': _delta([(0, 0), (0, 1), (1, 0), (1, 1)])},
    ]
    placements = replay_original_placements(steps)
    assert placements[3] == ('O', frozenset([(1, 0), (1, 1), (2, 0), (2, 1)]))


def test_replay_rejects_bad_delta():
    with pytest.raises(EncodingError):
        replay_original_placements(
            [{'placed_piece': 'I', 'delta': [0.0] * 40}])


def test_mirror_involution_and_swap():
    placements = [('L', frozenset([(0, 0), (0, 1), (0, 2), (1, 0)])),
                  ('S', frozenset([(0, 8), (0, 9), (1, 7), (1, 8)]))]
    color, ordinal = grids_from_order(placements, [0, 1])
    m_color, m_ordinal = mirror_sample(color, ordinal)
    # L became J, S became Z, positions flipped
    assert set(m_color) == {'J', 'Z', 'N'}
    assert m_color[cell_index(0, 9)] == 'J'
    assert m_ordinal[cell_index(0, 9)] == 1
    # mirroring twice is the identity
    assert mirror_sample(m_color, m_ordinal) == (color, ordinal)


def test_grids_to_tensors():
    placements = _column_i_placements()
    color, ordinal = grids_from_order(placements, list(range(10)))
    ct, ot = grids_to_tensors(color, ordinal)
    assert ct.shape == (4, 10) and ot.shape == (4, 10)
    assert int(ct.min()) == int(ct.max()) == 2  # 'I' index
    assert int(ot[0, 0]) == 1 and int(ot[0, 9]) == 10
