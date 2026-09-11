"""Grid encodings for solve samples.

Conventions:
- Board strings are 40 chars, row-major, row 0 = TOP of the PC window
  (matches v3 board strings and PCReview's ``completed`` rows).
- Placement cells are (y, x) with y=0 the BOTTOM row, in *original*
  (pre-any-line-clear) coordinates — the convention of v3's
  ``solution_order.glue``. The broken piece board is exactly the original-
  coordinate tiling: cleared lines stay where they were.
"""
import torch

from pcnn4.config import (BOARD_W, BOARD_H, BOARD_CELLS, COLOR_TO_IDX,
                          MAX_ORDINAL, MIRROR_MAP, PIECE_TYPES)


class EncodingError(ValueError):
    pass


def cell_index(y, x):
    """Original-space (y, x), y=0 bottom -> index into the 40-char string."""
    if not (0 <= y < BOARD_H and 0 <= x < BOARD_W):
        raise EncodingError("cell (%d, %d) outside the PC window" % (y, x))
    return (BOARD_H - 1 - y) * BOARD_W + x


def grids_from_order(placements, order_ks):
    """Broken-board grids for one play order of a solve.

    placements: [(piece_char, frozenset((y, x)))] in original coordinates.
    order_ks: placement indices in the order they are played.
    Returns (color_str 40 chars, ordinal list of 40 ints; 0 = empty,
    t+1 = filled by the t-th placed piece).
    """
    if len(order_ks) > MAX_ORDINAL:
        raise EncodingError("more than %d placements" % MAX_ORDINAL)
    color = ['N'] * BOARD_CELLS
    ordinal = [0] * BOARD_CELLS
    for t, k in enumerate(order_ks):
        piece, cells = placements[k]
        if piece not in PIECE_TYPES:
            raise EncodingError("unknown piece %r" % piece)
        for (y, x) in cells:
            i = cell_index(y, x)
            if color[i] != 'N':
                raise EncodingError("overlap at cell %d" % i)
            color[i] = piece
            ordinal[i] = t + 1
    return ''.join(color), ordinal


def replay_original_placements(steps):
    """Recover the player's placements in original coordinates from parsed
    replay steps (v3 ``prep_jstris`` schema: each step has ``placed_piece``
    and a 40-float pre-clear ``delta`` in current space).

    Returns an ordered list [(piece_char, frozenset((y, x)))]. Mirrors the
    row bookkeeping of ``solution_order.glue``: ``alive[cur_y] -> orig_y``.
    """
    alive = list(range(BOARD_H + 4))
    board = {}          # current-space {(y, x): piece}
    placements = []
    for st in steps:
        piece = st.get('placed_piece')
        if piece not in PIECE_TYPES:
            raise EncodingError("bad placed_piece %r" % piece)
        cells_cur = set()
        for i, v in enumerate(st['delta']):
            if v > 0.5:
                cells_cur.add((BOARD_H - 1 - (i // BOARD_W), i % BOARD_W))
        if len(cells_cur) != 4:
            raise EncodingError("delta has %d cells, expected 4"
                                % len(cells_cur))
        if cells_cur & set(board):
            raise EncodingError("delta overlaps existing cells")
        placements.append(
            (piece, frozenset((alive[y], x) for y, x in cells_cur)))
        for c in cells_cur:
            board[c] = piece
        full = sorted({y for y, _ in board
                       if all((y, x) in board for x in range(BOARD_W))},
                      reverse=True)
        for y in full:
            del alive[y]
            board = {((yy - 1, x) if yy > y else (yy, x)): ch
                     for (yy, x), ch in board.items() if yy != y}
    return placements


def mirror_sample(color, ordinal):
    """Horizontal mirror of one sample: flip each row, swap L<->J and S<->Z."""
    m_color = []
    m_ordinal = []
    for r in range(BOARD_H):
        row = color[r * BOARD_W:(r + 1) * BOARD_W]
        m_color.extend(MIRROR_MAP.get(c, c) for c in reversed(row))
        m_ordinal.extend(reversed(ordinal[r * BOARD_W:(r + 1) * BOARD_W]))
    return ''.join(m_color), m_ordinal


def grids_to_tensors(color, ordinal):
    """(color_str, ordinal list) -> two LongTensors of shape (4, 10)."""
    try:
        color_idx = [COLOR_TO_IDX[c] for c in color]
    except KeyError as e:
        raise EncodingError("unknown color char %s" % e)
    if len(color) != BOARD_CELLS or len(ordinal) != BOARD_CELLS:
        raise EncodingError("grids must have %d cells" % BOARD_CELLS)
    return (torch.tensor(color_idx, dtype=torch.long).view(BOARD_H, BOARD_W),
            torch.tensor(ordinal, dtype=torch.long).view(BOARD_H, BOARD_W))
